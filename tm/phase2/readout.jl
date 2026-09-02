# Fix the probability readout: interpolate the softmax-of-votes with each model's own
# unigram backoff (the discrete-TM analogue of KN smoothing):
#     P(t|ctx) = (1-α)·softmax(votes/τ)_t  +  α·P_unigram(t)
# G_A backs off to the KNOWN author's unigram, G_ref to the reference's. This floors the
# degenerate near-zero probabilities that softmax-of-votes produces and that capped AUC.
#
#   julia -t auto phase2/readout.jl        # negk=32 G_A per pair, sweeps an α×τ grid in ONE pass
#
# Per-token vote vectors are computed once; the α×τ grid is cheap arithmetic on top, so this
# costs ~one training pass and reports AUC for every (α,τ) combo (α=0 == old readout).

import Pkg
Pkg.activate(joinpath(@__DIR__, "..", "hdc"); io = devnull)
using HDC, Random, Printf, Serialization
using Base.Threads: @threads, threadid, maxthreadid
include(joinpath(@__DIR__, "..", "Tsetlin.jl-main", "src", "Tsetlin.jl"))
const TM = Tsetlin
include(joinpath(@__DIR__, "..", "phase1", "tm_ext.jl"))

const D = 8192; const WINDOW = 10; const MINCOUNT = 25
const CLAUSES = 64; const T_ = 512; const S_ = 8192; const L_ = 4096; const LF_ = 4096
const STATES = 256; const INCLUDE = 128
const NEGK = parse(Int, get(ENV, "P2_NEGK", "32"))
const GA_EPOCHS = parse(Int, get(ENV, "P2_GAEPOCHS", "20"))
const POS = let p = get(ENV, "P2_POS", "sharp")       # exact per-trigram position by default
    p == "sharp" ? POS_SHARP : p == "graded" ? POS_GRADED : p == "bands" ? POS_BANDS : POS_NONE
end
const EPS = 0.5                                        # add-ε unigram smoothing
const ALPHAS = [parse(Float64, x) for x in split(get(ENV, "P2_ALPHAS", "0.0,0.02,0.05,0.1,0.2,0.4"), ",")]
const TEMPS  = [parse(Float64, x) for x in split(get(ENV, "P2_TEMPS", "4,8,16"), ",")]
const P1 = joinpath(@__DIR__, "..", "phase1")

readsents(p) = [String.(split(l, '\t')) for l in eachline(p) if !isempty(l)]
function build_classes(sents; min_count)
    cnt = Dict{String,Int}(); for s in sents, w in s; cnt[w] = get(cnt, w, 0) + 1; end
    vocab = vcat(["<UNK>", "<BOS>"], sort([w for (w, c) in cnt if c >= min_count]))
    vocab, Dict(w => i for (i, w) in enumerate(vocab))
end
@inline tokid(w2i, w) = get(w2i, w, 1)
function tminput(hv::Vector{UInt64})
    m = Memory{UInt64}(undef, length(hv)); copyto!(m, hv); TM.TMInput(m, Int64(D))
end
function pairs(sents, w2i, cb)
    nt = maxthreadid()
    # SAME seed for every thread's encoder -> identical position codes across threads (the
    # inter-trigram slot codes must be globally consistent; per-thread seeds corrupted them).
    encs = [Encoder(cb; stride = 1, pos = POS, maxwin = WINDOW, rng = MersenneTwister(1000)) for t in 1:nt]
    Xs = [TM.TMInput[] for _ in 1:nt]; Ys = [Int[] for _ in 1:nt]; BOS = w2i["<BOS>"]
    @threads :static for si in eachindex(sents)
        t = threadid(); enc = encs[t]; rng = Random.default_rng()
        seq = Int[BOS]; for w in sents[si]; push!(seq, tokid(w2i, w)); end
        for p in 2:length(seq)
            ctx = @view seq[max(1, p - WINDOW):p - 1]
            push!(Xs[t], tminput(encode_window(enc, ctx, rng))); push!(Ys[t], seq[p])
        end
    end
    reduce(vcat, Xs), reduce(vcat, Ys)
end
function unigram(Y, V)                                 # add-ε smoothed marginal over classes
    c = fill(EPS, V); for y in Y; c[y] += 1.0; end; c ./ sum(c)
end
function train_model(X, Y, allids, k, epochs)
    tm = TM.TMClassifier(X[1], allids, CLAUSES, T_, S_, L_, LF_; states_num = STATES, include_limit = INCLUDE)
    for _ in 1:epochs
        k <= 0 ? TM.train!(tm, X, Y; shuffle = true, index = false) : train_ns_epoch!(tm, X, Y, k)
    end
    tm
end
@inline function margins!(v, tm, x)                    # raw (pos-neg) per class, computed once/token
    @inbounds for i in 1:tm.classes_num
        pos, neg = TM.vote(tm, tm.clauses[i], x); v[i] = Float64(pos - neg)
    end
    v
end
# log[ (1-α)·softmax(v/τ)_tgt + α·puni_tgt ] from a raw margin vector v
@inline function logp_interp(v, tgt, τ, α, puni)
    m = -Inf; @inbounds for x in v; z = x / τ; z > m && (m = z); end
    Z = 0.0; @inbounds for x in v; Z += exp(x / τ - m); end
    pTM = exp(v[tgt] / τ - m) / Z
    log((1 - α) * pTM + α * puni[tgt])
end

function auc(scores, labels)
    order = sortperm(scores); ranks = similar(scores, Float64); i = 1
    while i <= length(order)
        j = i; while j < length(order) && scores[order[j+1]] == scores[order[i]]; j += 1; end
        r = (i + j) / 2; for t in i:j; ranks[order[t]] = r; end; i = j + 1
    end
    pos = labels .== 1; np = count(pos); nn = length(labels) - np
    (sum(ranks[pos]) - np * (np + 1) / 2) / (np * nn)
end

function main()
    a_train = readsents(joinpath(P1, "a_train.tsv"))
    bankfiles = sort(filter(f -> endswith(f, ".tsv"), readdir(joinpath(P1, "bank"), join = true)))
    bank_sents = [readsents(f) for f in bankfiles]
    vocab, w2i = build_classes(vcat(a_train, reduce(vcat, bank_sents)); min_count = MINCOUNT)
    V = length(vocab); allids = collect(1:V)
    cb = Codebook(D, vocab, w2i, ones(Int, V); sub = 3, rng = MersenneTwister(1))
    combos = [(α, τ) for α in ALPHAS, τ in TEMPS][:]
    nc = length(combos)
    @printf("classes=%d  negk=%d  grid=%dα×%dτ=%d combos\n", V, NEGK, length(ALPHAS), length(TEMPS), nc); flush(stdout)

    # reference: reuse cached model; recompute its unigram from the same interleave
    function interleave(bs, n); out = Vector{String}[]; i = 1
        while length(out) < n; added = false
            for b in bs; i <= length(b) || continue; push!(out, b[i]); added = true; length(out) >= n && break; end
            added || break; i += 1; end; out; end
    ref_sents = interleave(bank_sents, length(a_train))
    Xr, Yr = pairs(ref_sents, w2i, cb)
    puni_ref = unigram(Yr, V)
    refcache = joinpath(P1, "ref_$(POS).jls")           # cache keyed by position scheme
    if isfile(refcache)
        tm_ref = deserialize(refcache); @printf("reference loaded (%s)\n", POS)
    else
        dt = @elapsed tm_ref = train_model(Xr, Yr, allids, 0, 10)
        serialize(refcache, tm_ref); @printf("reference trained (%s) in %.0fs\n", POS, dt)
    end
    flush(stdout)

    man = [split(l, '\t') for l in Iterators.drop(eachline(joinpath(P1, "pairs.tsv")), 1)]
    labels = Int[]; lam = [Float64[] for _ in 1:nc]      # λ_G per combo per pair
    tall = @elapsed for (ip, row) in enumerate(man)
        pid, lab = row[1], parse(Int, row[2])
        known = readsents(joinpath(P1, "pairs", "$(pid)_known.tsv"))
        q = readsents(joinpath(P1, "pairs", "$(pid)_q.tsv"))
        (isempty(known) || isempty(q)) && continue
        Xk, Yk = pairs(known, w2i, cb)
        tm_a = train_model(Xk, Yk, allids, NEGK, GA_EPOCHS)
        puni_a = unigram(Yk, V)
        Xq, Yq = pairs(q, w2i, cb); n = length(Xq)
        part = [zeros(nc) for _ in 1:maxthreadid()]      # per-thread per-combo sums
        @threads for k in 1:n
            t = threadid(); vA = Vector{Float64}(undef, V); vR = Vector{Float64}(undef, V)
            margins!(vA, tm_a, Xq[k]); margins!(vR, tm_ref, Xq[k]); tgt = Yq[k]
            @inbounds for (ci, (α, τ)) in enumerate(combos)
                part[t][ci] += logp_interp(vA, tgt, τ, α, puni_a) - logp_interp(vR, tgt, τ, α, puni_ref)
            end
        end
        tot = reduce((a, b) -> a .+ b, part)
        push!(labels, lab); for ci in 1:nc; push!(lam[ci], tot[ci] / n); end
        if ip % 20 == 0; @printf("  %d/%d pairs\n", ip, length(man)); flush(stdout); end
    end

    y = labels
    println("\n=== AUC grid ($(length(y)) pairs, negk=$NEGK) ===")
    @printf("   α \\ τ   %s\n", join([@sprintf("τ=%-6.0f", τ) for τ in TEMPS], ""))
    best = (-1.0, 0.0, 0.0)
    for (ai, α) in enumerate(ALPHAS)
        row = @sprintf("  %5.2f   ", α)
        for (ti, τ) in enumerate(TEMPS)
            ci = findfirst(==(( α, τ)), combos)
            a = auc(lam[ci], y); row *= @sprintf("%-8.3f", a)
            a > best[1] && (best = (a, α, τ))
        end
        println(row)
    end
    @printf("\nbest: AUC=%.4f at α=%.2f τ=%.0f   (α=0 baseline == old softmax readout)\n", best...)
end

main()
