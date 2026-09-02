# Multi-resolution context encoding (KN backoff, in HDC form):
#   F_k = ⊕_{j=0}^{k-1} ρ^j(E[t_{last-j}])   -- sharp, order-preserving fingerprint of the
#         EXACT k-gram ending at the target (ρ = word-rotation; D=8192=128 words, clean perm)
#   context = bundle(F_1, …, F_N)            -- every order superposed => implicit backoff
# No slot-position codes: order == recency. Then the same α×τ readout grid (α=1.0 == the pure
# unigram baseline, 0.943) so we can see whether the TM context now beats unigram.
#
#   P2_ORDER=6 julia -t auto phase2/multires.jl

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
const ORDER = parse(Int, get(ENV, "P2_ORDER", "6"))       # max n-gram order N in the stack
const EPS = 0.5
const ALPHAS = [parse(Float64, x) for x in split(get(ENV, "P2_ALPHAS", "0.0,0.2,0.4,0.6,0.8,1.0"), ",")]
const TEMPS  = [parse(Float64, x) for x in split(get(ENV, "P2_TEMPS", "8,16,32"), ",")]
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

# multi-resolution fingerprint bundle of a causal context (oldest..newest in ctx)
function encode_multires(ctx, cb, N, acc, rng)
    sp = cb.sp; E = cb.E; W = sp.W; K = length(ctx)
    out = zeros(UInt64, W)
    K == 0 && return out
    maxk = min(N, K)
    fps = Vector{Vector{UInt64}}(undef, maxk)
    fp = zeros(UInt64, W)
    @inbounds for k in 1:maxk
        tok = ctx[K - (k - 1)]                    # k-th token counting from the newest
        rot = circshift(@view(E[:, tok]), k - 1)   # ρ^{k-1}: rotate by k-1 words (a bit-perm)
        @. fp = fp ⊻ rot                           # running XOR -> F_k = exact k-gram fingerprint
        fps[k] = copy(fp)
    end
    bundle!(sp, out, fps, acc, rng)                # superpose all orders (deterministic tie)
    out
end

function pairs(sents, w2i, cb)
    nt = maxthreadid()
    accs = [Vector{Int32}(undef, cb.sp.W * 64) for _ in 1:nt]
    Xs = [TM.TMInput[] for _ in 1:nt]; Ys = [Int[] for _ in 1:nt]; BOS = w2i["<BOS>"]
    @threads :static for si in eachindex(sents)
        t = threadid(); acc = accs[t]; rng = Random.default_rng()
        seq = Int[BOS]; for w in sents[si]; push!(seq, tokid(w2i, w)); end
        for p in 2:length(seq)
            ctx = @view seq[max(1, p - WINDOW):p - 1]
            push!(Xs[t], tminput(encode_multires(ctx, cb, ORDER, acc, rng))); push!(Ys[t], seq[p])
        end
    end
    reduce(vcat, Xs), reduce(vcat, Ys)
end
function unigram(Y, V); c = fill(EPS, V); for y in Y; c[y] += 1.0; end; c ./ sum(c); end
function train_model(X, Y, allids, k, epochs)
    tm = TM.TMClassifier(X[1], allids, CLAUSES, T_, S_, L_, LF_; states_num = STATES, include_limit = INCLUDE)
    for _ in 1:epochs
        k <= 0 ? TM.train!(tm, X, Y; shuffle = true, index = false) : train_ns_epoch!(tm, X, Y, k)
    end
    tm
end
@inline function margins!(v, tm, x)
    @inbounds for i in 1:tm.classes_num
        pos, neg = TM.vote(tm, tm.clauses[i], x); v[i] = Float64(pos - neg)
    end
    v
end
@inline function logp_interp(v, tgt, τ, α, puni)
    m = -Inf; @inbounds for x in v; z = x / τ; z > m && (m = z); end
    Z = 0.0; @inbounds for x in v; Z += exp(x / τ - m); end
    log((1 - α) * (exp(v[tgt] / τ - m) / Z) + α * puni[tgt])
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
function interleave(bs, n); out = Vector{String}[]; i = 1
    while length(out) < n; added = false
        for b in bs; i <= length(b) || continue; push!(out, b[i]); added = true; length(out) >= n && break; end
        added || break; i += 1; end; out; end

function main()
    a_train = readsents(joinpath(P1, "a_train.tsv"))
    bankfiles = sort(filter(f -> endswith(f, ".tsv"), readdir(joinpath(P1, "bank"), join = true)))
    bank_sents = [readsents(f) for f in bankfiles]
    vocab, w2i = build_classes(vcat(a_train, reduce(vcat, bank_sents)); min_count = MINCOUNT)
    V = length(vocab); allids = collect(1:V)
    cb = Codebook(D, vocab, w2i, ones(Int, V); sub = 3, rng = MersenneTwister(1))
    let acc = Vector{Int32}(undef, cb.sp.W * 64), rng = MersenneTwister(0)   # encoder self-check
        a = encode_multires([3, 4, 5, 6], cb, ORDER, acc, rng)
        b = encode_multires([3, 4, 5, 6], cb, ORDER, acc, rng)
        c = encode_multires([6, 5, 4, 3], cb, ORDER, acc, rng)              # reversed order
        @printf("encoder self-check: deterministic=%s  order-sensitive=%s\n", a == b, a != c); flush(stdout)
    end
    combos = [(α, τ) for α in ALPHAS, τ in TEMPS][:]; nc = length(combos)
    @printf("classes=%d  negk=%d  ORDER=%d  grid=%d combos\n", V, NEGK, ORDER, nc); flush(stdout)

    ref_sents = interleave(bank_sents, length(a_train))
    Xr, Yr = pairs(ref_sents, w2i, cb); puni_ref = unigram(Yr, V)
    refcache = joinpath(P1, "ref_multires_O$(ORDER).jls")
    if isfile(refcache); tm_ref = deserialize(refcache); println("ref loaded")
    else; dt = @elapsed tm_ref = train_model(Xr, Yr, allids, 0, 10); serialize(refcache, tm_ref); @printf("ref trained %.0fs\n", dt); end
    flush(stdout)

    man = [split(l, '\t') for l in Iterators.drop(eachline(joinpath(P1, "pairs.tsv")), 1)]
    labels = Int[]; lam = [Float64[] for _ in 1:nc]
    for (ip, row) in enumerate(man)
        pid, lab = row[1], parse(Int, row[2])
        known = readsents(joinpath(P1, "pairs", "$(pid)_known.tsv")); q = readsents(joinpath(P1, "pairs", "$(pid)_q.tsv"))
        (isempty(known) || isempty(q)) && continue
        Xk, Yk = pairs(known, w2i, cb); tm_a = train_model(Xk, Yk, allids, NEGK, GA_EPOCHS); puni_a = unigram(Yk, V)
        Xq, Yq = pairs(q, w2i, cb); n = length(Xq)
        part = [zeros(nc) for _ in 1:maxthreadid()]
        @threads for k in 1:n
            t = threadid(); vA = Vector{Float64}(undef, V); vR = Vector{Float64}(undef, V)
            margins!(vA, tm_a, Xq[k]); margins!(vR, tm_ref, Xq[k]); tgt = Yq[k]
            @inbounds for (ci, (α, τ)) in enumerate(combos)
                part[t][ci] += logp_interp(vA, tgt, τ, α, puni_a) - logp_interp(vR, tgt, τ, α, puni_ref)
            end
        end
        tot = reduce((a, b) -> a .+ b, part)
        push!(labels, lab); for ci in 1:nc; push!(lam[ci], tot[ci] / n); end
        ip % 20 == 0 && (@printf("  %d/%d\n", ip, length(man)); flush(stdout))
    end

    println("\n=== AUC grid ($(length(labels)) pairs, ORDER=$ORDER negk=$NEGK) ===")
    @printf("   α \\ τ   %s\n", join([@sprintf("τ=%-6.0f", τ) for τ in TEMPS], ""))
    best = (-1.0, 0.0, 0.0)
    for α in ALPHAS
        row = @sprintf("  %5.2f   ", α)
        for τ in TEMPS
            a = auc(lam[findfirst(==((α, τ)), combos)], labels); row *= @sprintf("%-8.3f", a)
            a > best[1] && (best = (a, α, τ))
        end
        println(row)
    end
    @printf("\nbest: AUC=%.4f at α=%.2f τ=%.0f   (α=1.0 == pure unigram 0.943 baseline)\n", best...)
end

main()
