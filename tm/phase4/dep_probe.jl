# GO/NO-GO probe for the dependency-arc channel: unsupervised centered-cosine AUC of a
# rate-weighted ARC-BAG sketch -- bind(head-token ⊗ deprel ⊗ dependent-token) -- over the 500
# test pairs, no training anywhere. Campaign precedent: channels only survive into the TM if
# their standalone cosine is competitive with word/char (~0.86); the word-unigram cosine is
# computed on the same docs as the internal yardstick.
#
#   julia -t auto phase4/dep_probe.jl

import Pkg
Pkg.activate(joinpath(@__DIR__, "..", "hdc"); io = devnull)
using HDC, Random, Printf
using LinearAlgebra: dot
using Base.Threads: @threads

const D = 8192; const HERE = @__DIR__; const P3 = joinpath(HERE, "..", "phase3")

struct Tok; cur::String; dep::String; head::Int; end

function read_enr(p)
    sents = Vector{Vector{Tok}}()
    for line in eachline(p)
        isempty(line) && continue
        row = Tok[]
        for f in split(line, '\t')
            parts = split(f, '|')
            length(parts) < 4 && continue
            push!(row, Tok(String(parts[1]), String(parts[3]), parse(Int, parts[4])))
        end
        isempty(row) || push!(sents, row)
    end
    sents
end

@inline function circ_xor!(fp, src, r, W)
    @inbounds @simd for i in 1:W; j = i - r; j < 1 && (j += W); j > W && (j -= W); fp[i] ⊻= src[j]; end
end
@inline function add_bits!(acc, fp, W)
    @inbounds for c in 1:W; x = fp[c]; base = (c - 1) * 64
        while x != 0; acc[base+trailing_zeros(x)+1] += 1; x &= x - one(x); end
    end
end

function sketches(sents, w2i, d2i, Vw, Vd, sp)
    W = sp.W
    aw = zeros(Float64, sp.D); nw = 0            # word unigrams (yardstick)
    aa = zeros(Float64, sp.D); na = 0            # dependency arcs
    fp = newhv(sp)
    for s in sents
        for t in s
            id = get(w2i, t.cur, 0); id == 0 && continue
            add_bits!(aw, @view(Vw[:, id]), W); nw += 1
        end
        for (i, t) in enumerate(s)
            (t.head < 1 || t.head > length(s) || t.head == i) && continue
            hid = get(w2i, s[t.head].cur, 0); did_ = get(w2i, t.cur, 0); li = get(d2i, t.dep, 0)
            (hid == 0 || did_ == 0 || li == 0) && continue
            fill!(fp, 0)
            circ_xor!(fp, @view(Vw[:, hid]), 0, W)
            circ_xor!(fp, @view(Vd[:, li]), 1, W)
            circ_xor!(fp, @view(Vw[:, did_]), 2, W)
            add_bits!(aa, fp, W); na += 1
        end
    end
    (nw > 0 ? aw ./ nw : aw), (na > 0 ? aa ./ na : aa), na
end

function auc(scores, labels)
    o = sortperm(scores); r = similar(scores, Float64); i = 1
    while i <= length(o)
        j = i; while j < length(o) && scores[o[j+1]] == scores[o[i]]; j += 1; end
        m = (i + j) / 2; for t in i:j; r[o[t]] = m; end; i = j + 1
    end
    p = labels .== 1; np = count(p); nn = length(labels) - np
    (sum(r[p]) - np * (np + 1) / 2) / (np * nn)
end

function main()
    man = [split(l, '\t') for l in Iterators.drop(eachline(joinpath(P3, "pairs500.tsv")), 1)]
    pdir = joinpath(HERE, "pairs500_enr")
    docs = Dict{String,Vector{Vector{Tok}}}()
    for row in man, side in ("known", "q")
        docs["$(row[1])_$side"] = read_enr(joinpath(pdir, "$(row[1])_$side.tsv"))
    end

    wset = Set{String}(); dset = Set{String}()
    for ss in values(docs), s in ss, t in s
        push!(wset, t.cur); push!(dset, t.dep)
    end
    vocab = sort!(collect(wset)); w2i = Dict(w => i for (i, w) in enumerate(vocab))
    deps = sort!(collect(dset)); d2i = Dict(d => i for (i, d) in enumerate(deps))
    sp = Space(D)
    Vw = randcodes(sp, length(vocab), MersenneTwister(42))
    Vd = randcodes(sp, length(deps), MersenneTwister(47))
    @printf("vocab %d types, %d dep labels, %d docs\n", length(vocab), length(deps), length(docs))

    keys_ = sort!(collect(keys(docs)))
    SW_ = Dict{String,Vector{Float64}}(); SA_ = Dict{String,Vector{Float64}}()
    narcs = 0; lk = ReentrantLock()
    @threads for k in keys_
        w, a, na = sketches(docs[k], w2i, d2i, Vw, Vd, sp)
        lock(lk) do; SW_[k] = w; SA_[k] = a; narcs += na; end
    end
    @printf("%.1f arcs/doc average\n", narcs / length(keys_))

    mw = sum(values(SW_)) ./ length(SW_); ma = sum(values(SA_)) ./ length(SA_)
    nrm(x) = x ./ (sqrt(dot(x, x)) + 1e-9)
    for k in keys_; SW_[k] = nrm(SW_[k] .- mw); SA_[k] = nrm(SA_[k] .- ma); end

    y = Int[]; cw = Float64[]; ca = Float64[]
    for row in man
        pid = row[1]
        haskey(SW_, "$(pid)_known") || continue
        push!(y, parse(Int, row[2]))
        push!(cw, dot(SW_["$(pid)_known"], SW_["$(pid)_q"]))
        push!(ca, dot(SA_["$(pid)_known"], SA_["$(pid)_q"]))
    end
    @printf("\nword-unigram cosine AUC (yardstick) = %.4f\n", auc(cw, y))
    @printf("DEP-ARC cosine AUC                  = %.4f\n", auc(ca, y))
    @printf("combined (z-mean)                   = %.4f\n",
            auc((cw .- sum(cw)/length(cw)) ./ std_(cw) .+ (ca .- sum(ca)/length(ca)) ./ std_(ca), y))
end
std_(x) = sqrt(sum(abs2, x .- sum(x)/length(x)) / (length(x) - 1)) + 1e-12

main()
