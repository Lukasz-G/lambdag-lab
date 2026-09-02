# Add a CHARACTER-trigram document sketch (orthogonal to KN, which sees function words as
# atomic tokens). Reuses the subword inventory from subword_atoms.jls: each function-word token
# contributes its <word> char-trigrams (bag) to a char sketch. Emits three centered-cosine
# scores per pair -- word-only, char-only, word+char -- for the fusion test vs KN.
#
#   julia -t auto phase3/verify_char.jl   ->  phase3/hdc_char500.jsonl

import Pkg
Pkg.activate(joinpath(@__DIR__, "..", "hdc"); io = devnull)
using HDC, Random, Printf, Serialization
using LinearAlgebra: dot
using Base.Threads: @threads

const D = 8192; const HERE = @__DIR__
readsents(p) = [String.(split(l, '\t')) for l in eachline(p) if !isempty(l)]

@inline function circ_xor!(fp, src, r, W)
    @inbounds @simd for i in 1:W; j = i - r; j < 1 && (j += W); j > W && (j -= W); fp[i] ⊻= src[j]; end
end
@inline function add_bits!(acc, fp, W)
    @inbounds for c in 1:W; x = fp[c]; base = (c - 1) * 64
        while x != 0; acc[base+trailing_zeros(x)+1] += 1; x &= x - one(x); end
    end
end

# word sketch (uni+bi+tri+skip) AND char sketch (bag of function-word subwords)
function encode_doc(sents, Vw, SWc, word_subs, w2i, sp, GAP)
    W = sp.W; aw = zeros(Int32, sp.D); ac = zeros(Int32, sp.D); nw = 0; nc = 0; fp = newhv(sp)
    for s in sents
        ids = Int[]; for w in s; id = get(w2i, w, 0); id > 0 && push!(ids, id); end
        L = length(ids)
        @inbounds for i in 1:L
            add_bits!(aw, @view(Vw[:, ids[i]]), W); nw += 1               # word unigram
            for (sid, _) in word_subs[ids[i]]; add_bits!(ac, @view(SWc[:, sid]), W); nc += 1; end  # char trigrams
        end
        @inbounds for i in 1:L-1
            fill!(fp, 0); circ_xor!(fp, @view(Vw[:, ids[i]]), 0, W); circ_xor!(fp, @view(Vw[:, ids[i+1]]), 1, W); add_bits!(aw, fp, W); nw += 1
        end
        @inbounds for i in 1:L-2
            fill!(fp, 0); circ_xor!(fp, @view(Vw[:, ids[i]]), 0, W); circ_xor!(fp, @view(Vw[:, ids[i+1]]), 1, W); circ_xor!(fp, @view(Vw[:, ids[i+2]]), 2, W); add_bits!(aw, fp, W); nw += 1
        end
        @inbounds for i in 1:L-2
            fill!(fp, 0); circ_xor!(fp, @view(Vw[:, ids[i]]), 0, W); circ_xor!(fp, @view(Vw[:, ids[i+2]]), 1, W); circ_xor!(fp, @view(GAP[:, 1]), 0, W); add_bits!(aw, fp, W); nw += 1
        end
    end
    (nw > 0 ? aw ./ nw : zeros(Float64, sp.D)), (nc > 0 ? ac ./ nc : zeros(Float64, sp.D))
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
cos_pair(u, k) = dot(u, k) / (sqrt(dot(u, u)) * sqrt(dot(k, k)) + 1e-12)

function main()
    SW, Q, word_subs, vocab, w2i = deserialize(joinpath(HERE, "subword_atoms.jls"))
    sp = Space(D)
    Vw  = randcodes(sp, length(vocab), MersenneTwister(42))       # random WORD atoms (best for word n-grams)
    SWc = randcodes(sp, size(SW, 2), MersenneTwister(44))         # random CHAR-trigram atoms (distinct -> freq sketch)
    GAP = randcodes(sp, 2, MersenneTwister(43))

    man = [split(l, '\t') for l in Iterators.drop(eachline(joinpath(HERE, "pairs500.tsv")), 1)]
    ids = Int[]; labels = Int[]; kd = Vector{Vector{Vector{String}}}(); qd = Vector{Vector{Vector{String}}}()
    for row in man
        pid, lab = row[1], parse(Int, row[2])
        k = readsents(joinpath(HERE, "pairs500", "$(pid)_known.tsv")); q = readsents(joinpath(HERE, "pairs500", "$(pid)_q.tsv"))
        (isempty(k) || isempty(q)) && continue
        push!(ids, parse(Int, pid)); push!(labels, lab); push!(kd, k); push!(qd, q)
    end
    np = length(labels)
    kw = Vector{Vector{Float64}}(undef, np); kc = Vector{Vector{Float64}}(undef, np)
    qw = Vector{Vector{Float64}}(undef, np); qc = Vector{Vector{Float64}}(undef, np)
    @threads for i in 1:np; kw[i], kc[i] = encode_doc(kd[i], Vw, SWc, word_subs, w2i, sp, GAP); end
    @threads for i in 1:np; qw[i], qc[i] = encode_doc(qd[i], Vw, SWc, word_subs, w2i, sp, GAP); end
    pw = (sum(kw) .+ sum(qw)) ./ 2np; pc = (sum(kc) .+ sum(qc)) ./ 2np
    cos_w = [cos_pair(kw[i] .- pw, qw[i] .- pw) for i in 1:np]
    cos_c = [cos_pair(kc[i] .- pc, qc[i] .- pc) for i in 1:np]
    cos_wc = [cos_pair(vcat(kw[i] .- pw, kc[i] .- pc), vcat(qw[i] .- pw, qc[i] .- pc)) for i in 1:np]
    @printf("word cosine AUC=%.4f\nchar cosine AUC=%.4f\nword+char  AUC=%.4f\n",
            auc(cos_w, labels), auc(cos_c, labels), auc(cos_wc, labels))
    open(joinpath(HERE, "hdc_char500.jsonl"), "w") do io
        for i in eachindex(ids)
            write(io, """{"id":$(ids[i]),"label":$(labels[i]),"cos_word":$(cos_w[i]),"cos_char":$(cos_c[i]),"cos_wc":$(cos_wc[i])}\n""")
        end
    end
    println("wrote hdc_char500.jsonl")
end

main()
