# Length sweep (HDC side): truncate the QUESTIONED fragment to L tokens and recompute the
# word/char centered-cosine vs the full known doc, for each L. Exports per-pair scores so the
# Python side can score KN at the same lengths and compare AUC-vs-length.
#
#   julia -t auto phase3/lensweep.jl   ->  phase3/hdc_len500.jsonl

import Pkg
Pkg.activate(joinpath(@__DIR__, "..", "hdc"); io = devnull)
using HDC, Random, Printf, Serialization
using LinearAlgebra: dot
using Base.Threads: @threads

const D = 8192; const HERE = @__DIR__
const LENS = [parse(Int, x) for x in split(get(ENV, "P3_LENS", "10000,600,300,150,75"), ",")]
readsents(p) = [String.(split(l, '\t')) for l in eachline(p) if !isempty(l)]

@inline function circ_xor!(fp, src, r, W)
    @inbounds @simd for i in 1:W; j = i - r; j < 1 && (j += W); j > W && (j -= W); fp[i] ⊻= src[j]; end
end
@inline function add_bits!(acc, fp, W)
    @inbounds for c in 1:W; x = fp[c]; base = (c - 1) * 64
        while x != 0; acc[base+trailing_zeros(x)+1] += 1; x &= x - one(x); end
    end
end
function truncate_sents(sents, L)
    out = Vector{String}[]; n = 0
    for s in sents
        if n + length(s) <= L; push!(out, s); n += length(s)
        else; take = L - n; take > 0 && push!(out, s[1:take]); break; end
    end
    out
end
function encode_doc(sents, Vw, SWc, word_subs, w2i, sp, GAP)   # (word_rate, char_rate)
    W = sp.W; aw = zeros(Int32, sp.D); ac = zeros(Int32, sp.D); nw = 0; nc = 0; fp = newhv(sp)
    for s in sents
        ids = Int[]; for w in s; id = get(w2i, w, 0); id > 0 && push!(ids, id); end
        L = length(ids)
        @inbounds for i in 1:L
            add_bits!(aw, @view(Vw[:, ids[i]]), W); nw += 1
            for (sid, _) in word_subs[ids[i]]; add_bits!(ac, @view(SWc[:, sid]), W); nc += 1; end
        end
        @inbounds for i in 1:L-1
            fill!(fp, 0); circ_xor!(fp, @view(Vw[:, ids[i]]), 0, W); circ_xor!(fp, @view(Vw[:, ids[i+1]]), 1, W); add_bits!(aw, fp, W); nw += 1
        end
        @inbounds for i in 1:L-2
            fill!(fp, 0); circ_xor!(fp, @view(Vw[:, ids[i]]), 0, W); circ_xor!(fp, @view(Vw[:, ids[i+1]]), 1, W); circ_xor!(fp, @view(Vw[:, ids[i+2]]), 2, W); add_bits!(aw, fp, W); nw += 1
        end
    end
    (nw > 0 ? aw ./ nw : zeros(Float64, sp.D)), (nc > 0 ? ac ./ nc : zeros(Float64, sp.D))
end
cosp(u, k) = dot(u, k) / (sqrt(dot(u, u)) * sqrt(dot(k, k)) + 1e-12)

function main()
    SW, Q, word_subs, vocab, w2i = deserialize(joinpath(HERE, "subword_atoms.jls"))
    sp = Space(D)
    Vw = randcodes(sp, length(vocab), MersenneTwister(42)); SWc = randcodes(sp, size(SW, 2), MersenneTwister(44)); GAP = randcodes(sp, 2, MersenneTwister(43))
    man = [split(l, '\t') for l in Iterators.drop(eachline(joinpath(HERE, "pairs500.tsv")), 1)]
    ids = Int[]; labels = Int[]; kd = Vector{Vector{Vector{String}}}(); qd = Vector{Vector{Vector{String}}}()
    for row in man
        pid, lab = row[1], parse(Int, row[2])
        k = readsents(joinpath(HERE, "pairs500", "$(pid)_known.tsv")); q = readsents(joinpath(HERE, "pairs500", "$(pid)_q.tsv"))
        (isempty(k) || isempty(q)) && continue
        push!(ids, parse(Int, pid)); push!(labels, lab); push!(kd, k); push!(qd, q)
    end
    np = length(labels)
    kw = Vector{Vector{Float64}}(undef, np); kc = similar(kw); qwF = similar(kw); qcF = similar(kw)
    @threads for i in 1:np; kw[i], kc[i] = encode_doc(kd[i], Vw, SWc, word_subs, w2i, sp, GAP); end
    @threads for i in 1:np; qwF[i], qcF[i] = encode_doc(qd[i], Vw, SWc, word_subs, w2i, sp, GAP); end
    pw = (sum(kw) .+ sum(qwF)) ./ 2np; pc = (sum(kc) .+ sum(qcF)) ./ 2np   # population mean (fixed)

    cwL = Dict{Int,Vector{Float64}}(); ccL = Dict{Int,Vector{Float64}}()
    for L in LENS
        qw = Vector{Vector{Float64}}(undef, np); qc = similar(qw)
        @threads for i in 1:np; qw[i], qc[i] = encode_doc(truncate_sents(qd[i], L), Vw, SWc, word_subs, w2i, sp, GAP); end
        cwL[L] = [cosp(kw[i] .- pw, qw[i] .- pw) for i in 1:np]
        ccL[L] = [cosp(kc[i] .- pc, qc[i] .- pc) for i in 1:np]
        @printf("L=%-6d  word AUC=%.3f  char AUC=%.3f\n", L,
                let s = cwL[L]; o = sortperm(s); r = zeros(np); for (rk, ii) in enumerate(o); r[ii] = rk; end; p = labels .== 1; (sum(r[p]) - count(p) * (count(p) + 1) / 2) / (count(p) * (np - count(p))) end,
                let s = ccL[L]; o = sortperm(s); r = zeros(np); for (rk, ii) in enumerate(o); r[ii] = rk; end; p = labels .== 1; (sum(r[p]) - count(p) * (count(p) + 1) / 2) / (count(p) * (np - count(p))) end); flush(stdout)
    end
    open(joinpath(HERE, "hdc_len500.jsonl"), "w") do io
        for i in eachindex(ids)
            cols = join(["\"cw_$L\":$(cwL[L][i]),\"cc_$L\":$(ccL[L][i])" for L in LENS], ",")
            write(io, "{\"id\":$(ids[i]),\"label\":$(labels[i]),$cols}\n")
        end
    end
    println("wrote hdc_len500.jsonl")
end

main()
