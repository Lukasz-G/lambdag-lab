# Supervised TM verifier on ~500 pairs with AUTHOR-DISJOINT CV (folds grouped by the QUESTIONED
# author, so the TM is always tested on authors it never trained on — the fair protocol vs KN).
# Feature = per-bit sign-agreement of the two docs' population-centered unit sketches
# (uni+bi/tri+short-skip). Reports cosine (unsupervised) + TM(cv) for RANDOM & SUBWORD atoms.
#
#   julia -t auto phase3/verify500.jl

import Pkg
Pkg.activate(joinpath(@__DIR__, "..", "hdc"); io = devnull)
using HDC, Random, Printf, Serialization
using LinearAlgebra: dot
using Base.Threads: @threads
include(joinpath(@__DIR__, "..", "Tsetlin.jl-main", "src", "Tsetlin.jl"))
const TM = Tsetlin

const D = 8192; const HERE = @__DIR__
const CLAUSES = 300; const T_ = 48; const S_ = 8192; const L_ = 4096; const LF_ = 4096
const STATES = 256; const INCLUDE = 128; const EPOCHS = 60; const NFOLD = 5

readsents(p) = [String.(split(l, '\t')) for l in eachline(p) if !isempty(l)]

function compose_all(SW, Q, word_subs, sp)
    V = Matrix{UInt64}(undef, sp.W, length(word_subs))
    bound = [newhv(sp) for _ in 1:32]; acc = Vector{Int32}(undef, sp.W * 64); rng = MersenneTwister(0)
    for w in eachindex(word_subs)
        subs = word_subs[w]
        @inbounds for (k, (sid, slot)) in enumerate(subs)
            b = bound[k]; @simd for i in 1:sp.W; b[i] = SW[i, sid] ⊻ Q[i, slot]; end
        end
        bundle!(sp, @view(V[:, w]), view(bound, 1:length(subs)), acc, rng)
    end
    V
end
@inline function circ_xor!(fp, src, r, W)
    @inbounds @simd for i in 1:W; j = i - r; j < 1 && (j += W); j > W && (j -= W); fp[i] ⊻= src[j]; end
end
@inline function add_bits!(acc, fp, W)
    @inbounds for c in 1:W; x = fp[c]; base = (c - 1) * 64
        while x != 0; acc[base+trailing_zeros(x)+1] += 1; x &= x - one(x); end
    end
end
function encode_doc(sents, V, w2i, sp, GAP)          # uni + bi + tri + skip(1,2); own buffers (thread-safe)
    W = sp.W; acc = zeros(Int32, sp.D); n = 0; fp = newhv(sp)
    for s in sents
        ids = Int[]; for w in s; id = get(w2i, w, 0); id > 0 && push!(ids, id); end
        L = length(ids)
        @inbounds for i in 1:L; add_bits!(acc, @view(V[:, ids[i]]), W); n += 1; end
        @inbounds for i in 1:L-1
            fill!(fp, 0); circ_xor!(fp, @view(V[:, ids[i]]), 0, W); circ_xor!(fp, @view(V[:, ids[i+1]]), 1, W); add_bits!(acc, fp, W); n += 1
        end
        @inbounds for i in 1:L-2
            fill!(fp, 0); circ_xor!(fp, @view(V[:, ids[i]]), 0, W); circ_xor!(fp, @view(V[:, ids[i+1]]), 1, W); circ_xor!(fp, @view(V[:, ids[i+2]]), 2, W); add_bits!(acc, fp, W); n += 1
        end
        @inbounds for i in 1:L-2
            fill!(fp, 0); circ_xor!(fp, @view(V[:, ids[i]]), 0, W); circ_xor!(fp, @view(V[:, ids[i+2]]), 1, W); circ_xor!(fp, @view(GAP[:, 1]), 0, W); add_bits!(acc, fp, W); n += 1
        end
        @inbounds for i in 1:L-3
            fill!(fp, 0); circ_xor!(fp, @view(V[:, ids[i]]), 0, W); circ_xor!(fp, @view(V[:, ids[i+3]]), 1, W); circ_xor!(fp, @view(GAP[:, 2]), 0, W); add_bits!(acc, fp, W); n += 1
        end
    end
    n > 0 ? acc ./ n : zeros(Float64, sp.D)
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

function score_atoms(V, w2i, sp, GAP, kdocs, qdocs, labels, pair_fold)
    np = length(labels)
    ku = Vector{Vector{Float64}}(undef, np); qu = Vector{Vector{Float64}}(undef, np)
    @threads for i in 1:np; ku[i] = encode_doc(kdocs[i], V, w2i, sp, GAP); end
    @threads for i in 1:np; qu[i] = encode_doc(qdocs[i], V, w2i, sp, GAP); end
    pop = (sum(ku) .+ sum(qu)) ./ (2np)
    kc = [k .- pop for k in ku]; qc = [q .- pop for q in qu]
    cos = [dot(kc[i], qc[i]) / (sqrt(dot(kc[i], kc[i])) * sqrt(dot(qc[i], qc[i])) + 1e-12) for i in 1:np]
    feats = [Bool[(kc[i][b] > 0) == (qc[i][b] > 0) for b in 1:sp.D] for i in 1:np]
    margins = zeros(Float64, np)
    for f in 1:NFOLD
        tr = findall(pair_fold .!= f); te = findall(pair_fold .== f)
        (isempty(te) || length(unique(labels[tr])) < 2) && continue
        X = TM.TMInput[TM.TMInput(feats[i]) for i in tr]; Y = Bool[labels[i] == 1 for i in tr]
        tm = TM.TMClassifier(X[1], Y, CLAUSES, T_, S_, L_, LF_; states_num = STATES, include_limit = INCLUDE)
        for _ in 1:EPOCHS; TM.train!(tm, X, Y; shuffle = true, index = false); end
        for i in te
            pos, neg = TM.vote(tm, tm.clauses, TM.TMInput(feats[i])); margins[i] = pos - neg
        end
    end
    cos, margins
end

function main()
    SW, Q, word_subs, vocab, w2i = deserialize(joinpath(HERE, "subword_atoms.jls"))
    sp = Space(D)
    Vsub = compose_all(SW, Q, word_subs, sp)
    Vrnd = randcodes(sp, length(vocab), MersenneTwister(42))
    GAP = randcodes(sp, 2, MersenneTwister(43))

    man = [split(l, '\t') for l in Iterators.drop(eachline(joinpath(HERE, "pairs500.tsv")), 1)]
    ids = Int[]; labels = Int[]; qauth = String[]
    kdocs = Vector{Vector{Vector{String}}}(); qdocs = Vector{Vector{Vector{String}}}()
    for row in man
        pid, lab, qa = row[1], parse(Int, row[2]), row[4]
        k = readsents(joinpath(HERE, "pairs500", "$(pid)_known.tsv")); q = readsents(joinpath(HERE, "pairs500", "$(pid)_q.tsv"))
        (isempty(k) || isempty(q)) && continue
        push!(ids, parse(Int, pid)); push!(labels, lab); push!(qauth, qa); push!(kdocs, k); push!(qdocs, q)
    end
    # author-disjoint folds: assign each unique questioned author to a fold
    ua = shuffle(MersenneTwister(5), unique(qauth))
    afold = Dict(a => ((i - 1) % NFOLD) + 1 for (i, a) in enumerate(ua))
    pair_fold = [afold[a] for a in qauth]
    @printf("%d pairs, %d questioned authors, %d folds (author-disjoint)\n", length(labels), length(ua), NFOLD); flush(stdout)

    cos_r, tm_r = score_atoms(Vrnd, w2i, sp, GAP, kdocs, qdocs, labels, pair_fold)
    cos_s, tm_s = score_atoms(Vsub, w2i, sp, GAP, kdocs, qdocs, labels, pair_fold)
    @printf("\n  method                 AUC\n")
    @printf("  RANDOM  cosine        %.4f\n", auc(cos_r, labels))
    @printf("  RANDOM  TM (adCV)     %.4f\n", auc(tm_r, labels))
    @printf("  SUBWORD cosine        %.4f\n", auc(cos_s, labels))
    @printf("  SUBWORD TM (adCV)     %.4f\n", auc(tm_s, labels))
    open(joinpath(HERE, "hdc_scores500.jsonl"), "w") do io
        for i in eachindex(ids)
            write(io, """{"id":$(ids[i]),"label":$(labels[i]),"cos_rnd":$(cos_r[i]),"tm_rnd":$(tm_r[i]),"cos_sub":$(cos_s[i]),"tm_sub":$(tm_s[i])}\n""")
        end
    end
    println("wrote hdc_scores500.jsonl")
end

main()
