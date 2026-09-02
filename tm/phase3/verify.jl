# Representation test for the direct verifier (unsupervised, fast): encode each document as a
# rate-weighted, population-CENTERED sketch of short units (unigram + bi/tri-gram + short
# skip-grams), then score a pair by cosine(sketch_U, sketch_K) -> AUC over the 80 av_test pairs.
# This is Burrows's-Delta-in-HDC. Compares atom sets (pretrained SUBWORD vs RANDOM) and unit
# sets, to see whether pretrained morphological atoms give better verification signatures.
#
#   julia -t auto phase3/verify.jl
#
# Note: with RANDOM atoms the HDC sketch ~ classical n-gram-frequency Delta (random projection
# preserves cosines); SUBWORD atoms add morphological smoothing (related units share subwords).

import Pkg
Pkg.activate(joinpath(@__DIR__, "..", "hdc"); io = devnull)
using HDC, Random, Printf, Serialization
using LinearAlgebra: dot

const D = 8192
const HERE = @__DIR__
const P1 = joinpath(HERE, "..", "phase1")

readsents(p) = [String.(split(l, '\t')) for l in eachline(p) if !isempty(l)]

# ---- compose subword word vectors (same math as subword.jl) ----
function compose_all(SW, Q, word_subs, sp)
    V = Matrix{UInt64}(undef, sp.W, length(word_subs))
    bound = [newhv(sp) for _ in 1:32]; acc = Vector{Int32}(undef, sp.W * 64); rng = MersenneTwister(0)
    for w in eachindex(word_subs)
        subs = word_subs[w]; n = length(subs)
        @inbounds for (k, (sid, slot)) in enumerate(subs)
            b = bound[k]; @simd for i in 1:sp.W; b[i] = SW[i, sid] ⊻ Q[i, slot]; end
        end
        bundle!(sp, @view(V[:, w]), view(bound, 1:n), acc, rng)
    end
    V
end

@inline function circ_xor!(fp, src, r, W)             # fp ⊻= circshift(src, r) (word rotation)
    @inbounds @simd for i in 1:W
        j = i - r; j < 1 && (j += W); j > W && (j -= W)
        fp[i] ⊻= src[j]
    end
end
@inline function add_bits!(acc, fp, W)                # acc[bit] += 1 for each set bit of fp
    @inbounds for c in 1:W
        x = fp[c]; base = (c - 1) * 64
        while x != 0; acc[base + trailing_zeros(x) + 1] += 1; x &= x - one(x); end
    end
end

# encode one document into a per-bit unit-count vector; returns (acc::Vector{Int}, n_units)
function encode_doc(sents, V, w2i, sp, GAP, cfg)
    W = sp.W; acc = zeros(Int32, sp.D); n = 0; fp = newhv(sp)
    for s in sents
        ids = Int[]; for w in s; id = get(w2i, w, 0); id > 0 && push!(ids, id); end
        L = length(ids)
        @inbounds for i in 1:L; add_bits!(acc, @view(V[:, ids[i]]), W); n += 1; end     # unigram
        if cfg.bi
            @inbounds for i in 1:L-1
                fill!(fp, 0); circ_xor!(fp, @view(V[:, ids[i]]), 0, W); circ_xor!(fp, @view(V[:, ids[i+1]]), 1, W)
                add_bits!(acc, fp, W); n += 1
            end
        end
        if cfg.tri
            @inbounds for i in 1:L-2
                fill!(fp, 0); circ_xor!(fp, @view(V[:, ids[i]]), 0, W); circ_xor!(fp, @view(V[:, ids[i+1]]), 1, W); circ_xor!(fp, @view(V[:, ids[i+2]]), 2, W)
                add_bits!(acc, fp, W); n += 1
            end
        end
        if cfg.skip
            @inbounds for i in 1:L-2                    # skip gap 1: tokens (i, i+2)
                fill!(fp, 0); circ_xor!(fp, @view(V[:, ids[i]]), 0, W); circ_xor!(fp, @view(V[:, ids[i+2]]), 1, W); circ_xor!(fp, @view(GAP[:, 1]), 0, W)
                add_bits!(acc, fp, W); n += 1
            end
            @inbounds for i in 1:L-3                    # skip gap 2: tokens (i, i+3)
                fill!(fp, 0); circ_xor!(fp, @view(V[:, ids[i]]), 0, W); circ_xor!(fp, @view(V[:, ids[i+3]]), 1, W); circ_xor!(fp, @view(GAP[:, 2]), 0, W)
                add_bits!(acc, fp, W); n += 1
            end
        end
    end
    acc, n
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

function run(name, V, w2i, sp, GAP, docs, labels)
    for cfg in [(bi = false, tri = false, skip = false), (bi = true, tri = true, skip = false), (bi = true, tri = true, skip = true)]
        # sketches + population mean (Delta centering)
        rates = Vector{Vector{Float64}}(undef, length(docs))
        for (di, d) in enumerate(docs)
            acc, n = encode_doc(d, V, w2i, sp, GAP, cfg)
            rates[di] = n > 0 ? acc ./ n : zeros(Float64, sp.D)
        end
        mean_rate = sum(rates) ./ length(rates)
        cen = [r .- mean_rate for r in rates]
        # pair cosine (docs are stored known,q interleaved per pair)
        scores = Float64[]
        for pi in 1:2:length(cen)
            u = cen[pi]; k = cen[pi+1]
            scores = push!(scores, dot(u, k) / (sqrt(dot(u, u)) * sqrt(dot(k, k)) + 1e-12))
        end
        units = cfg.skip ? "uni+bi+tri+skip" : cfg.bi ? "uni+bi+tri" : "uni-only"
        @printf("  %-8s %-16s AUC=%.4f\n", name, units, auc(scores, labels)); flush(stdout)
    end
end

function main()
    SW, Q, word_subs, vocab, w2i = deserialize(joinpath(HERE, "subword_atoms.jls"))
    sp = Space(D)
    @printf("loaded subword atoms: vocab %d, subwords %d\n", length(vocab), size(SW, 2)); flush(stdout)
    Vsub = compose_all(SW, Q, word_subs, sp)
    Vrnd = randcodes(sp, length(vocab), MersenneTwister(42))
    GAP = randcodes(sp, 2, MersenneTwister(43))

    man = [split(l, '\t') for l in Iterators.drop(eachline(joinpath(P1, "pairs.tsv")), 1)]
    docs = Vector{Vector{Vector{String}}}(); labels = Int[]; oov = tot = 0
    for row in man
        pid, lab = row[1], parse(Int, row[2])
        k = readsents(joinpath(P1, "pairs", "$(pid)_known.tsv")); q = readsents(joinpath(P1, "pairs", "$(pid)_q.tsv"))
        (isempty(k) || isempty(q)) && continue
        push!(docs, k); push!(docs, q); push!(labels, lab)
        for s in vcat(k, q), w in s; tot += 1; haskey(w2i, w) || (oov += 1); end
    end
    @printf("%d pairs, OOV token rate %.1f%% (skipped)\n\n", length(labels), 100 * oov / tot); flush(stdout)

    println("=== centered-cosine verification AUC (80 pairs) ===")
    run("RANDOM", Vrnd, w2i, sp, GAP, docs, labels)
    run("SUBWORD", Vsub, w2i, sp, GAP, docs, labels)
end

main()
