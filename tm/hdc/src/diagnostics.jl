# Collapse instrumentation: turn "does the embedding space collapse?" into numbers.
#   mean_sim    -> 1  : words becoming identical (collapse)
#   sim_std     -> 0  : no spread left (collapse)
#   bit_entropy -> 0  : every bit is 0-or-1 across ALL words (collapse), regardless of weight
#   purity      high  : nearest neighbours share a label (useful structure)

"For labelled words, mean fraction of the top-`k` neighbours sharing the word's label."
function neighbor_purity(cb::Codebook, labels; k::Integer = 5)
    E = cb.E; V = size(E, 2); tot = 0.0; n = 0
    for v in 1:V
        haskey(labels, cb.vocab[v]) || continue
        q = @view E[:, v]
        ord = sortperm([hamming(q, @view E[:, u]) for u in 1:V])
        nb = [cb.vocab[u] for u in ord[2:min(k + 1, V)] if haskey(labels, cb.vocab[u])]
        isempty(nb) && continue
        lab = labels[cb.vocab[v]]
        tot += count(x -> labels[x] == lab, nb) / length(nb); n += 1
    end
    n == 0 ? NaN : tot / n
end

"""
    diagnostics(cb; labels=nothing, npairs=3000, rng)

Return `(mean_sim, sim_std, bit_entropy, purity)` — the collapse signature. `bit_entropy`
is the mean binary entropy of each bit's set-fraction across the vocabulary (capacity
actually in use); it hits 0 at consensus even when Hamming weight is conserved.
"""
function diagnostics(cb::Codebook; labels = nothing, npairs::Integer = 3000,
                     rng = Random.default_rng())
    sp = cb.sp; E = cb.E; V = size(E, 2)
    ss = 0.0; s2 = 0.0; m = 0
    for _ in 1:npairs
        a = rand(rng, 1:V); b = rand(rng, 1:V); a == b && continue
        s = sim(sp, @view(E[:, a]), @view(E[:, b])); ss += s; s2 += s * s; m += 1
    end
    μ = m == 0 ? 0.0 : ss / m
    σ = m == 0 ? 0.0 : sqrt(max(s2 / m - μ^2, 0.0))
    cnt = zeros(Int, sp.D)
    @inbounds for v in 1:V, i in 1:sp.W
        x = E[i, v]; base = (i - 1) * 64
        while x != 0; cnt[base + trailing_zeros(x) + 1] += 1; x &= x - one(x); end
    end
    H = 0.0
    for p in 1:sp.D
        q = cnt[p] / V
        (0 < q < 1) && (H -= q * log2(q) + (1 - q) * log2(1 - q))
    end
    (mean_sim = round(μ, digits = 3), sim_std = round(σ, digits = 3),
     bit_entropy = round(H / sp.D, digits = 3),
     purity = labels === nothing ? NaN : round(neighbor_purity(cb, labels), digits = 3))
end

"The `k` nearest words to `word` by Hamming similarity."
function nearest(cb::Codebook, word::AbstractString; k::Integer = 10)
    V = size(cb.E, 2); q = @view cb.E[:, cb.word2id[word]]
    ord = sortperm([hamming(q, @view cb.E[:, u]) for u in 1:V])
    [(cb.vocab[u], round(sim(cb.sp, q, @view cb.E[:, u]), digits = 3))
     for u in ord[2:min(k + 1, V)]]
end
