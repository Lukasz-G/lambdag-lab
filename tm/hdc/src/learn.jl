# Self-referential embedding learning with pluggable local update rules.
# Context is encoded from the CURRENT embeddings (in-place / recursive), then a rule edits
# the target atom. See diagnostics.jl for the collapse metrics that separate the rules.

# ---- binary edit primitives (flip probability = fold density 2^-folds = learning rate) ----
"Move `E` toward `C`: flip ~2^-folds of the DIFFERING bits (only where `writable`)."
function pull!(sp::Space, E, C, mask, tmp, folds, rng, writable = nothing)
    sparse_mask!(sp, mask, tmp, folds, rng)
    if writable !== nothing
        @inbounds @simd for i in eachindex(mask); mask[i] &= writable[i]; end
    end
    @inbounds @simd for i in eachindex(E); E[i] ⊻= (E[i] ⊻ C[i]) & mask[i]; end
end
"Move `E` away from `C`: flip ~2^-folds of the MATCHING bits (only where `writable`)."
function push_away!(sp::Space, E, C, mask, tmp, folds, rng, writable = nothing)
    sparse_mask!(sp, mask, tmp, folds, rng)
    if writable !== nothing
        @inbounds @simd for i in eachindex(mask); mask[i] &= writable[i]; end
    end
    @inbounds @simd for i in eachindex(E); E[i] ⊻= (~(E[i] ⊻ C[i])) & mask[i]; end
end

# ---- the rule zoo (see the handoff: which admit a non-collapsing equilibrium?) ----
abstract type UpdateRule end

"Pure attraction (control): should COLLAPSE — the binary voter/diffusion fixed point."
struct Surprise <: UpdateRule; folds::Int; end

"Attraction + repulsion (binary SGNS): a genuine non-trivial equilibrium."
struct AttractRepel <: UpdateRule; pos_folds::Int; neg_folds::Int; nneg::Int; end

"Capacity-conserving swap toward `C` (fixed Hamming weight): pure attraction, so still collapses — the control for 'does capacity conservation alone suffice?'."
struct SparseCapacity <: UpdateRule; nflips::Int; end

"Zero-temperature Ising update: align chosen bits to a local field (want `C`, want ≠ negatives). Repulsion folded into the field."
struct EnergyAlign <: UpdateRule; folds::Int; nneg::Int; end

mutable struct LearnBuf
    mask::Vector{UInt64}; tmp::Vector{UInt64}; a::Vector{UInt64}; b::Vector{UInt64}
end
LearnBuf(sp::Space) = LearnBuf(newhv(sp), newhv(sp), newhv(sp), newhv(sp))

update!(r::Surprise, cb, t, C, lb, rng) =
    pull!(cb.sp, @view(cb.E[:, t]), C, lb.mask, lb.tmp, r.folds, rng, cb.writable)

function update!(r::AttractRepel, cb, t, C, lb, rng)
    pull!(cb.sp, @view(cb.E[:, t]), C, lb.mask, lb.tmp, r.pos_folds, rng, cb.writable)
    for _ in 1:r.nneg
        n = sample_neg(cb, rng); n == t && continue
        push_away!(cb.sp, @view(cb.E[:, n]), C, lb.mask, lb.tmp, r.neg_folds, rng, cb.writable)
    end
end

function update!(r::SparseCapacity, cb, t, C, lb, rng)
    E = @view cb.E[:, t]; W = cb.sp.W; wr = cb.writable
    @inbounds for _ in 1:r.nflips
        for i in 1:W
            lb.a[i] = ~E[i] & C[i]                      # E=0,C=1 -> set (toward C, +1 weight)
            lb.b[i] =  E[i] & ~C[i]                      # E=1,C=0 -> clear (toward C, -1 weight)
            if wr !== nothing; lb.a[i] &= wr[i]; lb.b[i] &= wr[i]; end
        end
        p1 = random_setbit(lb.a, rng); p1 == 0 && break
        p0 = random_setbit(lb.b, rng); p0 == 0 && break
        setbit!(E, p1); clrbit!(E, p0)                 # net Hamming weight change = 0
    end
end

function update!(r::EnergyAlign, cb, t, C, lb, rng)
    sparse_mask!(cb.sp, lb.mask, lb.tmp, r.folds, rng)
    if cb.writable !== nothing
        @inbounds @simd for i in eachindex(lb.mask); lb.mask[i] &= cb.writable[i]; end
    end
    E = @view cb.E[:, t]
    negs = [@view cb.E[:, sample_neg(cb, rng)] for _ in 1:r.nneg]
    @inbounds for i in eachindex(E)
        x = lb.mask[i]
        while x != 0
            p = (i - 1) * 64 + trailing_zeros(x) + 1; x &= x - one(x)
            field = getbit(C, p) ? 1 : -1              # want to agree with context
            for nc in negs; field += getbit(nc, p) ? -1 : 1; end   # want to differ from negs
            field > 0 ? setbit!(E, p) : (field < 0 && clrbit!(E, p))
        end
    end
end

"""
    train!(enc, ids, rule; window=5, epochs=5, rng, log, labels) -> Codebook

Self-referential training. For every center token, the context (up to `window` tokens each
side) is encoded from the CURRENT embeddings via `enc`, then `rule` edits the center atom.
Set `log=stdout` to print collapse diagnostics per epoch (pass `labels` for neighbour purity).
"""
function train!(enc::Encoder, ids, rule::UpdateRule; window::Integer = 5, epochs::Integer = 5,
                rng = Random.default_rng(), log = nothing, labels = nothing)
    cb = enc.cb; C = newhv(cb.sp); lb = LearnBuf(cb.sp); ctx = Int[]
    for epoch in 1:epochs
        for sent in ids
            L = length(sent)
            for c in 1:L
                t = sent[c]; empty!(ctx)
                for o in max(1, c - window):min(L, c + window)
                    o == c || push!(ctx, sent[o])
                end
                isempty(ctx) && continue
                encode_window!(enc, C, ctx, rng)
                update!(rule, cb, t, C, lb, rng)
            end
        end
        log !== nothing && println(log, "  epoch $epoch  ", diagnostics(cb; labels = labels, rng = rng))
    end
    cb
end
