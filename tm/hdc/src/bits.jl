# Packed binary hypervector primitives.
# A hypervector = D bits stored in ceil(D/64) UInt64 chunks. Padding bits in the last
# chunk (positions D .. W*64-1) are kept 0 so Hamming/majority stay exact.

nchunks(D) = cld(D, 64)
tailmask(D) = (r = D & 63; r == 0 ? typemax(UInt64) : (one(UInt64) << r) - one(UInt64))

"""A binary hypervector space of dimension `D` (bits packed into `W` UInt64 chunks)."""
struct Space
    D::Int
    W::Int
    tail::UInt64
end
Space(D::Integer) = Space(Int(D), nchunks(Int(D)), tailmask(Int(D)))

newhv(sp::Space) = Vector{UInt64}(undef, sp.W)

function randhv!(sp::Space, v, rng)
    rand!(rng, v)
    @inbounds v[end] &= sp.tail          # zero the padding bits
    v
end
randhv(sp::Space, rng) = randhv!(sp, newhv(sp), rng)

# bind = XOR
@inline function bind!(dst, a, b)
    @inbounds @simd for i in eachindex(dst); dst[i] = a[i] ⊻ b[i]; end
    dst
end
@inline function xor!(dst, a)            # dst ⊻= a
    @inbounds @simd for i in eachindex(dst); dst[i] ⊻= a[i]; end
    dst
end

@inline function hamming(a, b)
    s = 0
    @inbounds @simd for i in eachindex(a); s += count_ones(a[i] ⊻ b[i]); end
    s
end
"Cosine-like similarity in [-1, 1] (1 = identical, 0 = orthogonal)."
sim(sp::Space, a, b) = (sp.D - 2 * hamming(a, b)) / sp.D
hamweight(v) = sum(count_ones, v)

# bit accessors (1-based bit positions)
@inline getbit(v, p) = ((v[(p - 1) >> 6 + 1] >> ((p - 1) & 63)) & one(UInt64)) == one(UInt64)
@inline setbit!(v, p) = (@inbounds v[(p - 1) >> 6 + 1] |= one(UInt64) << ((p - 1) & 63); nothing)
@inline clrbit!(v, p) = (@inbounds v[(p - 1) >> 6 + 1] &= ~(one(UInt64) << ((p - 1) & 63)); nothing)

# Deterministic, unbiased tie dither: a fixed ~50/50 function of the bit position, so an
# even-count majority tie at position p always resolves the same way. This makes bundle!
# (hence the whole encoder) a pure function of its inputs — reproducible scoring — while
# staying unbiased across positions. `rng` is kept in the signature for call-site
# compatibility but no longer consumed here.
@inline function _tiebit(p::Int)
    h = UInt64(p) * 0x9e3779b97f4a7c15
    h ⊻= h >> 29
    (h & one(UInt64)) == one(UInt64)
end

"""
    bundle!(sp, dst, hvs, acc, rng) -> dst

Majority vote (superposition) of the hypervectors in `hvs` into `dst`. `acc` is a reused
integer vote buffer of length ≥ `sp.W*64`. Even-count ties are broken by a deterministic
position dither (`_tiebit`) so identical inputs bundle identically. Only set bits are
visited, so cost ≈ (total set bits) + D.
"""
function bundle!(sp::Space, dst, hvs, acc, rng)
    n = length(hvs)
    fill!(acc, zero(Int32))
    @inbounds for h in hvs
        base = 0
        for i in eachindex(h)
            x = h[i]
            while x != 0
                acc[base + trailing_zeros(x) + 1] += one(Int32)
                x &= x - one(x)
            end
            base += 64
        end
    end
    thr = n >> 1; even = iseven(n)
    fill!(dst, zero(UInt64))
    @inbounds for p in 1:sp.D
        c = acc[p]
        if c > thr || (even && c == thr && _tiebit(p))
            dst[(p - 1) >> 6 + 1] |= one(UInt64) << ((p - 1) & 63)
        end
    end
    dst
end

"Uniformly random position (1-based) of a set bit of `v`, or 0 if none."
function random_setbit(v, rng)
    w = hamweight(v); w == 0 && return 0
    k = rand(rng, 1:w); c = 0
    @inbounds for i in eachindex(v)
        x = v[i]; pc = count_ones(x)
        if c + pc >= k
            for _ in 1:(k - c - 1); x &= x - one(x); end
            return (i - 1) * 64 + trailing_zeros(x) + 1
        end
        c += pc
    end
    0
end

"Random mask with bit density 2^-folds (AND of `folds` random HVs) — a purely-binary rate."
function sparse_mask!(sp::Space, mask, tmp, folds, rng)
    randhv!(sp, mask, rng)
    for _ in 2:folds
        randhv!(sp, tmp, rng)
        @inbounds @simd for i in eachindex(mask); mask[i] &= tmp[i]; end
    end
    mask
end
