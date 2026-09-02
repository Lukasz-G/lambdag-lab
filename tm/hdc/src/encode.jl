# Window encoder: overlapping trigram sub-bundles -> slot-bind -> majority bundle -> ONE
# fixed-D binary vector (the feature vector handed to the Tsetlin Machine).
#
#   sub-bundle  S_i = Majority_j( Pin_j ⊕ E[word_{i+j}] )      (odd `sub` => no ties)
#   vector      V   = Majority_i( slot_i ⊕ S_i )               (slot = position of trigram)

@enum PosScheme POS_NONE POS_SHARP POS_GRADED POS_BANDS

"""
    Encoder(cb; stride=1, pos=POS_GRADED, nbands=3, maxwin=40, rng)

Turns a window of token ids into one packed-binary feature vector. `sub` (trigram size) is
taken from the codebook (`size(cb.Pin, 2)`). `pos` sets how a trigram's position (its
distance-to-target, since the window is target-relative) is encoded:
`POS_NONE` (bag-of-trigrams), `POS_SHARP` (orthogonal per slot), `POS_GRADED` (thermometer,
neighbours similar — recommended), `POS_BANDS` (`nbands` coarse position buckets).
"""
struct Encoder
    cb::Codebook
    sub::Int
    stride::Int
    pos::PosScheme
    nbands::Int
    maxslots::Int
    slot::Matrix{UInt64}                 # W × (maxslots or nbands) position codes
    subhv::Vector{Vector{UInt64}}        # scratch: intra-trigram bound tokens
    subacc::Vector{Int32}
    sbuf::Vector{Vector{UInt64}}         # scratch: sub-bundles
    acc::Vector{Int32}
end

function Encoder(cb::Codebook; stride::Integer = 1, pos::PosScheme = POS_GRADED,
                 nbands::Integer = 3, maxwin::Integer = 40, rng = Random.default_rng())
    sp = cb.sp; sub = size(cb.Pin, 2)
    maxslots = max(1, (Int(maxwin) - sub) ÷ Int(stride) + 1)
    slot = pos == POS_SHARP  ? randcodes(sp, maxslots, rng) :
           pos == POS_GRADED ? level_codes(sp, maxslots, rng) :
           pos == POS_BANDS  ? randcodes(sp, nbands, rng) :
                               zeros(UInt64, sp.W, 1)
    Encoder(cb, sub, Int(stride), pos, Int(nbands), maxslots, slot,
            [newhv(sp) for _ in 1:sub], Vector{Int32}(undef, sp.W * 64),
            [newhv(sp) for _ in 1:(maxslots + 1)], Vector{Int32}(undef, sp.W * 64))
end

@inline function slot_index(enc::Encoder, si::Int)
    enc.pos == POS_BANDS ? clamp(cld(si * enc.nbands, enc.maxslots), 1, enc.nbands) :
                           clamp(si, 1, size(enc.slot, 2))
end

"""
    encode_window!(enc, out, tokens, rng) -> out

Encode a window of token ids into `out` (a hypervector). Deterministic for a fixed `rng`
state (only the majority tie-break consumes randomness). Windows shorter than `sub` are
bundled as a single group.
"""
function encode_window!(enc::Encoder, out, tokens::AbstractVector{<:Integer}, rng)
    sp = enc.cb.sp; E = enc.cb.E; Pin = enc.cb.Pin; sub = enc.sub
    K = length(tokens); nsub = 0
    if K >= sub
        si = 0
        s = 1
        while s <= K - sub + 1 && nsub < enc.maxslots
            si += 1; nsub += 1
            @inbounds for j in 1:sub
                bind!(enc.subhv[j], @view(Pin[:, j]), @view(E[:, tokens[s + j - 1]]))
            end
            bundle!(sp, enc.sbuf[nsub], view(enc.subhv, 1:sub), enc.subacc, rng)
            enc.pos != POS_NONE && xor!(enc.sbuf[nsub], @view enc.slot[:, slot_index(enc, si)])
            s += enc.stride
        end
    elseif K >= 1
        nsub = 1
        @inbounds for j in 1:K
            bind!(enc.subhv[j], @view(Pin[:, min(j, sub)]), @view(E[:, tokens[j]]))
        end
        bundle!(sp, enc.sbuf[1], view(enc.subhv, 1:K), enc.subacc, rng)
    else
        fill!(out, zero(UInt64)); return out
    end
    bundle!(sp, out, view(enc.sbuf, 1:nsub), enc.acc, rng)
    out
end

encode_window(enc::Encoder, tokens, rng) =
    encode_window!(enc, newhv(enc.cb.sp), tokens, rng)
