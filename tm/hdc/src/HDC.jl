"""
    HDC

Binary Hyperdimensional Computing core for the POSNoise → HDC → Tsetlin-Machine pipeline
(and for the self-referential-embedding research). Everything is packed-`UInt64` binary:
no floats in the vector math. The package emits **binary feature vectors**; the Tsetlin
Machine lives elsewhere.

Pieces:
- `Space`, packed-bit primitives (`bind!`, `bundle!`, `hamming`, `sim`, …)          — bits.jl
- `Codebook` random item memory + position codes + negative sampling               — codebook.jl
- `Encoder` overlapping-trigram window → one fixed-D feature vector                 — encode.jl
- `train!` self-referential learning with pluggable `UpdateRule`s                    — learn.jl
- `diagnostics` collapse metrics (`mean_sim`/`sim_std`/`bit_entropy`/`purity`)       — diagnostics.jl
"""
module HDC

using Random

export Space, newhv, randhv, randhv!, bind!, xor!, hamming, sim, hamweight,
       getbit, setbit!, clrbit!, bundle!, random_setbit, sparse_mask!,
       Codebook, build_vocab, sample_neg, randcodes, level_codes, form_anchor!,
       Encoder, PosScheme, POS_NONE, POS_SHARP, POS_GRADED, POS_BANDS,
       encode_window, encode_window!,
       UpdateRule, Surprise, AttractRepel, SparseCapacity, EnergyAlign,
       pull!, push_away!, train!,
       diagnostics, neighbor_purity, nearest,
       CharEncoder, word_form, word_form!, init_from_forms!

include("bits.jl")
include("codebook.jl")
include("diagnostics.jl")
include("encode.jl")
include("learn.jl")
include("chars.jl")

end # module HDC
