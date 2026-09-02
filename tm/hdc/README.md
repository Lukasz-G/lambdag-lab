# HDC.jl — binary Hyperdimensional Computing core

A self-contained, movable Julia package for the **POSNoise → HDC → Tsetlin-Machine** pipeline
and for the self-referential-embedding research. Everything is packed-`UInt64` binary — **no
floating-point in the vector math**. The package emits **binary feature vectors**; the Tsetlin
Machine lives elsewhere (features-out boundary).

## Layout

```
hdc/
  Project.toml
  src/
    HDC.jl          # module + exports
    bits.jl         # Space + packed-bit primitives (bind!, bundle!, hamming, sim, …)
    codebook.jl     # random item memory (word atoms + position codes) + neg-sampling
    encode.jl       # overlapping-trigram window  ->  ONE fixed-D feature vector
    learn.jl        # self-referential training with pluggable UpdateRules
    diagnostics.jl  # collapse metrics + nearest neighbours
  test/runtests.jl
```

## Quickstart

```julia
using Pkg; Pkg.activate("hdc"); Pkg.instantiate()
using HDC, Random

# 1. vocab + integer-encoded sentences (Vector{Vector{String}} in, ids out)
vocab, w2i, counts, ids = build_vocab(sentences; min_count=5)

# 2. item memory (D-bit random atoms) + the window encoder
cb  = Codebook(10_000, vocab, w2i, counts; sub=3)                 # trigram sub-bundles
enc = Encoder(cb; stride=1, pos=POS_GRADED, maxwin=20)            # overlapping, graded position

# 3a. feature vector for a window  ->  hand this to the Tsetlin Machine
feat = encode_window(enc, token_ids, MersenneTwister(0))         # Vector{UInt64}, D bits

# 3b. (optional) self-referentially LEARN the word atoms first
train!(enc, ids, AttractRepel(4,5,5); window=3, epochs=8, log=stdout, labels=labels)
nearest(cb, "king"; k=5)
```

## The encoder (one fixed-D vector)

`encode_window` builds, for a target-relative window:
`S_i = Majority_j(Pin_j ⊕ E[word])` over `sub`=3 overlapping tokens (odd → no ties), binds each
`S_i` to its slot, and majority-bundles them into one vector. Knobs: `sub` (from the codebook),
`stride` (overlap), and `pos` — `POS_NONE` (bag-of-trigrams), `POS_SHARP`, `POS_GRADED`
(thermometer, neighbours similar — recommended), `POS_BANDS(nbands)`.

## Character-aware embeddings (words that grasp their spelling)

`CharEncoder` reuses the trigram bundler **at the character level** (fastText-style char
n-grams over `<word>`), so a word atom is built from its characters — similar spellings start
similar. `init_from_forms!(cb, ce)` initialises every word embedding from its form; morphology,
compounds and OOV words all fall out (`unbekanntesfremdwort` → `unbekannt*`, from characters).

To keep that structure *through* context learning, build the codebook with a **form-anchor**:
`Codebook(D, …; formhold = D÷2)` (or `form_anchor!(cb, D÷2)`) freezes the low `formhold` bits —
the learner edits only the rest, so the spelling half is never overwritten. See
`examples/novels_de.jl` (real German prose): with the anchor, `haus → hause, wirtshaus`
survives training instead of decorrelating.

## Update rules & the collapse question

The central research question is *which local binary rule has a non-collapsing equilibrium*.
Rules are dispatched (`update!(::UpdateRule, …)`), so they share one loop and the diagnostics:

| Rule | mechanism | expectation |
|---|---|---|
| `Surprise` | pure attraction | collapses (voter/diffusion fixed point) |
| `AttractRepel` | attraction + repulsion | genuine equilibrium |
| `SparseCapacity` | attraction, fixed Hamming weight | still collapses (capacity alone insufficient) |
| `EnergyAlign` | Ising: align to (want C, want ≠ negatives) | equilibrium via frustration |

`diagnostics(cb)` returns the collapse signature: `mean_sim`→1, `sim_std`→0, **`bit_entropy`→0**
(consensus, weight-independent) mean collapse; high `purity` means useful structure.

## Boundary

Features out only. When you share your Julia TM's `fit/predict` API I'll add a thin adapter
(`feature vector → clause literals`, and `class votes → softmax → per-token P` for the LambdaG
`λ_G`). No Python/POSNoise bridge yet — the input is plain token ids.
