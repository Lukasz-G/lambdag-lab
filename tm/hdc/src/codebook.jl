# Item memory: the random atomic hypervectors (word "embeddings" + position codes) plus
# vocabulary bookkeeping and the negative-sampling table.

"`n` random (≈orthogonal) hypervectors as columns of a W×n matrix."
function randcodes(sp::Space, n::Integer, rng)
    P = Matrix{UInt64}(undef, sp.W, max(Int(n), 1))
    for j in 1:size(P, 2); @views randhv!(sp, P[:, j], rng); end
    P
end

"""
Graded / thermometer position codes `L_1 … L_n`: consecutive columns differ by a fixed
number of bit flips, so `sim(L_i, L_j)` decays ~linearly with `|i-j|` and the endpoints are
≈orthogonal. Use for smooth (translation-tolerant) position encoding.
"""
function level_codes(sp::Space, n::Integer, rng)
    n = Int(n)
    P = Matrix{UInt64}(undef, sp.W, max(n, 1))
    @views randhv!(sp, P[:, 1], rng)
    n <= 1 && return P
    per = max(1, (sp.D ÷ 2) ÷ (n - 1))        # total flips ≈ D/2 -> endpoints ≈orthogonal
    order = randperm(rng, sp.D)
    idx = 1
    for j in 2:n
        @views copyto!(P[:, j], P[:, j - 1])
        col = @view P[:, j]
        for _ in 1:per
            idx > sp.D && break
            p = order[idx]; idx += 1
            getbit(col, p) ? clrbit!(col, p) : setbit!(col, p)
        end
    end
    P
end

"""
Word item memory + position codes. `E` (W×V) holds one hypervector per vocabulary word; it
is the atom the encoder reads and the embedding the learner evolves. `Pin` (W×sub) are the
intra-trigram position codes.
"""
mutable struct Codebook
    sp::Space
    vocab::Vector{String}
    word2id::Dict{String,Int}
    counts::Vector{Int}
    E::Matrix{UInt64}          # W × V  — word atoms / embeddings (evolve if learned)
    Pin::Matrix{UInt64}        # W × sub — intra-trigram position codes (fixed)
    cum::Vector{Float64}       # cumulative count^0.75 (negative sampling)
    writable::Union{Nothing,Vector{UInt64}}   # bits the learner may flip (nothing = all)
end

"Mask with bits `formhold+1 … D` set (the region the learner may edit); `nothing` if formhold=0."
function anchor_mask(sp::Space, formhold::Integer)
    formhold = clamp(Int(formhold), 0, sp.D)
    formhold == 0 && return nothing
    w = zeros(UInt64, sp.W)
    @inbounds for p in (formhold + 1):sp.D; setbit!(w, p); end
    w
end

"""
    form_anchor!(cb, formhold)

Freeze bits `1 … formhold` of every word atom — the character-form region written by
`init_from_forms!`. Training only edits bits above `formhold`, so the character structure
persists in the learned embedding. `formhold=0` removes the anchor.
"""
form_anchor!(cb::Codebook, formhold::Integer) = (cb.writable = anchor_mask(cb.sp, formhold); cb)

"Vocabulary + integer-encoded sentences from tokenised text (drops words below `min_count`)."
function build_vocab(sentences; min_count::Integer = 5)
    cnt = Dict{String,Int}()
    for s in sentences, w in s; cnt[w] = get(cnt, w, 0) + 1; end
    vocab = sort!([w for (w, c) in cnt if c >= min_count])
    word2id = Dict(w => i for (i, w) in enumerate(vocab))
    counts = [cnt[w] for w in vocab]
    ids = [Int[word2id[w] for w in s if haskey(word2id, w)] for s in sentences]
    vocab, word2id, counts, ids
end

function Codebook(D::Integer, vocab, word2id, counts; sub::Integer = 3,
                  formhold::Integer = 0, rng = Random.default_rng())
    sp = Space(D); V = length(vocab)
    E = Matrix{UInt64}(undef, sp.W, V)
    for v in 1:V; @views randhv!(sp, E[:, v], rng); end   # random init breaks symmetry
    Pin = randcodes(sp, sub, rng)
    cum = cumsum(Float64.(counts) .^ 0.75); cum ./= cum[end]
    Codebook(sp, collect(String, vocab), Dict{String,Int}(word2id),
             collect(Int, counts), E, Pin, cum, anchor_mask(sp, formhold))
end

"Sample a word id ∝ count^0.75 (for negative sampling)."
@inline sample_neg(cb::Codebook, rng) = searchsortedfirst(cb.cum, rand(rng))
