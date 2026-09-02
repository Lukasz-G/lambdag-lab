# Character-structured word embeddings.
# Reuse the trigram bundler ONE level down -- over the characters of "<word>" (fastText-style
# boundary markers so prefixes/suffixes are distinct). Each word atom becomes a bundle of its
# character n-grams, so similarly-spelled words (inflections, compounds) start with similar
# hypervectors. The embedding stays per-WORD; the character structure is baked into it.

struct CharEncoder
    enc::Encoder
    char2id::Dict{Char,Int}
    lo::Int        # id of '<'  (word-start marker)
    hi::Int        # id of '>'  (word-end marker)
end

"""
    CharEncoder(vocab, D; sub=3, pos=POS_NONE, maxword=64, rng)

A character-level encoder over the alphabet of `vocab` (+ '<','>' markers). `pos=POS_NONE`
gives a fastText-style bag of character trigrams (order kept inside each trigram); use
`POS_GRADED` to make position within the word matter.
"""
function CharEncoder(vocab, D::Integer; sub::Integer = 3, stride::Integer = 1,
                     pos::PosScheme = POS_NONE, maxword::Integer = 64,
                     rng = Random.default_rng())
    chars = Set{Char}(('<', '>'))
    for w in vocab, c in w; push!(chars, c); end
    clist = sort!(collect(chars))
    cvocab = [string(c) for c in clist]
    c2i = Dict(c => i for (i, c) in enumerate(clist))
    ccb = Codebook(D, cvocab, Dict(cvocab[i] => i for i in eachindex(cvocab)),
                   fill(1, length(cvocab)); sub = sub, rng = rng)
    cenc = Encoder(ccb; stride = stride, pos = pos, maxwin = maxword, rng = rng)
    CharEncoder(cenc, c2i, c2i['<'], c2i['>'])
end

"Encode a word from its character structure into `out` (works for OOV words too)."
function word_form!(ce::CharEncoder, out, word::AbstractString, rng)
    ids = Int[ce.lo]
    for c in word
        i = get(ce.char2id, c, 0)
        i != 0 && push!(ids, i)          # skip characters unseen at build time
    end
    push!(ids, ce.hi)
    encode_window!(ce.enc, out, ids, rng)
end
word_form(ce::CharEncoder, word::AbstractString, rng) =
    word_form!(ce, newhv(ce.enc.cb.sp), word, rng)

"Initialise every word atom in `cb` from its character-structured form (morphology-aware init)."
function init_from_forms!(cb::Codebook, ce::CharEncoder; rng = Random.default_rng())
    for v in eachindex(cb.vocab)
        @views word_form!(ce, cb.E[:, v], cb.vocab[v], rng)
    end
    cb
end
