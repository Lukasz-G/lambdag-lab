# Character-aware HDC word embeddings on real German prose (the LambdaG novels reference set).
#
#   julia --project=. examples/novels_de.jl                # small default run
#   MAXDOCS=400 MINCOUNT=25 EPOCHS=4 julia --project=. examples/novels_de.jl
#
# Shows (a) morphological neighbours from the CHARACTER structure alone (no training), then
# (b) how a few epochs of self-referential context learning add semantics on top.

using HDC, Random

const PATH = get(ENV, "NOVELS",
    joinpath(@__DIR__, "..", "..", "german", "av_reference_novels_de.jsonl"))
const MAXDOCS  = parse(Int, get(ENV, "MAXDOCS", "120"))
const MAXSENTS = parse(Int, get(ENV, "MAXSENTS", "8000"))  # 0 = all (training scales with this)
const MINCOUNT = parse(Int, get(ENV, "MINCOUNT", "15"))
const D        = parse(Int, get(ENV, "D", "8192"))
const EPOCHS   = parse(Int, get(ENV, "EPOCHS", "3"))
const FORMHOLD = parse(Int, get(ENV, "FORMHOLD", string(D ÷ 2)))  # frozen char-form bits

# ---- minimal JSONL reader (no JSON dependency): pull out one string field ----
function json_string(s::AbstractString, i::Int)          # i = index of the opening quote
    io = IOBuffer(); i = nextind(s, i); n = lastindex(s)
    while i <= n
        c = s[i]
        if c == '\\'
            i = nextind(s, i); e = s[i]
            if     e == 'n'; write(io, '\n')
            elseif e == 't'; write(io, '\t')
            elseif e == 'r'; write(io, '\r')
            elseif e == 'u'
                h = s[nextind(s, i):nextind(s, i, 4)]
                write(io, Char(parse(UInt16, h; base = 16))); i = nextind(s, i, 4)
            else write(io, e)
            end
            i = nextind(s, i)
        elseif c == '"'
            return String(take!(io))
        else
            write(io, c); i = nextind(s, i)
        end
    end
    String(take!(io))
end

function field(line::AbstractString, key::String)
    p = findfirst("\"" * key * "\"", line); p === nothing && return ""
    i = nextind(line, last(p))                    # move PAST the key's closing quote
    while i <= lastindex(line) && line[i] != '"'; i = nextind(line, i); end
    i > lastindex(line) ? "" : json_string(line, i)
end

# ---- tokenise: lowercase, split into sentences, words = German letter runs ----
tokenize(text) = [toks for chunk in split(text, r"[.!?\n]+")
                  for toks in (String[m.match for m in eachmatch(r"[a-zäöüß]+", lowercase(chunk))],)
                  if length(toks) >= 2]

function load(path; maxdocs)
    sents = Vector{String}[]
    open(path) do io
        for (d, line) in enumerate(eachline(io))
            d > maxdocs && break
            isempty(strip(line)) && continue
            append!(sents, tokenize(field(line, "text")))
            MAXSENTS > 0 && length(sents) >= MAXSENTS && break
        end
    end
    MAXSENTS > 0 && length(sents) > MAXSENTS ? sents[1:MAXSENTS] : sents
end

println("loading up to $MAXDOCS docs from $(basename(PATH)) ...")
sents = load(PATH; maxdocs = MAXDOCS)
vocab, w2i, counts, ids = build_vocab(sents; min_count = MINCOUNT)
isempty(vocab) && (println("empty vocab -- raise MAXDOCS or lower MINCOUNT"); exit())
ntok = sum(length, ids; init = 0)
println("  $(length(sents)) sentences, $ntok tokens, vocab $(length(vocab)) (min_count=$MINCOUNT), D=$D")

# ---- character-structured initialisation (morphology-aware, no context yet) ----
# formhold freezes the low FORMHOLD bits = the character-form region, so context learning
# below can only edit the rest and the spelling structure PERSISTS in the trained embedding.
cb = Codebook(D, vocab, w2i, counts; sub = 3, formhold = FORMHOLD, rng = MersenneTwister(1))
ce = CharEncoder(vocab, D; sub = 3, pos = POS_NONE, rng = MersenneTwister(2))
init_from_forms!(cb, ce; rng = MersenneTwister(3))
println("  form-anchor: $FORMHOLD / $D bits frozen (character form), $(D - FORMHOLD) learnable")

probes = filter(w -> haskey(w2i, w),
                ["mann", "frau", "kind", "haus", "auge", "hand", "liebe", "leben"])
println("\n[character structure only] nearest neighbours ~ spelling relatives:")
for w in probes[1:min(5, end)]
    println("  $w -> ", nearest(cb, w; k = 6))
end
# OOV word gets a vector straight from its characters:
if !isempty(probes)
    oov = "unbekanntesfremdwort"
    v = word_form(ce, oov, MersenneTwister(9))
    best = sort([(sim(cb.sp, v, @view cb.E[:, u]), vocab[u]) for u in eachindex(vocab)]; rev = true)[1:5]
    println("  OOV \"$oov\" nearest in-vocab: ", [(round(s, digits = 3), w) for (s, w) in best])
end

# ---- self-referential context learning (adds semantics on top of the frozen form) ----
# Gentle repulsion (weak neg flips, few negatives) so context NUDGES rather than decorrelates;
# the frozen form-anchor guarantees the character structure cannot be washed out regardless.
println("\ntraining AttractRepel for $EPOCHS epochs (adds context on the learnable bits) ...")
enc = Encoder(cb; stride = 1, pos = POS_GRADED, maxwin = 20, rng = MersenneTwister(4))
train!(enc, ids, AttractRepel(3, 7, 3); window = 3, epochs = EPOCHS, rng = MersenneTwister(5), log = stdout)

println("\n[after context learning] nearest neighbours (frozen form + learned context):")
for w in probes[1:min(6, end)]
    println("  $w -> ", nearest(cb, w; k = 6))
end
