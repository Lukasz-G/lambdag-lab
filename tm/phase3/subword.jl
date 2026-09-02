# Learned-subword compositional atoms (fastText-in-HDC), with STRICT positional encoding.
#
#   word(w) = majority_bundle over slots of ( SW[subword_s] XOR Q[slot] )
#     - SW[s]  : a LEARNABLE atom per character-trigram subword (shared across all words)
#     - Q[slot]: STRICT (orthogonal) position code -> exact position of each fragment
#   No form-anchor: morphology is STRUCTURAL (the word IS its subwords), not a frozen bit region.
#
# Training (self-referential AttractRepel) updates the SUBWORDS, not word atoms. Because the
# composition binds each subword with XOR, moving a word toward context C means moving each of
# its subwords toward (C XOR Q[slot]) -- a clean unbind-by-position, so pull!/push_away! apply
# directly. Shared subwords => concatenative/umlaut families generalise; suppletion (sein/ist)
# still relies on context alone (disjoint subwords).
#
#   julia -t auto phase3/subword.jl

import Pkg
Pkg.activate(joinpath(@__DIR__, "..", "hdc"); io = devnull)
using HDC, Random, Printf, Serialization

const D = 8192
const NGRAM = 3
const EPOCHS = parse(Int, get(ENV, "P3_EPOCHS", "4"))
const MINCOUNT = 2
const MAXSENTS = parse(Int, get(ENV, "P3_MAXSENTS", "28000"))
const MAXSLOTS = 24                              # cap word length in trigram slots
const POS_FOLDS = 3; const NEG_FOLDS = 7; const NNEG = 3   # AttractRepel(3,7,3)
const WINDOW = 5
const HERE = @__DIR__

readsents(p) = [String.(split(l, '\t')) for l in eachline(p) if !isempty(l)]

# ---- subword inventory: char trigrams of <w>, each with its slot (position) ----
function build_subwords(vocab)
    sw2id = Dict{String,Int}()
    word_subs = Vector{Vector{Tuple{Int,Int}}}(undef, length(vocab))
    for (wi, w) in enumerate(vocab)
        chars = collect("<" * w * ">")
        subs = Tuple{Int,Int}[]
        for i in 1:max(0, length(chars) - NGRAM + 1)
            g = String(@view chars[i:i+NGRAM-1])
            id = get!(sw2id, g, length(sw2id) + 1)
            push!(subs, (id, min(i, MAXSLOTS)))
        end
        if isempty(subs)                          # word shorter than NGRAM
            id = get!(sw2id, String(chars), length(sw2id) + 1); push!(subs, (id, 1))
        end
        word_subs[wi] = subs
    end
    sw2id, word_subs
end

# ---- compose one word from its (current) subword atoms + strict position codes ----
function compose_into!(dst, SW, Q, subs, sp, bound, acc, rng)
    n = length(subs)
    @inbounds for (k, (sid, slot)) in enumerate(subs)
        b = bound[k]
        @simd for i in 1:sp.W; b[i] = SW[i, sid] ⊻ Q[i, slot]; end
    end
    bundle!(sp, dst, view(bound, 1:n), acc, rng)
end
function recompose!(cb, SW, Q, word_subs, bound, acc, rng)
    @inbounds for w in eachindex(word_subs)
        compose_into!(@view(cb.E[:, w]), SW, Q, word_subs[w], cb.sp, bound, acc, rng)
    end
end

# ---- one AttractRepel step, propagated to subwords via unbind-by-position ----
function sw_step!(SW, Q, word_subs, t, C, cb, sp, mask, tmp, ub, rng)
    @inbounds for (sid, slot) in word_subs[t]                 # pull target's subwords toward C
        @simd for i in 1:sp.W; ub[i] = C[i] ⊻ Q[i, slot]; end
        pull!(sp, @view(SW[:, sid]), ub, mask, tmp, POS_FOLDS, rng)
    end
    for _ in 1:NNEG                                           # push negatives' subwords away
        n = sample_neg(cb, rng); n == t && continue
        @inbounds for (sid, slot) in word_subs[n]
            @simd for i in 1:sp.W; ub[i] = C[i] ⊻ Q[i, slot]; end
            push_away!(sp, @view(SW[:, sid]), ub, mask, tmp, NEG_FOLDS, rng)
        end
    end
end

const PROBES = ["sein", "kann", "der", "ich", "müssen", "hat", "wird", "muss"]
show_nn(cb, w) = haskey(cb.word2id, w) &&
    @printf("  %-8s -> %s\n", w, join([@sprintf("%s(%.2f)", x, s) for (x, s) in nearest(cb, w; k = 8)], " "))

function main()
    sents = readsents(joinpath(HERE, "pretrain.tsv"))
    length(sents) > MAXSENTS && (shuffle!(MersenneTwister(0), sents); sents = sents[1:MAXSENTS])
    vocab, w2i, counts, ids = build_vocab(sents; min_count = MINCOUNT)
    V = length(vocab)
    sw2id, word_subs = build_subwords(vocab)
    nsw = length(sw2id)
    @printf("corpus %d tokens, vocab %d, subwords %d, D=%d, epochs=%d\n",
            sum(length, ids; init = 0), V, nsw, D, EPOCHS); flush(stdout)

    sp = Space(D); rng = MersenneTwister(7)
    SW = randcodes(sp, nsw, MersenneTwister(1))               # learnable subword atoms
    Q = randcodes(sp, MAXSLOTS, MersenneTwister(2))           # STRICT (orthogonal) position codes
    cb = Codebook(D, vocab, w2i, counts; sub = 3, rng = MersenneTwister(3))  # E := composed words
    bound = [newhv(sp) for _ in 1:MAXSLOTS]; acc = Vector{Int32}(undef, sp.W * 64)
    mask = newhv(sp); tmp = newhv(sp); ub = newhv(sp); C = newhv(sp)
    enc = Encoder(cb; stride = 1, pos = POS_GRADED, maxwin = 8, rng = MersenneTwister(4))

    recompose!(cb, SW, Q, word_subs, bound, acc, rng)
    # self-check: deterministic compose + morphological signal present at init
    let d1 = copy(cb.E), _ = recompose!(cb, SW, Q, word_subs, bound, acc, rng)
        det = cb.E == d1
        s(a, b) = sim(sp, @view(cb.E[:, w2i[a]]), @view(cb.E[:, w2i[b]]))
        @printf("self-check: compose deterministic=%s ; sim(muss,müssen)=%.2f vs sim(muss,der)=%.2f\n",
                det, s("muss", "müssen"), s("muss", "der")); flush(stdout)
    end

    println("\n=== BEFORE training (composed from random subwords) ===")
    for w in PROBES; show_nn(cb, w); end; flush(stdout)

    @printf("\ntraining %d epochs (subword AttractRepel, strict position) ...\n", EPOCHS); flush(stdout)
    for epoch in 1:EPOCHS
        t0 = @elapsed for sent in ids
            L = length(sent)
            for c in 1:L
                t = sent[c]
                ctx = Int[]
                for o in max(1, c - WINDOW):min(L, c + WINDOW); o == c || push!(ctx, sent[o]); end
                isempty(ctx) && continue
                encode_window!(enc, C, ctx, rng)             # context from CURRENT composed words
                sw_step!(SW, Q, word_subs, t, C, cb, sp, mask, tmp, ub, rng)
            end
        end
        recompose!(cb, SW, Q, word_subs, bound, acc, rng)    # refresh word vectors from updated subwords
        d = diagnostics(cb; rng = MersenneTwister(9))
        @printf("  epoch %d  %.0fs  mean_sim=%.3f bit_entropy=%.3f\n", epoch, t0, d.mean_sim, d.bit_entropy); flush(stdout)
    end

    println("\n=== AFTER training ===")
    for w in PROBES; show_nn(cb, w); end
    serialize(joinpath(HERE, "subword_atoms.jls"), (SW, Q, word_subs, vocab, w2i))
    println("saved -> subword_atoms.jls")
end

main()
