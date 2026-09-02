# Pretrain HDC token atoms on POSNoised German reference text:
#   char-init (init_from_forms!) + form-anchor (formhold=D/2) + AttractRepel context learning.
# Prints nearest-neighbour probes BEFORE (char-only) vs AFTER (context) so we can SEE whether
# morphological families pull together — especially the SUPPLETIVE ones (sein/ist/war) that
# character n-grams alone cannot connect. Serialises the frozen learned codebook.
#
#   julia -t auto phase3/pretrain.jl

import Pkg
Pkg.activate(joinpath(@__DIR__, "..", "hdc"); io = devnull)
using HDC, Random, Printf, Serialization

const D = parse(Int, get(ENV, "P3_D", "8192"))
const EPOCHS = parse(Int, get(ENV, "P3_EPOCHS", "4"))
const MINCOUNT = parse(Int, get(ENV, "P3_MINCOUNT", "2"))
const MAXSENTS = parse(Int, get(ENV, "P3_MAXSENTS", "28000"))   # cap (self-ref learner is 1-threaded)
const HERE = @__DIR__

readsents(p) = [String.(split(l, '\t')) for l in eachline(p) if !isempty(l)]

# probe families; the key test is suppletion (sein/haben) that char-similarity can't bridge
const PROBES = ["sein", "ist", "haben", "hat", "müssen", "können", "werden",
                "der", "ich", "kann", "wird", "nicht", "und", "war", "muss"]
show_nn(cb, w, k = 8) = haskey(cb.word2id, w) &&
    @printf("  %-9s -> %s\n", w, join([@sprintf("%s(%.2f)", x, s) for (x, s) in nearest(cb, w; k = k)], " "))

function main()
    sents = readsents(joinpath(HERE, "pretrain.tsv"))
    length(sents) > MAXSENTS && (shuffle!(MersenneTwister(0), sents); sents = sents[1:MAXSENTS])  # spread across authors
    vocab, w2i, counts, ids = build_vocab(sents; min_count = MINCOUNT)
    ntok = sum(length, ids; init = 0)
    @printf("corpus: %d sentences, %d tokens, vocab %d (min_count=%d), D=%d\n",
            length(sents), ntok, length(vocab), MINCOUNT, D); flush(stdout)

    cb = Codebook(D, vocab, w2i, counts; sub = 3, formhold = D ÷ 2, rng = MersenneTwister(1))
    ce = CharEncoder(vocab, D; sub = 3, pos = POS_NONE, rng = MersenneTwister(2))
    init_from_forms!(cb, ce; rng = MersenneTwister(3))
    @printf("char-init done (form-anchor: %d/%d bits frozen)\n\n", D ÷ 2, D); flush(stdout)

    println("=== BEFORE training (character form only) ===")
    for w in PROBES; show_nn(cb, w); end
    flush(stdout)

    enc = Encoder(cb; stride = 1, pos = POS_GRADED, maxwin = 8, rng = MersenneTwister(4))
    @printf("\ntraining AttractRepel(3,7,3) for %d epochs ...\n", EPOCHS); flush(stdout)
    train!(enc, ids, AttractRepel(3, 7, 3); window = 5, epochs = EPOCHS,
           rng = MersenneTwister(5), log = stdout)

    println("\n=== AFTER training (form + learned context) ===")
    for w in PROBES; show_nn(cb, w); end

    d = diagnostics(cb; rng = MersenneTwister(9))
    @printf("\nnon-collapse check: mean_sim=%.3f  sim_std=%.3f  bit_entropy=%.3f\n",
            d.mean_sim, d.sim_std, d.bit_entropy)

    serialize(joinpath(HERE, "atoms.jls"), cb)
    println("saved pretrained codebook -> atoms.jls")
end

main()
