# formhold sweep: how much char-form to freeze vs let context learning overwrite. D/2 (prev)
# kept morphology + rhymes but blocked suppletion. Test D/4, D/8, 0 (fully learnable) to see
# if less anchoring lets the context learner pull suppletive/functional families together
# (sein->ist/war, kann->können). One process (loops formhold), corpus + char forms reused.
#
#   julia -t auto phase3/formhold_sweep.jl

import Pkg
Pkg.activate(joinpath(@__DIR__, "..", "hdc"); io = devnull)
using HDC, Random, Printf

const D = 8192
const EPOCHS = parse(Int, get(ENV, "P3_EPOCHS", "3"))
const MINCOUNT = 2
const MAXSENTS = parse(Int, get(ENV, "P3_MAXSENTS", "28000"))
const FORMDIVS = [parse(Int, x) for x in split(get(ENV, "P3_FORMDIVS", "4,8,0"), ",")]  # 0 = no anchor
const HERE = @__DIR__

readsents(p) = [String.(split(l, '\t')) for l in eachline(p) if !isempty(l)]
const PROBES = ["sein", "kann", "der", "ich", "müssen", "hat", "wird"]
# targets we HOPE appear (functional/suppletive relatives char n-grams can't reach)
const TARGETS = Dict("sein" => ["ist", "war", "bin", "sind"], "kann" => ["können", "könnte", "muss", "will"],
                     "der" => ["die", "das", "dem", "den"], "ich" => ["mich", "dich", "sich", "er"],
                     "müssen" => ["können", "wollen", "sollen"], "hat" => ["hatte", "ist", "war"],
                     "wird" => ["ist", "war", "wurde", "sind"])
function probe(cb, w, k = 10)
    haskey(cb.word2id, w) || return
    nn = nearest(cb, w; k = k); hit = [x for (x, _) in nn if x in get(TARGETS, w, String[])]
    @printf("  %-8s -> %s   [targets hit: %s]\n", w,
            join([@sprintf("%s(%.2f)", x, s) for (x, s) in nn[1:min(6, end)]], " "),
            isempty(hit) ? "none" : join(hit, ","))
end

function main()
    sents = readsents(joinpath(HERE, "pretrain.tsv"))
    length(sents) > MAXSENTS && (shuffle!(MersenneTwister(0), sents); sents = sents[1:MAXSENTS])
    vocab, w2i, counts, ids = build_vocab(sents; min_count = MINCOUNT)
    @printf("corpus %d tokens, vocab %d, D=%d, epochs=%d\n", sum(length, ids; init = 0), length(vocab), D, EPOCHS)
    ce = CharEncoder(vocab, D; sub = 3, pos = POS_NONE, rng = MersenneTwister(2))

    for fd in FORMDIVS
        formhold = fd == 0 ? 0 : D ÷ fd
        @printf("\n===== formhold = %d / %d  (%s) =====\n", formhold, D, fd == 0 ? "no anchor" : "D/$fd"); flush(stdout)
        cb = Codebook(D, vocab, w2i, counts; sub = 3, formhold = formhold, rng = MersenneTwister(1))
        init_from_forms!(cb, ce; rng = MersenneTwister(3))
        enc = Encoder(cb; stride = 1, pos = POS_GRADED, maxwin = 8, rng = MersenneTwister(4))
        train!(enc, ids, AttractRepel(3, 7, 3); window = 5, epochs = EPOCHS, rng = MersenneTwister(5))
        d = diagnostics(cb; rng = MersenneTwister(9))
        @printf("  [mean_sim=%.3f bit_entropy=%.3f]\n", d.mean_sim, d.bit_entropy)
        for w in PROBES; probe(cb, w); end
        flush(stdout)
    end
end

main()
