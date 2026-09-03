# Cross-genre pairwise verification with a HELD-OUT (language, genre)
# combination, learned by a Tsetlin machine.
#
#   julia -t auto phase4/pair_tm_xgenre.jl german poetree [wknown] [wquest] [centre]
#
# The question: a questioned text in genre G* of language L*, whose candidate
# author is known only through another genre of L*, where NO (L*, G*) material
# may inform the verifier. Whatever "surviving a genre crossing" means must
# therefore be learned from OTHER languages' cross-genre pairs and from L*'s
# other genre pairs.
#
# A pair is (known window from genre g1, questioned window from genre g2) of
# the SAME language; positive when one author stands on both sides, negative
# when two authors do, always within the same (language, g1, g2) stratum -- so
# neither language nor genre separates the labels and the machine must learn
# agreement that outlives the genre change. An instance is HELD OUT when its
# language is L* and either of its genres is G*.
#
# Leakage control (every FITTED quantity excludes the held-out combination):
#   alphabet   the *_nopoe encodings, whose class-rank map is fitted on each
#              language's non-poetry genres -- correct exactly while G* is
#              poetry, which the script asserts
#   vocabulary accumulated over TRAINING windows only
#   centring   the mean profile of a (language, genre) is fitted on that
#              cell's training windows; the held-out cell has none, so it
#              borrows a legal substitute, and which substitute is an explicit
#              arm mirroring the generative experiment:
#                 known-genre  L*'s own known-genre mean  (right language)
#                 foreign      the pooled mean of G* in other languages
#   binarise   thermometer cut points from the TRAINING feature matrix only
#
# Both text lengths are parameters and are reported with every result, since a
# cross-genre comparison is only interpretable against the amount of evidence
# on each side. Defaults are SYMMETRIC (1000/1000).

using Random, Printf, Statistics
include(joinpath(@__DIR__, "..", "Tsetlin.jl-main", "src", "Tsetlin.jl"))
const TM = Tsetlin

const ROOT = joinpath(@__DIR__, "..", "masked_catsrank")
const LANGS = ["german", "czech", "italian", "english", "french", "hungarian"]
const GENRES = ["novels", "dracor", "poetree"]
const MAXWIN = 6
const NVOC = 2500
const NEG_X = 2
const CLAUSES = parse(Int, get(ENV, "PTM_CLAUSES", "256"))
const T_ = parse(Int, get(ENV, "PTM_T", "64"))
const S_ = parse(Int, get(ENV, "PTM_S", "4096"))
const LL = parse(Int, get(ENV, "PTM_LL", "32"))
const LF = parse(Int, get(ENV, "PTM_LF", "8"))
const EPOCHS = parse(Int, get(ENV, "PTM_EPOCHS", "300"))

read_tokens(f) = begin
    t = String[]
    for line in eachline(f)
        isempty(line) && continue
        append!(t, split(line, '\t'))
    end
    t
end

namekey(stem) = join(sort(filter(!isempty,
    split(replace(stem, r"^\d+_" => ""), '_'))), "_")

function load_bank(lang, genre)
    d = joinpath(ROOT, "$(lang)_$(genre)_nopoe", "bank")
    isdir(d) || return Dict{String,Vector{String}}()
    Dict(splitext(basename(f))[1] => read_tokens(f)
         for f in sort(readdir(d; join = true)))
end

chop(t, w, maxn) = [t[(i-1)*w+1 : i*w] for i in 1:min(length(t) ÷ w, maxn)]

function main()
    hl = length(ARGS) >= 1 ? ARGS[1] : "german"
    hg = length(ARGS) >= 2 ? ARGS[2] : "poetree"
    wk = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 1000
    wq = length(ARGS) >= 4 ? parse(Int, ARGS[4]) : 1000
    centre = length(ARGS) >= 5 ? ARGS[5] : "known-genre"
    @assert hg == "poetree" "the *_nopoe alphabet is only valid when the " *
                            "held-out genre is poetry"
    @printf("held out: (%s, %s) | known %d tok, questioned %d tok | centring %s\n",
            hl, hg, wk, wq, centre)

    banks = Dict((l, g) => load_bank(l, g) for l in LANGS for g in GENRES)
    # windows per (lang, genre, author), at both lengths
    winK = Dict{Tuple{String,String,String},Vector{Vector{String}}}()
    winQ = Dict{Tuple{String,String,String},Vector{Vector{String}}}()
    for ((l, g), b) in banks, (a, t) in b
        length(t) < 2 * max(wk, wq) && continue
        winK[(l, g, a)] = chop(t, wk, MAXWIN)
        winQ[(l, g, a)] = chop(t, wq, MAXWIN)
    end

    # cross-genre authors per language
    xauth = Dict{String,Dict{String,Dict{String,String}}}()
    for l in LANGS
        d = Dict{String,Dict{String,String}}()
        for g in GENRES, a in keys(get(banks, (l, g), Dict()))
            haskey(winK, (l, g, a)) || continue
            k = namekey(a)
            get!(d, k, Dict{String,String}())[g] = a
        end
        xauth[l] = filter(p -> length(p.second) >= 2, d)
    end
    for l in LANGS
        @printf("  %-10s cross-genre authors: %d\n", l, length(xauth[l]))
    end

    # instance list: (lang, g1, g2, key, a1, a2, heldout)
    insts = NamedTuple[]
    for l in LANGS, (k, gs) in xauth[l], g1 in keys(gs), g2 in keys(gs)
        g1 == g2 && continue
        ho = (l == hl) && (g1 == hg || g2 == hg)
        push!(insts, (lang = l, g1 = g1, g2 = g2, key = k,
                      a1 = gs[g1], a2 = gs[g2], heldout = ho))
    end
    ntest = count(i -> i.heldout, insts)
    @printf("%d instances, %d held out\n", length(insts), ntest)
    ntest == 0 && (println("nothing to test"); return)

    # vocabulary from TRAINING windows only
    cnt = Dict{Tuple,Int}()
    trainwins = Vector{Vector{String}}()
    for i in insts
        i.heldout && continue
        append!(trainwins, winK[(i.lang, i.g1, i.a1)])
        append!(trainwins, winQ[(i.lang, i.g2, i.a2)])
    end
    for w in trainwins, p in eachindex(w), n in 1:3
        p + n - 1 > length(w) && continue
        g = Tuple(w[p:p+n-1]); cnt[g] = get(cnt, g, 0) + 1
    end
    vocab = first.(sort(collect(cnt); by = x -> -x[2]))[1:min(NVOC, length(cnt))]
    vidx = Dict(g => i for (i, g) in enumerate(vocab))
    @printf("vocab %d n-grams (training windows only)\n", length(vocab))

    profile(w) = begin
        v = zeros(Float32, length(vocab))
        for p in eachindex(w), n in 1:3
            p + n - 1 > length(w) && continue
            j = get(vidx, Tuple(w[p:p+n-1]), 0)
            j > 0 && (v[j] += 1)
        end
        v ./ max(sum(v), 1)
    end

    # centring means per (lang, genre), fitted on TRAINING windows of that cell
    acc = Dict{Tuple{String,String},Vector{Vector{Float32}}}()
    for i in insts
        i.heldout && continue
        for w in winK[(i.lang, i.g1, i.a1)]
            push!(get!(acc, (i.lang, i.g1), Vector{Vector{Float32}}()), profile(w))
        end
        for w in winQ[(i.lang, i.g2, i.a2)]
            push!(get!(acc, (i.lang, i.g2), Vector{Vector{Float32}}()), profile(w))
        end
    end
    mu = Dict(k => mean(v) for (k, v) in acc)
    # legal substitute for the held-out cell
    sub = if centre == "known-genre"
        others = [mu[k] for k in keys(mu) if k[1] == hl]
        isempty(others) ? zeros(Float32, length(vocab)) : mean(others)
    else
        others = [mu[k] for k in keys(mu) if k[2] == hg && k[1] != hl]
        isempty(others) ? zeros(Float32, length(vocab)) : mean(others)
    end
    mu[(hl, hg)] = sub
    centred(w, l, g) = begin
        p = profile(w) .- get(mu, (l, g), zeros(Float32, length(vocab)))
        p ./ (sqrt(sum(abs2, p)) + 1f-9)
    end

    # pairs
    rng = MersenneTwister(11)
    X = Vector{Vector{Float32}}(); Y = Bool[]; HO = Bool[]; PA = String[]
    for i in insts
        kws = winK[(i.lang, i.g1, i.a1)]
        qws = winQ[(i.lang, i.g2, i.a2)]
        # same-language, same-genre-pair distractors
        others = [j for j in insts if j.lang == i.lang && j.g2 == i.g2 &&
                  j.key != i.key && haskey(winQ, (j.lang, j.g2, j.a2))]
        for (ki, kw) in enumerate(kws)
            ck = centred(kw, i.lang, i.g1)
            push!(X, ck .* centred(qws[min(ki, length(qws))], i.lang, i.g2))
            push!(Y, true); push!(HO, i.heldout); push!(PA, i.key)
            for _ in 1:NEG_X
                isempty(others) && break
                j = others[rand(rng, 1:length(others))]
                qw = winQ[(j.lang, j.g2, j.a2)]
                push!(X, ck .* centred(qw[rand(rng, 1:length(qw))], j.lang, j.g2))
                push!(Y, false); push!(HO, i.heldout); push!(PA, i.key)
            end
        end
    end
    @printf("%d pairs (%d positive) | test %d (%d positive)\n",
            length(Y), count(Y), count(HO), count(Y .& HO))

    if centre == "within-ceiling" || centre == "heldin-ceiling"
        # Two controls, without which a chance-level transfer number cannot be
        # interpreted:
        #   within-ceiling  train and test inside the HELD-OUT cell, split by
        #                   author -- is the task doable there at all?
        #   heldin-ceiling  train and test inside the TRAINING domain, split by
        #                   author -- does this pipeline generalise to unseen
        #                   authors even when no cell boundary is crossed?
        # If both are at chance the pipeline or the window size is at fault,
        # not the transfer.
        sel = centre == "within-ceiling" ? HO : .!HO
        auths = sort(unique(PA[sel]))
        half = Set(auths[1:2:end])
        tr = [i for i in eachindex(Y) if sel[i] && PA[i] in half]
        te = [i for i in eachindex(Y) if sel[i] && !(PA[i] in half)]
        @printf("%s: %d train authors, %d test authors\n", centre,
                length(half), length(auths) - length(half))
    else
        tr = findall(.!HO); te = findall(HO)
    end
    (isempty(tr) || isempty(te)) && (println("empty split"); return)

    # thermometer cut points from whichever rows are TRAINING in this mode
    M = reduce(hcat, X[tr])
    qs = [quantile(vec(M), q) for q in (0.2, 0.4, 0.6, 0.8)]
    tobits(v) = begin
        b = Vector{Bool}(undef, 4 * length(v))
        @inbounds for i in eachindex(v), k in 1:4
            b[4(i-1)+k] = v[i] > qs[k]
        end
        TM.TMInput(b)
    end
    XB = [tobits(v) for v in X]

    Random.seed!(7)
    tm = TM.TMClassifier(XB[tr[1]], Y[tr], CLAUSES, T_, S_, LL, LF;
                         states_num = 256, include_limit = 220)
    for _ in 1:EPOCHS
        TM.train!(tm, XB[tr], Y[tr]; shuffle = true, index = false)
    end
    margin(x) = begin p, n = TM.vote(tm, tm.clauses, x); Float64(p - n) end
    aucof(sc, yt) = begin
        pos = sc[yt]; neg = sc[.!yt]
        (isempty(pos) || isempty(neg)) && return NaN
        a = 0; tot = 0
        for s in pos, d in neg
            tot += 1; a += (s > d) + 0.5 * (s == d)
        end
        a / tot
    end
    sc = [margin(XB[i]) for i in te]
    yt = Y[te]
    npos = count(yt); nneg = count(.!yt)
    auc = aucof(sc, yt)
    # TRAINING auc: distinguishes "nothing transfers" from "nothing was learned"
    trauc = aucof([margin(XB[i]) for i in tr], Y[tr])
    @printf("train-set AUC %.3f (fit check)   held-out AUC %.3f\n", trauc, auc)
    # LINEAR CONTROL on the identical real-valued features: separates "the TM
    # is mis-tuned" from "these features carry no generalisable signal".
    pidx = [i for i in tr if Y[i]]; nidx = [i for i in tr if !Y[i]]
    wvec = mean(X[pidx]) .- mean(X[nidx])
    lin = [Float64(sum(wvec .* X[i])) for i in te]
    @printf("linear control on same features: held-out AUC %.3f\n",
            aucof(lin, Y[te]))
    @printf("\nRESULT heldout=(%s,%s) known=%d quest=%d centre=%-12s train=%d test=%d(+%d/-%d) AUC=%.3f\n",
            hl, hg, wk, wq, centre, length(tr), length(te),
            npos, nneg, auc)
    println("PAIR TM DONE")
end

main()
