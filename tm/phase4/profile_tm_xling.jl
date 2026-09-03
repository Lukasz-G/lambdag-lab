# Per-author profile Tsetlin Machines over the universal class-conditioned
# rank alphabet, with the negative population either native or borrowed from
# foreign languages (journal paper, cross-lingual section).
#
#   julia -t auto phase4/profile_tm_xling.jl native
#   julia -t auto phase4/profile_tm_xling.jl pooled
#
# Data: pre-encoded symbol streams (masked_catsrank/, from encode_catsrank.py).
# Case language: German novels; NAUTH profile authors, per-author compact-clause
# TMs, one-vs-population, verification by cohort contrast -- exactly the
# profile-TM protocol, with two arms differing ONLY in where the negative
# (population) windows come from:
#   native   other German authors (encoding control for the earlier pilot)
#   pooled   authors of the five foreign banks (the borrowed population)
# Smoke-test overrides via ENV: PT_EPOCHS, PT_NAUTH, PT_NNEGAUTH.

using Random, Printf, Statistics
include(joinpath(@__DIR__, "..", "Tsetlin.jl-main", "src", "Tsetlin.jl"))
const TM = Tsetlin

const ROOT = joinpath(@__DIR__, "..", "masked_catsrank")
const CASE = "german_novels"
const FOREIGN = ["english_novels", "french_novels", "polish_novels",
                 "czech_novels", "hungarian_novels"]
const WTOK = 400
const STRIDE = 200
const MAXWIN = 120
const NAUTH = parse(Int, get(ENV, "PT_NAUTH", "16"))
const NNEGAUTH = parse(Int, get(ENV, "PT_NNEGAUTH", "40"))  # foreign neg authors
const NVOC = 6000
const NEG_X = 3
const CLAUSES = 256; const T_ = 64; const S_ = 4096
const LL = 64; const LF = 8
const EPOCHS = parse(Int, get(ENV, "PT_EPOCHS", "800"))

function read_tokens(f)
    toks = String[]
    for line in eachline(f)
        isempty(line) && continue
        append!(toks, split(line, '\t'))
    end
    toks
end

function overlapping(t)
    ws = Vector{Vector{String}}()
    i = 1
    while i + WTOK - 1 <= length(t) && length(ws) < MAXWIN
        push!(ws, t[i:i+WTOK-1]); i += STRIDE
    end
    ws
end

function load_bank(ds, minlen)
    banks = Dict{String,Vector{String}}()
    for f in sort(readdir(joinpath(ROOT, ds, "bank"); join = true))
        t = read_tokens(f)
        length(t) >= minlen && (banks[splitext(basename(f))[1]] = t)
    end
    banks
end

function main()
    arm = isempty(ARGS) ? "native" : ARGS[1]
    rng = MersenneTwister(7)
    banks = load_bank(CASE, 16 * WTOK)
    authors = sort(collect(keys(banks)))
    length(authors) > NAUTH && (authors = sort(shuffle(rng, authors)[1:NAUTH]))
    @printf("arm=%s  %d case authors, window=%d symbols\n",
            arm, length(authors), WTOK)

    trainW = Dict{String,Vector{Vector{String}}}()
    evalW = Dict{String,Vector{Vector{String}}}()
    for a in authors
        t = banks[a]; h = length(t) ÷ 2
        trainW[a] = overlapping(t[1:h]); evalW[a] = overlapping(t[h+1:end])
    end

    # negative-population windows
    negW = Vector{Vector{String}}()
    if arm == "native"
        # handled per-author below (other case authors), as in the pilot
    else
        nrng = MersenneTwister(11)
        fb = Dict(ds => load_bank(ds, 2 * WTOK) for ds in FOREIGN)
        cand = [(ds, a) for ds in FOREIGN for a in sort(collect(keys(fb[ds])))]
        picks = shuffle(nrng, cand)[1:min(NNEGAUTH, length(cand))]
        for (ds, a) in picks
            t = fb[ds][a]
            append!(negW, overlapping(t[1:min(length(t), 12 * WTOK)]))
        end
        @printf("pooled negatives: %d windows from %d foreign authors\n",
                length(negW), length(picks))
    end

    # vocabulary from positive training windows + the negative pool
    cnt = Dict{Tuple,Int}()
    vocwins = [w for a in authors for w in trainW[a]]
    arm == "pooled" && append!(vocwins, negW)
    for w in vocwins
        for i in eachindex(w), n in 1:3
            i + n - 1 > length(w) && continue
            g = Tuple(w[i:i+n-1]); cnt[g] = get(cnt, g, 0) + 1
        end
    end
    vocab = first.(sort(collect(cnt); by = x -> -x[2]))[1:min(NVOC, length(cnt))]
    vidx = Dict(g => i for (i, g) in enumerate(vocab))
    @printf("vocab %d n-grams\n", length(vocab))

    profile(w) = begin
        v = zeros(Float32, length(vocab))
        for i in eachindex(w), n in 1:3
            i + n - 1 > length(w) && continue
            j = get(vidx, Tuple(w[i:i+n-1]), 0)
            j > 0 && (v[j] += 1)
        end
        v ./ max(sum(v), 1)
    end

    trP = Dict(a => [profile(w) for w in trainW[a]] for a in authors)
    evP = Dict(a => [profile(w) for w in evalW[a]] for a in authors)
    negP = [profile(w) for w in negW]
    Mtr = reduce(hcat, [p for a in authors for p in trP[a]])
    mu = vec(mean(Mtr; dims = 2)); sd = vec(std(Mtr; dims = 2)) .+ 1f-9
    qs = [quantile(vec((Mtr .- mu) ./ sd), q) for q in (0.2, 0.4, 0.6, 0.8)]
    tobits(p) = begin
        z = (p .- mu) ./ sd
        bits = Vector{Bool}(undef, 4 * length(z))
        @inbounds for i in eachindex(z), b in 1:4
            bits[4(i-1)+b] = z[i] > qs[b]
        end
        TM.TMInput(bits)
    end
    trX = Dict(a => [tobits(p) for p in trP[a]] for a in authors)
    evX = Dict(a => [tobits(p) for p in evP[a]] for a in authors)
    negX = [tobits(p) for p in negP]

    margins = Dict{String,Dict{Tuple{String,Int},Float64}}()
    for a in authors
        pos = trX[a]
        negpool = arm == "native" ?
            [x for b in authors if b != a for x in trX[b]] : negX
        neg = shuffle(MersenneTwister(hash(a)), negpool)[
            1:min(NEG_X * length(pos), length(negpool))]
        X = vcat(pos, neg); Y = vcat(fill(true, length(pos)),
                                     fill(false, length(neg)))
        Random.seed!(100 + hash(a) % 1000)
        tm = TM.TMClassifier(X[1], Y, CLAUSES, T_, S_, LL, LF;
                             states_num = 256, include_limit = 220)
        for _ in 1:EPOCHS
            TM.train!(tm, X, Y; shuffle = true, index = false)
        end
        d = Dict{Tuple{String,Int},Float64}()
        for b in authors, (k, x) in enumerate(evX[b])
            p, n = TM.vote(tm, tm.clauses, x); d[(b, k)] = p - n
        end
        margins[a] = d
        @printf("profile %s done\n", a); flush(stdout)
    end

    aucof(vals_same, vals_diff) = begin
        pos = 0; tot = 0
        for s in vals_same, d in vals_diff
            tot += 1; pos += (s > d) + 0.5 * (s == d)
        end
        pos / tot
    end
    same_r = Float64[]; diff_r = Float64[]
    same_t = Float64[]; diff_t = Float64[]
    for a in authors, (key, _) in margins[a]
        b, _ = key
        raw = margins[a][key]
        others = [margins[c][key] for c in authors if c != a]
        t = (raw - mean(others)) / (std(others) + 1e-9)
        if b == a
            push!(same_r, raw); push!(same_t, t)
        else
            push!(diff_r, raw); push!(diff_t, t)
        end
    end
    @printf("\narm=%s  TM profiles: raw AUC %.3f   cohort AUC %.3f\n",
            arm, aucof(same_r, diff_r), aucof(same_t, diff_t))
    println("XLING PROFILE DONE arm=", arm)
end

main()
