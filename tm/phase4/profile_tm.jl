# Per-author PROFILE Tsetlin Machines -- subset pilot of the theory-carrying
# architecture (journal paper, Sec. "Can a classifier carry the entrenchment
# account?"): one compact-clause TM per author, trained one-vs-population on
# single-window grammar profiles; verification by contrasting the candidate
# profile's margin against a cohort of reference profiles.
#
#   julia -t auto phase4/profile_tm.jl
#
# Data: masked German novels bank. Windows of WTOK masked tokens; per author,
# even-indexed windows train, odd-indexed evaluate (no text shared). Features:
# uni+bi+tri-gram profile over the masked stream, z-scored per dim on training
# windows, 4 thermometer bits per dim. Verdict statistics per case (q, cand a):
#   raw     margin_a(q)
#   cohort  (margin_a(q) - mean_j margin_j(q)) / sd_j(margin_j(q)), j != a
# AUC over balanced same/different cases; logistic profile control included.

import Pkg
Pkg.activate(joinpath(@__DIR__, "..", "hdc"); io = devnull)
using Random, Printf, Statistics
include(joinpath(@__DIR__, "..", "Tsetlin.jl-main", "src", "Tsetlin.jl"))
const TM = Tsetlin

const BANK = joinpath(@__DIR__, "..", "masked", "german_novels", "bank")
const WTOK = 400
const STRIDE = 200
const MAXWIN = 120
const NAUTH = 16
const NVOC = 6000
const NEG_X = 3          # negatives per positive in profile training
const CLAUSES = 256; const T_ = 64; const S_ = 4096
const LL = 64; const LF = 8            # compact clauses (the tuned small-L regime)
const EPOCHS = 800

function read_tokens(f)
    toks = String[]
    for line in eachline(f)
        isempty(line) && continue
        append!(toks, split(line, '\t'))
    end
    toks
end

function main()
    rng = MersenneTwister(7)
    banks = Dict{String,Vector{String}}()
    for f in sort(readdir(BANK; join = true))
        t = read_tokens(f)
        length(t) >= 16 * WTOK && (banks[splitext(basename(f))[1]] = t)
    end
    authors = sort(collect(keys(banks)))
    length(authors) > NAUTH && (authors = sort(shuffle(rng, authors)[1:NAUTH]))
    @printf("%d authors, window=%d tokens\n", length(authors), WTOK)

    # windows, split even=train / odd=eval
    trainW = Dict{String,Vector{Vector{String}}}(); evalW = Dict{String,Vector{Vector{String}}}()
    overlapping(t) = begin
        ws = Vector{Vector{String}}()
        i = 1
        while i + WTOK - 1 <= length(t) && length(ws) < MAXWIN
            push!(ws, t[i:i+WTOK-1]); i += STRIDE
        end
        ws
    end
    for a in authors
        t = banks[a]; h = length(t) ÷ 2          # text halves FIRST, then windows
        trainW[a] = overlapping(t[1:h]); evalW[a] = overlapping(t[h+1:end])
    end

    # n-gram vocabulary from training windows
    cnt = Dict{Tuple,Int}()
    for a in authors, w in trainW[a]
        for i in eachindex(w)
            for n in 1:3
                i + n - 1 > length(w) && break
                g = Tuple(w[i:i+n-1]); cnt[g] = get(cnt, g, 0) + 1
            end
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

    # per-author profile TMs + logistic-style linear control (perceptron-ish via
    # centroid difference is too weak; use ridge on z-profiles)
    margins = Dict{String,Dict{Tuple{String,Int},Float64}}()   # cand -> (author,widx) -> margin
    lin_w = Dict{String,Vector{Float32}}()
    for a in authors
        pos = trX[a]; negpool = [(b, x) for b in authors if b != a for x in trX[b]]
        neg = [x for (_, x) in shuffle(MersenneTwister(hash(a)), negpool)][1:min(NEG_X*length(pos), length(negpool))]
        X = vcat(pos, neg); Y = vcat(fill(true, length(pos)), fill(false, length(neg)))
        Random.seed!(100 + hash(a) % 1000)
        tm = TM.TMClassifier(X[1], Y, CLAUSES, T_, S_, LL, LF; states_num = 256, include_limit = 220)
        for _ in 1:EPOCHS
            TM.train!(tm, X, Y; shuffle = true, index = false)
        end
        d = Dict{Tuple{String,Int},Float64}()
        for b in authors, (k, x) in enumerate(evX[b])
            p, n = TM.vote(tm, tm.clauses, x); d[(b, k)] = p - n
        end
        margins[a] = d
        # linear control: ridge weights on z-profiles
        Zp = reduce(hcat, [(p .- mu) ./ sd for p in trP[a]])
        Zn = reduce(hcat, [(p .- mu) ./ sd for p in [q for b in authors if b != a for q in trP[b]]])
        w = vec(mean(Zp; dims = 2) .- mean(Zn; dims = 2))
        lin_w[a] = w
        @printf("profile %s done\n", a); flush(stdout)
    end

    # evaluation: balanced same/diff cases; raw vs cohort-contrast
    evalstat(get_margin) = begin
        same = Float64[]; diff = Float64[]
        for a in authors, (key, _) in margins[a]
            b, k = key
            others = [get_margin(c, key) for c in authors if c != a]
            mo, so = mean(others), std(others) + 1e-9
            raw = get_margin(a, key)
            t = (raw - mo) / so
            (b == a ? push!(same, t) : push!(diff, t))
        end
        pos = 0; tot = 0
        for s in same, d in diff
            tot += 1; pos += (s > d) + 0.5 * (s == d)
        end
        (length(same), length(diff), pos / tot)
    end
    aucraw(get_margin) = begin
        same = Float64[]; diff = Float64[]
        for a in authors, (key, _) in margins[a]
            (key[1] == a ? push!(same, get_margin(a, key)) : push!(diff, get_margin(a, key)))
        end
        pos = 0; tot = 0
        for s in same, d in diff
            tot += 1; pos += (s > d) + 0.5 * (s == d)
        end
        pos / tot
    end

    tm_get(c, key) = margins[c][key]
    lin_get(c, key) = begin
        b, k = key
        Float64(sum(lin_w[c] .* ((evP[b][k] .- mu) ./ sd)))
    end
    @printf("\nTM profiles:      raw AUC %.3f   cohort AUC %.3f\n",
            aucraw(tm_get), evalstat(tm_get)[3])
    @printf("linear profiles:  raw AUC %.3f   cohort AUC %.3f\n",
            aucraw(lin_get), evalstat(lin_get)[3])
    println("PROFILE PILOT DONE")
end

main()
