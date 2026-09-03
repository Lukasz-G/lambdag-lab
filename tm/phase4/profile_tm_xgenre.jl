# Route 6 -- profile-TM cross-genre impostor design (journal paper, cross-genre
# section): the discriminative mirror of run_ximpostor.py. One profile TM per
# (cross-genre author, known genre g1), trained one-vs-population against
# OTHER authors' g1 windows (population = the FULL g1 bank, not just the
# cross-genre subset). Verdict on a questioned genre-g2 window: the
# candidate's margin under TM_a(g1), standardised against the SAME margin
# computed under every OTHER cross-genre author's TM_b(g1) -- exactly route
# 1's impostor logic, on the discriminative side. A TM per (author, g1) is
# trained once and evaluated against ALL genres, so every direction sharing a
# known genre is served by one training pass.
#
#   julia -t auto phase4/profile_tm_xgenre.jl
#
# Env overrides for smoke tests: PTX_EPOCHS, PTX_MAXAUTH.

using Random, Printf, Statistics
include(joinpath(@__DIR__, "..", "Tsetlin.jl-main", "src", "Tsetlin.jl"))
const TM = Tsetlin

const ROOT = joinpath(@__DIR__, "..", "masked_catsrank")
const GENRES = ["novels", "dracor", "poetree"]
const WTOK = 400
const STRIDE = 200
const MAXWIN = 120
const NVOC = 6000
const NEG_X = 3
const CLAUSES = 256; const T_ = 64; const S_ = 4096
const LL = 64; const LF = 8
const EPOCHS = parse(Int, get(ENV, "PTX_EPOCHS", "800"))
const MAXAUTH = parse(Int, get(ENV, "PTX_MAXAUTH", "999"))
const MAXDISTRACT = 15

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

namekey(stem) = join(sort(filter(!isempty, split(replace(stem, r"^\d+_" => ""), '_'))), "_")

function load_bank(g)
    banks = Dict{String,Vector{String}}()
    for f in sort(readdir(joinpath(ROOT, "german_$(g)_shared", "bank"); join = true))
        banks[splitext(basename(f))[1]] = read_tokens(f)
    end
    banks
end

function main()
    rng = MersenneTwister(7)
    banks = Dict(g => load_bank(g) for g in GENRES)
    keys_of = Dict(g => Dict(namekey(a) => a for a in keys(banks[g])) for g in GENRES)
    # cross-genre authors: attested in >= 2 genres
    counts = Dict{String,Int}()
    for g in GENRES, k in keys(keys_of[g])
        counts[k] = get(counts, k, 0) + 1
    end
    xgenre = sort([k for (k, c) in counts if c >= 2])
    length(xgenre) > MAXAUTH && (xgenre = sort(shuffle(rng, xgenre)[1:MAXAUTH]))
    @printf("%d cross-genre authors\n", length(xgenre))

    # windows per (key, genre): train/eval halves for the author's OWN genres,
    # full windows for genres only used as evaluation targets
    trainW = Dict{Tuple{String,String},Vector{Vector{String}}}()
    evalW = Dict{Tuple{String,String},Vector{Vector{String}}}()
    for g in GENRES, k in xgenre
        haskey(keys_of[g], k) || continue
        a = keys_of[g][k]
        t = banks[g][a]
        length(t) < 8 * WTOK && continue
        h = length(t) ÷ 2
        trainW[(k, g)] = overlapping(t[1:h])
        evalW[(k, g)] = overlapping(t[h+1:end])
    end
    @printf("%d (author,genre) streams with enough text\n", length(trainW))

    # n-gram vocabulary, pooled over all training windows
    cnt = Dict{Tuple,Int}()
    for w in Iterators.flatten(values(trainW)), i in eachindex(w), n in 1:3
        i + n - 1 > length(w) && continue
        g = Tuple(w[i:i+n-1]); cnt[g] = get(cnt, g, 0) + 1
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
    Mtr = reduce(hcat, [profile(w) for w in Iterators.flatten(values(trainW))])
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

    trX = Dict(kg => [tobits(profile(w)) for w in ws] for (kg, ws) in trainW)
    evX = Dict(kg => [tobits(profile(w)) for w in ws] for (kg, ws) in evalW)

    tms = Dict{Tuple{String,String},Any}()
    for g in GENRES
        cohort = [k for k in xgenre if haskey(trainW, (k, g))]
        isempty(cohort) && continue
        for k in cohort
            pos = trX[(k, g)]
            negpool = [x for k2 in cohort if k2 != k for x in trX[(k2, g)]]
            # also draw from the FULL g-bank (not just cross-genre authors)
            # for a natural-sized population, matching route 1's fixed donor
            others = [a for a in keys(banks[g])
                     if namekey(a) != k && banks[g][a] |> length >= 2 * WTOK]
            prng = MersenneTwister(hash((k, g)))
            extra = shuffle(prng, others)[1:min(MAXDISTRACT, length(others))]
            for a in extra
                for w in overlapping(banks[g][a][1:min(end, 8 * WTOK)])
                    push!(negpool, tobits(profile(w)))
                end
            end
            neg = shuffle(MersenneTwister(hash((k, g, :neg))), negpool)[
                1:min(NEG_X * length(pos), length(negpool))]
            X = vcat(pos, neg); Y = vcat(fill(true, length(pos)),
                                         fill(false, length(neg)))
            Random.seed!(100 + hash((k, g)) % 1000)
            tm = TM.TMClassifier(X[1], Y, CLAUSES, T_, S_, LL, LF;
                                 states_num = 256, include_limit = 220)
            for _ in 1:EPOCHS
                TM.train!(tm, X, Y; shuffle = true, index = false)
            end
            tms[(k, g)] = tm
            @printf("TM(%s, %s) trained\n", k, g); flush(stdout)
        end
    end

    # margins[(cand,g1)][(otherkey,g2,widx)] = vote margin
    margins = Dict{Tuple{String,String},Dict{Tuple{String,String,Int},Float64}}()
    for (k, g1) in keys(tms)
        tm = tms[(k, g1)]
        d = Dict{Tuple{String,String,Int},Float64}()
        for g2 in GENRES, k2 in xgenre
            haskey(evalW, (k2, g2)) || continue
            ws = g2 == g1 && k2 == k ? evX[(k2, g2)] :
                 (haskey(trainW, (k2, g2)) ? trX[(k2, g2)] : evX[(k2, g2)])
            for (widx, x) in enumerate(ws)
                p, n = TM.vote(tm, tm.clauses, x)
                d[(k2, g2, widx)] = p - n
            end
        end
        margins[(k, g1)] = d
    end
    println("margins computed"); flush(stdout)

    aucof(same, diff) = begin
        pos = 0; tot = 0
        for s in same, dd in diff
            tot += 1; pos += (s > dd) + 0.5 * (s == dd)
        end
        tot == 0 ? NaN : pos / tot
    end

    println("\ndirection      n_pos n_neg  raw AUC  cohort AUC")
    for g1 in GENRES, g2 in GENRES
        g1 == g2 && continue
        cohort = [k for k in xgenre if haskey(tms, (k, g1))]
        length(cohort) < 3 && continue
        same_r = Float64[]; diff_r = Float64[]
        same_t = Float64[]; diff_t = Float64[]
        for k in cohort
            haskey(evalW, (k, g2)) || haskey(trainW, (k, g2)) || continue
            m = margins[(k, g1)]
            for (k2, g2b, widx) in keys(m)
                g2b == g2 || continue
                raw = m[(k2, g2b, widx)]
                others = [margins[(c, g1)][(k2, g2b, widx)] for c in cohort if c != k]
                isempty(others) && continue
                t = (raw - mean(others)) / (std(others) + 1e-9)
                if k2 == k
                    push!(same_r, raw); push!(same_t, t)
                else
                    push!(diff_r, raw); push!(diff_t, t)
                end
            end
        end
        isempty(same_r) && continue
        @printf("%-6s->%-6s  %5d %5d  %.3f    %.3f\n", g1, g2,
                length(same_r), length(diff_r),
                aucof(same_r, diff_r), aucof(same_t, diff_t))
    end
    println("XGENRE PROFILE DONE")
end

main()
