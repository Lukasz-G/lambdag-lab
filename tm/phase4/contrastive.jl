# Contrastive ATOM training (the SCL analog, HDC-native): reshape the codebook geometry with the
# TASK objective (same-author sketches should agree, different-author sketches should not) instead
# of the distributional proxy that phase3/subword.jl used.
#
# Why the codebook is the ONLY thing worth training here: with sketches U = Σ_f ΔU_f·A_f, the pair
# similarity is  cos(U,K) ∝ Σ_{f,g} ΔU_f·ΔK_g·⟨A_f,A_g⟩.  The rates Δ come from the text and the
# diagonal ⟨A_f,A_f⟩=1 is fixed, so the ONLY free parameters in the whole encoder are the atom
# cross-terms ⟨A_f,A_g⟩ — the Gram matrix of the codebook.
#
#   hard POSITIVE (same author, low cosine): the author expressed one habit two ways -> PULL the
#     up-deviating features of the two documents together (they become stylistic alternants).
#   hard NEGATIVE (different authors, high cosine): accidental atom overlap inflates similarity
#     -> PUSH the co-deviating features apart toward orthogonality.
#
# Updates hit SUBWORD atoms (word = bundle of SW[s] XOR Q[slot]), so morphological families move
# coherently and every n-gram/skip-gram inherits the change by XOR algebra — no n-gram is trained
# directly (they are derived, not stored).
#
# Safeguards, all lessons already paid for in this project:
#   - init from the pretrained atoms (never from scratch)
#   - partial write mask (the formhold lesson: full freedom destroys structure)
#   - author-disjoint validation split (atoms must not memorize reference authors)
#   - random-pair repulsion each epoch (contrastive collapse guard) + drift diagnostics
#
#   julia -t auto phase4/contrastive.jl   ->  phase4/contrastive_atoms.jls
#
# Env: P4C_EPOCHS P4C_PAIRS P4C_TOPK P4C_POSF P4C_NEGF P4C_HOLD P4C_POOLX P4C_VAL

import Pkg
Pkg.activate(joinpath(@__DIR__, "..", "hdc"); io = devnull)
using HDC, Random, Printf, Serialization
using LinearAlgebra: dot, mul!, BLAS
using Base.Threads: @threads

const D = 8192
const HERE = @__DIR__
const P3 = joinpath(HERE, "..", "phase3")
const EPOCHS = parse(Int, get(ENV, "P4C_EPOCHS", "6"))
const NPAIR = parse(Int, get(ENV, "P4C_PAIRS", "1500"))    # hard pairs of each polarity per epoch
const TOPK = parse(Int, get(ENV, "P4C_TOPK", "24"))        # features touched per pair
const POSF = parse(Int, get(ENV, "P4C_POSF", "6"))         # pull folds (higher = gentler)
const NEGF = parse(Int, get(ENV, "P4C_NEGF", "7"))
const HOLD = parse(Float64, get(ENV, "P4C_HOLD", "0.25"))  # fraction of bits held fixed
const POOLX = parse(Int, get(ENV, "P4C_POOLX", "6"))       # mining pool multiplier
const NVAL = parse(Int, get(ENV, "P4C_VAL", "8"))          # held-out authors (never trained on)
const KLEN = 1200; const QLEN = 600; const STRIDE = 2.0

readsents(p) = [String.(split(l, '\t')) for l in eachline(p) if !isempty(l)]

function fragments(sents, tok_target, stride_frac)
    frags = Vector{Vector{Vector{String}}}(); buf = [(s, length(s)) for s in sents]; i = 1
    while i <= length(buf)
        cur = Vector{Vector{String}}(); n = 0; j = i
        while j <= length(buf) && n < tok_target; push!(cur, buf[j][1]); n += buf[j][2]; j += 1; end
        n >= tok_target ÷ 2 && push!(frags, cur)
        j > length(buf) && break
        i += max(1, round(Int, (j - i) * stride_frac))
    end
    frags
end

# ---- float ±1 view of the composed codebook: lets sketches be one BLAS call ----
function atom_matrix!(Af, E, sp)
    @threads for w in 1:size(Af, 2)
        @inbounds for c in 1:sp.W
            x = E[c, w]; base = (c - 1) * 64
            for b in 0:63; Af[base+b+1, w] = ((x >> b) & one(UInt64)) == 1 ? 1.0f0 : -1.0f0; end
        end
    end
    Af
end

profile(f, w2i, V) = begin
    p = zeros(Float32, V); n = 0
    for s in f, w in s; id = get(w2i, w, 0); id > 0 && (p[id] += 1; n += 1); end
    n > 0 ? p ./ n : p
end

nrm!(x) = (s = sqrt(sum(abs2, x)) + 1f-9; x ./= s; x)

function compose_into!(dst, SW, Q, subs, sp, bound, acc, rng)
    @inbounds for (k, (sid, slot)) in enumerate(subs)
        b = bound[k]; @simd for i in 1:sp.W; b[i] = SW[i, sid] ⊻ Q[i, slot]; end
    end
    bundle!(sp, dst, view(bound, 1:length(subs)), acc, rng)
end
recompose!(E, SW, Q, word_subs, sp, bound, acc, rng) =
    for w in eachindex(word_subs); compose_into!(@view(E[:, w]), SW, Q, word_subs[w], sp, bound, acc, rng); end

# move word `a`'s subwords toward (+1) / away from (-1) word `b`'s CURRENT atom
function atom_step!(SW, Q, word_subs, a, Eb, sp, mask, tmp, ub, folds, rng, wmask, sgn)
    @inbounds for (sid, slot) in word_subs[a]
        @simd for i in 1:sp.W; ub[i] = Eb[i] ⊻ Q[i, slot]; end
        sgn > 0 ? pull!(sp, @view(SW[:, sid]), ub, mask, tmp, folds, rng, wmask) :
                  push_away!(sp, @view(SW[:, sid]), ub, mask, tmp, folds, rng, wmask)
    end
end

function main()
    BLAS.set_num_threads(Threads.nthreads())
    SW, Q, word_subs, vocab, w2i = deserialize(joinpath(P3, "subword_atoms.jls"))
    V = length(vocab); sp = Space(D); rng = MersenneTwister(17)
    E = Matrix{UInt64}(undef, sp.W, V)
    bound = [newhv(sp) for _ in 1:32]; acc = Vector{Int32}(undef, sp.W * 64)
    mask = newhv(sp); tmp = newhv(sp); ub = newhv(sp)
    wmask = newhv(sp)                                   # writable bits (partial hold)
    let r = MersenneTwister(23)
        for i in eachindex(wmask); wmask[i] = zero(UInt64); end
        for b in 1:D
            rand(r) > HOLD && (wmask[(b-1)÷64+1] |= one(UInt64) << ((b - 1) % 64))
        end
    end
    recompose!(E, SW, Q, word_subs, sp, bound, acc, rng)

    # ---- fragments per author, author-disjoint train/val split ----
    afiles = sort(filter(f -> endswith(f, ".tsv"), readdir(joinpath(HERE, "authors"), join = true)))
    nA = length(afiles)
    kfr = Vector{Vector{Vector{Vector{String}}}}(undef, nA)
    qfr = Vector{Vector{Vector{Vector{String}}}}(undef, nA)
    @threads for a in 1:nA
        sents = readsents(afiles[a])
        tot = sum(length, sents; init = 0); cut = 0; n = 0
        for (i, s) in enumerate(sents); n += length(s); if n >= 0.6 * tot; cut = i; break; end; end
        kfr[a] = fragments(sents[1:cut], KLEN, STRIDE)
        qfr[a] = fragments(sents[cut+1:end], QLEN, STRIDE)
    end
    vala = Set(nA-NVAL+1:nA); traina = [a for a in 1:nA if !(a in vala)]
    @printf("%d authors (%d train / %d val), k/author %.0f, q/author %.0f\n",
            nA, length(traina), NVAL, sum(length, kfr) / nA, sum(length, qfr) / nA); flush(stdout)

    # ---- rate profiles (fixed: they come from the text, not the atoms) ----
    KP = [hcat([profile(f, w2i, V) for f in kfr[a]]...) for a in 1:nA]
    QP = [hcat([profile(f, w2i, V) for f in qfr[a]]...) for a in 1:nA]
    mu = zeros(Float32, V); cnt = 0
    for a in 1:nA
        mu .+= vec(sum(KP[a], dims = 2)) .+ vec(sum(QP[a], dims = 2)); cnt += size(KP[a], 2) + size(QP[a], 2)
    end
    mu ./= cnt
    KD = [KP[a] .- mu for a in 1:nA]; QD = [QP[a] .- mu for a in 1:nA]   # Δ profiles

    Af = Matrix{Float32}(undef, D, V)
    KS = Vector{Matrix{Float32}}(undef, nA); QS = Vector{Matrix{Float32}}(undef, nA)
    function sketches!()
        atom_matrix!(Af, E, sp)
        for a in 1:nA
            KS[a] = Af * KD[a]; QS[a] = Af * QD[a]
            for j in 1:size(KS[a], 2); nrm!(@view KS[a][:, j]); end
            for j in 1:size(QS[a], 2); nrm!(@view QS[a][:, j]); end
        end
    end

    function val_auc()                                   # raw-cosine pair AUC on HELD-OUT authors
        sc = Float64[]; lb = Int[]; r = MersenneTwister(99)
        for a in sort(collect(vala)), _ in 1:150
            ki = rand(r, 1:size(KS[a], 2)); qi = rand(r, 1:size(QS[a], 2))
            push!(sc, dot(@view(KS[a][:, ki]), @view(QS[a][:, qi]))); push!(lb, 1)
            b = rand(r, setdiff(collect(vala), [a])); qj = rand(r, 1:size(QS[b], 2))
            push!(sc, dot(@view(KS[a][:, ki]), @view(QS[b][:, qj]))); push!(lb, 0)
        end
        o = sortperm(sc); rk = similar(sc); i = 1
        while i <= length(o)
            j = i; while j < length(o) && sc[o[j+1]] == sc[o[i]]; j += 1; end
            m = (i + j) / 2; for t in i:j; rk[o[t]] = m; end; i = j + 1
        end
        p = lb .== 1; np = count(p); nn = length(lb) - np
        (sum(rk[p]) - np * (np + 1) / 2) / (np * nn)
    end

    sketches!()
    @printf("epoch 0 (pretrained atoms)  val AUC = %.4f\n", val_auc()); flush(stdout)

    topidx(v, k) = partialsortperm(v, 1:min(k, length(v)), rev = true)
    for ep in 1:EPOCHS
        t0 = time()
        # mine hard pairs from a pool: same-author with LOW cosine, diff-author with HIGH cosine
        spool = Tuple{Int,Int,Int,Float64}[]; dpool = Tuple{Int,Int,Int,Int,Float64}[]
        for _ in 1:NPAIR*POOLX
            a = rand(rng, traina)
            (isempty(kfr[a]) || isempty(qfr[a])) && continue
            ki = rand(rng, 1:size(KS[a], 2)); qi = rand(rng, 1:size(QS[a], 2))
            push!(spool, (a, ki, qi, dot(@view(KS[a][:, ki]), @view(QS[a][:, qi]))))
            b = rand(rng, traina); b == a && continue
            qj = rand(rng, 1:size(QS[b], 2))
            push!(dpool, (a, ki, b, qj, dot(@view(KS[a][:, ki]), @view(QS[b][:, qj]))))
        end
        sort!(spool; by = x -> x[4])                      # hardest positives first (lowest cos)
        sort!(dpool; by = x -> -x[5])                     # hardest negatives first (highest cos)
        npos = min(NPAIR, length(spool)); nneg = min(NPAIR, length(dpool))

        for t in 1:npos                                   # PULL alternants together
            (a, ki, qi, _) = spool[t]
            fu = topidx(@view(QD[a][:, qi]), TOPK); fk = topidx(@view(KD[a][:, ki]), TOPK)
            for i in eachindex(fu)
                u = fu[i]; k = fk[i]; u == k && continue
                atom_step!(SW, Q, word_subs, u, @view(E[:, k]), sp, mask, tmp, ub, POSF, rng, wmask, +1)
                atom_step!(SW, Q, word_subs, k, @view(E[:, u]), sp, mask, tmp, ub, POSF, rng, wmask, +1)
            end
        end
        for t in 1:nneg                                   # PUSH spurious overlaps apart
            (a, ki, b, qj, _) = dpool[t]
            fu = topidx(@view(QD[b][:, qj]), TOPK); fk = topidx(@view(KD[a][:, ki]), TOPK)
            for i in eachindex(fu)
                u = fu[i]; k = fk[i]; u == k && continue
                atom_step!(SW, Q, word_subs, u, @view(E[:, k]), sp, mask, tmp, ub, NEGF, rng, wmask, -1)
            end
        end
        for _ in 1:NPAIR ÷ 2                              # collapse guard: repel random pairs
            u = rand(rng, 1:V); k = rand(rng, 1:V); u == k && continue
            atom_step!(SW, Q, word_subs, u, @view(E[:, k]), sp, mask, tmp, ub, NEGF, rng, wmask, -1)
        end

        recompose!(E, SW, Q, word_subs, sp, bound, acc, rng)
        sketches!()
        ms = 0.0; r2 = MersenneTwister(5)                 # drift/collapse diagnostic
        for _ in 1:400; ms += sim(sp, @view(E[:, rand(r2, 1:V)]), @view(E[:, rand(r2, 1:V)])); end
        @printf("epoch %d  %.0fs  val AUC = %.4f  mean_sim = %.3f\n", ep, time() - t0, val_auc(), ms / 400)
        flush(stdout)
    end

    serialize(joinpath(HERE, "contrastive_atoms.jls"), (SW, Q, word_subs, vocab, w2i))
    println("saved -> phase4/contrastive_atoms.jls")
end

main()
