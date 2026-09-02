# Phase 4: HDC encoding + TM classification at SCALE (the DeBERTa-analogy test).
#
#   - Training pairs are MASS-GENERATED from the reference authors (SCD-style): each author's
#     text splits into a known-pool (first 60% of tokens) and a questioned-pool (rest, so same-
#     author pairs never share text); fragments cut to match the test regime (known ~5900 tok,
#     questioned ~1190 tok; known fragments at half-stride for more combos).
#   - Pair feature FIXES the sign-agreement defect: per-dim product z[b] = Ucen[b]*Kcen[b] of the
#     centered word AND char sketches, thermometer-coded (2 pos + 2 neg thresholds per dim from
#     training-z quantiles). The cosine is sum(z), so the TM sees the cosine's ingredients plus
#     freedom to weight/combine non-linearly.
#   - Trains a binary TM at several TRAIN SIZES to expose the scaling curve, then scores the 500
#     av_test pairs (test authors verified disjoint from reference authors by prep_authors.py).
#
#   julia -t auto phase4/tm_scale.jl     ->  phase4/tm_scaled_scores.jsonl (largest-size margins)

import Pkg
Pkg.activate(joinpath(@__DIR__, "..", "hdc"); io = devnull)
using HDC, Random, Printf, Serialization
using LinearAlgebra: dot
using Statistics: quantile!
using Base.Threads: @threads, threadid, maxthreadid
include(joinpath(@__DIR__, "..", "Tsetlin.jl-main", "src", "Tsetlin.jl"))
const TM = Tsetlin

const D = 8192; const HERE = @__DIR__; const P3 = joinpath(HERE, "..", "phase3")
const KLENS = [parse(Int, x) for x in split(get(ENV, "P4_KLENS", "5900"), ",")]     # known scales
const QLENS = [parse(Int, x) for x in split(get(ENV, "P4_QLENS", "1190"), ",")]     # questioned scales
const NORM  = get(ENV, "P4_NORM", "0") == "1"          # unit-normalize centered sketches (multi-scale)
const EVALLENS = [parse(Int, x) for x in split(get(ENV, "P4_EVALLENS", "0"), ",")]  # 0 = full questioned
const EVALSYM = get(ENV, "P4_EVALSYM", "0") == "1"     # truncate KNOWN to the same L as questioned at eval
const KBITS = [parse(Int, x) for x in split(get(ENV, "P4_KBITS", "1000,2000,3500,5000"), ",")]  # klen thermometer
const QBITS = [parse(Int, x) for x in split(get(ENV, "P4_QBITS", "200,400,800,1100"), ",")]     # qlen thermometer
const STYLO = get(ENV, "P4_STYLO", "0") == "1"         # Tier A: classical stylometry bit-block
const NSTY = 29                                        # stylometry scalars (see stylo_vec)
const SURP = get(ENV, "P4_SURPRISAL", "0") == "1"      # Tier B: surprisal-weighted bundling
                                                       # (attention analog: weight units by -log2 p_ref)
const DIS = get(ENV, "P4_DIS", "0") == "1"             # Tier B: disentangled c2c/c2p/p2c channel
const ENR = get(ENV, "P4_ENRICHED", "0") == "1"        # inputs are 4-field cur|morph|dep|head TSVs
const TOKCOL = parse(Int, get(ENV, "P4_TOKCOL", "1"))  # which field is THE token (1=cur, 2=morph)
const DEP = get(ENV, "P4_DEP", "0") == "1"             # dependency-arc channel (needs P4_ENRICHED)
const NCH = (DIS || DEP) ? 12 : 8                      # thermometer bits per dim (4 per channel)
(DIS && DEP) && error("DIS and DEP share the third channel slot -- enable one")
(DEP && !ENR) && error("P4_DEP needs P4_ENRICHED=1 inputs")
_tok(w) = ENR ? split(w, '|')[TOKCOL] : w              # token view of a (possibly enriched) field
const HARD = parse(Float64, get(ENV, "P4_HARD", "0"))  # Tier C: fraction of pairs mined hard (0 = off)
const POOLX = parse(Int, get(ENV, "P4_POOLX", "3"))    # candidate-pool multiplier for mining
const PROF = get(ENV, "P4_PROF", "0") == "1"           # per-token KN surprisal-profile features
const EXPORTZ = get(ENV, "P4_EXPORTZ", "")             # dir: dump CONTINUOUS z features (word+char
                                                       # products, pre-thermometer) + exit -- for the
                                                       # XGBoost-vs-logistic adjudication
const NPROF = 23                                       # profile scalars (see prof_feats.py)
const SIZES = [parse(Int, x) for x in split(get(ENV, "P4_SIZES", "2000,8000,20000"), ",")]
const CLAUSES = parse(Int, get(ENV, "P4_CLAUSES", "512"))
const T_ = parse(Int, get(ENV, "P4_T", "128"))
const S_ = parse(Int, get(ENV, "P4_S", "4096"))
const L_ = parse(Int, get(ENV, "P4_L", "4096"))
const LF_ = parse(Int, get(ENV, "P4_LF", "2048"))
const STATES = parse(Int, get(ENV, "P4_STATES", "256"))
const INCLUDE = parse(Int, get(ENV, "P4_INCLUDE", "220"))
const EPOCHS = parse(Int, get(ENV, "P4_EPOCHS", "12"))
const KSTRIDE = parse(Float64, get(ENV, "P4_KSTRIDE", "0.5"))   # known-frag stride (fraction)
const QSTRIDE = parse(Float64, get(ENV, "P4_QSTRIDE", "1.0"))   # questioned-frag stride

readsents(p) = [String.(split(l, '\t')) for l in eachline(p) if !isempty(l)]

# ---- sketch encoder (word uni+bi+tri and char-trigram bag; same as phase3) ----
@inline function circ_xor!(fp, src, r, W)
    @inbounds @simd for i in 1:W; j = i - r; j < 1 && (j += W); j > W && (j -= W); fp[i] ⊻= src[j]; end
end
@inline function add_bits!(acc, fp, W, g)
    @inbounds for c in 1:W; x = fp[c]; base = (c - 1) * 64
        while x != 0; acc[base+trailing_zeros(x)+1] += g; x &= x - one(x); end
    end
end
function encode_doc(sents, Vw, SWc, word_subs, w2i, sp, GAP, wt, Vc, ROLES, Vd, d2i)
    W = sp.W; aw = zeros(Float64, sp.D); ac = zeros(Float64, sp.D); ad = zeros(Float64, sp.D)
    nw = 0.0; nc = 0.0; nd = 0.0; fp = newhv(sp)
    for s in sents
        # position-aligned parse (heads index the FULL emitted sentence, incl. OOV positions)
        L0 = length(s)
        pid_ = zeros(Int, L0)
        dep_ = DEP ? zeros(Int, L0) : Int[]; hed_ = DEP ? zeros(Int, L0) : Int[]
        for (j, w) in enumerate(s)
            if ENR
                p = split(w, '|')
                pid_[j] = get(w2i, p[TOKCOL], 0)
                if DEP && length(p) >= 4
                    dep_[j] = get(d2i, p[3], 0)
                    hed_[j] = something(tryparse(Int, p[4]), 0)
                end
            else
                pid_[j] = get(w2i, w, 0)
            end
        end
        ids = [pid_[j] for j in 1:L0 if pid_[j] > 0]
        L = length(ids)
        @inbounds for i in 1:L
            g = wt[ids[i]]
            add_bits!(aw, @view(Vw[:, ids[i]]), W, g); nw += g
            for (sid, _) in word_subs[ids[i]]; add_bits!(ac, @view(SWc[:, sid]), W, g); nc += g; end
        end
        @inbounds for i in 1:L-1
            g = 0.5 * (wt[ids[i]] + wt[ids[i+1]])
            fill!(fp, 0); circ_xor!(fp, @view(Vw[:, ids[i]]), 0, W); circ_xor!(fp, @view(Vw[:, ids[i+1]]), 1, W); add_bits!(aw, fp, W, g); nw += g
        end
        @inbounds for i in 1:L-2
            g = (wt[ids[i]] + wt[ids[i+1]] + wt[ids[i+2]]) / 3
            fill!(fp, 0); circ_xor!(fp, @view(Vw[:, ids[i]]), 0, W); circ_xor!(fp, @view(Vw[:, ids[i+1]]), 1, W); circ_xor!(fp, @view(Vw[:, ids[i+2]]), 2, W); add_bits!(aw, fp, W, g); nw += g
        end
        @inbounds for i in 1:L-2
            g = 0.5 * (wt[ids[i]] + wt[ids[i+2]])
            fill!(fp, 0); circ_xor!(fp, @view(Vw[:, ids[i]]), 0, W); circ_xor!(fp, @view(Vw[:, ids[i+2]]), 1, W); circ_xor!(fp, @view(GAP[:, 1]), 0, W); add_bits!(aw, fp, W, g); nw += g
        end
        # disentangled channel (DeBERTa decomposition): c2p = token⊗sentence-role (placement
        # habits marginalized over neighbors), p2c = class⊗role (syntax of positions, content
        # marginalized; placeholders self-class), c2c = commutative adjacent pair (association
        # habits, order marginalized). Marginals recur, so they survive bundling where the
        # entangled n-grams fragment below the noise floor.
        if DIS
            @inbounds for i in 1:L
                g = wt[ids[i]]
                r = i == 1 ? 1 : i == 2 ? 2 : i == L ? 5 : i == L - 1 ? 4 : 3
                fill!(fp, 0); circ_xor!(fp, @view(Vw[:, ids[i]]), 0, W); circ_xor!(fp, @view(ROLES[:, r]), 0, W); add_bits!(ad, fp, W, g); nd += g
                fill!(fp, 0); circ_xor!(fp, @view(Vc[:, ids[i]]), 0, W); circ_xor!(fp, @view(ROLES[:, r]), 0, W); add_bits!(ad, fp, W, g); nd += g
            end
            @inbounds for i in 1:L-1
                g = 0.5 * (wt[ids[i]] + wt[ids[i+1]])
                fill!(fp, 0); circ_xor!(fp, @view(Vw[:, ids[i]]), 0, W); circ_xor!(fp, @view(Vw[:, ids[i+1]]), 0, W); add_bits!(ad, fp, W, g); nd += g
            end
        end
        # dependency-arc channel: bind(head-token ⊗ deprel ⊗ dependent-token) per labelled arc.
        # Relational syntax at ANY distance -- the probe showed arcs beat word unigrams standalone.
        if DEP
            @inbounds for j in 1:L0
                h = hed_[j]
                (h < 1 || h > L0 || h == j) && continue
                (pid_[j] == 0 || pid_[h] == 0 || dep_[j] == 0) && continue
                g = 0.5 * (wt[pid_[h]] + wt[pid_[j]])
                fill!(fp, 0)
                circ_xor!(fp, @view(Vw[:, pid_[h]]), 0, W)
                circ_xor!(fp, @view(Vd[:, dep_[j]]), 1, W)
                circ_xor!(fp, @view(Vw[:, pid_[j]]), 2, W)
                add_bits!(ad, fp, W, g); nd += g
            end
        end
    end
    (nw > 0 ? aw ./ nw : zeros(Float64, sp.D)), (nc > 0 ? ac ./ nc : zeros(Float64, sp.D)),
    (nd > 0 ? ad ./ nd : zeros(Float64, sp.D))
end

# ---- fragment slicing ----
function fragments(sents, tok_target, stride_frac)
    frags = Vector{Vector{Vector{String}}}(); lens = Int[]
    cur = Vector{Vector{String}}(); n = 0
    starts = Vector{Vector{Vector{String}}}()   # unused; simple accumulation with stride via restart list
    # accumulate sentences; emit fragment when >= tok_target; overlap via half-restart
    buf = Tuple{Vector{String},Int}[]
    for s in sents; push!(buf, (s, length(s))); end
    i = 1
    while i <= length(buf)
        cur = Vector{Vector{String}}(); n = 0; j = i
        while j <= length(buf) && n < tok_target
            push!(cur, buf[j][1]); n += buf[j][2]; j += 1
        end
        n >= tok_target ÷ 2 && push!(frags, cur)            # keep near-full fragments only
        j > length(buf) && break
        # advance start by stride_frac of the fragment's sentences
        adv = max(1, round(Int, (j - i) * stride_frac)); i += adv
    end
    frags
end

function auc(scores, labels)
    o = sortperm(scores); r = similar(scores, Float64); i = 1
    while i <= length(o)
        j = i; while j < length(o) && scores[o[j+1]] == scores[o[i]]; j += 1; end
        m = (i + j) / 2; for t in i:j; r[o[t]] = m; end; i = j + 1
    end
    p = labels .== 1; np = count(p); nn = length(labels) - np
    (sum(r[p]) - np * (np + 1) / 2) / (np * nn)
end

# ---- pair feature: thermometer bands of z = Ucen .* Kcen (word + char [+ disentangled]) ----
function pair_feature!(bits::Vector{Bool}, uw, kw, uc, kc, ud, kd, tw::NTuple{2,Float64}, tc::NTuple{2,Float64}, td::NTuple{2,Float64}, Dd::Int)
    @inbounds for b in 1:Dd
        z = uw[b] * kw[b]; o = (b - 1) * 4
        bits[o+1] = z > tw[1]; bits[o+2] = z > tw[2]; bits[o+3] = z < -tw[1]; bits[o+4] = z < -tw[2]
    end
    off = 4Dd
    @inbounds for b in 1:Dd
        z = uc[b] * kc[b]; o = off + (b - 1) * 4
        bits[o+1] = z > tc[1]; bits[o+2] = z > tc[2]; bits[o+3] = z < -tc[1]; bits[o+4] = z < -tc[2]
    end
    if DIS
        off = 8Dd
        @inbounds for b in 1:Dd
            z = ud[b] * kd[b]; o = off + (b - 1) * 4
            bits[o+1] = z > td[1]; bits[o+2] = z > td[2]; bits[o+3] = z < -td[1]; bits[o+4] = z < -td[2]
        end
    end
    bits
end

# thermometer of the pair geometry (known/questioned token lengths) at offset `off` (8 bits):
# lets the TM learn length-conditional rules (e.g. tolerate mild disagreement on short questioned).
@inline function len_bits!(bits, off::Int, klen::Int, qlen::Int)
    @inbounds for i in 1:4
        bits[off+i] = klen >= KBITS[i]; bits[off+4+i] = qlen >= QBITS[i]
    end
    bits
end

# ---- Tier A: classical stylometry profile of a masked fragment (P4_STYLO=1) ----
# 29 scalars from what POSNoise keeps: sentence-length distribution (1-5), punctuation profile
# (6-13), lexical density total/per-placeholder-class/other (14-21), function-word length/TTR
# (22-24), dass/daß/ß orthography (25-27), commas per sentence (28), dialogue-sentence frac (29).
# Pair encoding: per-feature |Δ| of population-z-scored profiles, 4 thermometer bits each.
const QUOTECH = Set(["«", "»", "„", "“", "”", "\"", "'", "‚"])
const DASHCH = Set(["—", "–", "-", "--"])
const PUNCTCH = Set([",", ".", "!", "?", ";", ":", "(", ")", "…", "..", "...", "...."])
# NB: some POSNoise placeholders (Ø) are Unicode LETTERS — a single non-ASCII char is a
# placeholder candidate, not a word, else the verb mask leaks into the word features.
_isword(t) = any(isletter, t) && !(length(t) == 1 && !isascii(first(t)))
_issymtok(t) = !(t in PUNCTCH) && !(t in QUOTECH) && !(t in DASHCH) && !any(isdigit, t) &&
               (!any(isletter, t) || (length(t) == 1 && !isascii(first(t))))

function stylo_vec(sents, phidx::Dict{String,Int})
    v = zeros(Float64, NSTY)
    ns = length(sents); ns == 0 && return v
    slens = Float64[]; ntok = 0
    pc = zeros(Int, 8); ph = zeros(Int, 6); phother = 0; phtot = 0
    alens = Int[]; atypes = Set{String}(); ndass = 0; ndasz = 0; nesz = 0; ndlg = 0
    for s in sents
        push!(slens, length(s)); ntok += length(s); hasq = false
        for t in s
            if t == ","; pc[1] += 1
            elseif t == "."; pc[2] += 1
            elseif t == "!"; pc[3] += 1
            elseif t == "?"; pc[4] += 1
            elseif t == ";"; pc[5] += 1
            elseif t == ":"; pc[6] += 1
            elseif t in QUOTECH; pc[7] += 1; hasq = true
            elseif t in DASHCH; pc[8] += 1
            elseif _isword(t)
                push!(alens, length(t)); push!(atypes, t)
                lt = lowercase(t)
                lt == "dass" && (ndass += 1); lt == "daß" && (ndasz += 1)
                ('ß' in t) && (nesz += 1)
            elseif _issymtok(t)
                phtot += 1
                i = get(phidx, t, 0); i == 0 ? (phother += 1) : (ph[i] += 1)
            end
        end
        hasq && (ndlg += 1)
    end
    ntok == 0 && return v
    sort!(slens); nsl = length(slens); m = sum(slens) / nsl
    v[1] = m
    v[2] = nsl > 1 ? sqrt(sum(x -> (x - m)^2, slens) / (nsl - 1)) : 0.0
    v[3] = slens[max(1, ceil(Int, 0.25nsl))]; v[4] = slens[max(1, ceil(Int, 0.5nsl))]
    v[5] = slens[max(1, ceil(Int, 0.75nsl))]
    for i in 1:8; v[5+i] = pc[i] / ntok; end
    v[14] = phtot / ntok
    for i in 1:6; v[14+i] = ph[i] / ntok; end
    v[21] = phother / ntok
    na = length(alens)
    if na > 0
        ma = sum(alens) / na
        v[22] = ma; v[23] = na > 1 ? sqrt(sum(x -> (x - ma)^2, alens) / (na - 1)) : 0.0
        v[24] = length(atypes) / na
        v[25] = ndass / na; v[26] = ndasz / na; v[27] = nesz / na
    end
    v[28] = pc[1] / ns; v[29] = ndlg / ns
    v
end

@inline function stylo_bits!(bits, off::Int, zu, zk, TH)
    @inbounds for f in 1:NSTY
        d = abs(zu[f] - zk[f]); o = off + 4 * (f - 1)
        bits[o+1] = d >= TH[f, 1]; bits[o+2] = d >= TH[f, 2]
        bits[o+3] = d >= TH[f, 3]; bits[o+4] = d >= TH[f, 4]
    end
    bits
end

# per-token KN surprisal-profile pair block (P4_PROF=1): |Δz| thermometer over the 23 profile
# scalars from prof_feats.py — dense order-sensitive evidence the bag sketches cannot derive.
@inline function prof_bits!(bits, off::Int, zu, zk, TH)
    @inbounds for f in 1:NPROF
        d = abs(zu[f] - zk[f]); o = off + 4 * (f - 1)
        bits[o+1] = d >= TH[f, 1]; bits[o+2] = d >= TH[f, 2]
        bits[o+3] = d >= TH[f, 3]; bits[o+4] = d >= TH[f, 4]
    end
    bits
end

# within-stratum percentile rank (0 = lowest value in its stratum): used by hard-pair mining so
# "hard" is judged against pairs of the SAME length geometry, not conflated with short = sparse.
function percentile_within(str::Vector{Int}, vals::Vector{Float64})
    p = similar(vals)
    for s in unique(str)
        idx = findall(==(s), str)
        o = sortperm(vals[idx]); n = length(idx)
        for (r, j) in enumerate(o); p[idx[j]] = n > 1 ? (r - 1) / (n - 1) : 0.5; end
    end
    p
end

function compose_all(SW, Q, word_subs, sp)
    V = Matrix{UInt64}(undef, sp.W, length(word_subs))
    bound = [newhv(sp) for _ in 1:32]; acc = Vector{Int32}(undef, sp.W * 64); rng = MersenneTwister(0)
    for w in eachindex(word_subs)
        subs = word_subs[w]
        @inbounds for (k, (sid, slot)) in enumerate(subs)
            b = bound[k]; @simd for i in 1:sp.W; b[i] = SW[i, sid] ⊻ Q[i, slot]; end
        end
        bundle!(sp, @view(V[:, w]), view(bound, 1:length(subs)), acc, rng)
    end
    V
end

function main()
    rngG = MersenneTwister(11)
    SW, Q, word_subs, vocab, w2i = deserialize(get(ENV, "P4_ATOMFILE", joinpath(P3, "subword_atoms.jls")))
    sp = Space(D); nsw = size(SW, 2)
    if get(ENV, "P4_VOCAB", "atoms") == "corpus"
        # Build vocab + char-trigram subwords fresh from the training corpus. Needed whenever the
        # masking scheme emits token types absent from the pretrained atom vocabulary (e.g. the
        # morph placeholders Ø_pst/@s); forces random atoms — both arms of an A/B must use this.
        cntv = Dict{String,Int}()
        for f in filter(f -> endswith(f, ".tsv"),
                        readdir(get(ENV, "P4_AUTHORS", joinpath(HERE, "authors")), join = true))
            for l in eachline(f), w in split(l, '\t')
                isempty(w) && continue
                w2 = _tok(w); cntv[w2] = get(cntv, w2, 0) + 1
            end
        end
        vocab = sort!(collect(keys(cntv)))
        w2i = Dict{String,Int32}(w => Int32(i) for (i, w) in enumerate(vocab))
        sw2id = Dict{String,Int}()
        word_subs = Vector{Vector{Tuple{Int,Int}}}(undef, length(vocab))
        for (wi, w) in enumerate(vocab)
            chars = collect("<" * w * ">")
            subs = Tuple{Int,Int}[]
            for i in 1:max(0, length(chars) - 2)
                g = String(chars[i:i+2])
                push!(subs, (get!(sw2id, g, length(sw2id) + 1), min(i, 24)))
            end
            isempty(subs) && push!(subs, (get!(sw2id, String(chars), length(sw2id) + 1), 1))
            word_subs[wi] = subs
        end
        nsw = length(sw2id)
        ENV["P4_ATOMS"] = "random"
        println("vocab: CORPUS-built ($(length(vocab)) types, $(nsw) subwords) -> atoms forced RANDOM")
    end
    if get(ENV, "P4_ATOMS", "random") == "pretrained"
        Vw = compose_all(SW, Q, word_subs, sp)                 # subword-composed word atoms
        SWc = Matrix{UInt64}(SW)                               # learned char-trigram atoms
        println("atoms: PRETRAINED (subword-composed words + learned char trigrams)")
    else
        Vw = randcodes(sp, length(vocab), MersenneTwister(42)); SWc = randcodes(sp, nsw, MersenneTwister(44))
        println("atoms: RANDOM")
    end
    GAP = randcodes(sp, 2, MersenneTwister(43))
    ROLES = randcodes(sp, 5, MersenneTwister(45))          # sentence roles: 1st,2nd,medial,pre-final,final
    CLS = randcodes(sp, 2, MersenneTwister(46))            # class atoms: function word, punctuation
    Vc = Matrix{UInt64}(undef, sp.W, length(vocab))        # p2c class atom per vocab id
    for i in eachindex(vocab)
        t = vocab[i]
        if t in PUNCTCH || t in QUOTECH || t in DASHCH; Vc[:, i] = @view CLS[:, 2]
        elseif _issymtok(t); Vc[:, i] = @view Vw[:, i]     # POSNoise placeholder = its own class
        else; Vc[:, i] = @view CLS[:, 1]
        end
    end
    d2i = Dict{String,Int}(); Vd = Matrix{UInt64}(undef, sp.W, 1)
    if DEP                                                 # dependency-label atoms from the corpus
        for f in filter(f -> endswith(f, ".tsv"),
                        readdir(get(ENV, "P4_AUTHORS", joinpath(HERE, "authors")), join = true))
            for l in eachline(f), w in split(l, '\t')
                p = split(w, '|')
                length(p) >= 4 && !haskey(d2i, p[3]) && (d2i[String(p[3])] = length(d2i) + 1)
            end
        end
        Vd = randcodes(sp, length(d2i), MersenneTwister(47))
        println("dep-arc channel ON ($(length(d2i)) labels)")
    end

    # ---- author fragments ----
    afiles = sort(filter(f -> endswith(f, ".tsv"),
                         readdir(get(ENV, "P4_AUTHORS", joinpath(HERE, "authors")), join = true)))
    nA = length(afiles)
    kfr = Vector{Vector{Vector{Vector{String}}}}(undef, nA)   # per author: known fragments
    qfr = Vector{Vector{Vector{Vector{String}}}}(undef, nA)
    @threads for a in 1:nA
        sents = readsents(afiles[a])
        tot = sum(length, sents; init = 0); cut = 0; n = 0
        for (i, s) in enumerate(sents); n += length(s); if n >= 0.6 * tot; cut = i; break; end; end
        kfr[a] = reduce(vcat, [fragments(sents[1:cut], L, KSTRIDE) for L in KLENS])
        qfr[a] = reduce(vcat, [fragments(sents[cut+1:end], L, QSTRIDE) for L in QLENS])
    end
    @printf("%d authors, known frags/author ~%.1f, q frags/author ~%.1f\n",
            nA, sum(length, kfr) / nA, sum(length, qfr) / nA); flush(stdout)

    # ---- Tier B: surprisal weights (attention analog). A unit's contribution to the bundle is
    # its self-information -log2 p_ref under the REFERENCE population, so patterns that are
    # merely frequent in German weigh less than patterns that are unusual there. Uniform if off.
    # NB variable names must not collide with the @threads slicing loop above: a function-level
    # binding would capture the loop's locals and create a cross-thread race (bit us once).
    wt = ones(Float64, length(vocab))
    if SURP
        wcnt = zeros(Float64, length(vocab)); wtot = 0.0
        for a in 1:nA, f in kfr[a], s in f, w in s
            id = get(w2i, _tok(w), 0); id > 0 && (wcnt[id] += 1; wtot += 1)
        end
        for i in eachindex(wt)
            wt[i] = -log2((wcnt[i] + 0.5) / (wtot + 0.5 * length(vocab)))
        end
        wt ./= (sum(wt) / length(wt))          # mean 1: keeps the rate scale comparable
        @printf("surprisal weighting ON (range %.2f-%.2f, mean 1)\n", minimum(wt), maximum(wt))
    end

    # ---- encode all fragments once ----
    kv = [Vector{NTuple{3,Vector{Float64}}}(undef, length(kfr[a])) for a in 1:nA]
    qv = [Vector{NTuple{3,Vector{Float64}}}(undef, length(qfr[a])) for a in 1:nA]
    t0 = time()
    @threads for a in 1:nA
        for (i, f) in enumerate(kfr[a]); kv[a][i] = encode_doc(f, Vw, SWc, word_subs, w2i, sp, GAP, wt, Vc, ROLES, Vd, d2i); end
        for (i, f) in enumerate(qfr[a]); qv[a][i] = encode_doc(f, Vw, SWc, word_subs, w2i, sp, GAP, wt, Vc, ROLES, Vd, d2i); end
    end
    @printf("encoded %d fragments in %.0fs\n", sum(length, kv) + sum(length, qv), time() - t0); flush(stdout)

    # population centering from TRAINING fragments only
    allw = [v[1] for a in 1:nA for v in kv[a]]; allc = [v[2] for a in 1:nA for v in kv[a]]
    alld = [v[3] for a in 1:nA for v in kv[a]]
    for a in 1:nA, v in qv[a]; push!(allw, v[1]); push!(allc, v[2]); push!(alld, v[3]); end
    pw = sum(allw) ./ length(allw); pc = sum(allc) ./ length(allc); pd = sum(alld) ./ length(alld)
    _nrm(x) = NORM ? x ./ (sqrt(dot(x, x)) + 1e-9) : x     # length-invariant z scale (multi-scale)
    cen(v) = (_nrm(v[1] .- pw), _nrm(v[2] .- pc), _nrm(v[3] .- pd))
    kv = [[cen(v) for v in kv[a]] for a in 1:nA]; qv = [[cen(v) for v in qv[a]] for a in 1:nA]
    ktl = [[sum(length, f; init = 0) for f in kfr[a]] for a in 1:nA]   # fragment token lengths
    qtl = [[sum(length, f; init = 0) for f in qfr[a]] for a in 1:nA]

    # ---- Tier A: stylometry profiles, z-scored on the training-fragment population ----
    phidx = Dict{String,Int}(); zsty = identity
    ksz = Vector{Vector{Float64}}[]; qsz = Vector{Vector{Float64}}[]
    if STYLO
        cnt = Dict{String,Int}()
        for a in 1:nA, f in kfr[a], s in f, t in s
            _issymtok(t) && (cnt[t] = get(cnt, t, 0) + 1)
        end
        syms = sort!(collect(keys(cnt)); by = t -> (-cnt[t], t))
        phidx = Dict(t => i for (i, t) in enumerate(syms[1:min(6, end)]))
        ksy = [[stylo_vec(f, phidx) for f in kfr[a]] for a in 1:nA]
        qsy = [[stylo_vec(f, phidx) for f in qfr[a]] for a in 1:nA]
        alls = Vector{Vector{Float64}}()
        for a in 1:nA; append!(alls, ksy[a]); append!(alls, qsy[a]); end
        mu = sum(alls) ./ length(alls)
        sd = sqrt.(sum(v -> (v .- mu) .^ 2, alls) ./ max(length(alls) - 1, 1)) .+ 1e-9
        zsty = v -> (v .- mu) ./ sd
        ksz = [[zsty(v) for v in ksy[a]] for a in 1:nA]
        qsz = [[zsty(v) for v in qsy[a]] for a in 1:nA]
        println("stylometry block ON ($(NSTY) scalars, placeholders: $(join(syms[1:min(6, end)], ' ')))")
    end

    # ---- per-token surprisal profiles from prof_feats.py, z-scored on training frags ----
    kpf = Vector{Vector{Float64}}[]; qpf = Vector{Vector{Float64}}[]
    zprof = identity; ptest = Dict{Tuple{Int,Int,Char},Vector{Float64}}()
    if PROF
        kpf = [Vector{Vector{Float64}}(undef, length(kfr[a])) for a in 1:nA]
        qpf = [Vector{Vector{Float64}}(undef, length(qfr[a])) for a in 1:nA]
        for line in eachline(joinpath(HERE, "prof_authors.tsv"))
            p = split(line, '\t')
            a = parse(Int, p[2]) + 1; i = parse(Int, p[3]) + 1
            v = [parse(Float64, x) for x in p[4:end]]
            p[1] == "k" ? (kpf[a][i] = v) : (qpf[a][i] = v)
        end
        for a in 1:nA
            @assert all(isassigned(kpf[a], i) for i in eachindex(kpf[a])) "prof/kfr slicing mismatch (author $a)"
            @assert all(isassigned(qpf[a], i) for i in eachindex(qpf[a])) "prof/qfr slicing mismatch (author $a)"
        end
        pall = Vector{Vector{Float64}}()
        for a in 1:nA; append!(pall, kpf[a]); append!(pall, qpf[a]); end
        pmu = sum(pall) ./ length(pall)
        psd = sqrt.(sum(v -> (v .- pmu) .^ 2, pall) ./ max(length(pall) - 1, 1)) .+ 1e-9
        zprof = v -> (v .- pmu) ./ psd
        kpf = [[zprof(v) for v in kpf[a]] for a in 1:nA]
        qpf = [[zprof(v) for v in qpf[a]] for a in 1:nA]
        for line in eachline(joinpath(HERE, "prof_test.tsv"))
            p = split(line, '\t')
            ptest[(parse(Int, p[1]), parse(Int, p[2]), p[3][1])] = zprof([parse(Float64, x) for x in p[4:end]])
        end
        println("surprisal-profile block ON ($(NPROF) scalars/doc)")
    end

    # ---- generate the master balanced pair list (largest size; subsets share prefix) ----
    same = Tuple{Int,Int,Int}[]                                # (author, kidx, qidx)
    for a in 1:nA, ki in eachindex(kv[a]), qi in eachindex(qv[a]); push!(same, (a, ki, qi)); end
    shuffle!(rngG, same)
    nmax = min(SIZES[end] ÷ 2, length(same))
    ndiff = HARD > 0 ? nmax * POOLX : nmax
    diff = Tuple{Int,Int,Int,Int}[]                            # (authorK, kidx, authorQ, qidx)
    while length(diff) < ndiff
        a = rand(rngG, 1:nA); b = rand(rngG, 1:nA); a == b && continue
        (isempty(kv[a]) || isempty(qv[b])) && continue
        push!(diff, (a, rand(rngG, eachindex(kv[a])), b, rand(rngG, eachindex(qv[b]))))
    end

    # ---- Tier C: hard-pair mining (SCL analog). Judge difficulty by word-sketch cosine within
    # each length stratum; keep HARD fraction most-confusable (same: LOW cos, diff: HIGH cos)
    # plus random rest — pure hard training destabilizes, the mix is standard practice.
    if HARD > 0
        nsc = min(nmax * POOLX, length(same))
        scand = same[1:nsc]
        scos = Vector{Float64}(undef, nsc); sstr = Vector{Int}(undef, nsc)
        @threads for i in 1:nsc
            (a, ki, qi) = scand[i]
            scos[i] = dot(qv[a][qi][1], kv[a][ki][1])
            sstr[i] = findmin(abs.(KLENS .- ktl[a][ki]))[2] * 100 + findmin(abs.(QLENS .- qtl[a][qi]))[2]
        end
        sord = sortperm(percentile_within(sstr, scos))         # ascending: hardest same first
        nh = min(round(Int, HARD * nmax), nsc)
        srest = scand[sord[nh+1:end]]; shuffle!(rngG, srest)
        same = vcat(scand[sord[1:nh]], srest[1:max(0, nmax - nh)]); shuffle!(rngG, same)

        ndc = length(diff)
        dcos = Vector{Float64}(undef, ndc); dstr = Vector{Int}(undef, ndc)
        @threads for i in 1:ndc
            (ak, ki, aq, qi) = diff[i]
            dcos[i] = dot(qv[aq][qi][1], kv[ak][ki][1])
            dstr[i] = findmin(abs.(KLENS .- ktl[ak][ki]))[2] * 100 + findmin(abs.(QLENS .- qtl[aq][qi]))[2]
        end
        dord = sortperm(percentile_within(dstr, dcos); rev = true)   # descending: hardest diff first
        drest = diff[dord[nh+1:end]]; shuffle!(rngG, drest)
        diff = vcat(diff[dord[1:nh]], drest[1:max(0, nmax - nh)]); shuffle!(rngG, diff)
        @printf("hard mining ON: %.0f%% hard + rest random, pool x%d\n", 100HARD, POOLX)
    end
    @printf("pair pool: %d same, %d diff (target %d total)\n", nmax, nmax, SIZES[end]); flush(stdout)

    # ---- thermometer thresholds from a sample of training z values ----
    zs_w = Float64[]; zs_c = Float64[]; zs_d = Float64[]
    for t in 1:min(400, nmax)
        (a, ki, qi) = same[t]; uw, uc, ud = qv[a][qi]; kw, kc, kd = kv[a][ki]
        append!(zs_w, abs.(uw .* kw)); append!(zs_c, abs.(uc .* kc))
        DIS && append!(zs_d, abs.(ud .* kd))
    end
    tw = (quantile!(zs_w, 0.5), quantile!(zs_w, 0.85)); tc = (quantile!(zs_c, 0.5), quantile!(zs_c, 0.85))
    td = DIS ? (quantile!(zs_d, 0.5), quantile!(zs_d, 0.85)) : (0.0, 0.0)

    THS = zeros(NSTY, 4)
    if STYLO                                   # per-feature |Δz| thermometer thresholds
        dv = [Float64[] for _ in 1:NSTY]
        for t in 1:min(400, nmax)
            (a, ki, qi) = same[t]; d = abs.(qsz[a][qi] .- ksz[a][ki])
            for f in 1:NSTY; push!(dv[f], d[f]); end
        end
        for f in 1:NSTY
            THS[f, 1] = quantile!(dv[f], 0.5); THS[f, 2] = quantile!(dv[f], 0.75)
            THS[f, 3] = quantile!(dv[f], 0.9); THS[f, 4] = quantile!(dv[f], 0.97)
        end
    end

    THP = zeros(NPROF, 4)
    if PROF                                    # per-feature |Δz| thermometer thresholds
        dvp = [Float64[] for _ in 1:NPROF]
        for t in 1:min(400, nmax)
            (a, ki, qi) = same[t]; d = abs.(qpf[a][qi] .- kpf[a][ki])
            for f in 1:NPROF; push!(dvp[f], d[f]); end
        end
        for f in 1:NPROF
            THP[f, 1] = quantile!(dvp[f], 0.5); THP[f, 2] = quantile!(dvp[f], 0.75)
            THP[f, 3] = quantile!(dvp[f], 0.9); THP[f, 4] = quantile!(dvp[f], 0.97)
        end
    end

    # ---- build features for the master pool ----
    POFF = NCH * D + 8 + (STYLO ? 4 * NSTY : 0)            # profile block offset
    NB = POFF + (PROF ? 4 * NPROF : 0)
    Xsame = Vector{TM.TMInput}(undef, nmax); Xdiff = Vector{TM.TMInput}(undef, nmax)
    @threads for t in 1:nmax
        bits = Vector{Bool}(undef, NB)
        (a, ki, qi) = same[t]
        pair_feature!(bits, qv[a][qi][1], kv[a][ki][1], qv[a][qi][2], kv[a][ki][2], qv[a][qi][3], kv[a][ki][3], tw, tc, td, D)
        len_bits!(bits, NCH * D, ktl[a][ki], qtl[a][qi])
        STYLO && stylo_bits!(bits, NCH * D + 8, qsz[a][qi], ksz[a][ki], THS)
        PROF && prof_bits!(bits, POFF, qpf[a][qi], kpf[a][ki], THP)
        Xsame[t] = TM.TMInput(bits)
        (ak, ki2, aq, qi2) = diff[t]
        pair_feature!(bits, qv[aq][qi2][1], kv[ak][ki2][1], qv[aq][qi2][2], kv[ak][ki2][2], qv[aq][qi2][3], kv[ak][ki2][3], tw, tc, td, D)
        len_bits!(bits, NCH * D, ktl[ak][ki2], qtl[aq][qi2])
        STYLO && stylo_bits!(bits, NCH * D + 8, qsz[aq][qi2], ksz[ak][ki2], THS)
        PROF && prof_bits!(bits, POFF, qpf[aq][qi2], kpf[ak][ki2], THP)
        Xdiff[t] = TM.TMInput(bits)
    end
    println("features built ($(NB) bits/pair)"); flush(stdout)

    # ---- test pairs (500 av_test, disjoint authors) ----
    pairs_dir = get(ENV, "P4_PAIRS", joinpath(P3, "pairs500"))
    man = [split(l, '\t') for l in Iterators.drop(eachline(joinpath(P3, "pairs500.tsv")), 1)]
    function trunc_sents(sents, L)
        L <= 0 && return sents
        out = Vector{Vector{String}}(); n = 0
        for s in sents
            if n + length(s) <= L; push!(out, s); n += length(s)
            else; take = L - n; take > 0 && push!(out, s[1:take]); break; end
        end
        out
    end
    tid = Int[]; tlab = Int[]
    XteL = Dict(L => TM.TMInput[] for L in EVALLENS)
    zteRows = Vector{Vector{Float32}}(); zteK = Int[]; zteQ = Int[]   # EXPORTZ capture (L=0)
    for row in man
        pid, lab = row[1], parse(Int, row[2])
        k = readsents(joinpath(pairs_dir, "$(pid)_known.tsv")); q = readsents(joinpath(pairs_dir, "$(pid)_q.tsv"))
        (isempty(k) || isempty(q)) && continue
        kwv, kcv, kdv = encode_doc(k, Vw, SWc, word_subs, w2i, sp, GAP, wt, Vc, ROLES, Vd, d2i)
        kcw = _nrm(kwv .- pw); kcc = _nrm(kcv .- pc); kcd = _nrm(kdv .- pd)   # same normalization as training
        klen = sum(length, k; init = 0)
        ksv = STYLO ? zsty(stylo_vec(k, phidx)) : Float64[]
        pn = parse(Int, pid)
        push!(tid, pn); push!(tlab, lab)
        for L in EVALLENS
            qt = trunc_sents(q, L)
            qwv, qcv, qdv = encode_doc(qt, Vw, SWc, word_subs, w2i, sp, GAP, wt, Vc, ROLES, Vd, d2i)
            if EVALSYM && L > 0                    # symmetric protocol: known truncated to L too
                kt = trunc_sents(k, L)
                kwv2, kcv2, kdv2 = encode_doc(kt, Vw, SWc, word_subs, w2i, sp, GAP, wt, Vc, ROLES, Vd, d2i)
                kwL = _nrm(kwv2 .- pw); kcL = _nrm(kcv2 .- pc); kdL = _nrm(kdv2 .- pd); klL = sum(length, kt; init = 0)
                ksvL = STYLO ? zsty(stylo_vec(kt, phidx)) : Float64[]
            else
                kwL = kcw; kcL = kcc; kdL = kcd; klL = klen; ksvL = ksv
            end
            if EXPORTZ != "" && L == 0
                uwv = _nrm(qwv .- pw); ucv = _nrm(qcv .- pc)
                push!(zteRows, vcat(Float32.(uwv .* kwL), Float32.(ucv .* kcL)))
                push!(zteK, klL); push!(zteQ, sum(length, qt; init = 0))
            end
            bits = Vector{Bool}(undef, NB)
            pair_feature!(bits, _nrm(qwv .- pw), kwL, _nrm(qcv .- pc), kcL, _nrm(qdv .- pd), kdL, tw, tc, td, D)
            len_bits!(bits, NCH * D, klL, sum(length, qt; init = 0))
            STYLO && stylo_bits!(bits, NCH * D + 8, zsty(stylo_vec(qt, phidx)), ksvL, THS)
            PROF && prof_bits!(bits, POFF, ptest[(pn, L, 'q')],
                               ptest[(pn, (EVALSYM && L > 0) ? L : 0, 'k')], THP)
            push!(XteL[L], TM.TMInput(bits))
        end
    end
    @printf("test pairs encoded: %d (eval lens: %s)\n", length(tid), join(EVALLENS, ",")); flush(stdout)

    # ---- EXPORTZ: dump continuous z features + meta, then stop (no TM training) ----
    if EXPORTZ != ""
        mkpath(EXPORTZ)
        open(joinpath(EXPORTZ, "train_z.f32"), "w") do io
            for t in 1:nmax
                (a, ki, qi) = same[t]
                write(io, Float32.(qv[a][qi][1] .* kv[a][ki][1]))
                write(io, Float32.(qv[a][qi][2] .* kv[a][ki][2]))
            end
            for t in 1:nmax
                (ak, ki2, aq, qi2) = diff[t]
                write(io, Float32.(qv[aq][qi2][1] .* kv[ak][ki2][1]))
                write(io, Float32.(qv[aq][qi2][2] .* kv[ak][ki2][2]))
            end
        end
        open(joinpath(EXPORTZ, "train_meta.jsonl"), "w") do io
            for t in 1:nmax
                (a, ki, qi) = same[t]
                println(io, "{\"label\":1,\"klen\":$(ktl[a][ki]),\"qlen\":$(qtl[a][qi]),\"ka\":$(a),\"qa\":$(a)}")
            end
            for t in 1:nmax
                (ak, ki2, aq, qi2) = diff[t]
                println(io, "{\"label\":0,\"klen\":$(ktl[ak][ki2]),\"qlen\":$(qtl[aq][qi2]),\"ka\":$(ak),\"qa\":$(aq)}")
            end
        end
        open(joinpath(EXPORTZ, "test_z.f32"), "w") do io
            for r in zteRows; write(io, r); end
        end
        open(joinpath(EXPORTZ, "test_meta.jsonl"), "w") do io
            for (i, pn) in enumerate(tid)
                println(io, "{\"id\":$(pn),\"label\":$(tlab[i]),\"klen\":$(zteK[i]),\"qlen\":$(zteQ[i])}")
            end
        end
        open(joinpath(EXPORTZ, "meta.json"), "w") do io
            println(io, "{\"D\":$(D),\"row_dim\":$(2D),\"n_train\":$(2nmax),\"n_test\":$(length(zteRows))}")
        end
        @printf("EXPORTZ: %d train + %d test rows of %d floats -> %s\n",
                2nmax, length(zteRows), 2D, EXPORTZ); flush(stdout)
        return
    end

    # ---- scaling curve: train at each size (ensemble of E members, margins averaged) ----
    ENSEMBLE = parse(Int, get(ENV, "P4_ENSEMBLE", "1"))

    # ---- Tier B (depth): 2-fold cross-fitted TM STACKING (P4_STACK=1) ----
    # Stage 1 learns clauses on half the pairs; its per-clause activations (graded FPTM match
    # scores = LF - mismatches, thermometer-coded at per-clause quantiles measured on the OTHER
    # half) become stage-2 features + the 8 length bits. Cross-fitting keeps stage 2 from
    # reading overfit activations; test-time margins average both fold pipelines. This is
    # learned content-dependent composition -- depth -- with no gradients anywhere.
    if get(ENV, "P4_STACK", "0") == "1"
        E1 = parse(Int, get(ENV, "P4_STACK_E1", "3"))
        C2 = parse(Int, get(ENV, "P4_STACK_C2", "256")); T2 = parse(Int, get(ENV, "P4_STACK_T2", "64"))
        h = min(SIZES[end] ÷ 2, nmax)
        X = vcat(Xsame[1:h], Xdiff[1:h]); Y = vcat(fill(true, h), fill(false, h))
        n2 = length(X)
        stackedL = Dict(L => zeros(Float64, length(tid)) for L in EVALLENS)
        s1L = Dict(L => zeros(Float64, length(tid)) for L in EVALLENS)

        function acts!(A, tm1, Xs, off)
            Cn = tm1.clauses_num; ta = tm1.clauses
            @threads for j in eachindex(Xs)
                x = Xs[j]
                @inbounds for i in 1:Cn
                    A[off+i, j] = TM.check_clause(tm1, x, view(ta.positive_included_literals, :, i), view(ta.positive_included_literals_inverted, :, i))
                    A[off+Cn+i, j] = TM.check_clause(tm1, x, view(ta.negative_included_literals, :, i), view(ta.negative_included_literals_inverted, :, i))
                end
            end
        end

        for f in 0:1
            tr1 = findall(i -> i % 2 == f, 1:n2); tr2 = findall(i -> i % 2 != f, 1:n2)
            X1 = X[tr1]; Y1 = Y[tr1]; X2 = X[tr2]; Y2 = Y[tr2]
            R = 2 * CLAUSES
            A2 = Matrix{Int32}(undef, E1 * R, length(X2))
            AteL = Dict(L => Matrix{Int32}(undef, E1 * R, length(tid)) for L in EVALLENS)
            for m in 1:E1
                Random.seed!(3000 + 10 * f + m)
                tm1 = TM.TMClassifier(X[1], Y, CLAUSES, T_, S_, L_, LF_; states_num = STATES, include_limit = INCLUDE)
                tt = @elapsed for e in 1:EPOCHS; TM.train!(tm1, X1, Y1; shuffle = true, index = false); end
                acts!(A2, tm1, X2, (m - 1) * R)
                for L in EVALLENS; acts!(AteL[L], tm1, XteL[L], (m - 1) * R); end
                Cn = tm1.clauses_num
                for L in EVALLENS                              # stage-1-alone reference margins
                    AL = AteL[L]
                    for j in 1:length(tid)
                        s1L[L][j] += (sum(@view AL[(m-1)*R+1:(m-1)*R+Cn, j]) - sum(@view AL[(m-1)*R+Cn+1:m*R, j])) / (2 * E1)
                    end
                end
                @printf("STACK fold %d  stage1 member %d/%d (%.0fs)\n", f, m, E1, tt); flush(stdout)
            end
            NR = E1 * R
            th = Matrix{Int32}(undef, NR, 2)                   # per-clause activation quantiles
            tmpv = Vector{Int32}(undef, size(A2, 2))
            for r in 1:NR
                copyto!(tmpv, @view A2[r, :]); sort!(tmpv)
                th[r, 1] = tmpv[max(1, round(Int, 0.5 * length(tmpv)))]
                th[r, 2] = tmpv[max(1, round(Int, 0.85 * length(tmpv)))]
            end
            NB2 = 2 * NR + 8
            tobits = (A, j, xsrc) -> begin
                b = Vector{Bool}(undef, NB2)
                @inbounds for r in 1:NR
                    b[2r-1] = A[r, j] > th[r, 1]; b[2r] = A[r, j] > th[r, 2]
                end
                @inbounds for k in 1:8; b[2NR+k] = xsrc[NCH*D+k]; end   # carry len bits through
                TM.TMInput(b)
            end
            X2b = [tobits(A2, j, X2[j]) for j in eachindex(X2)]
            XteB = Dict(L => [tobits(AteL[L], j, XteL[L][j]) for j in 1:length(tid)] for L in EVALLENS)
            S2 = max(1, round(Int, NB2 / 16))                  # keep s ~= 16 at stage 2
            for m in 1:ENSEMBLE
                Random.seed!(5000 + 10 * f + m)
                tm2 = TM.TMClassifier(X2b[1], Y2, C2, T2, S2, NB2, NB2 ÷ 2; states_num = STATES, include_limit = INCLUDE)
                tt = @elapsed for e in 1:EPOCHS; TM.train!(tm2, X2b, Y2; shuffle = true, index = false); end
                for L in EVALLENS
                    XB = XteB[L]
                    margins = Vector{Float64}(undef, length(XB))
                    @threads for i in eachindex(XB)
                        pos, neg = TM.vote(tm2, tm2.clauses, XB[i]); margins[i] = pos - neg
                    end
                    stackedL[L] .+= margins ./ (2 * ENSEMBLE)
                end
                @printf("STACK fold %d  stage2 member %d/%d (%.0fs)\n", f, m, ENSEMBLE, tt); flush(stdout)
            end
        end
        for L in EVALLENS
            @printf("STACK evalL=%-6s  stage1-alone AUC = %.4f   STACKED AUC = %.4f\n",
                    L <= 0 ? "full" : string(L), auc(s1L[L], tlab), auc(stackedL[L], tlab))
        end
        flush(stdout)
        sbase = get(ENV, "P4_OUT", "tm_stack.jsonl")
        for L in EVALLENS
            fn = L <= 0 ? sbase : replace(sbase, ".jsonl" => "_L$(L).jsonl")
            open(joinpath(HERE, fn), "w") do io
                for i in eachindex(tid)
                    write(io, """{"id":$(tid[i]),"label":$(tlab[i]),"tm":$(stackedL[L][i])}\n""")
                end
            end
        end
        println("wrote stacked scores")
        return
    end
    last_margins = Float64[]
    for sz in SIZES
        h = min(sz ÷ 2, nmax)
        X = vcat(Xsame[1:h], Xdiff[1:h]); Y = vcat(fill(true, h), fill(false, h))
        ensL = Dict(L => zeros(Float64, length(tid)) for L in EVALLENS)
        for m in 1:ENSEMBLE
            Random.seed!(1000 + m)                 # member-specific seed for the training shuffle
            tm = TM.TMClassifier(X[1], Y, CLAUSES, T_, S_, L_, LF_; states_num = STATES, include_limit = INCLUDE)
            tt = @elapsed for e in 1:EPOCHS; TM.train!(tm, X, Y; shuffle = true, index = false); end
            for L in EVALLENS
                Xte = XteL[L]
                margins = Vector{Float64}(undef, length(Xte))
                @threads for i in eachindex(Xte)
                    pos, neg = TM.vote(tm, tm.clauses, Xte[i]); margins[i] = pos - neg
                end
                ensL[L] .+= margins
                L == EVALLENS[1] && @printf("TRAIN %6d pairs  member %d/%d (%.0fs, %d ep)  AUC = %.4f\n",
                                            2h, m, ENSEMBLE, tt, EPOCHS, auc(margins, tlab))
            end
            flush(stdout)
        end
        for L in EVALLENS
            ensL[L] ./= ENSEMBLE
            @printf("TRAIN %6d pairs  ENSEMBLE(%d)  evalL=%-6s ->  test AUC = %.4f\n",
                    2h, ENSEMBLE, L <= 0 ? "full" : string(L), auc(ensL[L], tlab))
        end
        flush(stdout)
        last_margins = ensL
    end

    base = get(ENV, "P4_OUT", "tm_scaled_scores.jsonl")
    for L in EVALLENS
        fn = L <= 0 ? base : replace(base, ".jsonl" => "_L$(L).jsonl")
        open(joinpath(HERE, fn), "w") do io
            for i in eachindex(tid)
                write(io, """{"id":$(tid[i]),"label":$(tlab[i]),"tm":$(last_margins[L][i])}\n""")
            end
        end
    end
    println("wrote scores (largest size, all eval lens)")
end

using Statistics: quantile!
using Base: time
main()
