# FORENSIC EXPLAINABILITY for the HDC-TM verifier: turn a margin into a report an examiner can
# read, audit, and cross-examine. Three layers, only the last of which is approximate:
#
#   1. margin -> clauses          EXACT. The verdict IS sum(positive clause votes) - sum(negative).
#      Unlike attention weights, this decomposition is the computation, not a post-hoc story.
#   2. clause -> input bits       EXACT. Every literal names a thermometer band of one dimension
#      of one channel (or a length bit) -- readable as a conjunction.
#   3. bits -> LINGUISTIC features   APPROXIMATE, and honestly so: an HDC dimension is a
#      superposition, so no dimension "means" a word. We therefore attribute by OCCLUSION --
#      delete a word type from BOTH documents, re-encode, re-vote, and report the margin shift.
#      TM inference is bitwise, so thousands of probes per case are affordable (this is the
#      practical advantage over refitting KN or re-running a transformer).
#
# Also reports per-member agreement across the ensemble, since a single TM's clause set is one
# sample of a stochastic training process (documented ±0.03 AUC swings).
#
#   julia -t auto phase4/explain.jl        -> phase4/explain_report.html
#
# Env: P4X_PAIRS (comma-separated test pair ids; default = auto-pick 4 illustrative cases)
#      P4X_TOPW (word types probed per case), P4X_MEMBERS, plus the usual P4_* encoder knobs.

import Pkg
Pkg.activate(joinpath(@__DIR__, "..", "hdc"); io = devnull)
using HDC, Random, Printf, Serialization
using LinearAlgebra: dot
using Statistics: quantile!
using Base.Threads: @threads
include(joinpath(@__DIR__, "..", "Tsetlin.jl-main", "src", "Tsetlin.jl"))
const TM = Tsetlin

const D = 8192; const HERE = @__DIR__; const P3 = joinpath(HERE, "..", "phase3")
const KLENS = [150, 300, 600, 1200, 3000]; const QLENS = [150, 300, 600, 1200]
const KSTRIDE = 2.0; const QSTRIDE = 2.0
const CLAUSES = parse(Int, get(ENV, "P4_CLAUSES", "512"))
const T_ = 128; const S_ = 4096; const L_ = 4096; const LF_ = 2048
const STATES = 256; const INCLUDE = 220
const EPOCHS = parse(Int, get(ENV, "P4_EPOCHS", "60"))
const MEMBERS = parse(Int, get(ENV, "P4X_MEMBERS", "3"))
const TOPW = parse(Int, get(ENV, "P4X_TOPW", "60"))
const NPAIRS = 18000
const KBITS = [200, 400, 800, 2000]; const QBITS = [200, 400, 800, 1100]

readsents(p) = [String.(split(l, '\t')) for l in eachline(p) if !isempty(l)]

@inline function circ_xor!(fp, src, r, W)
    @inbounds @simd for i in 1:W; j = i - r; j < 1 && (j += W); j > W && (j -= W); fp[i] ⊻= src[j]; end
end
@inline function add_bits!(acc, fp, W, g)
    @inbounds for c in 1:W; x = fp[c]; base = (c - 1) * 64
        while x != 0; acc[base+trailing_zeros(x)+1] += g; x &= x - one(x); end
    end
end

# `skip`: vocabulary id to delete from the stream (0 = none) -- the occlusion probe.
function encode_doc(sents, Vw, SWc, word_subs, w2i, sp, GAP, wt; skip::Int = 0)
    W = sp.W; aw = zeros(Float64, sp.D); ac = zeros(Float64, sp.D); nw = 0.0; nc = 0.0; fp = newhv(sp)
    for s in sents
        ids = Int[]
        for w in s
            id = get(w2i, w, 0)
            (id > 0 && id != skip) && push!(ids, id)
        end
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
    end
    (nw > 0 ? aw ./ nw : zeros(Float64, sp.D)), (nc > 0 ? ac ./ nc : zeros(Float64, sp.D))
end

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

function pair_feature!(bits, uw, kw, uc, kc, tw, tc, Dd)
    @inbounds for b in 1:Dd
        z = uw[b] * kw[b]; o = (b - 1) * 4
        bits[o+1] = z > tw[1]; bits[o+2] = z > tw[2]; bits[o+3] = z < -tw[1]; bits[o+4] = z < -tw[2]
    end
    off = 4Dd
    @inbounds for b in 1:Dd
        z = uc[b] * kc[b]; o = off + (b - 1) * 4
        bits[o+1] = z > tc[1]; bits[o+2] = z > tc[2]; bits[o+3] = z < -tc[1]; bits[o+4] = z < -tc[2]
    end
    bits
end
@inline function len_bits!(bits, off, klen, qlen)
    @inbounds for i in 1:4
        bits[off+i] = klen >= KBITS[i]; bits[off+4+i] = qlen >= QBITS[i]
    end
    bits
end

compose_all(SW, Q, word_subs, sp) = begin
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

esc(s) = replace(s, "&" => "&amp;", "<" => "&lt;", ">" => "&gt;", "\"" => "&quot;")
colour(v, scale) = begin                       # same palette as lambdag.heatmap_html
    x = max(-1.0, min(1.0, v / (scale + 1e-12)))
    x >= 0 ? @sprintf("rgba(214,39,40,%.3f)", 0.08 + 0.62x) : @sprintf("rgba(31,119,180,%.3f)", 0.08 - 0.62x)
end

function main()
    SW, Q, word_subs, vocab, w2i = deserialize(joinpath(P3, "subword_atoms.jls"))
    sp = Space(D); rngG = MersenneTwister(11)
    Vw = compose_all(SW, Q, word_subs, sp); SWc = Matrix{UInt64}(SW)
    GAP = randcodes(sp, 2, MersenneTwister(43))

    afiles = sort(filter(f -> endswith(f, ".tsv"), readdir(joinpath(HERE, "authors"), join = true)))
    nA = length(afiles)
    kfr = Vector{Vector{Vector{Vector{String}}}}(undef, nA); qfr = similar(kfr)
    @threads for a in 1:nA
        sents = readsents(afiles[a])
        tot = sum(length, sents; init = 0); cut = 0; n = 0
        for (i, s) in enumerate(sents); n += length(s); if n >= 0.6 * tot; cut = i; break; end; end
        kfr[a] = reduce(vcat, [fragments(sents[1:cut], L, KSTRIDE) for L in KLENS])
        qfr[a] = reduce(vcat, [fragments(sents[cut+1:end], L, QSTRIDE) for L in QLENS])
    end

    wcnt = zeros(Float64, length(vocab)); wtot = 0.0     # surprisal weights (best config)
    for a in 1:nA, f in kfr[a], s in f, w in s
        id = get(w2i, w, 0); id > 0 && (wcnt[id] += 1; wtot += 1)
    end
    wt = [-log2((wcnt[i] + 0.5) / (wtot + 0.5 * length(vocab))) for i in eachindex(wcnt)]
    wt ./= (sum(wt) / length(wt))

    kv = [Vector{Tuple{Vector{Float64},Vector{Float64}}}(undef, length(kfr[a])) for a in 1:nA]
    qv = [Vector{Tuple{Vector{Float64},Vector{Float64}}}(undef, length(qfr[a])) for a in 1:nA]
    @threads for a in 1:nA
        for (i, f) in enumerate(kfr[a]); kv[a][i] = encode_doc(f, Vw, SWc, word_subs, w2i, sp, GAP, wt); end
        for (i, f) in enumerate(qfr[a]); qv[a][i] = encode_doc(f, Vw, SWc, word_subs, w2i, sp, GAP, wt); end
    end
    allw = [v[1] for a in 1:nA for v in kv[a]]; allc = [v[2] for a in 1:nA for v in kv[a]]
    for a in 1:nA, v in qv[a]; push!(allw, v[1]); push!(allc, v[2]); end
    pw = sum(allw) ./ length(allw); pc = sum(allc) ./ length(allc)
    _nrm(x) = x ./ (sqrt(dot(x, x)) + 1e-9)
    cen(v) = (_nrm(v[1] .- pw), _nrm(v[2] .- pc))
    kv = [[cen(v) for v in kv[a]] for a in 1:nA]; qv = [[cen(v) for v in qv[a]] for a in 1:nA]
    ktl = [[sum(length, f; init = 0) for f in kfr[a]] for a in 1:nA]
    qtl = [[sum(length, f; init = 0) for f in qfr[a]] for a in 1:nA]

    same = Tuple{Int,Int,Int}[]
    for a in 1:nA, ki in eachindex(kv[a]), qi in eachindex(qv[a]); push!(same, (a, ki, qi)); end
    shuffle!(rngG, same); nmax = min(NPAIRS ÷ 2, length(same))
    diff = Tuple{Int,Int,Int,Int}[]
    while length(diff) < nmax
        a = rand(rngG, 1:nA); b = rand(rngG, 1:nA); a == b && continue
        push!(diff, (a, rand(rngG, eachindex(kv[a])), b, rand(rngG, eachindex(qv[b]))))
    end
    zs_w = Float64[]; zs_c = Float64[]
    for t in 1:min(400, nmax)
        (a, ki, qi) = same[t]; uw, uc = qv[a][qi]; kw, kc = kv[a][ki]
        append!(zs_w, abs.(uw .* kw)); append!(zs_c, abs.(uc .* kc))
    end
    tw = (quantile!(zs_w, 0.5), quantile!(zs_w, 0.85)); tc = (quantile!(zs_c, 0.5), quantile!(zs_c, 0.85))

    NB = 8D + 8
    X = Vector{TM.TMInput}(undef, 2nmax); Y = Vector{Bool}(undef, 2nmax)
    @threads for t in 1:nmax
        bits = Vector{Bool}(undef, NB)
        (a, ki, qi) = same[t]
        pair_feature!(bits, qv[a][qi][1], kv[a][ki][1], qv[a][qi][2], kv[a][ki][2], tw, tc, D)
        len_bits!(bits, 8D, ktl[a][ki], qtl[a][qi]); X[t] = TM.TMInput(bits); Y[t] = true
        (ak, ki2, aq, qi2) = diff[t]
        pair_feature!(bits, qv[aq][qi2][1], kv[ak][ki2][1], qv[aq][qi2][2], kv[ak][ki2][2], tw, tc, D)
        len_bits!(bits, 8D, ktl[ak][ki2], qtl[aq][qi2]); X[nmax+t] = TM.TMInput(bits); Y[nmax+t] = false
    end
    println("training $(MEMBERS) explainable member(s) on $(2nmax) pairs ..."); flush(stdout)
    tms = Vector{Any}(undef, MEMBERS)
    for m in 1:MEMBERS
        Random.seed!(1000 + m)
        tm = TM.TMClassifier(X[1], Y, CLAUSES, T_, S_, L_, LF_; states_num = STATES, include_limit = INCLUDE)
        tt = @elapsed for e in 1:EPOCHS; TM.train!(tm, X, Y; shuffle = true, index = false); end
        tms[m] = tm; @printf("  member %d/%d (%.0fs)\n", m, MEMBERS, tt); flush(stdout)
    end

    # ---- test pairs ----
    man = [split(l, '\t') for l in Iterators.drop(eachline(joinpath(P3, "pairs500.tsv")), 1)]
    pairs_dir = joinpath(P3, "pairs500")
    feat(kw, kc, qw, qc, klen, qlen) = begin
        bits = Vector{Bool}(undef, NB)
        pair_feature!(bits, qw, kw, qc, kc, tw, tc, D); len_bits!(bits, 8D, klen, qlen)
        TM.TMInput(bits)
    end
    margins_of(x) = [(p = TM.vote(tm, tm.clauses, x); p[1] - p[2]) for tm in tms]

    docs = Dict{Int,Any}(); scores = Dict{Int,Float64}(); labels = Dict{Int,Int}()
    for row in man
        pid = parse(Int, row[1]); lab = parse(Int, row[2])
        k = readsents(joinpath(pairs_dir, "$(pid)_known.tsv")); q = readsents(joinpath(pairs_dir, "$(pid)_q.tsv"))
        (isempty(k) || isempty(q)) && continue
        kwv, kcv = encode_doc(k, Vw, SWc, word_subs, w2i, sp, GAP, wt)
        qwv, qcv = encode_doc(q, Vw, SWc, word_subs, w2i, sp, GAP, wt)
        x = feat(_nrm(kwv .- pw), _nrm(kcv .- pc), _nrm(qwv .- pw), _nrm(qcv .- pc),
                 sum(length, k; init = 0), sum(length, q; init = 0))
        ms = margins_of(x)
        docs[pid] = (k, q); scores[pid] = sum(ms) / length(ms); labels[pid] = lab
    end
    ids = sort(collect(keys(scores)))
    sel = if haskey(ENV, "P4X_PAIRS")
        [parse(Int, s) for s in split(ENV["P4X_PAIRS"], ",")]
    else                       # confident-correct SA, confident-correct DA, and the two worst errors
        sa = sort([i for i in ids if labels[i] == 1]; by = i -> -scores[i])
        da = sort([i for i in ids if labels[i] == 0]; by = i -> scores[i])
        errSA = sort([i for i in ids if labels[i] == 1]; by = i -> scores[i])
        errDA = sort([i for i in ids if labels[i] == 0]; by = i -> -scores[i])
        [sa[1], da[1], errSA[1], errDA[1]]
    end
    @printf("explaining pairs: %s\n", join(sel, ", ")); flush(stdout)

    out = IOBuffer()
    write(out, """<div style="font-family:system-ui,-apple-system,Segoe UI,sans-serif;max-width:1100px;margin:0 auto;padding:1.5em 2em;color:#1a1a1a;background:#fff">
<!-- deliberately single-theme: a forensic exhibit is a light printed document; background painted explicitly so the page holds on any viewer ground -->
<h1 style="font-size:1.5em">HDC&ndash;TM authorship verifier &mdash; forensic report</h1>
<p style="color:#555">Margin = &Sigma;(positive clause votes) &minus; &Sigma;(negative clause votes), averaged over $(MEMBERS) independently seeded ensemble members. Positive favours <b>same author</b>. Attributions below are <b>occlusion</b> measurements: a word type is deleted from <i>both</i> documents, both are re-encoded, and the change in margin is recorded. Occlusion is used because an HDC dimension is a superposition of all features &mdash; per-dimension &ldquo;decoding&rdquo; would be ill-posed, aggregate deletion is not.</p>
""")

    for pid in sel
        (k, q) = docs[pid]
        kwv0, kcv0 = encode_doc(k, Vw, SWc, word_subs, w2i, sp, GAP, wt)
        qwv0, qcv0 = encode_doc(q, Vw, SWc, word_subs, w2i, sp, GAP, wt)
        klen = sum(length, k; init = 0); qlen = sum(length, q; init = 0)
        x0 = feat(_nrm(kwv0 .- pw), _nrm(kcv0 .- pc), _nrm(qwv0 .- pw), _nrm(qcv0 .- pc), klen, qlen)
        m0 = margins_of(x0); mm0 = sum(m0) / length(m0)

        # candidate word types: those with the largest |Δ| rate deviation in either document
        cnt = Dict{Int,Int}()
        for s in k, w in s; id = get(w2i, w, 0); id > 0 && (cnt[id] = get(cnt, id, 0) + 1); end
        for s in q, w in s; id = get(w2i, w, 0); id > 0 && (cnt[id] = get(cnt, id, 0) + 1); end
        cand = sort(collect(keys(cnt)); by = i -> -cnt[i])[1:min(TOPW, length(cnt))]
        attrib = Vector{Tuple{String,Float64}}(undef, length(cand))
        @threads for t in eachindex(cand)
            id = cand[t]
            kw2, kc2 = encode_doc(k, Vw, SWc, word_subs, w2i, sp, GAP, wt; skip = id)
            qw2, qc2 = encode_doc(q, Vw, SWc, word_subs, w2i, sp, GAP, wt; skip = id)
            x2 = feat(_nrm(kw2 .- pw), _nrm(kc2 .- pc), _nrm(qw2 .- pw), _nrm(qc2 .- pc), klen, qlen)
            m2 = margins_of(x2)
            attrib[t] = (vocab[id], mm0 - sum(m2) / length(m2))   # >0 = supports same-author
        end
        sort!(attrib; by = x -> -abs(x[2]))
        amap = Dict(w => v for (w, v) in attrib)
        scale = maximum(abs(v) for (_, v) in attrib; init = 1.0)

        # clause-level view on the first member (exact decomposition of ITS vote)
        tm1 = tms[1]; ta = tm1.clauses; Cn = tm1.clauses_num
        posv = [TM.check_clause(tm1, x0, view(ta.positive_included_literals, :, i), view(ta.positive_included_literals_inverted, :, i)) for i in 1:Cn]
        negv = [TM.check_clause(tm1, x0, view(ta.negative_included_literals, :, i), view(ta.negative_included_literals_inverted, :, i)) for i in 1:Cn]
        npf = count(>(0), posv); nnf = count(>(0), negv)

        verdict = mm0 > 0 ? "SAME author" : "DIFFERENT authors"
        truth = labels[pid] == 1 ? "same author" : "different authors"
        ok = (mm0 > 0) == (labels[pid] == 1)
        write(out, """<hr style="margin:2em 0;border:none;border-top:1px solid #ddd">
<h2 style="font-size:1.2em">Case $(pid) &mdash; margin $(@sprintf("%+.1f", mm0)) &rarr; <b>$(verdict)</b>
<span style="font-weight:400;color:$(ok ? "#2a7" : "#c33")">(ground truth: $(truth)$(ok ? "" : " &mdash; ERROR"))</span></h2>
<p style="color:#555">Known document $(klen) tokens, questioned $(qlen) tokens (POSNoise-masked). Per-member margins: $(join([@sprintf("%+.0f", v) for v in m0], ", ")) &mdash; $(all(v -> (v > 0) == (mm0 > 0), m0) ? "all members agree" : "<b>members disagree</b>") on the direction. Member 1 fired $(npf) same-author and $(nnf) different-author clauses of $(Cn) each.</p>
<h3 style="font-size:1em">Occlusion attributions (top $(min(20, length(attrib))) of $(length(attrib)) probed word types)</h3>
<table style="border-collapse:collapse;font-size:13px;width:100%"><tr style="text-align:left;color:#666">
<th style="padding:3px 8px">token</th><th style="padding:3px 8px">&Delta;margin if deleted</th><th style="padding:3px 8px">reads as</th></tr>""")
        for (w, v) in attrib[1:min(20, length(attrib))]
            reads = v > 0 ? "shared usage supporting <b>same author</b>" : "usage difference supporting <b>different authors</b>"
            write(out, """<tr><td style="padding:3px 8px;font-family:ui-monospace,monospace;background:$(colour(v, scale))">$(esc(w))</td>
<td style="padding:3px 8px;font-family:ui-monospace,monospace">$(@sprintf("%+.2f", v))</td><td style="padding:3px 8px;color:#555">$(reads)</td></tr>""")
        end
        write(out, "</table>\n")

        write(out, """<h3 style="font-size:1em;margin-top:1.2em">Questioned document, coloured by attribution</h3>
<p style="color:#555;font-size:13px"><span style="background:rgba(214,39,40,.55);padding:1px 6px">supports same author</span>
<span style="background:rgba(31,119,180,.55);padding:1px 6px">supports different authors</span> &mdash; hover a token for its measured &Delta;margin.</p>
<div style="font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:13px;line-height:2.1;max-height:320px;overflow:auto;border:1px solid #eee;padding:.6em">""")
        for s in q[1:min(25, length(q))]
            for w in s
                v = get(amap, w, 0.0)
                write(out, """<span title="$(@sprintf("%+.3f", v))" style="background:$(colour(v, scale));padding:1px 2px;margin:0 1px">$(esc(w))</span> """)
            end
            write(out, "<br>")
        end
        write(out, "</div>\n")
    end

    write(out, """<hr style="margin:2em 0;border:none;border-top:1px solid #ddd">
<h2 style="font-size:1.1em">How to read this &mdash; and its limits</h2>
<ul style="color:#444;line-height:1.6">
<li><b>Exact where it claims to be.</b> The margin is literally the clause-vote sum; clause literals are literally thermometer bands of specific sketch dimensions. No surrogate model stands between the report and the decision.</li>
<li><b>Attributions are aggregate, not per-dimension.</b> Each number is a measured deletion effect on the whole pipeline. Individual HDC dimensions carry no isolated meaning (superposition), so they are deliberately not reported.</li>
<li><b>Thermometer coding makes effects piecewise.</b> A small change can be invisible until it crosses a band edge; attributions are therefore not smooth derivatives.</li>
<li><b>Ensemble disagreement is diagnostic.</b> Training is stochastic (unseeded shuffle + HogWild threading cause documented AUC swings); cases where members split on direction should be treated as low-confidence regardless of the averaged margin.</li>
<li><b>The margin is not a likelihood ratio.</b> For forensic reporting it must pass through the calibrator (and, in the deployed stack, fusion with KN &lambda;<sub>G</sub>) before being expressed as an LR.</li>
</ul></div>""")
    open(joinpath(HERE, "explain_report.html"), "w") do io; write(io, String(take!(out))); end
    println("wrote phase4/explain_report.html")
end

main()
