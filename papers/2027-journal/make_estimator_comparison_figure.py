# What we keep of LambdaG, and what we change.
#
# THE POINT OF THE FIGURE, and the reason for its shape: the score equation is
# NOT one of the things we change. Everything we do is a change to one of three
# inputs -- the evidence unit S, the reference grammars G_j, and the route from
# a score to a likelihood ratio. Drawing the score as a full-width band above
# three two-column rows says that before any of the text is read.
#
# Equations are verbatim from the paper, which takes the first three from the
# sources: Eq. (lambdag) and the size-matched reference sampling from Nini et
# al. 2026, Eqs. (sqrtcorr) and (hapaxcorr) from Barlow, Nini & Manino 2026;
# the CLT line (clt) is ours. matplotlib mathtext lacks \big -- use
# \left(...\right) -- and its failures surface at DRAW time, not at text() time,
# so render a suspect equation on its own to find the broken one.
#
# WHY THE UNIT GETS A ROW OF ITS OWN. S does three jobs at once: the unit of
# independence in the sum, the unit the reference grammars are size-matched on,
# and the unit the contiguous run is measured in. Nini et al. fix it as the
# sentence; Section 7 retires that, and the machinery is already unit-agnostic
# because lambdag.py matches on the number of objects in the store, whatever
# they are. An earlier draft of this figure buried all that in a caption, which
# had the effect of silently keeping fixed one of the very things the paper
# changes.
#
# EVERY b HERE WAS MEASURED ON SENTENCES, and the figure says so, because a
# declared parameter that goes undeclared is worth nothing. Neither
# run_symmeter.py nor run_xref.py rechunked at the time these were run; both
# now take --seg, and the w100 spot-check is what will license dropping that
# qualifier.
#
# Sources for every measured number:
#   b pooled        -0.056..-0.117   tab_b_measured, 38 datasets
#   b size-matched  -0.004..-0.030   tab_symmeter, sampled donors, 8 datasets
#   b shape-matched -0.009..+0.011   tab_symmeter, contiguous donors, 8 datasets
#   meter, register +0.058..+0.139   symmeter genre-mismatch runs, contiguous
#
# NO IN-HOUSE SHORTHAND ON THE PAGE. Two words the by-product band cannot do
# without -- what ONE POINT is, and what the gauge reading measures -- are
# glossed directly under its header, because a reader who has to reconstruct
# either of them is reading a worse figure than the data deserves. Earlier
# drafts called each point an "arm", which is a word from our own bench notes
# and from nowhere the reader has been.
#
# THE THIRD CHART ANSWERS THE OBVIOUS OBJECTION to the second. Two matched
# datasets sit above C_llr 1, which invites the reading that the estimator
# fails there; it does not. The studentised score is in donor standard
# deviations, not in decades of odds, so read straight off as a log10 LR its
# magnitude is simply in the wrong units -- and the floors say so (C_llr^min
# 0.13..0.36 on all eight). The chart fits the missing unit constant, a slope
# and an intercept, ALWAYS ON OTHER DATASETS: leave-one-dataset-out for the
# matched ones, and for the mismatched ones the map fitted on all the matched
# ones, which is the position a caseworker is in. So the chart is not a
# demonstration that calibration works. It is a test of whether calibration
# can rescue a wrong reference population, and the answer is no -- three of
# the four genre-mismatched runs stay above 1.
#
# THE BAND IS DRAWN ON THE CONTIGUOUS RUNS, and that is not a detail. The band
# says "with b set to zero, a residual reading warns you", so the readings must
# come from the estimator in which b IS set to zero -- the size- AND
# shape-matched donors of panel ②, the *_L2000_contig score files. An earlier
# version globbed the SAMPLED-donor files instead, so it illustrated a claim
# about one estimator with readings from another; the matched datasets then sat
# at -0.030..-0.004 while the panel above them claimed -0.009..+0.011. The
# tidier picture was the wrong one: on the correct files two of the eight
# matched datasets sit just above C_llr 1, and that is reported, not cropped.
#
# THE BY-PRODUCT BAND SHOWS ONE CONTRAST AND NO MORE: a reference population
# matched to the case in language and in genre, against one that differs in
# genre with nothing done to adjust for the difference. Both blocks are the
# same experiment at the same settings -- POSNoise, 2,000 tokens a side,
# contiguous size-matched donors -- so every point on the chart is comparable
# with every other, which is the whole reason for narrowing it. Mismatches of
# language belong to the paper, where they can be set out at length.
#
#   python journal/figures/make_estimator_comparison_figure.py
#
# Output: journal/figures/estimator_comparison.png

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from lambdag import cllr  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
SYM = ROOT / "experiments" / "scores"
OUT = HERE / "estimator_comparison.png"

FIGW, FIGH = 15.5, 19.4

BLUE = "#2a78d6"          # the method as it stands
ORANGE = "#eb6834"        # what this paper alters
SURFACE = "#fcfcfb"
PANEL = "#f4f4f1"
INK = "#1a1a19"
MUTED = "#6b6b68"
RULE = "#dededa"

L, R, W = 0.038, 0.518, 0.444          # left col, right col, col width
FW = 0.924                              # full width


def pts(n):
    """n typographic points as a fraction of figure height."""
    return n / 72.0 / FIGH


class Col:
    """A cursor running down one panel, so spacing is computed not guessed."""

    def __init__(self, ax, x, top):
        self.ax, self.x, self.y = ax, x, top

    def gap(self, n):
        self.y -= pts(n)
        return self

    def head(self, text, accent, sub=None):
        self.gap(20)
        self.ax.text(self.x, self.y, text, fontsize=12, color=accent,
                     weight="semibold", va="top", zorder=3)
        if sub:
            self.gap(19)
            self.ax.text(self.x, self.y, sub, fontsize=9.5, color=MUTED,
                         va="top", style="italic", zorder=3)
        return self

    def eq(self, text, width, size=14.5):
        self.gap(38)
        self.ax.text(self.x + width / 2 - pts(0) , self.y, text, fontsize=size,
                     color=INK, ha="center", va="center", zorder=3)
        self.gap(26)
        return self

    def body(self, lines, size=10.2):
        self.gap(12)
        for t in lines:
            self.gap(15)
            self.ax.text(self.x, self.y, t, fontsize=size,
                         color=INK if t.startswith("→") else MUTED,
                         va="top", zorder=3)
        return self

    def foot(self, text, accent):
        self.gap(28)
        self.ax.text(self.x, self.y, text, fontsize=11, color=accent,
                     va="top", weight="semibold", zorder=3)
        return self


def dataset_stats(fn):
    """One dataset's gauge reading and what it costs, from its score file.

    Three numbers, because the whole point is that they come apart. b says where
    the different-author band sits. AUC says whether the two hypotheses are still
    separable -- and it barely moves. C_llr says whether the score can be uttered
    as odds, and THAT is what a wrong reference population destroys: 1.0 is the
    value of saying nothing at all, and a mismatched one reaches twenty.
    """
    per, y, sq, t = {}, [], [], []
    for line in open(fn, encoding="utf-8"):
        r = json.loads(line)
        if not r["within"]:
            per.setdefault(r["known"], []).append(r["lambda_G"] / r["n_q"])
        y.append(r["within"])
        sq.append(r["lambda_G"] / np.sqrt(r["n_q"]))
        lj = r["lam_j"]
        t.append(float(np.mean(lj) / (np.std(lj) + 1e-9)))
    if not per or len(set(y)) < 2:
        return None
    y, sq, t = np.array(y), np.array(sq), np.array(t)
    return (float(np.mean([np.mean(v) for v in per.values()])),
            float(roc_auc_score(y, sq)),
            float(cllr(sq[y == 1], sq[y == 0])),
            float(cllr(t[y == 1], t[y == 0])))


SHORT = {"english": "En", "french": "Fr", "german": "De", "polish": "Pl",
         "czech": "Cs", "hungarian": "Hu", "dracor": "drama",
         "poetree": "verse", "novels": "novels"}


def nick(name):
    return " ".join(SHORT.get(p, p) for p in name.split("_"))


def meter_readings():
    """The gauge readings, from the one experiment that produced them.

    A single block -- POSNoise at 2,000 tokens a side, contiguous size-matched
    donors -- so the matched runs and the mismatched ones differ in the
    reference population and in nothing else. Read from the score files at
    render time, so nothing here can drift from the runs.
    """
    reg, reg_ctl = [], []
    for fn in sorted(SYM.glob("*__symref-*__L2000_contig.jsonl")):
        ds, ref = fn.name.replace("__L2000_contig.jsonl", "").split("__symref-")
        st = dataset_stats(fn)
        if st:
            (reg_ctl if ds == ref else reg).append(
                st + (f"{nick(ds)} ← {nick(ref)}",))
    reg = [r[:-1] + (r[-1].replace("Fr novels ← Fr", "Fr novels ←")
                     .replace("De novels ← De", "De novels ←")
                     .replace("En novels ← En", "En novels ←")
                     .replace("De verse ← De", "De verse ←"),) for r in reg]
    spec = [("reference matched: same language, same genre", reg_ctl, MUTED, "o"),
            ("reference of another genre, nothing done to adjust", reg, ORANGE, "o")]
    out = []
    for name, rows, colour, marker in spec:
        out.append(dict(name=name, b=np.array([r[0] for r in rows]),
                        auc=np.array([r[1] for r in rows]),
                        cllr=np.array([r[2] for r in rows]),
                        cllr_t=np.array([r[3] for r in rows]),
                        tags=[r[4] for r in rows],
                        colour=colour, marker=marker))
    return out


def calibration_readings():
    """What one transported calibration constant does to the readings above.

    The studentised score has the right location and a per-case scale, but it is
    measured in donor standard deviations, not in decades of odds. Read straight
    off as a log10 likelihood ratio it can therefore still cost more than saying
    nothing, even where the two hypotheses are perfectly separable -- which is
    why two matched datasets sit above C_llr 1 in the chart to its left. The step
    that converts the units is a monotone map, a slope and an intercept, and the
    point worth making is WHERE it is fitted: never on the case in hand.

    Matched datasets are calibrated leave-one-dataset-out. Mismatched ones get
    the map fitted on all the matched datasets, which is the situation a
    caseworker is actually in -- a constant learned where the reference
    population was sound, carried to a case where it is not. So the chart is not
    a demonstration that calibration works; it is a test of whether calibration
    can rescue a wrong population, and the answer it returns is no.
    """
    runs = {}
    for fn in sorted(SYM.glob("*__symref-*__L2000_contig.jsonl")):
        ds, ref = fn.name.replace("__L2000_contig.jsonl", "").split("__symref-")
        y, t = [], []
        for line in open(fn, encoding="utf-8"):
            r = json.loads(line)
            y.append(r["within"])
            lj = r["lam_j"]
            t.append(float(np.mean(lj) / (np.std(lj) + 1e-9)))
        runs[(ds, ref)] = (np.asarray(t, float), np.asarray(y, int), ds == ref)
    matched = [k for k, v in runs.items() if v[2]]

    def fit(keys):
        X = np.concatenate([runs[k][0] for k in keys])[:, None]
        yy = np.concatenate([runs[k][1] for k in keys])
        m = LogisticRegression(C=1e12).fit(X, yy)
        return m, float(np.log(yy.mean() / (1 - yy.mean())))

    rows = []
    for k, (t, y, mt) in runs.items():
        m, prior = fit([j for j in matched if j != k] if mt else matched)
        llr = (m.decision_function(t[:, None]) - prior) / np.log(10)
        rows.append(dict(matched=mt,
                         before=float(cllr(t[y == 1], t[y == 0])),
                         after=float(cllr(llr[y == 1], llr[y == 0]))))
    return rows, float(fit(matched)[0].coef_[0, 0] / np.log(10))


def panel(ax, x, y, w, h, accent, fill=PANEL):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0,rounding_size=0.010",
        linewidth=0, facecolor=fill, zorder=1, mutation_aspect=1.6))
    ax.add_patch(plt.Rectangle((x, y), 0.0052, h, linewidth=0,
                               facecolor=accent, zorder=2))


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    fig = plt.figure(figsize=(FIGW, FIGH))
    fig.patch.set_facecolor(SURFACE)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(L, 0.982, "λ$_G$: what is kept, and what is changed",
            fontsize=17, color=INK, va="top", weight="semibold")
    ax.text(L, 0.960,
            "Nini et al. (2026) and Barlow, Nini & Manino (2026) against the "
            "present paper. The score itself is not among the changes.",
            fontsize=11, color=MUTED, va="top")

    # ---- band A: the score, shared ------------------------------------- #
    aT, aH = 0.940, 0.0704
    panel(ax, L, aT - aH, FW, aH, MUTED, fill="#f0efec")
    ax.text(L + 0.020, aT - pts(20), "THE SCORE   ·   unchanged", fontsize=12,
            color=INK, weight="semibold", va="top", zorder=3)
    ax.text(0.5, aT - pts(58),
            r"$\lambda_G(Q,K,R)=\sum_{S\in Q}\ \sum_{t_i\in S}\left[\ "
            r"\log_{10}P(t_i\mid \mathrm{ctx}_i,\,G_K)\ -\ \frac{1}{r}"
            r"\sum_{j=1}^{r}\log_{10}P(t_i\mid \mathrm{ctx}_i,\,G_j)\ \right]$",
            fontsize=15, color=INK, ha="center", va="center", zorder=3)
    ax.text(0.5, aT - pts(92),
            "order-10 Kneser–Ney grammars over POS-masked tokens; $r=30$ "
            "reference grammars.   Everything we change is a change to one of "
            "the three inputs below — never to the ratio.",
            fontsize=10.2, color=MUTED, ha="center", va="center", zorder=3)

    # ---- band B: the evidence unit ------------------------------------- #
    bT, bH = aT - aH - 0.017, 0.1366
    panel(ax, L, bT - bH, W, bH, BLUE)
    panel(ax, R, bT - bH, W, bH, ORANGE)

    Col(ax, L + 0.020, bT).head("① THE EVIDENCE UNIT   $S$", BLUE,
                                "Nini et al. 2026").body([
        "the SENTENCE, fixed — assumed available and apt for",
        "every genre and every length of evidence",
        "→  what counts as one piece of independent evidence",
        "     is never varied, so it is never examined",
        "→  and where the text resists segmentation — verse,",
        "     damaged or unpunctuated — there is no fallback",
    ]).foot("an unexamined default", BLUE)

    Col(ax, R + 0.020, bT).head("① THE EVIDENCE UNIT   $S$", ORANGE,
                                "this paper, §7").body([
        "a DECLARED PARAMETER: sentences for well-punctuated",
        "prose at generous $L$; $w=100$ tokens where evidence is",
        "scarce; $w=20$ for verse, whose calibration it rescues",
        "→  unit-agnostic machinery: the size-matching and the",
        "     contiguous run below count whatever units are in",
        "     the store, so nothing downstream needs changing",
    ]).foot("declared, varied and reported", ORANGE)

    # ---- band C: the reference grammars -------------------------------- #
    cT, cH = bT - bH - 0.017, 0.1601
    panel(ax, L, cT - cH, W, cH, BLUE)
    panel(ax, R, cT - cH, W, cH, ORANGE)

    Col(ax, L + 0.020, cT).head("② THE REFERENCE GRAMMARS   $G_1 … G_r$", BLUE,
                                "Nini et al. 2026").eq(
        r"$G_j\ \leftarrow\ \mathrm{fit}\left(\mathrm{sample}_{|S_K|}"
        r"\left(S_{\mathrm{ref}}\right)\right)$", W - 0.040).body([
        "$r$ samples of $|S_K|$ units drawn at random from the",
        "reference corpus, POOLED over every author",
        "→  each $G_j$ is a MIXTURE near the population centroid,",
        "     a different kind of object from single-author $G_K$",
    ]).foot("measured   $b$ = −0.056 to −0.117 per token", BLUE)

    Col(ax, R + 0.020, cT).head("② THE REFERENCE GRAMMARS   $G_1 … G_r$",
                                ORANGE, "this paper").eq(
        r"$G_j\ \leftarrow\ \mathrm{fit}\left(\mathrm{run}_{|S_K|}"
        r"\left(S_{a_j}\right)\right)$", W - 0.040).body([
        "one grammar per reference author $a_j$, on a CONTIGUOUS",
        "run of $|S_K|$ units — built exactly as $G_K$ is",
        "→  each $G_j$ is a COMPONENT matched to $G_K$ in size and",
        "     in shape: the two sides differ in nothing but the author",
    ]).foot("measured   $b$ = −0.009 to +0.011   (zero within noise)", ORANGE)

    # ---- the hinge ----------------------------------------------------- #
    hY = cT - cH - 0.016
    ax.plot([L, L + FW], [hY, hY], color=RULE, lw=1, zorder=1)
    ax.text(0.5, hY - pts(24),
            r"$\lambda_G\mid H_d\ \sim\ \mathcal{N}\left(bN,\ \sigma^2 N\right)$"
            "          the location is $bN$, the spread $\\sigma\\sqrt{N}$ — "
            "and a correction to one is not a correction to the other",
            fontsize=12.5, color=INK, ha="center", va="center", zorder=3)

    # ---- band D: score to LLR ------------------------------------------ #
    dT, dH = hY - pts(48), 0.1592
    panel(ax, L, dT - dH, W, dH, BLUE)
    panel(ax, R, dT - dH, W, dH, ORANGE)

    Col(ax, L + 0.020, dT).head("③ FROM SCORE TO LLR", BLUE,
                                "Barlow, Nini & Manino 2026").eq(
        r"$\mathrm{LLR}_{\sqrt{\ }}=\dfrac{\lambda_G}{\sqrt{N(Q)}}"
        r"\qquad \mathrm{LLR}_{\mathrm{hapax}}=\lambda_G\cdot"
        r"\dfrac{V_1(Q)}{N(Q)}$", W - 0.040, size=14).body([
        "both corrections are MULTIPLICATIVE: they normalise the",
        "spread $\\sigma\\sqrt{N}$ and inherit the mean untouched",
        "→  $bN/\\sqrt{N}=b\\sqrt{N}$, growing with the evidence length:",
        "     more text, better ranking, worse calibration",
    ]).foot("scale without location", BLUE)

    Col(ax, R + 0.020, dT).head("③ FROM SCORE TO LLR", ORANGE,
                                "this paper").eq(
        r"$b=0$  by construction $\qquad$  "
        r"$t=\dfrac{s_A-\overline{s_j}}{\mathrm{sd}_j(s_j)}$",
        W - 0.040, size=14).body([
        "the location is removed by the pool above; the scale comes",
        "from the case's own cohort of $r$ donor scores",
        "→  nothing is fitted on the case: ONE constant is left to fit,",
        "     the decades of odds per donor sd, and it transports",
    ]).foot("location by construction, scale from the cohort", ORANGE)

    # ---- band E: the by-product, drawn as cause against consequence ----- #
    eT, eH = dT - dH - 0.018, 0.2930
    panel(ax, L, eT - eH, FW, eH, ORANGE, fill="#faeee8")
    ax.text(L + 0.020, eT - pts(20),
            "THE BY-PRODUCT   ·   with $b$ set to zero, a residual reading "
            "warns you the reference population does not suit the case",
            fontsize=12, color=ORANGE, va="top", weight="semibold", zorder=3)

    # The two words the chart cannot avoid are glossed here rather than left to
    # the reader: what ONE POINT is, and what the gauge reading actually
    # measures. Both were in-house shorthand in earlier drafts, which is exactly
    # the kind of thing a figure must not ask its reader to reconstruct.
    ax.text(L + 0.020, eT - pts(40),
            "Each point is one dataset scored against one reference population "
            "— 432 known/questioned pairs, 2,000 tokens a side, POSNoise.",
            fontsize=9.8, color=MUTED, va="top", zorder=3)
    ax.text(L + 0.020, eT - pts(55),
            "The GAUGE READING $b$ is the average per-token score of the "
            "different-author pairs: at $b=0$ the reference population suits "
            "the case, and the further out it reads, the less it does.",
            fontsize=9.8, color=MUTED, va="top", zorder=3)

    groups = meter_readings()
    cal_rows, cal_slope = calibration_readings()
    gy, gh = eT - eH + pts(88), eH - pts(182)
    c_ax = fig.add_axes([L + 0.062, gy, 0.300, gh])
    a_ax = fig.add_axes([L + 0.430, gy, 0.196, gh])
    k_ax = fig.add_axes([L + 0.702, gy, 0.180, gh])

    ctl = np.concatenate([g["b"] for g in groups if g["colour"] is MUTED])
    lim = max(0.02, float(np.abs(ctl).max()) * 1.15)

    for axx in (c_ax, a_ax):
        axx.set_facecolor("#faeee8")
        axx.axvspan(-lim, lim, color="#efe2da", zorder=0)
        axx.set_xlim(-0.055, 0.205)
        axx.set_xticks([-0.05, 0, 0.05, 0.10, 0.15, 0.20])
        axx.tick_params(colors=MUTED, labelsize=9, length=0, pad=4)
        axx.grid(color="#eddfd6", lw=0.8, zorder=0)
        axx.set_axisbelow(True)
        axx.set_xlabel("gauge reading   $b$ per token", fontsize=9.5,
                       color=MUTED, labelpad=3)
        for sp in ("top", "right"):
            axx.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            axx.spines[sp].set_color("#e0cfc6")

    c_ax.set_yscale("log")
    c_ax.axhline(1.0, color=INK, lw=1.2, ls=(0, (5, 3)), zorder=2)
    c_ax.text(0.200, 1.05, "$C_{llr}=1$: the value of saying nothing",
              ha="right", va="bottom", fontsize=9, color=INK, zorder=5)

    for g in groups:
        for axx, key in ((c_ax, "cllr_t"), (a_ax, "auc")):
            axx.scatter(g["b"], g[key], s=46, marker=g["marker"],
                        facecolor=g["colour"], edgecolor=SURFACE,
                        linewidth=1.1, alpha=0.9, zorder=3,
                        label=g["name"] if axx is c_ax else None)
        # Every mismatched point is named -- there are only four of them, and
        # a named point is one the reader can check. Placement is keyed to
        # its RANK in b, not hand-tuned one by one: the three rightmost labels
        # would otherwise run through each other's markers, so the second one
        # is thrown left into the empty upper middle. Verified against the
        # render; if the set of points changes, look at the picture again.
        if g["colour"] is not MUTED:
            PLACE = {0: ("left", 0.005, 1.34), 1: ("right", -0.005, 1.02),
                     2: ("left", 0.005, 1.28), 3: ("left", 0.005, 0.72)}
            for rank, k in enumerate(np.argsort(g["b"])):
                ha, dx, dy = PLACE.get(rank, ("left", 0.005, 1.0))
                c_ax.annotate(g["tags"][k], xy=(g["b"][k], g["cllr_t"][k]),
                              xytext=(g["b"][k] + dx, g["cllr_t"][k] * dy),
                              ha=ha, va="center", fontsize=8.5,
                              color=g["colour"], zorder=5,
                              arrowprops=dict(arrowstyle="-", color=g["colour"],
                                              lw=0.7, alpha=0.6))

    c_ax.set_ylim(0.14, 20)
    c_ax.set_yticks([0.2, 0.5, 1, 2, 5, 10])
    c_ax.set_yticklabels(["0.2", "0.5", "1", "2", "5", "10"])
    c_ax.set_ylabel("$C_{llr}$ of the studentised score   (log)",
                fontsize=9.5, color=MUTED, labelpad=2)
    c_ax.set_title("can you still state odds?", fontsize=10.5, color=INK, pad=7)
    # the legend lives in the AUC panel: its lower-left is the only corner of
    # either chart that no point occupies
    h, lb = c_ax.get_legend_handles_labels()
    leg = a_ax.legend(h, lb, loc="lower left", frameon=False, fontsize=8.5,
                      handletextpad=0.4, borderpad=0.2, labelspacing=0.35)
    for txt in leg.get_texts():
        txt.set_color(INK)

    a_ax.set_ylim(0.862, 1.008)
    a_ax.set_yticks([0.90, 0.95, 1.0])
    a_ax.set_ylabel("AUC", fontsize=9.5, color=MUTED, labelpad=2)
    a_ax.set_title("can you still tell the authors apart?", fontsize=10.5,
                   color=INK, pad=7)

    # ---- the calibration step, drawn as what it moves and what it cannot --- #
    k_ax.set_facecolor("#faeee8")
    k_ax.grid(color="#eddfd6", lw=0.8, axis="y", zorder=0)
    k_ax.set_axisbelow(True)
    k_ax.set_yscale("log")
    k_ax.set_xlim(-0.42, 1.42)
    k_ax.set_ylim(0.14, 20)          # the same scale as c_ax, so the left
    k_ax.set_yticks([0.2, 0.5, 1, 2, 5, 10])   # column of this chart IS that
    k_ax.set_yticklabels(["0.2", "0.5", "1", "2", "5", "10"])
    k_ax.set_xticks([0, 1])
    k_ax.set_xticklabels(["studentised\nscore", "after the\nconstant"])
    k_ax.tick_params(colors=MUTED, labelsize=9, length=0, pad=4)
    for sp in ("top", "right"):
        k_ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        k_ax.spines[sp].set_color("#e0cfc6")
    k_ax.axhline(1.0, color=INK, lw=1.2, ls=(0, (5, 3)), zorder=2)
    for row in sorted(cal_rows, key=lambda r: r["matched"]):
        col = MUTED if row["matched"] else ORANGE
        k_ax.plot([0, 1], [row["before"], row["after"]], color=col, lw=1.2,
                  alpha=0.75, zorder=3, solid_capstyle="round")
        k_ax.scatter([0, 1], [row["before"], row["after"]], s=34, marker="o",
                     facecolor=col, edgecolor=SURFACE, linewidth=1.0,
                     alpha=0.95, zorder=4)
    k_ax.set_ylabel("$C_{llr}$   (log, the scale at left)", fontsize=9.5,
                    color=MUTED, labelpad=2)
    k_ax.set_title("can calibration repair it?", fontsize=10.5, color=INK, pad=7)
    k_ax.text(0.5, 17.5, "one slope and one intercept,\nfitted on the OTHER "
              "datasets\n($%.2f$ decades of odds per donor sd)" % cal_slope,
              ha="center", va="top", fontsize=8.2, color=MUTED, zorder=5)

    ax.text(L + 0.020, eT - eH + pts(36),
            "The gauge separates the two conditions completely — every matched "
            "dataset reads inside the band, every mismatched one outside it — "
            "and the damage falls on the ODDS, not on the ranking.",
            fontsize=10.5, color=INK, va="top", zorder=3)
    ax.text(L + 0.020, eT - eH + pts(17),
            "Calibration converts the units, nothing more: every matched case "
            "goes under $C_{llr}$ 1 (0.17–0.85), three of the four mismatched "
            "stay above. Detectable, not repairable.",
            fontsize=10, color=MUTED, va="top", style="italic", zorder=3)

    fig.savefig(OUT, dpi=170, facecolor=SURFACE)
    n = sum(g["b"].size for g in groups)
    print(f"-> {OUT}   ({n} points plotted; band E bottom y={eT - eH:.3f})")


if __name__ == "__main__":
    main()
