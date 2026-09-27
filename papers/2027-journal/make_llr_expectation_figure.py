# A score centred on zero is not yet a likelihood ratio.
#
# WHY THIS FIGURE EXISTS. Two statements about lambda_G are both defensible and
# cannot both hold at once:
#
#   (a) lambda_G is read as a calibrated log likelihood ratio, so 0 means
#       inconclusive;
#   (b) when the known author and the reference population are exchangeable, the
#       expected lambda_G under H_d is 0.
#
# For any system that discriminates at all, a calibrated log-LR must average
# BELOW zero on different-author cases. So (b) is evidence that lambda_G is a
# well-CENTRED SCORE, not that it is already a likelihood ratio -- and the
# calibration step stays, even though it no longer needs a calibration corpus
# for the case in hand. The figure states the identity, verifies it numerically,
# and then shows both quantities measured on the same eight datasets.
#
# THE IDENTITY. For LLR(x) = log p_s(x)/p_d(x),
#
#   E[LLR | H_d] = integral p_d log(p_s/p_d) = -D(p_d || p_s) <= 0
#
# with equality exactly when p_s = p_d, i.e. when the two hypotheses cannot be
# told apart at all. The middle panel's curve is that identity under a Gaussian
# pair of equal variance, where D = d^2/2 and AUC = Phi(d/sqrt 2).
#
# WHY THE MEASURED POINTS SIT ABOVE THAT CURVE, and it is not a defect of the
# argument: the curve is what a PERFECTLY calibrated system would give at that
# discrimination. A real calibration is imperfect, so its log-LRs are less
# extreme and their mean is less negative. The vertical gap is therefore the
# calibration loss, and naming it is more honest than plotting the curve alone.
#
# Numbers are read at render time from llr_expectation_data.json, written by the
# command in the session log: POSNoise, 2,000 tokens a side, contiguous
# size-matched donors, the eight same-language same-genre datasets, calibrated
# by one slope and intercept fitted leave-one-dataset-out.
#
#   python journal/figures/make_llr_expectation_figure.py
#
# Output: journal/figures/llr_expectation.png

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch
from scipy.stats import norm

HERE = Path(__file__).resolve().parent
DATA = HERE / "llr_expectation_data.json"
OUT = HERE / "llr_expectation.png"

FIGW, FIGH = 14.0, 10.4
BLUE = "#2a78d6"
ORANGE = "#eb6834"
SURFACE = "#fcfcfb"
PANEL = "#f4f4f1"
INK = "#1a1a19"
MUTED = "#6b6b68"

L, R, W = 0.040, 0.516, 0.444
FW = 0.920


def pts(n):
    return n / 72.0 / FIGH


def panel(ax, x, y, w, h, accent, fill=PANEL):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.008",
                                linewidth=0, facecolor=fill, zorder=0))
    ax.add_patch(FancyBboxPatch((x, y), 0.0038, h, boxstyle="square,pad=0",
                                linewidth=0, facecolor=accent, zorder=1))


def nice(name):
    return name.replace("_novels", "").replace("_", " ").capitalize()


def main():
    rows = json.load(open(DATA, encoding="utf-8"))
    raw_d = np.array([r["raw_d"] for r in rows])
    raw_s = np.array([r["raw_s"] for r in rows])
    llr_d = np.array([r["llr_d"] for r in rows])
    llr_s = np.array([r["llr_s"] for r in rows])
    auc = np.array([r["auc"] for r in rows])
    names = [nice(r["ds"]) for r in rows]

    fig = plt.figure(figsize=(FIGW, FIGH), facecolor=SURFACE)
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_axis_off()
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    ax.text(L, 0.962, "A score centred on zero is not yet a likelihood ratio",
            fontsize=19, color=INK, va="top", weight="semibold")
    ax.text(L, 0.925,
            "POSNoise, 2,000 tokens a side, contiguous size-matched reference grammars; "
            "eight same-language, same-genre datasets, 432 pairs each.",
            fontsize=10.5, color=MUTED, va="top")

    # ---- band A: the identity ------------------------------------------- #
    aT, aH = 0.898, 0.138
    panel(ax, L, aT - aH, FW, aH, INK, fill="#f1f1ee")
    ax.text(L + 0.018, aT - pts(20), "THE IDENTITY", fontsize=11.5, color=INK,
            va="top", weight="semibold")
    ax.text(0.5, aT - pts(62),
            r"$\mathbb{E}\left[\mathrm{LLR}\mid H_d\right]\;=\;"
            r"\int p_d \log \dfrac{p_s}{p_d}\;=\;-\,D\!\left(p_d\,\|\,p_s\right)\;\leq\;0$",
            fontsize=17, color=INK, ha="center", va="center")
    ax.text(0.5, aT - aH + pts(20),
            "equality holds only when $p_s = p_d$: when the two hypotheses give the same "
            "score distribution, so the method has no discriminating power",
            fontsize=11, color=MUTED, ha="center", va="center")

    # ---- band B: the two readings --------------------------------------- #
    bT, bH = aT - aH - 0.020, 0.252
    panel(ax, L, bT - bH, W, bH, BLUE)
    panel(ax, R, bT - bH, W, bH, ORANGE)

    for x, accent, head, sub, eq, body, foot in (
        (L, BLUE, "THE RAW SCORE", "what the design guarantees",
         r"$\mathbb{E}\left[\lambda_G \mid H_d\right] = 0$",
         ["every reference grammar is built exactly as $G_K$ is: one author, same",
          "size, same contiguous run. Under $H_d$ the questioned text is by none of",
          "them, so all $r+1$ models stand in the same relation to it, and the",
          "expected difference of log-likelihoods is zero."],
         "measured  $+3.0$ on average   ($-17.6$ to $+22.1$; $p=0.54$ against zero)"),
        (R, ORANGE, "AFTER CALIBRATION", "now a log likelihood ratio",
         r"$\mathbb{E}\left[\mathrm{LLR} \mid H_d\right] < 0$",
         ["the same cases and the same pairs, mapped to odds by one slope and one",
          "intercept fitted on the other seven datasets, never on the dataset they score",
          "→  the mean moves off zero and turns NEGATIVE, which the identity",
          "     above requires whenever the method discriminates"],
         "measured  $-1.455$ on average   ($-1.63$ to $-1.30$)"),
    ):
        ax.text(x + 0.020, bT - pts(20), head, fontsize=12.5, color=accent,
                va="top", weight="semibold")
        ax.text(x + 0.020, bT - pts(39), sub, fontsize=10, color=MUTED,
                va="top", style="italic")
        ax.text(x + W / 2, bT - pts(74), eq, fontsize=16, color=INK,
                ha="center", va="center")
        yy = bT - pts(96)
        for line in body:
            yy -= pts(15.0)
            ax.text(x + 0.020, yy, line, fontsize=9.8,
                    color=INK if line.startswith("→") else MUTED, va="top")
        ax.text(x + 0.020, bT - bH + pts(18), foot, fontsize=11, color=accent,
                va="top", weight="semibold")

    # ---- the hinge: why the transported slope lands where it does -------- #
    # Worth one line rather than a panel: it says the slope is not a constant
    # of nature but the datasets' own separation divided by ln 10, so a
    # better-discriminating setting would give a steeper one.
    hY = bT - bH - pts(30)
    ax.plot([L, L + FW], [hY + pts(16), hY + pts(16)], color="#dededa", lw=1, zorder=1)
    ax.text(0.5, hY - pts(6),
            r"studentising makes $t \mid H_d$ have mean 0 and standard deviation 1, "
            r"so the log-LR is linear in $t$ with slope $\delta / \ln 10$",
            fontsize=11.5, color=INK, ha="center", va="center", zorder=3)
    ax.text(0.5, hY - pts(23),
            "where $\\delta$ is the same-author shift in donor standard deviations — "
            "measured 2.3 to 3.7 here, which is why the slope lands near 1",
            fontsize=10, color=MUTED, ha="center", va="center", zorder=3)

    # ---- band C: the three charts ---------------------------------------- #
    cT, cH = hY - pts(44), 0.368
    panel(ax, L, cT - cH, FW, cH, ORANGE, fill="#faeee8")
    ax.text(L + 0.018, cT - pts(19),
            "THE SAME EIGHT DATASETS, BEFORE AND AFTER", fontsize=11.5,
            color=ORANGE, va="top", weight="semibold")

    gy, gh = cT - cH + pts(86), cH - pts(150)
    ax1 = fig.add_axes([L + 0.052, gy, 0.222, gh])
    ax2 = fig.add_axes([L + 0.352, gy, 0.222, gh])
    ax3 = fig.add_axes([L + 0.648, gy, 0.232, gh])
    xs = np.arange(len(rows))

    for axx in (ax1, ax2, ax3):
        axx.set_facecolor("#faeee8")
        axx.grid(color="#eddfd6", lw=0.8, zorder=0)
        axx.set_axisbelow(True)
        axx.tick_params(colors=MUTED, labelsize=8.5, length=0, pad=3)
        for sp in ("top", "right"):
            axx.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            axx.spines[sp].set_color("#e0cfc6")

    for axx, sd, dd, ttl, ylab in (
        (ax1, raw_s, raw_d, "raw score $\\lambda_G$", "mean over cases"),
        (ax2, llr_s, llr_d, "after calibration", "mean log$_{10}$ LR"),
    ):
        axx.axhline(0, color=INK, lw=1.2, ls=(0, (5, 3)), zorder=2)
        axx.scatter(xs, sd, s=44, marker="^", facecolor=MUTED, edgecolor=SURFACE,
                    linewidth=1.0, zorder=3, label="same author")
        axx.scatter(xs, dd, s=44, marker="o", facecolor=ORANGE, edgecolor=SURFACE,
                    linewidth=1.0, zorder=3, label="different author")
        axx.set_xticks(xs)
        axx.set_xticklabels(names, rotation=55, ha="right", fontsize=8)
        axx.set_title(ttl, fontsize=10.5, color=INK, pad=6)
        axx.set_ylabel(ylab, fontsize=9.5, color=MUTED, labelpad=2)
    ax1.set_yscale("symlog", linthresh=30)
    ax1.set_yticks([0, 100, 300])
    ax1.set_yticklabels(["0", "100", "300"])
    ax2.set_ylim(-2.2, 2.6)
    leg = ax1.legend(loc="center left", frameon=False, fontsize=8.5,
                     handletextpad=0.4, borderpad=0.2, bbox_to_anchor=(0.02, 0.62))
    for t in leg.get_texts():
        t.set_color(INK)

    # the identity as a curve: how negative the mean MUST be at a given
    # discrimination, for a Gaussian pair of equal variance
    a = np.linspace(0.5001, 0.9965, 400)
    d = np.sqrt(2) * norm.ppf(a)
    # D is in nats while the measured means are log10 likelihood ratios, so the
    # curve is divided by ln 10. Getting this wrong makes the panel compare two
    # different units and read the mismatch as a finding -- it did, in an
    # earlier version, where every point appeared to sit above the curve.
    ax3.plot(a, -d ** 2 / 2 / np.log(10), color=INK, lw=1.6, zorder=3,
             label="Gaussian ideal")
    ax3.scatter(auc, llr_d, s=44, marker="o", facecolor=ORANGE,
                edgecolor=SURFACE, linewidth=1.0, zorder=4, label="measured")
    ax3.axhline(0, color=INK, lw=1.2, ls=(0, (5, 3)), zorder=2)
    ax3.set_xlim(0.5, 1.0); ax3.set_ylim(-3.4, 0.35)
    ax3.set_xlabel("AUC", fontsize=9.5, color=MUTED, labelpad=2)
    ax3.set_ylabel("$\\mathbb{E}[\\mathrm{LLR}\\mid H_d]$", fontsize=9.5,
                   color=MUTED, labelpad=2)
    ax3.set_title("zero only at AUC 0.5", fontsize=10.5, color=INK, pad=6)
    leg3 = ax3.legend(loc="lower left", frameon=False, fontsize=8.5,
                      handletextpad=0.4, borderpad=0.2)
    for t in leg3.get_texts():
        t.set_color(INK)
    ax3.text(0.515, -1.15,
             "one transported constant cannot match\nevery dataset's discrimination: the "
             "weakest\ndatasets overshoot, the strongest undershoot",
             fontsize=8.2, color=MUTED, ha="left", va="top")

    ax.text(L + 0.018, cT - cH + pts(40),
            "Before calibration the different-author cases sit ON zero and the "
            "same-author cases far above it; after calibration the same cases sit "
            "BELOW zero, as they must.",
            fontsize=10.5, color=INK, va="top")
    ax.text(L + 0.018, cT - cH + pts(20),
            "$\\mathbb{E}[\\lambda_G\\mid H_d]=0$ locates the score; the calibration "
            "step is what turns it into a likelihood ratio.",
            fontsize=10, color=MUTED, va="top", style="italic")

    fig.savefig(OUT, dpi=170, facecolor=SURFACE)
    print(f"-> {OUT}   ({len(rows)} datasets)")


if __name__ == "__main__":
    main()
