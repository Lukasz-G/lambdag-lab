# Why b is negative: a sharp model on the numerator, a broad one on the
# denominator.
#
# WHY NOT THE OBVIOUS PICTURE. The first version of this figure drew grammars as
# points and divergence as distance, with the reference models at the centroid.
# It fails on its own terms twice over. The claim is about the AVERAGE
# individual, and many single grammars sit nearer an outsider than the centroid
# does, so any one highlighted individual can argue the opposite of the text.
# Worse, once the questioned text is far from the population -- which is the
# case of interest -- the centroid's advantage is second order in the spread and
# vanishes on the page: measured 4.84 against 4.79. The mechanism is not
# proximity. It is COVERAGE.
#
# WHAT IS DRAWN. The estimator sets log P(Q | G_A), fitted on ONE author's
# sentences, against the mean over reference grammars each fitted on sentences
# POOLED across the population. A pooled model is close to the average of the
# individual grammars, and the log of an average is never below the average of
# the logs. That is Jensen's inequality, and b is its size. Under H_d the
# candidate's grammar is sharp in the wrong place and has only its backoff mass
# left for the questioned text, while the pooled model covers everyone
# moderately -- so the pooled model wins, at almost every token.
#
# The individual grammars are drawn with the backoff floor Kneser-Ney gives
# them. Without it a stranger's text would be flatly impossible rather than
# merely unlikely, and the figure would overstate its own case.
#
# SCHEMATIC IN SHAPE, EXACT IN SIGN. The horizontal axis is an abstract arrangement
# of grammatical constructions and the densities are illustrative, so no number
# is printed on either axis. What is not illustrative is the direction: for the
# population drawn here, every one of the 42 wrong-author pairings gives b < 0,
# mean -0.66 per token. The check is an assertion below, not a hope.
#
#   python journal/figures/make_b_mechanism_figure.py
#
# Output: journal/figures/b_mechanism.png

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "b_mechanism.png"

BLUE = "#2a78d6"          # the pooled reference models (denominator)
ORANGE = "#eb6834"        # the candidate's own grammar (numerator)
SURFACE = "#fcfcfb"
INK = "#1a1a19"
MUTED = "#6b6b68"
FAINT = "#c3c3be"

MUS = np.array([1.1, 2.6, 3.9, 5.2, 6.4, 7.8, 9.1])
SD = 0.52
FLOOR = 0.16              # backoff mass, i.e. what smoothing leaves for the unseen
LO, HI = 0.0, 10.2
CANDIDATE = 1             # index into MUS: the wrong author under H_d
X0 = 7.55                 # a construction of the questioned text


def dens(x, mu):
    g = np.exp(-0.5 * ((x - mu) / SD) ** 2) / (SD * np.sqrt(2 * np.pi))
    return (1 - FLOOR) * g + FLOOR / (HI - LO)


def check_sign(x, mix):
    """Every wrong-author pairing must give b < 0, or the figure is a lie."""
    rng = np.random.default_rng(3)
    bs = []
    for i, mu_i in enumerate(MUS):
        xs = rng.normal(mu_i, SD, 40000)
        xs = xs[(xs > LO) & (xs < HI)]
        for j, mu_j in enumerate(MUS):
            if i == j:
                continue
            bs.append(np.mean(np.log10(dens(xs, mu_j))
                              - np.log10(np.interp(xs, x, mix))))
    bs = np.array(bs)
    assert (bs < 0).all(), "the figure claims a sign the population does not have"
    return bs.mean()


def main():
    sys.stdout.reconfigure(encoding="utf-8")

    x = np.linspace(LO, HI, 3000)
    ind = np.array([dens(x, m) for m in MUS])
    mix = ind.mean(axis=0)
    b_mean = check_sign(x, mix)

    y_cand = dens(X0, MUS[CANDIDATE])
    y_pool = np.interp(X0, x, mix)

    fig, ax = plt.subplots(figsize=(11.2, 6.2))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    ax.set_yscale("log")

    # stops below the label, so the label's leader continues the same line
    ax.plot([X0, X0], [0.0034, 1.20], color=MUTED, lw=1, ls=(0, (3, 4)),
            zorder=2)

    for k, row in enumerate(ind):
        if k == CANDIDATE:
            continue
        ax.plot(x, row, color=FAINT, lw=1.3, zorder=3)
    ax.plot(x, ind[CANDIDATE], color=ORANGE, lw=2.2, zorder=5,
            solid_capstyle="round")
    ax.plot(x, mix, color=BLUE, lw=2.8, zorder=4, solid_capstyle="round")

    # the two numbers the estimator actually differences, at one token
    ax.scatter([X0, X0], [y_cand, y_pool], s=[110, 110],
               facecolor=[ORANGE, BLUE], edgecolor=SURFACE, linewidth=2,
               zorder=7)
    ax.annotate("", xy=(X0, y_pool), xytext=(X0, y_cand),
                arrowprops=dict(arrowstyle="<|-|>", color=INK, lw=1.6,
                                shrinkA=4, shrinkB=4), zorder=6)
    ax.annotate("the gap this token\ncontributes to λ$_G$",
                xy=(X0, np.sqrt(y_cand * y_pool)), xytext=(6.30, 0.030),
                ha="center", fontsize=10.5, color=INK, linespacing=1.45,
                zorder=8,
                bbox=dict(fc=SURFACE, ec="none", pad=2),
                arrowprops=dict(arrowstyle="-", color=INK, lw=1))

    ax.annotate("the candidate's own grammar:\nsharp, and in the wrong place",
                xy=(MUS[CANDIDATE], dens(MUS[CANDIDATE], MUS[CANDIDATE])),
                xytext=(MUS[CANDIDATE] - 0.15, 2.45), ha="center",
                fontsize=11, color=ORANGE, linespacing=1.45, zorder=8,
                arrowprops=dict(arrowstyle="-", color=ORANGE, lw=1.2))
    ax.annotate("a reference model, fitted on pooled sentences:\n"
                "covers everybody moderately, nobody sharply",
                xy=(3.25, np.interp(3.25, x, mix)), xytext=(3.05, 0.0060),
                ha="center", fontsize=11, color=BLUE, linespacing=1.45,
                zorder=8, bbox=dict(fc=SURFACE, ec="none", pad=2),
                arrowprops=dict(arrowstyle="-", color=BLUE, lw=1.2))
    ax.annotate("the other individual grammars",
                xy=(5.2, dens(5.2, 5.2)), xytext=(5.2, 1.45), ha="center",
                fontsize=10.5, color=MUTED, style="italic", zorder=8,
                bbox=dict(fc=SURFACE, ec="none", pad=2),
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=1))
    ax.annotate("a construction in\nthe questioned text",
                xy=(X0, 1.15), xytext=(X0, 2.45), ha="center",
                fontsize=10.5, color=INK, linespacing=1.45, zorder=8,
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=1))

    ax.set_xlabel("grammatical constructions, schematically arranged",
                  fontsize=10, color=MUTED, labelpad=6)
    ax.set_ylabel("probability the model assigns  (log scale)",
                  fontsize=10, color=MUTED, labelpad=8)
    ax.set_xlim(LO, HI)
    ax.set_ylim(0.0034, 4.6)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.minorticks_off()
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#e4e4e1")

    fig.text(0.5, 0.205,
             "Under $H_d$ the numerator is one author's grammar; the denominator "
             "is an average over the population.\nA pooled model is close to the "
             "average of the individual grammars, and the log of an average is "
             "never\nbelow the average of the logs. So the denominator wins at "
             "almost every token, and $b$ is the size of that inequality.",
             ha="center", va="top", fontsize=10.5, color=INK, linespacing=1.6)
    fig.text(0.5, 0.030,
             "Curves are schematic and the axes carry no units; the sign is not. "
             f"All 42 wrong-author pairings here give $b$ < 0, mean {b_mean:.2f} per token.",
             ha="center", va="top", fontsize=9, color=MUTED)

    fig.subplots_adjust(left=0.062, right=0.978, top=0.965, bottom=0.285)
    fig.savefig(OUT, dpi=200, facecolor=SURFACE)
    print(f"-> {OUT}   mean b = {b_mean:.3f} per token")


if __name__ == "__main__":
    main()
