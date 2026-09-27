# The b-offset figure: where the different-author distribution sits, and why
# moving it costs no discrimination.
#
# SCHEMATIC IN SHAPE, MEASURED IN LOCATION. The curves are illustrative
# densities; the only quantities taken from experiment are the per-token
# different-author offsets, which are the point of the figure. Three panels,
# because the offset comes off in two distinct steps and each is a separate
# claim:
#
#   1. POOLED reference grammars, fitted on sentences drawn from across the
#      whole population: b = -0.056 .. -0.117 per token. A mixture model sits
#      nearer a stranger than any individual does, so the denominator wins.
#   2. PER-AUTHOR grammars, size-matched to the candidate's: b = -0.004 ..
#      -0.030. Most of the offset goes, but not all, because the two sides are
#      still unlike in SHAPE -- the candidate's grammar is fitted on one
#      contiguous window, each donor's on sentences scattered over a whole
#      output.
#   3. SIZE- AND SHAPE-MATCHED, each donor fitted on a contiguous run of |S_A|
#      sentences: b = -0.009 .. +0.011, zero within noise. The eight arms, every
#      one negative at step 2, now fall three negative and five positive -- the
#      signature of noise about a true zero rather than a residual bias.
#
# The separation between the hypotheses is deliberately drawn IDENTICAL in all
# three panels and carries no number, because the claim it illustrates is that
# correcting the estimator moves the location without touching the gap. That is
# measured, not assumed: across the eight matched arms the separation moves 0.7
# per cent and AUC by 0.002, in a direction that is a coin flip.
#
#   python journal/figures/make_b_offset_figure.py
#
# Output: journal/figures/b_offset_concept.png

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "b_offset_concept.png"

# Reference categorical theme, slots 1 and 2 (documented as validating on all
# pairs in both modes for the first three slots).
BLUE = "#2a78d6"          # different author
ORANGE = "#eb6834"        # same author
SURFACE = "#fcfcfb"
INK = "#1a1a19"
MUTED = "#6b6b68"
GRID = "#e4e4e1"

GAP = 0.155               # schematic separation, identical in every panel
SD_D, SD_S = 0.030, 0.034

PANELS = [
    (-0.087, "Pooled reference models",
     "reference grammars pooled across the population",
     "measured  $b$ = −0.056 to −0.117 per token"),
    (-0.017, "Per-author, size-matched",
     "one grammar per reference author, matched in size",
     "measured  $b$ = −0.004 to −0.030 per token"),
    (+0.000, "Per-author, size- and shape-matched",
     "each fitted on a contiguous run, as the candidate's is",
     "measured  $b$ = −0.009 to +0.011 per token"),
]


def density(x, mu, sd):
    return np.exp(-0.5 * ((x - mu) / sd) ** 2)


def panel(ax, b, title, sub1, sub2, show_zero_flag):
    x = np.linspace(-0.26, 0.26, 900)
    d = density(x, b, SD_D)
    s = density(x, b + GAP, SD_S)

    ax.axvline(0, color=MUTED, lw=1.1, ls=(0, (4, 3)), zorder=1)

    for vals, colour in ((d, BLUE), (s, ORANGE)):
        ax.fill_between(x, vals, color=colour, alpha=0.16, lw=0, zorder=2)
        ax.plot(x, vals, color=colour, lw=2, zorder=3, solid_capstyle="round")

    # direct labels on the curves, so identity is never colour alone
    ax.text(b, 1.04, "different author", ha="center", va="bottom",
            fontsize=9.5, color=INK)
    ax.text(b + GAP, 1.04, "same author", ha="center", va="bottom",
            fontsize=9.5, color=INK)

    if abs(b) > 0.005:
        ax.annotate("", xy=(b, 0.50), xytext=(0, 0.50),
                    arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.5,
                                    shrinkA=0, shrinkB=0))
        ax.text(b / 2, 0.55, "offset $b$", ha="center", va="bottom",
                fontsize=10.5, color=INK)
    if show_zero_flag:
        ax.text(0, 0.55, "$b = 0$", ha="center", va="bottom",
                fontsize=11, color=INK,
                bbox=dict(fc=SURFACE, ec="none", pad=2))

    # the gap, drawn identically in every panel
    y = 0.085
    ax.annotate("", xy=(b + GAP, y), xytext=(b, y),
                arrowprops=dict(arrowstyle="<|-|>", color=MUTED, lw=1.2,
                                shrinkA=0, shrinkB=0))
    ax.text(b + GAP / 2, y + 0.035, "separation", ha="center", va="bottom",
            fontsize=9, color=MUTED, style="italic",
            bbox=dict(fc=SURFACE, ec="none", pad=1.5))

    ax.set_title(f"{title}\n{sub1}\n{sub2}", fontsize=10.5, color=INK, pad=11,
                 loc="left", linespacing=1.65)
    ax.set_xlim(-0.26, 0.26)
    ax.set_ylim(0, 1.32)
    ax.set_yticks([])
    ax.set_xticks([-0.2, -0.1, 0, 0.1, 0.2])
    ax.tick_params(axis="x", colors=MUTED, labelsize=9, length=0, pad=6)
    ax.grid(axis="x", color=GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    fig, axes = plt.subplots(1, 3, figsize=(15.2, 5.2), sharey=True)
    fig.patch.set_facecolor(SURFACE)
    for ax in axes:
        ax.set_facecolor(SURFACE)

    for i, (ax, (b, t, s1, s2)) in enumerate(zip(axes, PANELS)):
        panel(ax, b, t, s1, s2, show_zero_flag=(i == 2))

    fig.text(0.5, 0.055,
             "λ$_G$ per token.  The gap between the hypotheses is drawn the same in all three panels because "
             "that is what the measurement shows: correcting the\nestimator moves where the different-author "
             "distribution sits, not how far apart the two are — across the eight matched arms the separation "
             "moves 0.7 %\nand AUC 0.002, in a direction that is a coin flip. Curve shapes are schematic; every "
             "offset is measured.",
             ha="center", va="bottom", fontsize=9.5, color=MUTED,
             linespacing=1.55)

    fig.subplots_adjust(left=0.035, right=0.985, top=0.70, bottom=0.235,
                        wspace=0.10)
    fig.savefig(OUT, dpi=200, facecolor=SURFACE)
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
