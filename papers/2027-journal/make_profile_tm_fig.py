# Schematic of the per-author profile Tsetlin machine, contrasted with the
# pairwise-product architecture it outperforms cross-genre.
#
#   python experiments/make_profile_tm_fig.py
# -> journal/figures/fig_profile_tm.png

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "journal" / "figures"

AUTHOR = "#1F4E9C"     # the candidate author's own text
POP = "#8C8C8C"        # the population it is contrasted against
QUEST = "#B85C00"      # the questioned document
INK = "#1A1A1A"
MUTED = "#585858"
HAIR = "#C9C9C9"
PANEL = "#F5F5F3"

AUTHORS = ["Keller", "Meyer", "Tieck", "Heine"]
# illustrative margins, not measured values
MARGINS = [18.0, -4.0, 2.0, -7.0]


def box(ax, x, y, w, h, fc, ec, lw=1.0, r=0.02, z=2, alpha=1.0):
    p = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                       facecolor=fc, edgecolor=ec, linewidth=lw, zorder=z,
                       alpha=alpha)
    ax.add_patch(p)
    return p


def arrow(ax, x1, y1, x2, y2, color=MUTED, lw=1.3, z=3, style="-|>"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style,
                                 mutation_scale=11, color=color, lw=lw,
                                 zorder=z, shrinkA=2, shrinkB=2))


def rect(ax, x, y, w, h, fc, z=4):
    """Plain rectangle. FancyBboxPatch is not used for small marks: its
    rounding radius is absolute, so on a 0.02-wide box it rounds the corners
    all the way and the block renders as an ellipse."""
    ax.add_patch(Rectangle((x, y), w, h, facecolor=fc, edgecolor="none",
                           zorder=z))


def strip(ax, x, y, n, fc, w=0.016, h=0.03, gap=0.007):
    """A row of little text-window blocks."""
    for i in range(n):
        rect(ax, x + i * (w + gap), y, w, h, fc)
    return x + n * (w + gap) - gap


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(13.2, 8.6), dpi=200)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    fig.patch.set_facecolor("white")

    fig.text(0.035, 0.955, "The per-author profile Tsetlin machine",
             fontsize=19, color=INK, fontweight="bold")
    fig.text(0.035, 0.923,
             "One machine per candidate, each asking “what does this author "
             "look like?” — not “do these two texts agree?”",
             fontsize=11.5, color=MUTED)

    # ---------------------------------------------------------------- panel 1
    box(ax, 0.035, 0.545, 0.43, 0.335, PANEL, HAIR, 0.9, r=0.012, z=1)
    fig.text(0.052, 0.845, "1  TRAINING — one machine per author",
             fontsize=12, color=INK, fontweight="bold")
    fig.text(0.052, 0.818,
             "each author’s own windows against the population’s",
             fontsize=9.8, color=MUTED)

    y = 0.755
    for i, a in enumerate(AUTHORS[:3]):
        fig.text(0.052, y + 0.009, a, fontsize=9.5, color=INK, ha="left")
        xe = strip(ax, 0.108, y, 4, AUTHOR)
        fig.text(xe + 0.011, y + 0.009, "vs", fontsize=8.5, color=MUTED)
        strip(ax, xe + 0.032, y, 6, POP)
        arrow(ax, 0.352, y + 0.019, 0.379, y + 0.019)
        box(ax, 0.381, y - 0.004, 0.062, 0.046, "white", AUTHOR, 1.2, r=0.008)
        fig.text(0.412, y + 0.012, f"TM$_{{{a[0]}}}$", fontsize=9.5,
                 color=AUTHOR, ha="center", fontweight="bold")
        y -= 0.068

    fig.text(0.108, 0.565, "■", fontsize=9, color=AUTHOR)
    fig.text(0.121, 0.566, "this author’s windows", fontsize=9, color=MUTED)
    fig.text(0.253, 0.565, "■", fontsize=9, color=POP)
    fig.text(0.266, 0.566, "population windows", fontsize=9, color=MUTED)

    # ---------------------------------------------------------------- panel 2
    box(ax, 0.5, 0.545, 0.465, 0.335, PANEL, HAIR, 0.9, r=0.012, z=1)
    fig.text(0.517, 0.845, "2  WHAT IT LEARNS — legible clauses",
             fontsize=12, color=INK, fontweight="bold")
    fig.text(0.517, 0.818,
             "conjunctions over named grammatical events, readable aloud",
             fontsize=9.8, color=MUTED)

    box(ax, 0.517, 0.632, 0.431, 0.168, "white", HAIR, 0.9, r=0.008)
    fig.text(0.532, 0.762, "clause 7   vote FOR", fontsize=9, color=AUTHOR,
             fontweight="bold")
    fig.text(0.532, 0.733,
             "IF  rate⟨AUX1 DET2⟩ HIGH   AND   rate⟨ADP4⟩ LOW",
             fontsize=9.6, color=INK, family="DejaVu Sans Mono")
    fig.text(0.532, 0.706,
             "    AND  rate⟨PRON1 ADV2⟩ HIGH",
             fontsize=9.6, color=INK, family="DejaVu Sans Mono")
    fig.text(0.532, 0.668, "clause 12  vote AGAINST", fontsize=9, color="#7A1F1F",
             fontweight="bold")
    fig.text(0.532, 0.645,
             "IF  rate⟨SCONJ1 DET1⟩ HIGH",
             fontsize=9.6, color=INK, family="DejaVu Sans Mono")

    fig.text(0.517, 0.594,
             "An author has signature combinations — leans on this "
             "construction",
             fontsize=9.3, color=MUTED)
    fig.text(0.517, 0.570,
             "and avoids that one — and a combination is exactly what a "
             "clause is.",
             fontsize=9.3, color=MUTED)

    # ---------------------------------------------------------------- panel 3
    box(ax, 0.035, 0.075, 0.93, 0.42, PANEL, HAIR, 0.9, r=0.012, z=1)
    fig.text(0.052, 0.455, "3  VERIFYING A QUESTIONED TEXT",
             fontsize=12, color=INK, fontweight="bold")
    fig.text(0.052, 0.428,
             "every machine reads the questioned document; the candidate is "
             "judged against the rest",
             fontsize=9.8, color=MUTED)

    box(ax, 0.052, 0.255, 0.105, 0.105, "white", QUEST, 1.4, r=0.008)
    fig.text(0.1045, 0.325, "questioned", fontsize=9.5, color=QUEST,
             ha="center", fontweight="bold")
    fig.text(0.1045, 0.298, "document", fontsize=9.5, color=QUEST, ha="center",
             fontweight="bold")
    fig.text(0.1045, 0.271, "(a poem)", fontsize=8.8, color=MUTED, ha="center")

    ys = [0.357, 0.297, 0.237, 0.177]
    for i, (a, m) in enumerate(zip(AUTHORS, MARGINS)):
        cand = i == 0
        col = AUTHOR if cand else POP
        arrow(ax, 0.16, 0.307, 0.243, ys[i] + 0.018, color=col if cand else HAIR,
              lw=1.5 if cand else 1.0)
        box(ax, 0.245, ys[i], 0.075, 0.037, "white", col, 1.4 if cand else 0.9,
            r=0.007)
        fig.text(0.2825, ys[i] + 0.011, f"TM$_{{{a[0]}}}$", fontsize=9.5,
                 color=col, ha="center", fontweight="bold" if cand else "normal")
        fig.text(0.331, ys[i] + 0.011, a, fontsize=9, color=INK if cand else MUTED,
                 fontweight="bold" if cand else "normal")

        # margin bar
        x0 = 0.46
        scale = 0.0032
        w = abs(m) * scale
        xb = x0 if m >= 0 else x0 - w
        rect(ax, xb, ys[i] + 0.008, w, 0.021, col)
        fig.text(x0 + (w + 0.008 if m >= 0 else -w - 0.008), ys[i] + 0.011,
                 f"{m:+.0f}", fontsize=9, color=col, ha="left" if m >= 0 else "right",
                 va="center", fontweight="bold" if cand else "normal")
    ax.plot([0.46, 0.46], [0.170, 0.390], color=HAIR, lw=0.9, zorder=3)
    fig.text(0.46, 0.148, "0", fontsize=8.5, color=MUTED, ha="center")
    fig.text(0.46, 0.400, "vote margin", fontsize=9, color=MUTED, ha="center")
    fig.text(0.46, 0.122, "illustrative values", fontsize=8.3, color=MUTED,
             ha="center", style="italic")

    arrow(ax, 0.60, 0.29, 0.655, 0.29, color=MUTED, lw=1.4)
    box(ax, 0.658, 0.205, 0.29, 0.175, "white", AUTHOR, 1.3, r=0.008)
    fig.text(0.803, 0.345, "cohort contrast", fontsize=10.5, color=AUTHOR,
             ha="center", fontweight="bold")
    fig.text(0.803, 0.297,
             r"$t=\dfrac{m_{\mathrm{candidate}}-\mathrm{mean}(m_{\mathrm{others}})}"
             r"{\mathrm{sd}(m_{\mathrm{others}})}$",
             fontsize=13, color=INK, ha="center")
    fig.text(0.803, 0.232,
             "the candidate’s standing among his peers,\nnot his raw score",
             fontsize=8.8, color=MUTED, ha="center", linespacing=1.5)

    fig.text(0.803, 0.168,
             "cross-genre AUC 0.70–0.80, against ~0.50 for the pairwise route",
             fontsize=9.2, color=INK, ha="center", fontweight="bold")

    p = OUT / "fig_profile_tm.png"
    fig.savefig(p, dpi=200, facecolor="white")
    print(f"wrote {p}")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()
