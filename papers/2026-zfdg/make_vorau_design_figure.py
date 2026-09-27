# Figure: what is compared with what, in the Vorau codex test.
#
# The result figure (medieval/make_vorau_rolling_figure.py) shows the profile;
# this one shows the arrangement that produced it, because a reader cannot judge
# a rolling score without knowing which text played which part. Four things have
# to be visible at a glance:
#
#   the MANUSCRIPT and which of its texts the reference corpus annotates,
#   the QUESTIONED stream and the window that rolls along it,
#   the KNOWN passages, and that they are disjoint from what is scored,
#   the REFERENCE texts that supply the denominator, and the further passages
#     that locate a stranger.
#
# The hatching over the known bar is measured, not drawn by hand: it is the
# stretches of the edition's text that the corpus's excerpt also carries, found
# by the same shingle match the reuse work uses, and the known passages are
# taken from what is left.
#
#   python medieval/make_vorau_design_figure.py
#
# Output: medieval/ZfdG_Kaiserchronik/figures/vorau_design.png

import json
import textwrap
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, Rectangle

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(HERE))
import run_vorau_rolling as R  # noqa: E402
import run_witness_cells as W  # noqa: E402
from kc_dates import GREY, INK  # noqa: E402

OUT = HERE / "ZfdG_Kaiserchronik" / "figures" / "vorau_design.png"
SCORES = ROOT / "scores" / "vorau_rolling" / "lemma__L2000__w100__edition.jsonl"
CHRON_C, ALEX_C, COH_C, DON_C = "#1a4f7a", "#b8860b", "#8c2d2d", "#5a7d5a"
PALE = "#d9d9d9"

X0, X1 = 5.0, 70.0          # the schematic's left and right edge
WIDE = X1 - X0


def bar(ax, x, y, w, h, fc, ec=INK, lw=0.8, z=3, **kw):
    ax.add_patch(Rectangle((x, y), w, h, facecolor=fc, edgecolor=ec, lw=lw,
                           zorder=z, **kw))


def label(ax, x, y, s, size=9, colour=INK, ha="left", va="center", **kw):
    ax.text(x, y, s, fontsize=size, color=colour, ha=ha, va=va, zorder=6, **kw)


def arrow(ax, p, q, colour=GREY, lw=1.1, style="-|>", rad=0.0):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=11,
                                 color=colour, lw=lw, zorder=5,
                                 connectionstyle=f"arc3,rad={rad}"))


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    rows = [json.loads(l) for l in
            SCORES.read_text(encoding="utf-8").splitlines() if l]
    meta = rows[0]
    wins = [r for r in rows[1:] if r["kind"] == "window"]
    coh = np.array([r["lambda_G"] for r in rows[1:] if r["kind"] == "cohort"])
    ch = np.array([r["lambda_G"] for r in wins if r["text"] == "kaiserchronik"])
    al = np.array([r["lambda_G"] for r in wins if r["text"] == "alexanderlied"])
    L = meta["L"]

    bank = W.load("lemma")
    chron = R.read_bank("lemma", R.CHRON)
    alex = [t for s in bank[R.ALEX]["sents"] for t in s]
    ed = [t for s in bank[R.EDITION]["sents"] for t in s]
    hit = R.excerpt_mask(ed, chron, 200)
    kstarts = [int(s.split("@")[1]) for s in meta["known_from"]]
    print(f"  known passages at {kstarts}; the excerpt covers {hit.mean():.1%} "
          f"of the edition")

    fig = plt.figure(figsize=(14.2, 8.6))
    ax = fig.add_axes([0.0, 0.075, 0.74, 0.815])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    # ---- the manuscript, by leaves ------------------------------------- #
    label(ax, X0, 95, "the manuscript", size=11, colour=INK,
          fontweight="bold")
    label(ax, X0, 91.2, "Vorau, Stiftsbibliothek, Cod. 276 — one codex, "
                        "one scribe's work as far as the leaves show",
          size=9, colour=GREY)
    leaves = [(1, 73, "Kaiserchronik\nBl. 1ra–73vb", CHRON_C),
              (74, 108, "eleven further texts,\nnot in the corpus", PALE),
              (109, 115, "Alexanderlied\n109ra–115va", ALEX_C),
              (116, 135, "four further texts,\nnot in the corpus", PALE)]
    for i, (a, b, name, c) in enumerate(leaves):
        x = X0 + WIDE * (a - 1) / 135
        w = WIDE * (b - a + 1) / 135
        bar(ax, x, 82, w, 5.4, c)
        # the Alexanderlied is six leaves of a hundred and thirty-five, so the
        # labels alternate in height and reach their segment by a leader line
        y = 78.6 if i % 2 == 0 else 73.4
        ax.plot([x + w / 2, x + w / 2], [81.8, y + 0.4], color=GREY, lw=0.6,
                zorder=1)
        label(ax, x + w / 2, y, name, size=8.2,
              colour=INK if c is PALE else c, ha="center", va="top",
              linespacing=1.4)

    # ---- the questioned stream ----------------------------------------- #
    label(ax, X0, 68.5, "the questioned stream", size=11, fontweight="bold")
    label(ax, X0, 64.8, "the two annotated texts, read one after the other; "
                        "a window of 2,000 tokens rolls along it in steps of "
                        "250", size=9, colour=GREY)
    tot = len(chron) + len(alex)
    wc = WIDE * len(chron) / tot
    bar(ax, X0, 54, wc, 5.4, CHRON_C, z=3)
    bar(ax, X0 + wc, 54, WIDE - wc, 5.4, ALEX_C, z=3)
    label(ax, X0 + wc / 2, 56.7, f"{len(chron):,} tokens", size=8.4,
          colour="white", ha="center", fontweight="bold")
    label(ax, X0 + wc + (WIDE - wc) / 2, 56.7, f"{len(alex):,}", size=8.4,
          colour="white", ha="center", fontweight="bold")
    # the two texts come down from the manuscript above
    for a, b, xa, xb in ((1, 73, X0, X0 + wc),
                         (109, 115, X0 + wc, X1)):
        x = X0 + WIDE * (a - 1) / 135
        w = WIDE * (b - a + 1) / 135
        for p, q in ((x, xa), (x + w, xb)):
            ax.plot([p, q], [82, 59.4], color=GREY, lw=0.5, ls=":",
                    alpha=0.5, zorder=0)
    ww = WIDE * L / tot
    for i, off in enumerate((0.10, 0.28, 0.46)):
        bar(ax, X0 + WIDE * off, 52.6, ww, 8.2, "none", ec=GREY, lw=0.9, z=4,
            ls=(0, (3, 2)))
    bar(ax, X0 + WIDE * 0.64, 52.6, ww, 8.2, "none", ec=INK, lw=1.6, z=5)
    label(ax, X0 + WIDE * 0.64 + ww / 2, 62.2, "one window", size=8.4,
          ha="center", colour=INK, fontweight="bold")
    ax.annotate("", xy=(X0 + WIDE * 0.62, 51.4), xytext=(X0 + WIDE * 0.06, 51.4),
                arrowprops=dict(arrowstyle="<|-", color=GREY, lw=1.0))
    label(ax, X0 + WIDE * 0.22, 49.6, "the window rolls this way", size=8.2,
          colour=GREY)

    # ---- the known passages -------------------------------------------- #
    label(ax, X0, 42.5, "the known passages", size=11, fontweight="bold")
    label(ax, X0, 38.8, "four passages of 2,000 tokens of the Kaiserchronik, "
                        "from the edition's text of this same manuscript",
          size=9, colour=GREY)
    bar(ax, X0, 28, WIDE, 5.4, "white", ec=GREY, lw=0.8, z=2)
    # the stretches the corpus's excerpt also carries, measured
    nb = 400
    e = np.linspace(0, len(ed), nb + 1).astype(int)
    for i in range(nb):
        if hit[e[i]:e[i + 1]].mean() > 0.5:
            bar(ax, X0 + WIDE * i / nb, 28, WIDE / nb + 0.02, 5.4, PALE,
                ec="none", z=2)
    for s in kstarts:
        bar(ax, X0 + WIDE * s / len(ed), 28, max(WIDE * L / len(ed), 0.7), 5.4,
            CHRON_C, ec=INK, lw=0.6, z=4)
    label(ax, X1, 35.0, f"{len(ed):,} tokens", size=8.2, colour=GREY,
          ha="right")
    label(ax, X0, 25.2, "grey: the stretches the corpus's excerpt also carries, "
                        "measured and kept clear, so nothing scored is also "
                        "known", size=8.2, colour=GREY)

    # ---- the reference and the strangers -------------------------------- #
    label(ax, X0, 18.5, "the reference", size=11, fontweight="bold")
    label(ax, X0, 14.8, "fifteen texts by other authors, 2,000 tokens each, "
                        "one grammar per text", size=9, colour=GREY)
    for i in range(15):
        bar(ax, X0 + i * 2.4, 7.0, 1.9, 4.2, DON_C, ec="none", z=3)
    label(ax, X0 + 15 * 2.4 + 1.5, 9.8, "and forty further passages by other "
                                        "authors,\nscored the same way, to "
                                        "locate a stranger", size=8.6,
          colour=COH_C, linespacing=1.4)
    for i in range(40):
        bar(ax, X0 + 15 * 2.4 + 1.5 + (i % 20) * 0.62, 5.0 - (i // 20) * 1.2,
            0.45, 0.9, COH_C, ec="none", z=3)

    # ---- what the comparison is ----------------------------------------- #
    bx, by, bw, bh = 74.5, 42.0, 24.0, 26.0
    bar(ax, bx, by, bw, bh, "#f4f6f8", ec=GREY, lw=0.9, z=2)
    label(ax, bx + 1.2, by + bh - 3.0, "the score of one window",
          size=10, fontweight="bold")
    label(ax, bx + 1.2, by + bh - 11.5,
          "how much better the\nKaiserchronik's own grammar\npredicts that "
          "window than the\nfifteen unrelated grammars do,\naveraged over "
          "them", size=9, colour=INK, va="center", linespacing=1.6)
    label(ax, bx + 1.2, by + 4.2,
          "one number per window,\non the same scale throughout",
          size=8.4, colour=GREY, va="center", linespacing=1.5)
    arrow(ax, (X0 + WIDE * 0.64 + ww, 58), (bx, by + bh - 4), rad=-0.18)
    arrow(ax, (X1 + 1.0, 31.0), (bx, by + 13), rad=-0.15)
    arrow(ax, (X0 + 15 * 2.4, 9.0), (bx, by + 4), rad=-0.30)

    # ---- what is read off it -------------------------------------------- #
    axr = fig.add_axes([0.80, 0.115, 0.135, 0.52])
    med = float(np.median(ch))
    sd = 1.4826 * float(np.median(np.abs(ch - med)))
    axr.axhspan((med - 3 * sd) / 1000, (med + 3 * sd) / 1000, color=CHRON_C,
                alpha=0.12, lw=0)
    for v, c, m, lab in ((ch, CHRON_C, "o", "the chronicle's\nown windows"),
                         (al, ALEX_C, "s", "the Alexanderlied's\nwindows"),
                         (coh, COH_C, "^", "passages by\nother authors")):
        x = {CHRON_C: 0, ALEX_C: 1, COH_C: 2}[c]
        axr.scatter(np.full(len(v), x) + np.random.default_rng(x)
                    .normal(0, 0.06, len(v)), v / 1000, s=10, color=c,
                    marker=m, linewidths=0, alpha=0.85, zorder=3)
        axr.plot([x - 0.25, x + 0.25], [np.median(v) / 1000] * 2, color=INK,
                 lw=1.6, zorder=4)
    axr.set_xticks([0, 1, 2])
    axr.set_xticklabels(["the\nchronicle", "the Alexan-\nderlied",
                         "other\nauthors"], fontsize=8.4, color=INK)
    axr.set_ylabel("score (thousands)", fontsize=9, color=INK)
    axr.set_title("what is read off it", fontsize=10.5, loc="left", color=INK,
                  fontweight="bold", pad=34)
    axr.text(0, 1.015, "the band, the step,\nand where a stranger falls",
             transform=axr.transAxes, fontsize=8.4, color=GREY, va="bottom")
    for s in ("top", "right"):
        axr.spines[s].set_visible(False)
    for s in axr.spines.values():
        s.set_color(GREY)
    axr.tick_params(colors=INK, labelsize=8.4)
    axr.grid(axis="y", alpha=0.12, lw=0.6)

    fig.suptitle("What is compared with what: the Vorau codex test",
                 fontsize=14, x=0.037, ha="left", y=0.972, color=INK,
                 fontweight="bold")
    fig.text(0.037, 0.022, textwrap.fill(
        "Middle High German verse, POSNoise-masked lemma stream, units of 100 "
        "tokens. Known and questioned never share a token: the questioned "
        "stream is the reference corpus's annotation of the codex, the known "
        "passages come from the edition's text of the same manuscript, and the "
        "stretches the two have in common are measured and kept clear.", 152),
        fontsize=9, color=GREY, ha="left", va="bottom", linespacing=1.5)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=190)
    plt.close(fig)
    print(f"  wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
