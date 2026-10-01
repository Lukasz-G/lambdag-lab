# Figure: what a copyist leaves alone, and what it is worth for authorship.
#
# The tradition's argument for reading the rhyme is that a scribe alters
# inconspicuous words as he goes but cannot touch the word the verse rhymes on.
# Measured on works of named authors, the argument holds only in a weaker and more
# interesting form: what the copyist preserves is not the rhyme WORD but the
# rhyme's sound, and a profile of the rime -- the syllable from the last stressed
# vowel, with spelling folded together -- discriminates authors better than either
# the whole line or the word at the rhyme.
#
# Every pair here crosses a work, so the comparison is the one an attribution
# needs and not the easier one of two copies of the same poem.
#
#   python medieval/make_rhyme_figure.py
#
# Output: medieval/ZfdG_Kaiserchronik/figures/rhyme_rime.png

import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from kc_dates import GREY, INK  # noqa: E402

SRC = HERE / "rhyme_mfw_authors.tsv"
OUT = HERE / "ZfdG_Kaiserchronik" / "figures" / "rhyme_rime.png"
SURF_C, LEM_C = "#b8860b", "#1a4f7a"
SETS = [("all", "the whole line"), ("rhyme", "the word at the rhyme"),
        ("rime", "the rhyme's sound\n(from the last stressed vowel)")]


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    rows = [r for r in csv.DictReader(SRC.open(encoding="utf-8"), delimiter="\t")
            if r["top"] == "200" and r["verses"] == "2000"]
    by = {(r["stream"], r["positions"]): r for r in rows}
    x = np.arange(len(SETS))
    w = 0.36
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    for i, (stream, colour, label) in enumerate(
            (("surface_full", SURF_C, "the scribe's own spelling"),
             ("lemma_full", LEM_C, "the canonical lemma"))):
        ys, es, xs = [], [], []
        for j, (pos, _) in enumerate(SETS):
            r = by.get((stream, pos))
            if r is None:
                continue
            ys.append(float(r["auc"]))
            es.append(float(r["auc_sd"]))
            xs.append(j + (i - 0.5) * w)
        ax.bar(xs, ys, w, yerr=es, capsize=3, color=colour, label=label,
               zorder=3, error_kw=dict(lw=0.9, ecolor=INK))
        for xx, yy in zip(xs, ys):
            ax.text(xx, yy + 0.012, f"{yy:.3f}", ha="center", fontsize=8.4,
                    color=INK)
    ax.axhline(0.5, color=GREY, lw=0.9, ls=":", zorder=2)
    ax.text(len(SETS) - 0.55, 0.507, "chance", fontsize=8.4, color=GREY)
    n = by[("surface_full", "all")]
    ax.set_xticks(x)
    ax.set_xticklabels([lab for _, lab in SETS], fontsize=9.2, color=INK)
    ax.set_ylabel("works of one author told from works of others\n"
                  "(area under the curve)", fontsize=9.5, color=INK)
    ax.set_title("What survives copying, and what identifies the poet",
                 fontsize=11.5, loc="left", color=INK, fontweight="bold",
                 pad=22)
    ax.text(0, 1.02, f"normalised editions, {n['n_identities']} named authors, "
                     f"{n['n_same_pairs']} pairs, every pair crossing a work · "
                     f"blocks of {int(n['verses']):,} verses · profiles over the "
                     f"200 commonest forms, compared by Burrows's delta",
            transform=ax.transAxes, fontsize=8.8, color=GREY, va="bottom")
    ax.set_ylim(0.45, 0.97)
    ax.legend(fontsize=9, frameon=False, loc="upper left")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ax.spines.values():
        s.set_color(GREY)
    ax.tick_params(colors=INK, labelsize=9)
    ax.grid(axis="y", alpha=0.12, lw=0.6)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"wrote {OUT.relative_to(ROOT)}")
    for (stream, pos), r in sorted(by.items()):
        print(f"  {stream:14} {pos:6} AUC {float(r['auc']):.3f} "
              f"+- {float(r['auc_sd']):.3f}")


if __name__ == "__main__":
    main()
