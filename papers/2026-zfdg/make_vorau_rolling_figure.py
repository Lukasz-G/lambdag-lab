# Figure: the rolling profile across a known author boundary in one codex.
#
# Vorau, Stiftsbibliothek, Cod. 276 carries the Kaiserchronik and, thirty-five
# leaves later, Pfaffe Lambrecht's Alexanderlied. The reference corpus annotates
# both, so the two texts come from one manuscript in one annotation and the
# boundary between the authors is documented. The profile is therefore read
# where the answer is known, which is what makes it usable where it is not.
#
# What the figure has to show, and nothing else:
#   the BAND, the spread of the score over windows that are all by the
#     chronicle's author, which is the fluctuation that means nothing;
#   the STEP at the boundary, in the band's own units;
#   the COHORT, passages by other authors scored against the same known
#     material, which is where a different author is expected to fall -- and
#     the Alexanderlied does not fall there, being much closer to the chronicle
#     than a stranger is. That gap is the finding, not a detail.
#
#   python medieval/make_vorau_rolling_figure.py
#
# Output: medieval/ZfdG_Kaiserchronik/figures/vorau_rolling.png

import json
import sys
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from kc_dates import GREY, INK  # noqa: E402

SCORES = ROOT / "scores" / "vorau_rolling"
OUT = HERE / "ZfdG_Kaiserchronik" / "figures" / "vorau_rolling.png"

CHRON_C, ALEX_C, BOUND_C, COH_C = "#1a4f7a", "#b8860b", "#7d7d7d", "#8c2d2d"


def load(name):
    rows = [json.loads(l) for l in
            (SCORES / name).read_text(encoding="utf-8").splitlines() if l]
    return rows[0], rows[1:]


def band(ch):
    med = float(np.median(ch))
    sd = 1.4826 * float(np.median(np.abs(ch - med))) or float(np.std(ch))
    return med, sd


def panel(ax, meta, rows, letter, title):
    wins = [r for r in rows if r["kind"] == "window"]
    coh = np.array([r["lambda_G"] for r in rows if r["kind"] == "cohort"])
    ch = np.array([r["lambda_G"] for r in wins if r["text"] == "kaiserchronik"])
    al = np.array([r["lambda_G"] for r in wins if r["text"] == "alexanderlied"])
    med, sd = band(ch)
    L, b = meta["L"], meta["boundary"]

    # the band first, so the marks sit on it
    ax.axhspan((med - 3 * sd) / 1000, (med + 3 * sd) / 1000, color=CHRON_C,
               alpha=0.10, lw=0, zorder=0)
    ax.axhline(med / 1000, color=CHRON_C, lw=1.0, ls="--", alpha=0.8, zorder=1)
    ax.axhspan((np.median(coh) - 3 * band(coh)[1]) / 1000,
               (np.median(coh) + 3 * band(coh)[1]) / 1000,
               color=COH_C, alpha=0.08, lw=0, zorder=0)
    ax.axhline(float(np.median(coh)) / 1000, color=COH_C, lw=1.0, ls=":",
               alpha=0.9, zorder=1)

    # the two texts, and the windows that straddle the join between them
    for text, colour, marker, lab in (
            ("kaiserchronik", CHRON_C, "o", "window inside the Kaiserchronik"),
            ("boundary", BOUND_C, "D", "window straddling the join"),
            ("alexanderlied", ALEX_C, "s", "window inside the Alexanderlied")):
        xs = [(r["start"] + L / 2) / 1000 for r in wins if r["text"] == text]
        ys = [r["lambda_G"] / 1000 for r in wins if r["text"] == text]
        ax.plot(xs, ys, marker=marker, ms=4.5, lw=1.0, color=colour,
                alpha=0.9, label=lab, zorder=3)
    ax.axvline(b / 1000, color=INK, lw=1.2, zorder=2)
    ax.set_xlabel("position in the codex stream (thousands of masked tokens)",
                  fontsize=9.5, color=INK)
    ax.set_ylabel("score of the window against the\nknown passages "
                  "(thousands)", fontsize=9.5, color=INK)
    ax.set_title(f"{letter}   {title}", fontsize=11.5, loc="left", color=INK,
                 fontweight="bold", pad=18)
    auc = float(np.mean([(a < c) + 0.5 * (a == c) for c in ch for a in al]))
    ax.text(0, 1.015, f"{len(ch)} chronicle windows, {len(al)} Alexanderlied · "
            f"median {(np.median(al) - med) / sd:.1f} robust standard "
            f"deviations below · AUC {auc:.3f}",
            transform=ax.transAxes, fontsize=8.8, color=GREY, va="bottom")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ax.spines.values():
        s.set_color(GREY)
    ax.tick_params(colors=INK, labelsize=9)
    ax.grid(alpha=0.12, lw=0.6)
    return med, sd, ch, al, coh


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    m2, r2 = load("lemma__L2000__w100__edition.jsonl")
    m1, r1 = load("lemma__L1000__w100__edition.jsonl")

    fig = plt.figure(figsize=(14.0, 8.8))
    gs = GridSpec(2, 2, width_ratios=[4.2, 1.45], hspace=0.46, wspace=0.20,
                  left=0.068, right=0.985, top=0.775, bottom=0.155)
    ax2 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[1, 0], sharex=ax2)
    med2, sd2, ch2, al2, co2 = panel(
        ax2, m2, r2, "a", "windows of 2,000 tokens")
    med1, sd1, ch1, al1, co1 = panel(
        ax1, m1, r1, "b", "windows of 1,000 tokens: the same boundary, "
                          "half the evidence")

    # the three levels side by side, for both window lengths
    axd = fig.add_subplot(gs[:, 1])
    groups = [("the\nchronicle", ch2, ch1, CHRON_C),
              ("the Alexan-\nderlied", al2, al1, ALEX_C),
              ("other\nauthors", co2, co1, COH_C)]
    for i, (lab, a, b_, colour) in enumerate(groups):
        for j, (v, alpha, mk) in enumerate(((a, 0.95, "o"), (b_, 0.45, "s"))):
            x = i + (j - 0.5) * 0.34
            axd.scatter(np.full(len(v), x) + np.random.default_rng(i * 2 + j)
                        .normal(0, 0.045, len(v)), v / 1000, s=11, color=colour,
                        alpha=alpha, marker=mk, linewidths=0, zorder=3)
            axd.plot([x - 0.14, x + 0.14], [np.median(v) / 1000] * 2,
                     color=INK, lw=1.6, zorder=4)
    axd.set_xticks(range(3))
    axd.set_xticklabels([g[0] for g in groups], fontsize=8.8, color=INK)
    axd.set_ylabel("score (thousands)", fontsize=9.5, color=INK)
    axd.set_title("c   every score", fontsize=11.5, loc="left", color=INK,
                  fontweight="bold", pad=30)
    axd.text(0, 1.015, "circles 2,000 tokens,\nsquares 1,000",
             transform=axd.transAxes, fontsize=8.8, color=GREY, va="bottom")
    for s in ("top", "right"):
        axd.spines[s].set_visible(False)
    for s in axd.spines.values():
        s.set_color(GREY)
    axd.tick_params(colors=INK, labelsize=9)
    axd.grid(axis="y", alpha=0.12, lw=0.6)

    h, lab = ax2.get_legend_handles_labels()
    extra = [plt.Line2D([], [], color=CHRON_C, ls="--", lw=1.0),
             plt.Line2D([], [], color=COH_C, ls=":", lw=1.0),
             plt.Line2D([], [], color=INK, lw=1.2)]
    fig.legend(h + extra,
               lab + ["the chronicle's own level, with three robust "
                      "standard deviations",
                      "the level of passages by other authors",
                      "the join between the two texts"],
               loc="upper left", bbox_to_anchor=(0.068, 0.888), ncol=3,
               frameon=False, fontsize=9.2)

    fig.suptitle("One codex, two authors: the rolling score across a boundary "
                 "that is documented",
                 fontsize=14, x=0.068, ha="left", y=0.975, color=INK,
                 fontweight="bold")
    fig.text(0.068, 0.945, textwrap.fill(
        "Vorau, Stiftsbibliothek, Cod. 276: the Kaiserchronik (leaves "
        "1ra–73vb) and Pfaffe Lambrecht's Alexanderlied (109ra–115va), both in "
        "the reference corpus's annotation of this manuscript, read one after "
        "the other. The codex carries other texts between them, which the "
        "corpus does not annotate, so the join in the middle of the figure "
        "joins two texts of one manuscript rather than reading its leaves "
        "continuously.", 150),
        fontsize=9.5, color=INK, ha="left", va="top", linespacing=1.5)
    fig.text(0.068, 0.095, textwrap.fill(
        f"Middle High German verse, POSNoise-masked lemma stream. Each window "
        f"is scored against {len(m2['known_from'])} known passages of "
        f"{m2['L']:,} tokens taken from the edition's text of the same "
        f"manuscript, from stretches the corpus's excerpt does not carry, and "
        f"against {len(m2['donors'])} reference grammars, one per donor text, "
        f"averaged. Scribe, manuscript, period and annotation are therefore "
        f"the same on both sides of the join, and the author is not. The "
        f"Alexanderlied sits far below the chronicle's band and far above the "
        f"level of passages by other authors: a different author in the same "
        f"hand is the hard case, and a cohort of strangers would overstate "
        f"what the method can do.", 152),
        fontsize=9.2, color=GREY, ha="left", va="top", linespacing=1.5)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=190)
    plt.close(fig)
    print(f"  chronicle {med2:.0f} +- {sd2:.0f} (L2000), "
          f"{med1:.0f} +- {sd1:.0f} (L1000)")
    print(f"  wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
