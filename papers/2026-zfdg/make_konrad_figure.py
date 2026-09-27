# Figure: the Kaiserchronik read against the Rolandslied.
#
# The question is whether any stretch of the chronicle is Pfaffe Konrad's, and
# the figure has to let a reader answer it without taking the author's word for
# the scale. So the profile is drawn against three levels measured on the same
# known passages: the Rolandslied's other witnesses at the top, which is what
# the same author in another hand looks like; passages by other authors at the
# bottom; and the Alexanderlied, a named different author of the same milieu,
# between them.
#
# Both runs are drawn, with the shared wording left in and taken out, because
# for this work the shared wording IS the question: borrowing and common
# authorship produce it alike.
#
#   python medieval/make_konrad_figure.py
#
# Output: medieval/ZfdG_Kaiserchronik/figures/konrad_profile.png

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

SCORES = ROOT / "scores" / "konrad_profile"
OUT = HERE / "ZfdG_Kaiserchronik" / "figures" / "konrad_profile.png"
WIN_C, CEIL_C, ALEX_C, COH_C = "#1a4f7a", "#2e7d32", "#b8860b", "#8c2d2d"


def load(mask):
    p = SCORES / f"lemma__L2000__w100__A1__{mask}.jsonl"
    rows = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines()
            if l]
    meta = rows[0]
    g = {k: np.array([r["lambda_G"] for r in rows[1:] if r["kind"] == k])
         for k in ("window", "ceiling", "alexanderlied", "cohort")}
    xs = np.array([r["start"] for r in rows[1:] if r["kind"] == "window"])
    return meta, g, xs


def panel(ax, meta, g, xs, letter, title):
    L = meta["L"]
    coh, win = g["cohort"], g["window"]
    ax.axhspan(np.percentile(coh, 5) / 1000, np.percentile(coh, 95) / 1000,
               color=COH_C, alpha=0.10, lw=0, zorder=0)
    for v, c, ls, lab in ((g["ceiling"], CEIL_C, "--",
                           "the Rolandslied's other witnesses"),
                          (g["alexanderlied"], ALEX_C, "-.",
                           "the Alexanderlied"),
                          (coh, COH_C, ":", "passages by other authors")):
        ax.axhline(float(np.median(v)) / 1000, color=c, lw=1.1, ls=ls,
                   alpha=0.9, zorder=2, label=lab)
    ax.plot((xs + L / 2) / 1000, win / 1000, color=WIN_C, lw=1.0, marker="o",
            ms=2.6, alpha=0.9, zorder=3,
            label="a window of the Kaiserchronik")
    hi = win >= np.percentile(coh, 95)
    ax.scatter((xs[hi] + L / 2) / 1000, win[hi] / 1000, s=26, facecolor="none",
               edgecolor=INK, lw=0.9, zorder=4,
               label="window above every 19 in 20 of them")
    ax.set_xlabel("position in the witness that was scored "
                  "(thousands of masked tokens)", fontsize=9.5, color=INK)
    ax.set_ylabel("score against the Rolandslied\n(thousands)", fontsize=9.5,
                  color=INK)
    ax.set_title(f"{letter}   {title}", fontsize=11.5, loc="left", color=INK,
                 fontweight="bold", pad=18)
    ax.text(0, 1.015,
            f"{len(win)} windows of {L:,} tokens · "
            f"{np.mean(win >= np.percentile(coh, 90)):.0%} of them stand above "
            f"the cohort's ninth decile, where a tenth is expected · "
            f"{np.mean(win > coh.max()):.0%} above every passage by another "
            f"author", transform=ax.transAxes, fontsize=8.8, color=GREY,
            va="bottom")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ax.spines.values():
        s.set_color(GREY)
    ax.tick_params(colors=INK, labelsize=9)
    ax.grid(alpha=0.12, lw=0.6)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    mn, gn, xn = load("none")
    mm, gm, xm = load("reuse")

    fig = plt.figure(figsize=(14.2, 9.0))
    gs = GridSpec(2, 2, width_ratios=[4.3, 1.25], hspace=0.46, wspace=0.20,
                  left=0.068, right=0.985, top=0.775, bottom=0.165)
    a = fig.add_subplot(gs[0, 0])
    b = fig.add_subplot(gs[1, 0], sharey=a)
    panel(a, mn, gn, xn, "a", "with everything the chronicle shares with the "
                              "Rolandslied left in")
    panel(b, mm, gm, xm, "b", f"with the shared wording taken out "
                              f"({mm['dropped']:,} tokens dropped)")

    axd = fig.add_subplot(gs[:, 1])
    sets = [("the Rolands-\nlied's other\nwitnesses", gn["ceiling"], CEIL_C, "^"),
            ("the chron-\nicle's\nwindows", gn["window"], WIN_C, "o"),
            ("the Alexan-\nderlied", gn["alexanderlied"], ALEX_C, "D"),
            ("other\nauthors", gn["cohort"], COH_C, "s")]
    for i, (lab, v, c, mk) in enumerate(sets):
        axd.scatter(np.full(len(v), i) + np.random.default_rng(i)
                    .normal(0, 0.07, len(v)), v / 1000, s=12, color=c,
                    marker=mk, linewidths=0, alpha=0.8, zorder=3)
        axd.plot([i - 0.28, i + 0.28], [np.median(v) / 1000] * 2, color=INK,
                 lw=1.6, zorder=4)
    axd.set_xticks(range(len(sets)))
    axd.set_xticklabels([s[0] for s in sets], fontsize=8.2, color=INK)
    axd.set_ylabel("score (thousands)", fontsize=9.5, color=INK)
    axd.set_title("c   the scale", fontsize=11.5, loc="left", color=INK,
                  fontweight="bold", pad=30)
    axd.text(0, 1.015, "everything scored against\nthe same known passages",
             transform=axd.transAxes, fontsize=8.6, color=GREY, va="bottom")
    for s in ("top", "right"):
        axd.spines[s].set_visible(False)
    for s in axd.spines.values():
        s.set_color(GREY)
    axd.tick_params(colors=INK, labelsize=9)
    axd.grid(axis="y", alpha=0.12, lw=0.6)

    h, lab = a.get_legend_handles_labels()
    fig.legend(h, lab, loc="upper left", bbox_to_anchor=(0.068, 0.885), ncol=3,
               frameon=False, fontsize=9.2)
    fig.suptitle("Is any of the Kaiserchronik Pfaffe Konrad's? The chronicle "
                 "read against the Rolandslied",
                 fontsize=14, x=0.068, ha="left", y=0.975, color=INK,
                 fontweight="bold")
    fig.text(0.068, 0.945, textwrap.fill(
        f"Known: {len(mn['known_starts'])} passages of {mn['L']:,} tokens of "
        f"the Rolandslied (Heidelberg, Cpg 112). Questioned: windows rolled "
        f"along the Vorau Kaiserchronik. Reference: {len(mn['donors'])} "
        f"grammars from other texts, averaged. The two works are in different "
        f"manuscripts, so the cost of crossing hands is paid everywhere in this "
        f"figure and works against the chronicle.", 150),
        fontsize=9.5, color=INK, ha="left", va="top", linespacing=1.5)
    fig.text(0.068, 0.105, textwrap.fill(
        "Middle High German verse, POSNoise-masked lemma stream, units of 100 "
        "tokens. The Rolandslied's other witnesses give the top of the scale: "
        "the same author's text in another hand. Shared wording is not evidence "
        "of authorship by itself — one author writing both works and one work "
        "borrowing from the other produce it alike — so panel b removes every "
        "stretch the reuse mask marks and asks the question again of what is "
        "left.", 152),
        fontsize=9.2, color=GREY, ha="left", va="top", linespacing=1.5)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=190)
    plt.close(fig)
    for name, g in (("unmasked", gn), ("masked", gm)):
        print(f"  {name:9} windows median {np.median(g['window']):8.1f}  "
              f"max {g['window'].max():8.1f}  ceiling "
              f"{np.median(g['ceiling']):8.1f}  cohort "
              f"{np.median(g['cohort']):8.1f}")
    print(f"  wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
