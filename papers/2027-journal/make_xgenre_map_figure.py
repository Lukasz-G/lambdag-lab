# The cross-genre map: AUC and calibrated cost against known length, one
# panel per questioned length, one line per score, mean over the six directed
# genre pairs of the German three-genre panel.
#
# Data: journal/figures/xgenre_map.json, written by
#   python experiments/analyze_xgenre_map.py --dump journal/figures/xgenre_map.json
# Output: journal/figures/xgenre_map.png

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
D = json.load(open(HERE / "xgenre_map.json", encoding="utf-8"))
T = {}
for k, v in D.items():
    pair, K, L, score = k.split("|")
    T[(pair, int(K), int(L), score)] = v
PAIRS = sorted({k[0] for k in T})
Ks = sorted({k[1] for k in T})
Ls = sorted({k[2] for k in T})
STYLE = {
    "lambdaG": ("λG, grammar score (mean over donors)", "#d62728", "D"),
    "composition": ("composition score, as it is", "0.55", "o"),
    "comp_projected": ("composition score, 10 genre directions removed", "#1f77b4", "s"),
    "fusion": ("fusion of λG and the projected composition score", "#2ca02c", "^"),
}


def mean_over_pairs(K, L, score, metric):
    vals = [T[(p, K, L, score)][metric] for p in PAIRS if (p, K, L, score) in T]
    return (np.mean(vals), len(vals)) if vals else (np.nan, 0)


plt.rcParams.update({"font.size": 10, "axes.titlesize": 10.5})
fig, axes = plt.subplots(2, len(Ls), figsize=(4.6 * len(Ls), 8.6), sharex=True, squeeze=False)
fig.subplots_adjust(hspace=0.3, wspace=0.22, top=0.84, bottom=0.1, left=0.06, right=0.99)
fig.suptitle("Cross-genre authorship verification: what known text and questioned text buy",
             fontsize=13, fontweight="bold", y=0.985)
fig.text(0.5, 0.925,
         "German (TextGrid), 17 authors who wrote prose, drama and verse; six directed genre pairs, "
         "mean over pairs. POSNoise-masked text, 100-token units.\n"
         "λG: Kneser-Ney 10-gram, known grammar on the first K known-genre tokens, 15 size-matched "
         "donors from the questioned genre. Composition: cosine of standardised symbol rates.\n"
         "Calibrated cost: logistic calibrator fitted on the other 16 authors of the same arm, case author "
         "held out. Dashed lines: AUC 0.90 and Cllr 0.5.",
         ha="center", va="top", fontsize=9)

for j, L in enumerate(Ls):
    for i, metric in enumerate(("auc", "cllr")):
        ax = axes[i, j]
        for score, (label, col, mk) in STYLE.items():
            ys, ns = zip(*[mean_over_pairs(K, L, score, metric) for K in Ks])
            ax.plot(Ks, ys, marker=mk, color=col, lw=1.8, ms=7, label=label)
            if i == 0 and score == "lambdaG":
                for K, y, n in zip(Ks, ys, ns):
                    if n and n < len(PAIRS):
                        ax.annotate("%d pairs" % n, (K, y), textcoords="offset points",
                                    xytext=(0, 9), ha="center", fontsize=7.5, color="0.4")
        ax.set_xscale("log")
        ax.set_xticks(Ks); ax.set_xticklabels([format(K, ",") for K in Ks])
        ax.grid(color="0.92")
        if i == 0:
            ax.set_title("questioned text: %s tokens" % format(L, ","))
            ax.set_ylim(0.55, 1.0)
            ax.axhline(0.90, color="0.5", lw=1, ls="--")
            if j == 0:
                ax.set_ylabel("AUC (same author vs. other authors)")
        else:
            ax.set_ylim(0.2, 1.1)
            ax.axhline(0.5, color="0.5", lw=1, ls="--")
            ax.axhline(1.0, color="0.8", lw=1)
            ax.set_xlabel("known text (tokens)")
            if j == 0:
                ax.set_ylabel("Cllr, calibrated on the other authors")
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=9, frameon=False,
           bbox_to_anchor=(0.5, 0.0))
fig.subplots_adjust(bottom=0.15)
out = HERE / "xgenre_map.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
fig.savefig(HERE / "xgenre_map.pdf", bbox_inches="tight")      # the paper's copy
print("->", out)
