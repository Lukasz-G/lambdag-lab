# The register ladder as a figure, with the uncertainty the table cannot show.
#
# The register table reports one AUC per cell; the letters cells rest on ten
# same-author cases against thirty different-author cases per length, and a
# reader cannot tell from the table whether the prose-to-letters run
# (0.720, 0.550, 0.600 at 500, 5,000 and 20,000 tokens) is a finding or the
# sampling noise of ten cases. This draws every cell of the ladder across all
# six lengths with a bootstrap band (cases resampled within each label,
# 2,000 draws, central 90 %) so that the answer is visible.
#
#   python journal/figures/make_register_figure.py
#
# Data: experiments/scores/xgenre_ladder/<a>2<b>__symmetric__d-known*.jsonl
# (symmetric protocol, known and questioned truncated to the same length,
# donors from the known register). Output: fig_register.png and .pdf here.

import glob
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
LADDER = ROOT / "experiments" / "scores" / "xgenre_ladder"
NAME = {"eprose": "prose", "everse": "verse", "eessay": "essays", "eletters": "letters", "ememoir": "memoir"}
WITHIN = [("eprose", "eprose"), ("eessay", "eessay"), ("eletters", "eletters"), ("ememoir", "ememoir")]
ACROSS = [("eprose", "eessay"), ("eprose", "ememoir"), ("eprose", "eletters"),
          ("eessay", "everse"), ("eletters", "everse"), ("ememoir", "everse")]
LENGTHS = [500, 1000, 2000, 5000, 10000, 20000]
DRAWS = 2000
# one colour and one marker per cell, fixed
STYLE = {
    ("eprose", "eprose"): ("#7f7f7f", "o"), ("eessay", "eessay"): ("#a5a5a5", "s"),
    ("eletters", "eletters"): ("#bdbdbd", "^"), ("ememoir", "ememoir"): ("#8c8c8c", "D"),
    ("eprose", "eessay"): ("#1f77b4", "o"), ("eprose", "ememoir"): ("#2ca02c", "s"),
    ("eprose", "eletters"): ("#d62728", "^"), ("eessay", "everse"): ("#9467bd", "D"),
    ("eletters", "everse"): ("#ff7f0e", "v"), ("ememoir", "everse"): ("#17becf", "P"),
}


def rows_of(a, b):
    rows = []
    for fn in sorted(glob.glob(str(LADDER / (a + "2" + b + "__symmetric__d-known*.jsonl")))):
        if "catsrank" in fn or "dep_" in fn:
            continue
        rows += [json.loads(l) for l in open(fn, encoding="utf-8")]
    return rows


def fast_auc(s1, s0):
    """AUC as the Mann-Whitney statistic, vectorised over bootstrap draws:
    s1 (draws, n1), s0 (draws, n0)"""
    gt = (s1[:, :, None] > s0[:, None, :]).mean(axis=(1, 2))
    eq = (s1[:, :, None] == s0[:, None, :]).mean(axis=(1, 2))
    return gt + 0.5 * eq


def auc_band(rows, q, rng):
    sel = [r for r in rows if r["Q"] == q]
    s1 = np.array([r["lambda_G"] for r in sel if r["label"] == 1])
    s0 = np.array([r["lambda_G"] for r in sel if r["label"] == 0])
    if len(s1) == 0 or len(s0) == 0:
        return None
    a = float(fast_auc(s1[None, :], s0[None, :])[0])
    assert abs(a - roc_auc_score(np.r_[np.ones(len(s1)), np.zeros(len(s0))], np.r_[s1, s0])) < 1e-9
    b1 = rng.choice(s1, (DRAWS, len(s1))); b0 = rng.choice(s0, (DRAWS, len(s0)))
    boots = fast_auc(b1, b0)
    lo, hi = np.percentile(boots, [5, 95])
    return a, lo, hi, len(s1), len(s0)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    rng = np.random.default_rng(0)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
    counts = {}
    for ax, block, title in ((axes[0], WITHIN, "within one register"),
                             (axes[1], ACROSS, "across registers")):
        for a, b in block:
            rows = rows_of(a, b)
            if not rows:
                continue
            pts = []
            for q in LENGTHS:
                band = auc_band(rows, q, rng)
                if band:
                    pts.append((q,) + band)
            if not pts:
                continue
            x = [p[0] for p in pts]; y = [p[1] for p in pts]; lo = [p[2] for p in pts]; hi = [p[3] for p in pts]
            counts[(a, b)] = (pts[0][4], pts[0][5])
            col, mk = STYLE[(a, b)]
            label = "%s → %s (%d same, %d different)" % (NAME[a], NAME[b], pts[0][4], pts[0][5])
            emph = (a, b) == ("eprose", "eletters")
            if emph:
                ax.fill_between(x, lo, hi, color=col, alpha=0.2, lw=0)
                ax.plot(x, y, color=col, marker=mk, ms=6, lw=2.0, label=label)
                for xi, yi in zip(x, y):
                    ax.annotate("%.2f" % yi, (xi, yi), textcoords="offset points", xytext=(0, 7),
                                ha="center", fontsize=7, color=col)
                print("prose -> letters, 90 % bands:", "; ".join("%d: %.2f [%.2f, %.2f]" % (xi, yi, l, h)
                                                                  for xi, yi, l, h in zip(x, y, lo, hi)))
            else:
                ax.errorbar(x, y, yerr=[np.array(y) - np.array(lo), np.array(hi) - np.array(y)], color=col,
                            marker=mk, ms=4.5, lw=1.3, elinewidth=0.7, capsize=2, alpha=0.9, label=label)
        ax.set_xscale("log")
        ax.set_xticks(LENGTHS); ax.set_xticklabels(["500", "1k", "2k", "5k", "10k", "20k"])
        ax.set_xlabel("tokens a side (known and questioned truncated alike)")
        ax.axhline(0.5, color="#bbb", lw=0.7, ls=":")
        ax.set_title(title, fontsize=10, loc="left")
        ax.legend(fontsize=7, loc="lower right", frameon=False)
        ax.grid(alpha=0.25, lw=0.5)
    axes[0].set_ylabel("AUC of $\\lambda_G$")
    axes[0].set_ylim(0.35, 1.02)
    fig.suptitle("English registers, POSNoise-masked, symmetric protocol, donors from the known register; "
                 "band and error bars = central 90 % of 2,000 case bootstraps", fontsize=9.5, y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    for ext in ("png", "pdf"):
        fig.savefig(HERE / ("fig_register." + ext), dpi=200, bbox_inches="tight", facecolor="white")
    print("wrote fig_register.png/.pdf;", {"%s->%s" % (NAME[a], NAME[b]): n for (a, b), n in counts.items()})


if __name__ == "__main__":
    main()
