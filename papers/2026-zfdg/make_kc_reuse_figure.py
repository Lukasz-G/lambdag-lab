# Figure: textual reuse in the Kaiserchronik.
#
# Four things the detector established, one panel each:
#   a  the threshold is measured rather than chosen -- the distribution of
#      8-gram sharing over every comparison text, with the known partners
#      standing clear of it
#   b  why a block detector finds nothing -- the lengths of the runs actually
#      shared, whose maximum is seventeen tokens
#   c  where the echo falls inside the chronicle -- twenty slices per witness,
#      the Annolied against the Rolandslied
#   d  how the echoes weaken as the witnesses grow later
#
# Everything is recomputed from the banks and from medieval/kc_reuse/, so the
# figure cannot drift from the numbers. The rank correlations in panel d are
# computed here and printed, so the caption never states a trend the data does
# not carry -- a first version asserted that the Annolied echo declines and the
# Rolandslied's does not, and the measurement says the Rolandslied A echo
# declines MORE steeply than the Annolied's.
#
#   python medieval/make_kc_reuse_figure.py
#
# Output: medieval/ZfdG_Kaiserchronik/figures/kc_reuse.png

import csv
import json
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from scipy import stats

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(HERE))
import run_witness_cells as W  # noqa: E402
from kc_dates import GREY, INK, STYLE, year_of  # noqa: E402

REUSE = HERE / "kc_reuse"
OUT = HERE / "ZfdG_Kaiserchronik" / "figures" / "kc_reuse.png"

K = 8


def shared_runs(a, b, k):
    """lengths of the maximal runs of `a` that also occur in `b`"""
    s = {tuple(b[i:i + k]) for i in range(len(b) - k + 1)}
    hit = np.zeros(len(a), bool)
    for i in range(len(a) - k + 1):
        if tuple(a[i:i + k]) in s:
            hit[i:i + k] = True
    runs, n = [], 0
    for h in hit:
        if h:
            n += 1
        elif n:
            runs.append(n); n = 0
    if n:
        runs.append(n)
    return runs


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    bank = W.load("lemma")
    with (ROOT / "masked" / "mhg_witnesses_lemma" / "MANIFEST.tsv").open(
            encoding="utf-8") as fh:
        man = {r["file"]: r for r in csv.DictReader(fh, delimiter="\t")}

    recs = {}
    for f in sorted(REUSE.glob("kcd*.json")):
        r = json.loads(f.read_text(encoding="utf-8"))
        if r["tokens"] >= 20000:
            recs[r["witness"]] = r
    order = sorted(recs, key=lambda n: (year_of(man[n]["date"]) or 9999))
    sig = {n: man[n]["siglum"] for n in order}

    A1 = "kcd__anon__kaiserchronik__A1"
    a1 = [t for s in bank[A1]["sents"] for t in s]
    a1sh = {tuple(a1[i:i + K]) for i in range(len(a1) - K + 1)}

    print("  recomputing the baseline distribution against the Vorau witness")
    cov, names = [], []
    for m in bank:
        if m == A1 or m.startswith("kcd__") or "kaiserchronik" in bank[m]["work"]:
            continue
        if bank[m]["ntok"] < 2000:
            continue
        t = [x for s in bank[m]["sents"] for x in s]
        tot = max(len(t) - K + 1, 1)
        cov.append(sum(1 for i in range(tot) if tuple(t[i:i + K]) in a1sh) / tot)
        names.append(m)
    cov = np.array(cov)
    med = float(np.median(cov))
    rsd = 1.4826 * float(np.median(np.abs(cov - med)))
    cut = med + 3 * rsd

    fig = plt.figure(figsize=(13.6, 13.4))
    gs = GridSpec(3, 2, figure=fig, height_ratios=[1.0, 1.25, 0.95],
                  hspace=0.60, wspace=0.30,
                  left=0.085, right=0.955, top=0.855, bottom=0.062)

    # --- a: the measured baseline --------------------------------------------
    ax = fig.add_subplot(gs[0, 0])
    ax.hist(cov * 1000, bins=44, color="#c9d6e4", edgecolor="#5b7fa6", lw=0.6)
    top = ax.get_ylim()[1]
    ax.set_ylim(0, top * 1.32)
    ax.axvline(cut * 1000, color=GREY, ls="--", lw=1.3)
    ax.text(cut * 1000 + 0.25, top * 1.24, "threshold\nmedian + 3 robust sd",
            color=GREY, fontsize=8.4, va="top", ha="left", linespacing=1.3)
    for i, (key, (c, mk, lab)) in enumerate(STYLE.items()):
        if key not in names:
            continue
        v = cov[names.index(key)] * 1000
        y = top * (0.50 + 0.22 * i)
        ax.plot([v], [y], marker=mk, color=c, ms=9, clip_on=False, zorder=5)
        ax.annotate(lab, (v, y), xytext=(0, 13), textcoords="offset points",
                    ha="center", fontsize=8.8, color=c, fontweight="bold")
    ax.set_xlabel("share of a text's 8-grams that also occur in the\n"
                  "Vorau witness (per thousand)", fontsize=9.4)
    ax.set_ylabel("number of texts", fontsize=9.4)
    ax.set_title("a   the threshold is measured, not chosen",
                 fontsize=11.5, loc="left", fontweight="bold", pad=10)
    ax.text(0.985, 0.985, f"{len(cov)} comparison texts,\n2,000 tokens or more",
            transform=ax.transAxes, ha="right", va="top", fontsize=8.4,
            color=GREY, linespacing=1.3)

    # --- b: why a block detector finds nothing -------------------------------
    ax = fig.add_subplot(gs[0, 1])
    anno = [t for s in bank["rem__M013O"]["sents"] for t in s]
    ctrl = [t for s in bank["rem__M317"]["sents"] for t in s]
    ra, rc = shared_runs(anno, a1, K), shared_runs(ctrl, a1, K)
    bins = np.arange(K, max(max(ra), max(rc)) + 2) - 0.5
    ax.hist([ra, rc], bins=bins,
            label=["Annolied", "a text of the same form,\nHugo von Langenstein"],
            color=["#0b5fa5", "#dcdcdc"], edgecolor=INK, lw=0.5)
    ax.set_yscale("log")
    ax.set_xlim(K - 1, 32)
    ax.axvline(30, color="#8c2d2d", ls=":", lw=1.6)
    ax.annotate("a block detector asked for\nruns of 30 and found nothing",
                (29.4, 1.02), ha="right", va="bottom", fontsize=8.6,
                color="#8c2d2d", linespacing=1.35)
    ax.annotate(f"longest run shared with\nthe chronicle: {max(ra)} tokens",
                (max(ra), 1.0), xytext=(18, 74), textcoords="offset points",
                ha="left", fontsize=8.6, color="#0b5fa5", linespacing=1.35,
                arrowprops=dict(arrowstyle="->", color="#0b5fa5", lw=0.9))
    ax.set_xlabel("length in tokens of a run shared with the chronicle",
                  fontsize=9.4)
    ax.set_ylabel("number of runs (log scale)", fontsize=9.4)
    ax.set_title("b   the reuse is echo, not a copied block",
                 fontsize=11.5, loc="left", fontweight="bold", pad=10)
    ax.legend(fontsize=8.2, frameon=False, loc="upper right",
              bbox_to_anchor=(1.0, 1.0))

    # --- c: where the echo falls ---------------------------------------------
    for col, key in ((0, "rem__M013O"), (1, "rem__M205P")):
        ax = fig.add_subplot(gs[1, col])
        colr, mk, lab = STYLE[key]
        mat = np.full((len(order), 20), np.nan)
        for i, n in enumerate(order):
            p = recs[n]["partners"].get(key)
            if p:
                mat[i] = p["profile"]
        finite = mat[np.isfinite(mat)]
        vmax = float(np.quantile(finite, 0.98)) if len(finite) else 1.0
        cmap = plt.get_cmap("Blues" if col == 0 else "YlOrBr").copy()
        cmap.set_bad("#efefef")
        im = ax.imshow(np.ma.masked_invalid(mat), aspect="auto", cmap=cmap,
                       vmin=0, vmax=vmax, interpolation="nearest")
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels([f"{sig[n]}   {year_of(man[n]['date']) or '?'}"
                            for n in order], fontsize=8.6)
        ax.set_xticks([0, 4, 9, 14, 19])
        ax.set_xticklabels(["opening", "20%", "50%", "75%", "end"], fontsize=8.6)
        ax.set_xlabel("position in the chronicle", fontsize=9.4)
        ax.set_title(f"c{col + 1}   where the {lab} echo falls",
                     fontsize=11.5, loc="left", fontweight="bold", color=colr,
                     pad=10)
        cb = fig.colorbar(im, ax=ax, fraction=0.036, pad=0.025)
        cb.set_label("share of tokens echoing", fontsize=8.0)
        cb.ax.tick_params(labelsize=7.4)
        if col == 0:
            ax.add_patch(plt.Rectangle((-0.48, -0.48), 3.0, len(order) - 0.05,
                                       fill=False, ec="#0b5fa5", lw=2.0,
                                       zorder=6))
            ax.annotate("the opening, where the\nchronicle's Roman matter stands",
                        (2.6, 0.9), xytext=(34, 6), textcoords="offset points",
                        fontsize=8.4, color="#0b5fa5", linespacing=1.3,
                        arrowprops=dict(arrowstyle="->", color="#0b5fa5",
                                        lw=0.9))
        ax.text(0.0, -0.20, "pale grey: this text is below the witness's own "
                            "threshold", transform=ax.transAxes, fontsize=8.0,
                color=GREY)

    # --- d: the trend with date ----------------------------------------------
    ax = fig.add_subplot(gs[2, :])
    notes = []
    for key, (c, mk, lab) in STYLE.items():
        xs, ys, tags = [], [], []
        for n in order:
            p = recs[n]["partners"].get(key)
            y = year_of(man[n]["date"])
            if p and y:
                xs.append(y); ys.append(p["z"]); tags.append(sig[n])
        rho, pv = stats.spearmanr(xs, ys)
        notes.append(f"{lab}: rho {rho:+.2f}, p {pv:.3f} (n {len(xs)})")
        print(f"  {lab:15} Spearman rho {rho:+.3f}  p {pv:.3f}  n {len(xs)}")
        ax.plot(xs, ys, marker=mk, color=c, ms=8, lw=1.3, label=lab, alpha=0.92)
        if key == "rem__M013O":
            for x, y, t in zip(xs, ys, tags):
                ax.annotate(t, (x, y), xytext=(0, 10),
                            textcoords="offset points", ha="center",
                            fontsize=8.4, color=c)
    ax.set_ylim(0, 54)
    ax.axhline(3, color=GREY, ls="--", lw=1.1)
    ax.text(1183, 4.0, "threshold", color=GREY, fontsize=8.4, ha="left")
    ax.set_xlabel("approximate date of the witness", fontsize=9.4)
    ax.set_ylabel("strength of the echo\n(robust standard deviations\n"
                  "above that witness's baseline)", fontsize=9.4)
    ax.set_title("d   the Annolied and Rolandslied A echoes weaken as the "
                 "witnesses grow later; the Rolandslied P trend is not "
                 "significant",
                 fontsize=11.5, loc="left", fontweight="bold", pad=10)
    ax.legend(fontsize=9, frameon=False, loc="upper right")
    ax.text(0.985, 0.60, "\n".join(notes), transform=ax.transAxes, ha="right",
            va="top", fontsize=8.4, color="#333333", linespacing=1.5)
    ax.text(0.0, -0.26,
            "Eleven of the twelve witnesses carry a date; the B recension's "
            "does not. With so few points the ranks are suggestive and no more, "
            "and redaction alters length as well as wording.",
            transform=ax.transAxes, fontsize=8.2, color=GREY)

    fig.suptitle("Textual reuse in the Kaiserchronik", fontsize=16,
                 fontweight="bold", x=0.085, ha="left", y=0.978)
    fig.text(0.085, 0.938,
             "Twelve witnesses of 20,000 tokens or more, against 188 other "
             "Middle High German verse texts. Function-word stream after\n"
             "lemmatisation and grammatical masking; sharing measured on "
             "8-grams. Reuse is located before any authorship profile is\n"
             "computed, because a passage the chronicle shares with another "
             "work would register whatever its style.",
             fontsize=9.4, ha="left", va="top", color="#333333",
             linespacing=1.55)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=190, facecolor="white")
    print(f"  wrote {OUT.relative_to(ROOT)}")
    print(f"  baseline median {med:.5f}, robust sd {rsd:.5f}, cut {cut:.5f}")
    print(f"  longest shared run: Annolied {max(ra)}, control {max(rc)}")


if __name__ == "__main__":
    main()
