# Figure: the chronicle against itself, and what the pattern across it is made of.
#
# Two claims have to be seen at once for the attribution to be read correctly.
# The first is that the work is one profile from end to end: every region claims
# every other, and no part of the poem stands apart -- neither the opening that
# reworks the Annolied nor the closing regions. The second is that the variation
# which does exist belongs to the regions as KNOWN TEXTS and not to their
# affinities, so that the pattern of the attribution across the poem says nothing
# about where in the chronicle anything sits.
#
# The matrix carries the first claim and the scatter the second, and they are
# drawn side by side because a reader who sees only the matrix will go looking for
# a seam in it.
#
#   python medieval/make_kc_internal_figure.py
#
# Output: medieval/ZfdG_Kaiserchronik/figures/kc_internal.png

import csv
import json
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

RUN = ROOT / "scores" / "konrad_profile" / "internal__A1__r20__q4000.jsonl"
INT = HERE / "kc_reuse" / "kc_internal.tsv"
OUT = HERE / "ZfdG_Kaiserchronik" / "figures" / "kc_internal.png"
CAND_C = "#1a4f7a"


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    lines = RUN.open(encoding="utf-8").read().splitlines()
    meta = json.loads(lines[0])
    n = int(meta["regions"])
    Z = np.full((n, n), np.nan)
    for l in lines[1:]:
        r = json.loads(l)
        Z[r["known"], r["quest"]] = r["z"]
    rows = list(csv.DictReader(INT.open(encoding="utf-8"), delimiter="\t"))
    claimed = np.array([float(r["claimed_by_rest"]) for r in rows])
    claims = np.array([float(r["claims_rest"]) for r in rows])
    cand = np.array([float(r["candidate"]) for r in rows])

    fig = plt.figure(figsize=(12.6, 5.4))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 1, 1], wspace=0.34)
    a, b, c = (fig.add_subplot(gs[0]), fig.add_subplot(gs[1]),
               fig.add_subplot(gs[2]))

    im = a.imshow(Z, cmap="Blues", vmin=0, vmax=np.nanmax(Z), origin="upper")
    a.set_xlabel("the region that was scored", fontsize=9.5, color=INK)
    a.set_ylabel("the region used as the known text", fontsize=9.5, color=INK)
    a.set_title("a   Every region against every other", fontsize=11.5,
                loc="left", color=INK, fontweight="bold", pad=20)
    a.text(0, 1.03, f"{n} regions of {meta['region_len']:,} masked tokens",
           transform=a.transAxes, fontsize=8.6, color=GREY, va="bottom")
    a.set_xticks(range(0, n, 4))
    a.set_yticks(range(0, n, 4))
    cb = fig.colorbar(im, ax=a, fraction=0.046, pad=0.03)
    cb.set_label("deviations of the cohort of strangers", fontsize=8.6,
                 color=INK)
    cb.ax.tick_params(labelsize=8, colors=INK)

    x = np.arange(n)
    b.plot(x, claimed, color="#2e7d32", lw=1.3, marker="s", ms=4.5,
           label="as a questioned passage: how the rest claims it")
    b.plot(x, claims, color="#8c2d2d", lw=1.3, ls="--", marker="D", ms=4.2,
           label="as a known text: how it claims the rest")
    b.set_xlabel("the region, from the chronicle's opening to its close",
                 fontsize=9.5, color=INK)
    b.set_ylabel("deviations of the cohort", fontsize=9.5, color=INK)
    b.set_title("b   A region read two ways", fontsize=11.5, loc="left",
                color=INK, fontweight="bold", pad=20)
    b.text(0, 1.03, f"claimed: {claimed.min():.2f} to {claimed.max():.2f} · "
                    f"claiming: {claims.min():.2f} to {claims.max():.2f}",
           transform=b.transAxes, fontsize=8.6, color=GREY, va="bottom")
    b.legend(fontsize=8.4, frameon=False, loc="upper left")
    b.annotate("the opening,\nreworked from\nthe Annolied",
               xy=(0, claimed[0]), xytext=(0.02, 0.06), textcoords="axes fraction", fontsize=8.2,
               color=GREY,
               arrowprops=dict(arrowstyle="-", color=GREY, lw=0.8))
    b.annotate("the two weakest\nknown texts", xy=(18, claims[18]),
               xytext=(0.66, 0.10), textcoords="axes fraction", fontsize=8.2,
               color=GREY,
               arrowprops=dict(arrowstyle="-", color=GREY, lw=0.8))

    c.scatter(claims, cand, s=46, color=CAND_C, zorder=3,
              label=f"against the region as a known text (r = "
                    f"{np.corrcoef(claims, cand)[0, 1]:+.3f})")
    c.scatter(claimed, cand, s=46, facecolor="none", edgecolor="#8c2d2d",
              lw=1.2, zorder=3,
              label=f"against how the rest claims it (r = "
                    f"{np.corrcoef(claimed, cand)[0, 1]:+.3f})")
    lo, hi = 2.8, 7.4
    c.plot([lo, hi], np.polyval(np.polyfit(claims, cand, 1), [lo, hi]),
           color=CAND_C, lw=1.0, ls=":", zorder=2)
    c.set_xlabel("the chronicle's own score for that region", fontsize=9.5,
                 color=INK)
    c.set_ylabel("the score Konrad's poem takes there", fontsize=9.5,
                 color=INK)
    c.set_title("c   What the pattern is made of", fontsize=11.5, loc="left",
                color=INK, fontweight="bold", pad=20)
    c.text(0, 1.03, "each point is one region of the chronicle",
           transform=c.transAxes, fontsize=8.6, color=GREY, va="bottom")
    c.legend(fontsize=8.2, frameon=False, loc="upper left")

    for ax in (b, c):
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.grid(alpha=0.12, lw=0.6)
    for ax in (a, b, c):
        for s in ax.spines.values():
            s.set_color(GREY)
        ax.tick_params(colors=INK, labelsize=9)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"wrote {OUT.relative_to(ROOT)}")
    print(f"  claimed by the rest {np.median(claimed):.2f} "
          f"({claimed.min():.2f}-{claimed.max():.2f}); "
          f"r with the candidate {np.corrcoef(claims, cand)[0, 1]:+.3f} as a "
          f"known text, {np.corrcoef(claimed, cand)[0, 1]:+.3f} as a questioned "
          f"one")


if __name__ == "__main__":
    main()
