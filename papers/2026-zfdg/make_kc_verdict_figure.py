# Figure: what each region of the chronicle says, and how much it is worth.
#
# The reader has to be able to see two different things at once and not confuse
# them. The upper panel is the measurement: every region of the oldest witness as
# a known text, with the candidate, the nearest poem of the same milieu and the
# chronicle's own other regions scored against it, all in deviations of that
# region's own cohort of strangers -- so the three levels are comparable down the
# poem. The lower panel is the verdict: the same comparison after calibration on
# one author's two different works, where what matters is not how high a bar
# stands but whether the candidate's bar clears the odds the calibration material
# can support while the control's does not.
#
# The ceiling is drawn because it is a property of the material rather than of the
# poem, and a reader who does not see it will read the flat top of the candidate's
# series as a measurement instead of a bound.
#
#   python medieval/make_kc_verdict_figure.py
#
# Output: medieval/ZfdG_Kaiserchronik/figures/kc_verdict.png

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from kc_dates import GREY, INK  # noqa: E402

RUN = ROOT / "scores" / "konrad_profile" / "regions__A1__r20__q4000__ed4000.jsonl"
XW = HERE / "kc_reuse" / "kc_regions_crosswork.tsv"
INT = HERE / "kc_reuse" / "kc_internal.tsv"
OUT = HERE / "ZfdG_Kaiserchronik" / "figures" / "kc_verdict.png"

CAND_C, CTRL_C, OWN_C = "#1a4f7a", "#b8860b", "#2e7d32"


def spread(v):
    m = float(np.median(v))
    return m, max(1.4826 * float(np.median(np.abs(v - m))),
                  (float(np.quantile(v, 0.9)) - m) / 1.2816, 1e-9)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    rows = [json.loads(l) for l in RUN.open(encoding="utf-8")]
    meta, rows = rows[0], rows[1:]
    by = defaultdict(lambda: defaultdict(list))
    for r in rows:
        by[r["region"]][r["kind"]].append(r["lambda_G"])
    regions = sorted(by)
    xw = {int(r["region"]): r for r in
          csv.DictReader(XW.open(encoding="utf-8"), delimiter="\t")}
    own = {int(r["region"]): float(r["claimed_by_rest"]) for r in
           csv.DictReader(INT.open(encoding="utf-8"), delimiter="\t")}

    z = {"candidate": [], "control": []}
    for ri in regions:
        m, sd = spread(np.array(by[ri]["cohort"]))
        for k in z:
            z[k].append(float(np.median([(v - m) / sd for v in by[ri][k]])))
    ceiling = float(xw[regions[0]]["elub_hi"])
    llr = {k: [float(xw[ri]["llr" if k == "candidate" else "llr_control"])
               for ri in regions] for k in z}
    x = np.arange(len(regions))
    verses = [xw[ri]["verses"] for ri in regions]

    fig, (a, b) = plt.subplots(2, 1, figsize=(10.5, 7.4), sharex=True,
                               gridspec_kw=dict(height_ratios=[1.25, 1],
                                                hspace=0.22))
    a.axhspan(-1, 1, color=GREY, alpha=0.13, lw=0, zorder=0)
    a.axhline(0, color=GREY, lw=0.9, zorder=1)
    a.plot(x, [own[ri] for ri in regions], color=OWN_C, lw=1.1, ls="--",
           marker="s", ms=4.5, zorder=3,
           label="the chronicle's own other regions (the same work)")
    a.plot(x, z["candidate"], color=CAND_C, lw=1.4, marker="o", ms=5.5,
           zorder=4, label="Konrad's Rolandslied")
    a.plot(x, z["control"], color=CTRL_C, lw=1.1, ls="-.", marker="^", ms=4.8,
           zorder=3, label="Lambrecht's Alexanderlied and three other poems "
                           "of the same milieu")
    a.set_ylabel("deviations of that region's\nown cohort of strangers",
                 fontsize=9.5, color=INK)
    a.set_title("a   What each region of the Kaiserchronik claims",
                fontsize=11.5, loc="left", color=INK, fontweight="bold",
                pad=20)
    a.text(0, 1.015,
           f"Vorau, Cod. 276, lemmatised and topic-masked · each region "
           f"{meta['region_len']:,} masked tokens as the known text · questioned "
           f"passages {meta['L']:,} · {len(meta['donors'])} reference grammars · "
           f"the grey band is the cohort's own middle",
           transform=a.transAxes, fontsize=8.8, color=GREY, va="bottom")
    # the series are named where they end rather than in a box, which would have
    # to sit on top of them: twenty regions leave no free corner
    a.set_xlim(-0.6, len(regions) + 5.4)
    for y, c, lab in ((own[regions[-1]], OWN_C,
                       "the chronicle's own\nother regions"),
                      (z["candidate"][-1], CAND_C, "Konrad's\nRolandslied"),
                      (z["control"][-1], CTRL_C,
                       "Lambrecht's Alexanderlied\nand three other poems\nof the "
                       "same milieu")):
        a.annotate(lab, xy=(len(regions) - 1, y),
                   xytext=(len(regions) + 0.4, y), fontsize=8.6, color=c,
                   va="center", ha="left")

    w = 0.38
    b.axhline(0, color=GREY, lw=0.9, zorder=1)
    b.axhline(ceiling, color=INK, lw=1.0, ls=":", zorder=2)
    b.text(len(regions) + 0.3, ceiling - 0.12,
           f"{10 ** ceiling:.0f} to one:\nthe most this calibration\nmaterial can "
           f"support", fontsize=8.6, color=INK, va="top", ha="left")
    b.set_xlim(-0.6, len(regions) + 5.4)
    b.bar(x - w / 2, llr["candidate"], w, color=CAND_C, zorder=3,
          label="Konrad's Rolandslied")
    b.bar(x + w / 2, llr["control"], w, color=CTRL_C, zorder=3,
          label="the nearest poem of the same milieu")
    b.set_ylabel("weight of evidence\n(log$_{10}$ likelihood ratio)",
                 fontsize=9.5, color=INK)
    b.set_xlabel("the region of the chronicle, by verse", fontsize=9.5,
                 color=INK)
    b.set_title("b   The same comparison, calibrated on one author's two "
                "different works", fontsize=11.5, loc="left", color=INK,
                fontweight="bold", pad=20)
    b.text(0, 1.015,
           f"calibration class: eight poets of 1150-1250, 63 passages of their "
           f"other works against 176 by strangers · "
           f"AUC {float(xw[0]['auc_class']):.3f}, "
           f"Cllr {float(xw[0]['cllr_class']):.3f} · a bar above zero favours "
           f"the same author, below it a different one",
           transform=b.transAxes, fontsize=8.8, color=GREY, va="bottom")
    b.legend(fontsize=8.6, frameon=False, loc="lower left")
    b.set_xticks(x)
    b.set_xticklabels([v.split("-")[0] for v in verses], rotation=45,
                      ha="right", fontsize=8.2)

    for ax in (a, b):
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ax.spines.values():
            s.set_color(GREY)
        ax.tick_params(colors=INK, labelsize=9)
        ax.grid(axis="y", alpha=0.12, lw=0.6)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"wrote {OUT.relative_to(ROOT)}")
    print(f"  candidate at the ceiling in "
          f"{sum(abs(v - ceiling) < 1e-4 for v in llr['candidate'])} of "
          f"{len(regions)} regions; the control above zero in "
          f"{sum(v > 0 for v in llr['control'])}")


if __name__ == "__main__":
    main()
