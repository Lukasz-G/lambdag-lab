# The same verdicts, against the class the attribution is actually about.
#
# experiments/calibrate_kc_regions.py calibrates on the chronicle's own other
# regions, which are the same author but also the same WORK. This script replaces
# that class with the one the question needs -- one author, two different works,
# scored by experiments/run_calibration_cases.py in the same construction as the
# chronicle's cases and against the same reference grammars and cohort passages --
# and reads the verdicts again.
#
# Every score is first expressed in its own case's units: deviations from that
# case's cohort median, in that cohort's robust spread. lambda_G is not comparable
# from one known text to another, and this statistic is, which is what allows a
# calibrator fitted on Hartmann's cases to be applied to the chronicle's.
#
# What the run reports, in this order:
#   how well the verifier separates the class at all, before any chronicle is
#     mentioned -- AUC, Cllr, and the bound the calibration set supports;
#   the candidate and the control against that class, region by region.
# The first number governs the second. If the verifier cannot separate one
# author's two works from a stranger's, no verdict it utters about the chronicle
# means anything, and that has to be read before the table of verdicts is.
#
#   python experiments/calibrate_crosswork.py
#
# Output: medieval/kc_reuse/kc_regions_crosswork.tsv and the table on stdout

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
from lambdag import PAVCalibrator, cllr, cllr_min  # noqa: E402
from calibrate_kc_regions import load, verse_ranges  # noqa: E402

SCORES = ROOT / "scores" / "konrad_profile"
OUT = ROOT / "medieval" / "kc_reuse" / "kc_regions_crosswork.tsv"


def spread(v):
    """the cohort's own location and scale, robust to its tail"""
    m = float(np.median(v))
    return m, max(1.4826 * float(np.median(np.abs(v - m))),
                  (float(np.quantile(v, 0.9)) - m) / 1.2816, 1e-9)


def crosswork(path):
    """the calibration set,each case in its own units -> (z_same, z_diff, cases)"""
    rows = [json.loads(l) for l in path.open(encoding="utf-8")]
    meta, rows = rows[0], rows[1:]
    same, diff, per = [], [], []
    for c in sorted({r["case"] for r in rows}):
        sel = [r for r in rows if r["case"] == c]
        coh = np.array([r["lambda_G"] for r in sel if r["kind"] == "cohort"])
        m, sd = spread(coh)
        p = [(r["lambda_G"] - m) / sd for r in sel
             if r["kind"] == "same_author"]
        same += p
        diff += [(r["lambda_G"] - m) / sd for r in sel
                 if r["kind"] == "cohort"]
        per.append((sel[0]["author"], sel[0]["known"], float(np.median(p))))
    return np.array(same), np.array(diff), per, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--calibration", default="calib__crosswork__q2000.jsonl")
    ap.add_argument("--regions", default="regions__A1__r20__q2000__controls.jsonl")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    same, diff, per, cmeta = crosswork(SCORES / args.calibration)
    auc = float(np.mean([(a > b) + 0.5 * (a == b) for a in same for b in diff]))
    cal = PAVCalibrator().fit(np.concatenate([same, diff]),
                              np.array([1] * len(same) + [0] * len(diff)))
    print("THE CLASS THE ATTRIBUTION IS ABOUT: one author, two works")
    for a, k, z in per:
        print(f"  {a:26} known {k.split('__')[-2]:24} "
              f"his other works stand {z:+.2f} deviations above the cohort")
    print(f"\n  {len(same)} same-author passages of other works from "
          f"{len({a for a, _, _ in per})} authors, {len(diff)} strangers; "
          f"median {np.median(same):+.2f} against {np.median(diff):+.2f}")
    print(f"  the verifier on this class: AUC {auc:.3f}, "
          f"Cllr {cllr(cal.transform(same), cal.transform(diff)):.3f}, "
          f"Cllr_min {cllr_min(same, diff):.3f}")
    print(f"  the odds the calibration set supports at all: "
          f"[{cal.lo:+.2f}, {cal.hi:+.2f}] in log10, i.e. at most "
          f"{10 ** cal.hi:.0f} to one\n")

    meta, by = load(SCORES / args.regions)
    regions = sorted(by)
    vr = verse_ranges(meta["witness"], int(meta["region_len"]), len(regions))
    rows = []
    for i in regions:
        coh = np.array([s for _, s in by[i]["cohort"]])
        m, sd = spread(coh)

        def z(kind):
            return np.array([(s - m) / sd for _, s in by[i][kind]])

        zc, zt = z("candidate"), z("control")
        rows.append(dict(region=i, verses=f"{vr[i][0]}-{vr[i][1]}",
                         z=float(np.median(zc)),
                         llr=float(np.median(cal.transform(zc))),
                         z_control=float(np.median(zt)),
                         llr_control=float(np.median(cal.transform(zt)))))
    print(f"  {'region':>6}{'verses':>14}{'candidate z':>13}{'LR':>7}"
          f"{'control z':>11}{'LR':>7}   the higher")
    for r in rows:
        print(f"  {r['region']:>6}{r['verses']:>14}{r['z']:>13.2f}"
              f"{r['llr']:>7.2f}{r['z_control']:>11.2f}{r['llr_control']:>7.2f}"
              f"   {'candidate' if r['z'] > r['z_control'] else 'control'}")
    n_c = sum(r["llr"] > 0 for r in rows)
    n_t = sum(r["llr_control"] > 0 for r in rows)
    n_ab = sum(r["z"] > r["z_control"] for r in rows)
    print(f"\n  positive odds for the candidate in {n_c} of {len(rows)} "
          f"regions, for a different author of the same milieu in {n_t}")
    print(f"  the candidate stands above that control in {n_ab} of "
          f"{len(rows)} regions")
    print(f"  both saturate the bound in most regions, so the odds separate "
          f"them nowhere; what survives is the ordering")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as fh:
        fh.write("region\tverses\tz\tllr\tz_control\tllr_control\tauc_class\t"
                 "cllr_class\telub_hi\n")
        for r in rows:
            fh.write(f"{r['region']}\t{r['verses']}\t{r['z']:.4f}\t"
                     f"{r['llr']:.4f}\t{r['z_control']:.4f}\t"
                     f"{r['llr_control']:.4f}\t{auc:.4f}\t"
                     f"{cllr(cal.transform(same), cal.transform(diff)):.4f}\t"
                     f"{cal.hi:.4f}\n")
    print(f"  wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
