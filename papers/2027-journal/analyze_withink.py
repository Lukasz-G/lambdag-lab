# Analysis of the within-K experiment (run_withink.py): does the known author's
# within-author variation predict the different-author band around him?
# (Journal paper, calibration chapter: the range of variation around K.)
#
# Per author (as KNOWN side), from per-token scores w = lambda_G / n_q:
#   within_mean, within_sd   over the author's within pairs (their windows vs each other)
#   cross_mean,  cross_sd    over the author's cross pairs  (other authors' windows vs K)
# cross_mean is the author-level H_d LOCATION (per-token b around this K);
# cross_sd the author-level H_d SCALE.
#
# The serious test is WITHIN-DATASET: per (ds, L), Spearman correlation across authors,
# then the distribution of those correlations (median, sign test). The pooled author-level
# and dataset-level numbers are reported as context (the pooled one is confounded
# by dataset).
#
#   python experiments/analyze_withink.py

import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
SCORES = HERE / "scores"


def author_stats(fn):
    """{author: (within_mean, within_sd, cross_mean, cross_sd, n_within, n_cross)}"""
    within, cross = defaultdict(list), defaultdict(list)
    for line in open(fn, encoding="utf-8"):
        r = json.loads(line)
        w = r["lambda_G"] / r["n_q"]
        (within if r["within"] else cross)[r["known"]].append(w)
    out = {}
    for a in within:
        if len(within[a]) >= 6 and len(cross.get(a, [])) >= 6:
            wi, cr = np.array(within[a]), np.array(cross[a])
            out[a] = (wi.mean(), wi.std(ddof=1), cr.mean(), cr.std(ddof=1),
                      len(wi), len(cr))
    return out


def per_dataset_corrs(L):
    rows = []           # (ds, n_authors, r_loc_sd, r_loc_mean, r_scale_sd)
    pooled = []         # (ds, within_mean, within_sd, cross_mean, cross_sd) per author
    for fn in sorted(SCORES.glob(f"*__kn__withink__L{L}.jsonl")):
        ds = fn.name.split("__")[0]
        st = author_stats(fn)
        if len(st) < 8:
            continue
        wm = np.array([v[0] for v in st.values()])
        ws = np.array([v[1] for v in st.values()])
        cm = np.array([v[2] for v in st.values()])
        cs = np.array([v[3] for v in st.values()])
        r_loc_sd = stats.spearmanr(ws, cm).statistic
        r_loc_mean = stats.spearmanr(wm, cm).statistic
        r_scale_sd = stats.spearmanr(ws, cs).statistic
        rows.append((ds, len(st), r_loc_sd, r_loc_mean, r_scale_sd))
        for a, v in st.items():
            pooled.append((ds, *v[:4]))
    return rows, pooled


def sign_test(rs):
    rs = [r for r in rs if not math.isnan(r)]
    pos = sum(1 for r in rs if r > 0)
    p = stats.binomtest(pos, len(rs), 0.5).pvalue if rs else float("nan")
    return pos, len(rs), p


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    for L in (2000, 5000):
        rows, pooled = per_dataset_corrs(L)
        if not rows:
            continue
        print(f"\n=== L = {L}  ({len(rows)} datasets, {len(pooled)} authors) ===")
        print(f"{'dataset':28s} {'n':>3s} {'r(loc~wSD)':>11s} {'r(loc~wMEAN)':>13s} "
              f"{'r(scale~wSD)':>13s}")
        for ds, n, r1, r2, r3 in rows:
            print(f"{ds:28s} {n:3d} {r1:11.2f} {r2:13.2f} {r3:13.2f}")

        for name, idx in [("LOCATION ~ within_SD", 2), ("LOCATION ~ within_MEAN", 3),
                          ("SCALE ~ within_SD", 4)]:
            rs = [row[idx] for row in rows]
            med = float(np.median([r for r in rs if not math.isnan(r)]))
            pos, n, p = sign_test(rs)
            print(f"  {name:24s} median r {med:+.2f}   positive {pos}/{n}   "
                  f"sign-test p {p:.4f}")

        # dataset-level (ecological) view: median author stats vs each other
        ds_ws = defaultdict(list); ds_cm = defaultdict(list); ds_cs = defaultdict(list)
        for ds, wm, wsd, cm, csd in pooled:
            ds_ws[ds].append(wsd); ds_cm[ds].append(cm); ds_cs[ds].append(csd)
        a = np.array([np.median(ds_ws[d]) for d in ds_ws])
        b = np.array([np.median(ds_cm[d]) for d in ds_ws])
        c = np.array([np.median(ds_cs[d]) for d in ds_ws])
        print(f"  dataset-level (n={len(a)}): loc~wSD Spearman "
              f"{stats.spearmanr(a, b).statistic:+.2f} (p {stats.spearmanr(a, b).pvalue:.3f}), "
              f"scale~wSD {stats.spearmanr(a, c).statistic:+.2f} "
              f"(p {stats.spearmanr(a, c).pvalue:.3f})")


if __name__ == "__main__":
    main()
