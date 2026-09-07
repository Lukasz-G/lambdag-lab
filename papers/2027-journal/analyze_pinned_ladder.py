# Read the accumulation exponent off the pinned-known ladder.
#
# Reads experiments/scores/{ds}__kn__pinned__K{K}.jsonl (run_pinned_ladder.py)
# and fits log-log slopes of three quantities against the questioned length:
#
#   signal(N_q) = mean lambda[same] - mean lambda[different]   -> beta_signal
#   drift(N_q)  = |mean lambda[different]|                     -> beta_drift
#   noise(N_q)  = pooled sd of lambda                          -> beta_noise
#
# beta_signal = 1 is LambdaG's own premise (a constant per-token evidence rate);
# beta_signal < 1 is a Hilberg-type sub-linear drag. beta_drift = 1 is the
# definition of the offset b as a constant per-token rate, so it doubles as a
# sanity check on the run.
#
# Confidence intervals are a CASE-level bootstrap: case ids are resampled with
# all their rungs kept together, so the nesting of the questioned windows is
# respected and the rungs are not treated as independent observations.
#
# TWO READINGS THIS OUTPUT DOES NOT SUPPORT, both tempting:
#   * d' = signal/noise is reported for completeness but is distorted by the
#     heavy right tail of the same-author distribution; AUC is the honest
#     discrimination readout and is printed beside it.
#   * beta_noise is a BETWEEN-CASE spread and contains real author
#     heterogeneity, so it does not test whether lambda random-walks under H_d.
#     That needs the per-donor cohort spread, which run_xgenre.py persists.
#
#   python experiments/analyze_pinned_ladder.py
#   python experiments/analyze_pinned_ladder.py --k 10000 --boot 5000

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SCORES = HERE / "scores"


def load(fn):
    rows, labels, qs = {}, {}, set()
    for line in open(fn, encoding="utf-8"):
        if not line.strip():
            continue
        d = json.loads(line)
        rows[(d["id"], d["Q"])] = (d["lambda_G"], d["label"])
        labels[d["id"]] = d["label"]
        qs.add(d["Q"])
    return rows, labels, sorted(qs)


def curves(rows, ids, qs):
    """signal / noise / drift at each rung, for the given case ids."""
    sig, noi, dri = [], [], []
    for q in qs:
        one = np.array([rows[(i, q)][0] for i in ids if rows[(i, q)][1] == 1])
        zero = np.array([rows[(i, q)][0] for i in ids if rows[(i, q)][1] == 0])
        if len(one) < 3 or len(zero) < 3:
            return None
        sd = np.sqrt(((len(one) - 1) * one.var(ddof=1)
                      + (len(zero) - 1) * zero.var(ddof=1))
                     / (len(one) + len(zero) - 2))
        sig.append(one.mean() - zero.mean())
        noi.append(sd)
        dri.append(abs(zero.mean()))
    return np.array(sig), np.array(noi), np.array(dri)


def slope(x, y):
    return np.nan if np.any(y <= 0) else float(np.polyfit(np.log(x), np.log(y), 1)[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=5000)
    ap.add_argument("--boot", type=int, default=2000)
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    from sklearn.metrics import roc_auc_score

    files = sorted(SCORES.glob(f"*__kn__pinned__K{args.k}.jsonl"))
    if not files:
        print(f"no pinned ladders at K={args.k}; run run_pinned_ladder.py first")
        return
    print(f"PINNED-KNOWN LADDER   K={args.k} fixed, only the questioned block grows")
    print(f"{'dataset':17s} {'beta_signal [95% CI]':>22s} {'beta_drift':>11s} "
          f"{'beta_noise':>11s}   AUC per rung")
    agg = []
    for f in files:
        ds = f.name.split("__")[0]
        rows, labels, qs = load(f)
        ids = sorted(labels)
        x = np.array(qs, float)
        c = curves(rows, ids, qs)
        if c is None:
            print(f"{ds:17s} (too few cases)")
            continue
        sig, noi, dri = c
        b_sig, b_dri, b_noi = slope(x, sig), slope(x, dri), slope(x, noi)

        rng = np.random.default_rng(0)
        boot = []
        for _ in range(args.boot):
            bc = curves(rows, list(rng.choice(ids, len(ids), replace=True)), qs)
            if bc is not None:
                v = slope(x, bc[0])
                if not np.isnan(v):
                    boot.append(v)
        lo, hi = np.percentile(boot, [2.5, 97.5]) if boot else (np.nan, np.nan)

        auc = []
        for q in qs:
            lam = np.array([rows[(i, q)][0] for i in ids])
            y = np.array([rows[(i, q)][1] for i in ids])
            auc.append(roc_auc_score(y, lam))
        agg.append((b_sig, b_dri, b_noi))
        print(f"{ds:17s} {b_sig:9.3f} [{lo:5.2f},{hi:5.2f}] {b_dri:11.3f} "
              f"{b_noi:11.3f}   " + " ".join(f"{a:.3f}" for a in auc))
    if agg:
        a = np.array(agg)
        print(f"{'-- median --':17s} {np.median(a[:, 0]):9.3f}"
              f"{'':14s}{np.median(a[:, 1]):11.3f} {np.median(a[:, 2]):11.3f}")
    print(f"\nrungs (questioned tokens): {qs}")
    print("beta_signal = 1 -> constant per-token evidence rate (LambdaG's premise)")
    print("beta_signal < 1 -> sub-linear accumulation; n* = (E/c)^(1/beta)")


if __name__ == "__main__":
    main()
