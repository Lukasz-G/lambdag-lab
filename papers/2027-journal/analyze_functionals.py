# Aggregation-functional sweep (Layer 3): what do per-token/per-sentence SHAPE
# statistics carry that the sum discards? Runs offline on the details dumps
# (run_details.py). Metrics per functional: AUC and Cllr_min (both
# monotone-invariant, so no calibration map is presupposed), plus an H_d
# location-stability check (does the functional's different-author mean drift
# with L the way the sum's does?) and a 5-fold two-feature test (sqrt +
# functional) for residual signal on top of lambda/sqrt(N).
#
#   python experiments/analyze_functionals.py

import gzip
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats as sps

HERE = Path(__file__).resolve().parent
SCORES = HERE / "scores"
sys.path.insert(0, str(HERE.parent))
from lambdag import cllr_min  # noqa: E402

CELLS = sorted(SCORES.glob("*__kn__details-*__L*.jsonl.gz"))


def load(fn):
    return [json.loads(l) for l in gzip.open(fn, "rt", encoding="utf-8")]


def functionals(rows):
    """Per-case functional values; pooled H_d tail thresholds computed first."""
    hd = np.concatenate([np.asarray(r["tlam"]) for r in rows if r["label"] == 0][:400])
    t_lo, t_hi = np.quantile(hd, [0.05, 0.95])
    hd_sub = np.random.default_rng(0).choice(hd, size=min(len(hd), 20000),
                                             replace=False)
    out = defaultdict(list)
    for r in rows:
        t = np.asarray(r["tlam"], dtype=float)
        s = np.asarray(r["slam"], dtype=float)
        n = max(r["n_q"], 1)
        out["y"].append(r["label"])
        out["sqrt"].append(r["lambda_G"] / np.sqrt(n))
        out["fsign"].append(float((s > 0).mean()))
        k = max(1, int(0.05 * len(t)))
        st = np.sort(t)
        out["trim5"].append(float(st[k:-k].sum()) / np.sqrt(n))
        out["median"].append(float(np.median(t)))
        out["tail_lo"].append(float((t < t_lo).mean()))
        out["tail_hi"].append(float((t > t_hi).mean()))
        out["skew"].append(float(sps.skew(t)))
        by_type = defaultdict(list)
        for tok, v in zip(r["toks"], r["tlam"]):
            by_type[tok].append(v)
        cons = [np.mean(v) > 0 for v in by_type.values() if len(v) >= 3]
        out["habit"].append(float(np.mean(cons)) if cons else 0.5)
        out["wass"].append(float(sps.wasserstein_distance(
            t[:3000], hd_sub[:3000])))
    return {k: np.asarray(v) for k, v in out.items()}


def two_feature_gain(y, base, extra):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold
    X1 = base.reshape(-1, 1)
    X2 = np.column_stack([base, extra])
    aucs1, aucs2 = [], []
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(X2, y):
        m1 = LogisticRegression(max_iter=1000).fit(X1[tr], y[tr])
        m2 = LogisticRegression(max_iter=1000).fit(X2[tr], y[tr])
        aucs1.append(roc_auc_score(y[te], m1.decision_function(X1[te])))
        aucs2.append(roc_auc_score(y[te], m2.decision_function(X2[te])))
    return float(np.mean(aucs2) - np.mean(aucs1))


FUNCS = ["sqrt", "fsign", "trim5", "median", "tail_lo", "tail_hi", "skew",
         "habit", "wass"]


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    from sklearn.metrics import roc_auc_score
    drift = defaultdict(dict)
    for fn in CELLS:
        name = fn.name.replace(".jsonl.gz", "")
        ds, _, segL = name.split("__kn__")[0], None, name.split("__")[-2:]
        rows = load(fn)
        F = functionals(rows)
        y = F["y"]
        if len(set(y.tolist())) < 2:
            continue
        line = f"{name.replace('__kn__details-', ' '):46s}"
        for k in FUNCS:
            v = F[k]
            auc = roc_auc_score(y, v)
            cmin = cllr_min(v[y == 1], v[y == 0])
            line += f" {auc:.3f}/{cmin:.2f}"
        print(line)
        # residual signal on top of sqrt
        gains = {k: two_feature_gain(y, F["sqrt"], F[k])
                 for k in FUNCS if k != "sqrt"}
        best = max(gains, key=gains.get)
        print(f"{'':46s}  best 2-feature: sqrt+{best} "
              f"dAUC {gains[best]:+.4f}  (fsign {gains['fsign']:+.4f}, "
              f"habit {gains['habit']:+.4f}, trim5 {gains['trim5']:+.4f})")
        # H_d location stability across L (per dataset/seg)
        key = name.rsplit("__L", 1)[0]
        L = name.rsplit("__L", 1)[1]
        drift[key][L] = (float(F["sqrt"][y == 0].mean()),
                        float(F["fsign"][y == 0].mean()))
    print("\nH_d location drift (different-author mean at L600 -> L0):")
    print(f"{'cell':44s} {'sqrt':>16s} {'fsign':>16s}")
    for key, d in drift.items():
        if "600" in d and "0" in d:
            print(f"{key:44s} {d['600'][0]:7.2f}->{d['0'][0]:7.2f} "
                  f"{d['600'][1]:7.3f}->{d['0'][1]:7.3f}")


if __name__ == "__main__":
    print(f"{'cell':46s} " + " ".join(f"{k:>10s}" for k in FUNCS)
          + "   (AUC/Cllr_min)")
    main()
