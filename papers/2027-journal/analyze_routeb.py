# ROUTE B: case-internal studentisation, for free from the symmeter files.
# Each case carries its own H_d cohort: the per-donor lambdas lam_j = S_A - S_j
# persisted by run_symmeter.py. The studentised statistic
#     t = (S_A - mean_j S_j) / sd_j(S_j) = mean(lam_j) / sd(lam_j)
# needs no population constant at all. The rank variant F = frac(lam_j > 0) is
# the impostors-method percentile (resolution 1/R).
#
# Evaluated on the same case sets, three renderings:
#   sqrt  lambda_sym / sqrt(N)     (location still population-dependent)
#   stud  mean(lam_j) / sd(lam_j)  (case-internal location AND scale)
#   rank  frac(lam_j > 0)          (AUC only; bounded, resolution-limited)
# Matched and MISMATCHED arms reported separately: studentisation is
# shift-invariant per case cohort, so it should survive corpus mismatch.
#
#   python experiments/analyze_routeb.py

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SCORES = HERE / "scores"
sys.path.insert(0, str(HERE.parent))
from lambdag import cllr, cllr_min  # noqa: E402


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    from sklearn.metrics import roc_auc_score
    agg = {"matched": {"sqrt": [], "stud": [], "rank": []},
           "mismatch": {"sqrt": [], "stud": [], "rank": []}}
    print(f"{'arm':52s} {'n':>4s}  {'AUC s/t/r':>19s}  "
          f"{'Cllr sqrt':>9s} {'stud':>6s}  {'floor sqrt/stud':>15s}")
    for fn in sorted(SCORES.glob("*__symref-*__L2000.jsonl")):
        rows = [json.loads(l) for l in open(fn, encoding="utf-8")]
        if not rows:
            continue
        ds, refds = fn.name.replace("__L2000.jsonl", "").split("__symref-")
        kind = "matched" if ds == refds else "mismatch"
        y = np.array([r["within"] for r in rows])
        sq = np.array([r["lambda_G"] / np.sqrt(r["n_q"]) for r in rows])
        st, rk = [], []
        for r in rows:
            lj = np.asarray(r["lam_j"], dtype=float)
            st.append(float(np.mean(lj) / (np.std(lj, ddof=1) + 1e-9)))
            rk.append(float((lj > 0).mean()))
        st, rk = np.array(st), np.array(rk)
        out = {}
        for name, s in (("sqrt", sq), ("stud", st), ("rank", rk)):
            auc = roc_auc_score(y, s)
            c = float(cllr(s[y == 1], s[y == 0])) if name != "rank" else np.nan
            cm = float(cllr_min(s[y == 1], s[y == 0]))
            out[name] = (auc, c, cm)
            agg[kind][name].append((auc, c, cm))
        name = f"{ds} | ref={refds}"
        print(f"{name:52s} {len(y):4d}  "
              f"{out['sqrt'][0]:.3f}/{out['stud'][0]:.3f}/{out['rank'][0]:.3f}  "
              f"{out['sqrt'][1]:9.3f} {out['stud'][1]:6.3f}  "
              f"{out['sqrt'][2]:.3f}/{out['stud'][2]:.3f}")
    for kind in ("matched", "mismatch"):
        if not agg[kind]["sqrt"]:
            continue
        print(f"\n{kind} medians:")
        for name in ("sqrt", "stud", "rank"):
            v = agg[kind][name]
            a = np.median([x[0] for x in v])
            c = np.nanmedian([x[1] for x in v])
            m = np.median([x[2] for x in v])
            print(f"  {name}: AUC {a:.3f}  Cllr {c:.3f}  floor {m:.3f}")


if __name__ == "__main__":
    main()
