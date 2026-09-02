# Cohort studentisation on REAL grid test pairs -- the analysis. Three renderings on the
# identical (subsampled) case sets at symmetric L=1200:
#   grid   the paper's pooled-reference sqrt score (joined on stored id)
#   ssqrt  symmetrised sqrt: mean(lam_j) / sqrt(N)   (per-author donors)
#   stud   mean(lam_j) / sd(lam_j)                   (case-internal cohort)
#
#   python experiments/analyze_routeb_real.py

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SCORES = HERE / "scores"
sys.path.insert(0, str(HERE.parent))
from lambdag import cllr, cllr_min  # noqa: E402

L = 1200


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    from sklearn.metrics import roc_auc_score
    med = {k: [] for k in ("grid", "ssqrt", "stud")}
    print(f"{'dataset':20s} {'n':>4s}  {'AUC g/s/t':>19s}  "
          f"{'Cllr grid':>9s} {'ssqrt':>7s} {'stud':>7s}  "
          f"{'floor grid/stud':>15s}")
    for fn in sorted(SCORES.glob(f"*__routebreal__L{L}.jsonl")):
        ds = fn.name.split("__")[0]
        rows = [json.loads(l) for l in open(fn, encoding="utf-8")]
        if not rows:
            continue
        gfn = SCORES / f"{ds}__kn__sent__L{L}.jsonl"
        grid = {r["id"]: r for r in
                (json.loads(l) for l in open(gfn, encoding="utf-8"))}
        y, g, ss, st = [], [], [], []
        for r in rows:
            if r["id"] not in grid:
                continue
            assert grid[r["id"]]["label"] == r["label"], f"{ds} id {r['id']}"
            lj = np.asarray(r["lam_j"], dtype=float)
            y.append(r["label"])
            g.append(grid[r["id"]]["sqrt"])
            ss.append(float(np.mean(lj)) / np.sqrt(r["n_q"]))
            st.append(float(np.mean(lj) / (np.std(lj, ddof=1) + 1e-9)))
        y = np.array(y); scores = {"grid": np.array(g), "ssqrt": np.array(ss),
                                   "stud": np.array(st)}
        if len(set(y.tolist())) < 2:
            continue
        out = {}
        for k, s in scores.items():
            out[k] = (roc_auc_score(y, s), float(cllr(s[y == 1], s[y == 0])),
                      float(cllr_min(s[y == 1], s[y == 0])))
            med[k].append(out[k])
        print(f"{ds:20s} {len(y):4d}  "
              f"{out['grid'][0]:.3f}/{out['ssqrt'][0]:.3f}/{out['stud'][0]:.3f}  "
              f"{out['grid'][1]:9.3f} {out['ssqrt'][1]:7.3f} {out['stud'][1]:7.3f}  "
              f"{out['grid'][2]:.3f}/{out['stud'][2]:.3f}")
    print("\nmedians:")
    for k, v in med.items():
        print(f"  {k}: AUC {np.median([x[0] for x in v]):.3f}  "
              f"Cllr {np.median([x[1] for x in v]):.3f}  "
              f"floor {np.median([x[2] for x in v]):.3f}")
    wins = sum(1 for a, b in zip(med['stud'], med['grid']) if a[1] < b[1])
    print(f"stud Cllr < grid Cllr in {wins}/{len(med['stud'])} datasets")


if __name__ == "__main__":
    main()
