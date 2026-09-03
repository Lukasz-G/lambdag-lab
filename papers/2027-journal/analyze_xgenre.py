# Analysis of the German cross-genre pilot (run_xgenre.py): per direction and
# donor arm, discrimination and calibration of the symmetrised lambda, its
# sqrt(N) form, and the cohort-studentised statistic, plus the adequacy
# reading b_sym.
#
#   python experiments/analyze_xgenre.py

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
XS = HERE.parent / "scores" / "xgenre"
sys.path.insert(0, str(HERE.parent))
from lambdag import cllr_min  # noqa: E402

from sklearn.metrics import roc_auc_score  # noqa: E402


def cell(fn):
    rows = [json.loads(l) for l in open(fn, encoding="utf-8")]
    y = np.array([r["within"] for r in rows])
    lam = np.array([r["lambda_G"] for r in rows])
    n = np.array([r["n_q"] for r in rows], dtype=float)
    t = np.array([np.mean(r["lam_j"]) / (np.std(r["lam_j"]) + 1e-9)
                  for r in rows])
    sq = lam / np.sqrt(n)
    out = {"n": len(y), "npos": int(y.sum()),
           "b_sym": float(np.mean(lam[y == 0] / n[y == 0])),
           "b_same": float(np.mean(lam[y == 1] / n[y == 1]))}
    for name, s in (("sym", lam), ("sqrt", sq), ("t", t)):
        out[f"auc_{name}"] = float(roc_auc_score(y, s))
        out[f"cmin_{name}"] = float(cllr_min(s[y == 1], s[y == 0]))
    return out


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    print(f"{'direction':16s} {'arm':8s} {'n':>4s} {'b_same':>8s} {'b_sym':>8s} "
          f"{'AUC sym':>8s} {'AUC t':>7s} {'Cmin sqrt':>9s} {'Cmin t':>7s}")
    for fn in sorted(XS.glob("*__L1000.jsonl")):
        d, arm = fn.name.split("__")[:2]
        c = cell(fn)
        print(f"{d:16s} {arm:8s} {c['n']:4d} {c['b_same']:8.4f} "
              f"{c['b_sym']:8.4f} {c['auc_sym']:8.3f} {c['auc_t']:7.3f} "
              f"{c['cmin_sqrt']:9.3f} {c['cmin_t']:7.3f}")


if __name__ == "__main__":
    main()
