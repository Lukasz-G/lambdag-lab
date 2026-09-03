# Analysis of the borrowed-reference grid (run_xref.py): per (case language,
# donor arm) the discrimination and calibration of three statistics --
#   sym    the symmetrised lambda (mean over per-author donor grammars)
#   sqrt   sym / sqrt(N)
#   t      cohort studentisation mean(lam_j) / sd(lam_j)
# -- plus the corpus-adequacy reading b_sym (mean per-token lambda on
# different-author rows), and the gauge-vs-degradation correlation across
# donor arms.
#
#   python experiments/analyze_xref.py

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
XS = HERE.parent / "scores" / "xref"
sys.path.insert(0, str(HERE.parent))
from lambdag import cllr_min  # noqa: E402

from scipy import stats as sps  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

DATASETS = ["german_novels", "english_novels", "french_novels",
            "polish_novels", "czech_novels", "hungarian_novels"]


def cell(fn):
    rows = [json.loads(l) for l in open(fn, encoding="utf-8")]
    y = np.array([r["within"] for r in rows])
    lam = np.array([r["lambda_G"] for r in rows])
    n = np.array([r["n_q"] for r in rows], dtype=float)
    t = np.array([np.mean(r["lam_j"]) / (np.std(r["lam_j"]) + 1e-9)
                  for r in rows])
    sq = lam / np.sqrt(n)
    out = {"n": len(y), "b_sym": float(np.mean(lam[y == 0] / n[y == 0]))}
    for name, s in (("sym", lam), ("sqrt", sq), ("t", t)):
        out[f"auc_{name}"] = float(roc_auc_score(y, s))
        out[f"cmin_{name}"] = float(cllr_min(s[y == 1], s[y == 0]))
    return out


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    res = {}
    for fn in sorted(XS.glob("*__xref-*__L1000.jsonl")):
        ds, ref = fn.name.split("__")[0], fn.name.split("__")[1][5:]
        res[(ds, ref)] = cell(fn)
    print(f"{'case':10s} {'donors':10s} {'n':>4s} {'b_sym':>8s} "
          f"{'AUC sym':>8s} {'AUC sqrt':>9s} {'AUC t':>7s} "
          f"{'Cmin sqrt':>9s} {'Cmin t':>7s}")
    degr, gauges = [], []
    for ds in DATASETS:
        arms = sorted(r for d, r in res if d == ds)
        arms = (["native"] + [a for a in arms if a not in ("native", "pooled")]
                + (["pooled"] if "pooled" in arms else []))
        for ref in arms:
            c = res.get((ds, ref))
            if not c:
                continue
            short = ref if ref in ("native", "pooled") else ref.split("_")[0]
            print(f"{ds.split('_')[0]:10s} {short:10s} {c['n']:4d} "
                  f"{c['b_sym']:8.4f} {c['auc_sym']:8.3f} "
                  f"{c['auc_sqrt']:9.3f} {c['auc_t']:7.3f} "
                  f"{c['cmin_sqrt']:9.3f} {c['cmin_t']:7.3f}")
            if ref != "native" and (ds, "native") in res:
                base = res[(ds, "native")]
                degr.append(base["auc_t"] - c["auc_t"])
                gauges.append(abs(c["b_sym"] - base["b_sym"]))
        print()
    if len(degr) >= 6:
        rho, p = sps.spearmanr(gauges, degr)
        print(f"gauge shift |b_sym - b_sym(native)| vs AUC-t degradation "
              f"across {len(degr)} borrowed arms: Spearman rho={rho:.3f} "
              f"(p={p:.4f})")


if __name__ == "__main__":
    main()
