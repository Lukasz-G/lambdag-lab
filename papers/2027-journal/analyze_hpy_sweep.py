# HPY parameter sweep table: theta x table-estimator x discount vs the KN
# anchor, on lithuanian/hungarian/english novels at L in {full, 600}.
#   python experiments/analyze_hpy_sweep.py
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SCORES = HERE / "scores"
sys.path.insert(0, str(HERE.parent))
from lambdag import cllr, cllr_min  # noqa: E402

ENGS = ["kn", "hpy_t0_min", "hpy_t1_min", "hpy_t5_min", "hpy_t10_min",
        "hpy_t0_exp", "hpy_t1_exp", "hpy_t5_exp", "hpy_t10_exp",
        "hpy_t0_min_d50", "hpy_t0_min_d90", "hpy_t1_exp_d50", "hpy_t1_exp_d90"]
DS = ["lithuanian_novels", "hungarian_novels", "english_novels"]


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    from sklearn.metrics import roc_auc_score
    for L in (0, 600):
        print(f"--- L={L or 'full'}   (AUC / Cllr_sqrt / floor) ---")
        print(f"{'variant':16s}" + "".join(f"{d.split('_')[0]:>22s}" for d in DS))
        for e in ENGS:
            row = f"{e:16s}"
            for ds in DS:
                fn = SCORES / f"{ds}__{e}__sent__L{L}.jsonl"
                if not fn.exists():
                    row += f"{'--':>22s}"
                    continue
                recs = [json.loads(l) for l in open(fn, encoding="utf-8")]
                y = np.array([r["label"] for r in recs])
                lam = np.array([r["lambda_G"] for r in recs])
                s = np.array([r["sqrt"] for r in recs])
                row += (f"  {roc_auc_score(y, lam):.3f}/"
                        f"{cllr(s[y == 1], s[y == 0]):5.2f}/"
                        f"{cllr_min(s[y == 1], s[y == 0]):4.2f}")
            print(row)
        print()


if __name__ == "__main__":
    main()
