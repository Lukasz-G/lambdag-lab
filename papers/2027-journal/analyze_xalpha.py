# Analysis of the universal-alphabet LambdaG pilot (run_lambdag_catsrank.py):
# per dataset and arm (surface vs catsrank), AUC for lambda_G and lambda/sqrt(N),
# plus Cllr_min of the sqrt statistic (PAV, from lambdag.py's own machinery).
#
#   python experiments/analyze_xalpha.py

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SCORES = HERE.parent / "scores"
sys.path.insert(0, str(HERE.parent))
from lambdag import cllr_min  # noqa: E402

from sklearn.metrics import roc_auc_score  # noqa: E402


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    files = sorted(SCORES.glob("*__xalpha__L*.jsonl"))
    print(f"{'dataset':18s} {'arm':9s} {'n':>5s} {'AUC lam':>8s} "
          f"{'AUC sqrt':>9s} {'Cllr_min':>9s}")
    for fn in files:
        ds, arm = fn.name.split("__")[:2]
        rows = [json.loads(l) for l in open(fn, encoding="utf-8")]
        if not rows:
            continue
        y = np.array([r["within"] for r in rows])
        lam = np.array([r["lambda_G"] for r in rows])
        sq = np.array([r["sqrt"] for r in rows])
        cm = cllr_min(sq[y == 1], sq[y == 0])
        print(f"{ds:18s} {arm:9s} {len(y):5d} {roc_auc_score(y, lam):8.3f} "
              f"{roc_auc_score(y, sq):9.3f} {cm:9.3f}")


if __name__ == "__main__":
    main()
