# Final comparison with the SCALED TM: KN vs cosine vs TM(scaled, author-disjoint by
# construction) and all fusions, on the same 500 pairs. AUC + CV-calibrated Cllr.
#
#   python phase4/fuse_scaled.py

import json, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent; ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
from lambdag import LambdaGCalibrator, cllr
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

def jl(path, keys):
    d = {}
    for line in open(path, encoding="utf-8"):
        r = json.loads(line); d[r["id"]] = tuple(r[k] for k in keys)
    return d

def cv(X, y, seed=0):
    X = np.atleast_2d(np.asarray(X, float))
    if X.shape[0] != len(y): X = X.T
    y = np.asarray(y, int); LR = np.empty(len(y))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(X, y):
        LR[te] = LambdaGCalibrator().fit(X[tr], y[tr]).transform(X[te])
    return roc_auc_score(y, LR), cllr(LR[y == 1], LR[y == 0])

def main():
    kn = jl(ROOT / "phase3" / "kn500.jsonl", ["kn"])
    cosd = jl(ROOT / "phase3" / "hdc_char500.jsonl", ["label", "cos_word", "cos_char"])
    tmd = jl(HERE / (sys.argv[1] if len(sys.argv) > 1 else "tm_scaled_scores.jsonl"), ["label", "tm"])
    ids = sorted(set(kn) & set(cosd) & set(tmd))
    y = np.array([cosd[i][0] for i in ids])
    KN = np.array([kn[i][0] for i in ids])
    CW = np.array([cosd[i][1] for i in ids]); CC = np.array([cosd[i][2] for i in ids])
    T = np.array([tmd[i][1] for i in ids])
    print(f"=== {len(ids)} pairs ===")
    for name, X in [("KN alone", KN), ("cosine char", CC), ("TM scaled", T),
                    ("TM + cosines", np.c_[T, CW, CC]),
                    ("KN + cosine", np.c_[KN, CC]), ("KN + TM", np.c_[KN, T]),
                    ("KN + TM + cosines", np.c_[KN, T, CW, CC])]:
        a, c = cv(X, y); print(f"  {name:20} AUC={a:.4f}  Cllr={c:.3f}")

if __name__ == "__main__":
    main()
