# Unified head-to-head on the same 80 av_test pairs: KN oracle vs the failed LM vs the direct
# HDC verifiers (unsupervised cosine + supervised TM), for RANDOM and SUBWORD atoms.
# Reports AUC, Cllr_min (calibration-free discrimination floor), and 5-fold CV Cllr.
#
#   python phase3/compare_all.py

import json, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
from lambdag import LambdaGCalibrator, cllr, cllr_min
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

def load(path, key):
    d = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line); d[r["id"]] = (r["label"], r[key])
    return d

def cv_cllr(lam, y, seed=0):
    lam = np.asarray(lam, float); y = np.asarray(y, int); Lam = np.empty_like(lam)
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(lam, y):
        Lam[te] = LambdaGCalibrator().fit(lam[tr], y[tr]).transform(lam[te])
    return cllr(Lam[y == 1], Lam[y == 0])

def report(name, lam, y):
    lam = np.asarray(lam, float); y = np.asarray(y, int)
    print(f"  {name:24} AUC={roc_auc_score(y,lam):.3f}  Cllr_min={cllr_min(lam[y==1],lam[y==0]):.3f}  Cllr(cv)={cv_cllr(lam,y):.3f}")

def main():
    P1, P2 = ROOT / "phase1", ROOT / "phase2"
    kn  = load(P2 / "kn_lambdas.jsonl", "lambda")
    lm  = load(P1 / "lambdas.jsonl", "lambda")
    hdc = {}
    with open(HERE / "hdc_scores.jsonl", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line); hdc[r["id"]] = r

    ids = sorted(set(kn) & set(lm) & set(hdc))
    y = [kn[i][0] for i in ids]
    print(f"=== {len(ids)} pairs, same/diff = {sum(y)}/{len(y)-sum(y)} ===")
    report("KN oracle (LM)",        [kn[i][1] for i in ids], y)
    report("HDC-LM lambda_G",       [lm[i][1] for i in ids], y)
    report("HDC cosine  RANDOM",    [hdc[i]["cos_rnd"] for i in ids], y)
    report("HDC cosine  SUBWORD",   [hdc[i]["cos_sub"] for i in ids], y)
    report("HDC TM(cv)  RANDOM",    [hdc[i]["tm_rnd"] for i in ids], y)
    report("HDC TM(cv)  SUBWORD",   [hdc[i]["tm_sub"] for i in ids], y)
    print("\n(Cllr: 0=perfect, ~1=uninformative. Cllr_min=discrimination floor.)")

if __name__ == "__main__":
    main()
