# Decisive diagnostic: does HDC carry signal ORTHOGONAL to KN? Fuse HDC-cosine (and HDC-TM)
# with KN lambda_G via the calibrator (2-D score fusion, Brummer-du Preez), author-agnostic
# 5-fold CV, and compare AUC/Cllr to KN alone. If fusion > KN, HDC has a foothold to beat it.
#
#   python phase3/fuse500.py

import json, sys, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent; ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
from lambdag import LambdaG, LambdaGCalibrator, cllr
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

def read_tsv(p):
    out = []
    with open(p, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if line: out.append(line.split("\t"))
    return out

def run_kn(man):
    ref = []
    for f in sorted((ROOT / "phase1" / "bank").glob("*.tsv")): ref += read_tsv(f)
    lg = LambdaG(N=10, r=30, engine="kn", random_state=0); lg.set_reference(ref)
    print(f"KN scoring {len(man)} pairs ...", flush=True); out = {}; t0 = time.time()
    for i, (pid, lab) in enumerate(man):
        k = read_tsv(HERE / "pairs500" / f"{pid}_known.tsv"); q = read_tsv(HERE / "pairs500" / f"{pid}_q.tsv")
        if k and q: out[int(pid)] = float(lg.score(q, k, with_details=False).lambda_G)
    print(f"KN done ({time.time()-t0:.0f}s)", flush=True); return out

# CV AUC + Cllr for a (possibly multi-column) score matrix X calibrated to an LR
def cv_metrics(X, y, seed=0):
    X = np.atleast_2d(np.asarray(X, float));
    if X.shape[0] != len(y): X = X.T
    y = np.asarray(y, int); LR = np.empty(len(y))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(X, y):
        cal = LambdaGCalibrator().fit(X[tr], y[tr]); LR[te] = cal.transform(X[te])
    return roc_auc_score(y, LR), cllr(LR[y == 1], LR[y == 0])

def main():
    man = []
    with open(HERE / "pairs500.tsv", encoding="utf-8") as f:
        next(f)
        for line in f:
            pid, lab, ka, qa = line.rstrip("\n").split("\t"); man.append((pid, int(lab)))
    hdc = {}
    with open(HERE / "hdc_scores500.jsonl", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line); hdc[r["id"]] = r
    kn = run_kn(man)

    ids = sorted(set(hdc) & set(kn)); y = np.array([hdc[i]["label"] for i in ids])
    knv = np.array([kn[i] for i in ids])
    cos = np.array([hdc[i]["cos_rnd"] for i in ids]); tm = np.array([hdc[i]["tm_rnd"] for i in ids])
    print(f"\n=== {len(ids)} pairs ===")
    a, c = cv_metrics(knv, y);                 print(f"  KN alone            AUC={a:.3f}  Cllr={c:.3f}")
    a, c = cv_metrics(cos, y);                 print(f"  HDC cosine alone    AUC={a:.3f}  Cllr={c:.3f}")
    a, c = cv_metrics(np.c_[knv, cos], y);     print(f"  FUSE KN + cosine    AUC={a:.3f}  Cllr={c:.3f}")
    a, c = cv_metrics(np.c_[knv, tm], y);      print(f"  FUSE KN + TM        AUC={a:.3f}  Cllr={c:.3f}")
    a, c = cv_metrics(np.c_[knv, cos, tm], y); print(f"  FUSE KN + cos + TM  AUC={a:.3f}  Cllr={c:.3f}")
    print("\n(fusion > KN  =>  HDC has orthogonal signal.)")

if __name__ == "__main__":
    main()
