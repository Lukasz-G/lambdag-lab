# Does the CHAR-trigram feature add signal orthogonal to KN? Fuse KN with word / char / word+char
# HDC-cosine scores (calibrator 2-D fusion, 5-fold CV) and compare AUC/Cllr to KN alone and to
# the earlier KN+word fusion (0.944). KN scores cached to kn500.jsonl on first run.

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

def get_kn(man):
    cache = HERE / "kn500.jsonl"
    if cache.exists():
        d = {}
        for line in open(cache, encoding="utf-8"):
            r = json.loads(line); d[r["id"]] = r["kn"]
        return d
    ref = []
    for f in sorted((ROOT / "phase1" / "bank").glob("*.tsv")): ref += read_tsv(f)
    lg = LambdaG(N=10, r=30, engine="kn", random_state=0); lg.set_reference(ref)
    print(f"KN scoring {len(man)} pairs ...", flush=True); out = {}; t0 = time.time()
    for pid, lab in man:
        k = read_tsv(HERE / "pairs500" / f"{pid}_known.tsv"); q = read_tsv(HERE / "pairs500" / f"{pid}_q.tsv")
        if k and q: out[int(pid)] = float(lg.score(q, k, with_details=False).lambda_G)
    with open(cache, "w", encoding="utf-8") as f:
        for i, v in out.items(): f.write(json.dumps({"id": i, "kn": v}) + "\n")
    print(f"KN done ({time.time()-t0:.0f}s)", flush=True); return out

def cv(X, y, seed=0):
    X = np.atleast_2d(np.asarray(X, float))
    if X.shape[0] != len(y): X = X.T
    y = np.asarray(y, int); LR = np.empty(len(y))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(X, y):
        LR[te] = LambdaGCalibrator().fit(X[tr], y[tr]).transform(X[te])
    return roc_auc_score(y, LR), cllr(LR[y == 1], LR[y == 0])

def main():
    man = []
    with open(HERE / "pairs500.tsv", encoding="utf-8") as f:
        next(f)
        for line in f:
            pid, lab, ka, qa = line.rstrip("\n").split("\t"); man.append((pid, int(lab)))
    hc = {}
    for line in open(HERE / "hdc_char500.jsonl", encoding="utf-8"):
        r = json.loads(line); hc[r["id"]] = r
    kn = get_kn(man)
    ids = sorted(set(hc) & set(kn)); y = np.array([hc[i]["label"] for i in ids])
    knv = np.array([kn[i] for i in ids])
    W = np.array([hc[i]["cos_word"] for i in ids]); C = np.array([hc[i]["cos_char"] for i in ids])
    print(f"\n=== {len(ids)} pairs ===")
    for name, X in [("KN alone", knv), ("KN + word", np.c_[knv, W]), ("KN + char", np.c_[knv, C]),
                    ("KN + word + char", np.c_[knv, W, C])]:
        a, c = cv(X, y); print(f"  {name:18} AUC={a:.3f}  Cllr={c:.3f}")
    print("\n(KN+char > KN+word => char is MORE orthogonal to KN than word n-grams.)")

if __name__ == "__main__":
    main()
