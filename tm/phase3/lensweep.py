# Length sweep (KN side + comparison): score KN with the QUESTIONED text truncated to L tokens,
# for each L, and compare AUC to the HDC char/word cosines (from lensweep.jl) and their fusion.
# The question: does KN fall faster than HDC as the fragment shrinks (a crossover where HDC wins)?
#
#   python phase3/lensweep.py

import json, sys, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent; ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
from lambdag import LambdaG, LambdaGCalibrator, cllr
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

LENS = [10000, 600, 300, 150, 75]

def read_tsv(p):
    out = []
    for line in open(p, encoding="utf-8"):
        line = line.rstrip("\n")
        if line: out.append(line.split("\t"))
    return out

def truncate(sents, L):
    out, n = [], 0
    for s in sents:
        if n + len(s) <= L: out.append(s); n += len(s)
        else:
            take = L - n
            if take > 0: out.append(s[:take])
            break
    return out

def cv_auc(X, y, seed=0):
    X = np.atleast_2d(np.asarray(X, float))
    if X.shape[0] != len(y): X = X.T
    y = np.asarray(y, int); LR = np.empty(len(y))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(X, y):
        LR[te] = LambdaGCalibrator().fit(X[tr], y[tr]).transform(X[te])
    return roc_auc_score(y, LR)

def main():
    man = []
    with open(HERE / "pairs500.tsv", encoding="utf-8") as f:
        next(f)
        for line in f:
            pid, lab, ka, qa = line.rstrip("\n").split("\t"); man.append((pid, int(lab)))
    hdc = {}
    for line in open(HERE / "hdc_len500.jsonl", encoding="utf-8"):
        r = json.loads(line); hdc[r["id"]] = r

    ref = []
    for f in sorted((ROOT / "phase1" / "bank").glob("*.tsv")): ref += read_tsv(f)
    lg = LambdaG(N=10, r=30, engine="kn", random_state=0); lg.set_reference(ref)

    # precache known/q sentences
    KQ = {}
    for pid, lab in man:
        KQ[pid] = (read_tsv(HERE / "pairs500" / f"{pid}_known.tsv"), read_tsv(HERE / "pairs500" / f"{pid}_q.tsv"))

    print(f"{'L':>6}  {'KN':>6} {'HDCchar':>7} {'HDCword':>7} {'KN+char':>7} {'KN+wc':>6}")
    for L in LENS:
        kn = {}; t0 = time.time()
        for pid, lab in man:
            k, q = KQ[pid]
            if not k or not q: continue
            qt = truncate(q, L)
            if not qt: continue
            kn[int(pid)] = float(lg.score(qt, k, with_details=False).lambda_G)
        ids = sorted(set(kn) & set(hdc)); y = np.array([hdc[i]["label"] for i in ids])
        knv = np.array([kn[i] for i in ids])
        cc = np.array([hdc[i][f"cc_{L}"] for i in ids]); cw = np.array([hdc[i][f"cw_{L}"] for i in ids])
        a_kn = roc_auc_score(y, knv); a_c = roc_auc_score(y, cc); a_w = roc_auc_score(y, cw)
        a_fc = cv_auc(np.c_[knv, cc], y); a_fwc = cv_auc(np.c_[knv, cw, cc], y)
        print(f"{L:>6}  {a_kn:6.3f} {a_c:7.3f} {a_w:7.3f} {a_fc:7.3f} {a_fwc:6.3f}   ({time.time()-t0:.0f}s)", flush=True)
    print("\n(crossover: HDCchar/KN+char > KN at short L => HDC wins on short fragments)")

if __name__ == "__main__":
    main()
