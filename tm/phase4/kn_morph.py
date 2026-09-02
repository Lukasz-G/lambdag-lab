# KN A/B: morph-enriched masking vs current POSNoise, best KN config (sentence, N=10),
# full-length + symmetric lengths. Reference = bank_morph, pairs = pairs500_morph.
# Baselines are the cached kn500_L0.jsonl / kn500sym_L{X}.jsonl scores.
#
#   python phase4/kn_morph.py

import json, sys, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent; ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
from lambdag import LambdaG, LambdaGCalibrator, cllr
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

LENS = [0, 1200, 600, 300, 150]

def read_tsv(p):
    out = []
    for line in open(p, encoding="utf-8"):
        line = line.rstrip("\n")
        if line: out.append(line.split("\t"))
    return out

def trunc(sents, L):
    if L <= 0: return sents
    out, n = [], 0
    for s in sents:
        if n + len(s) <= L: out.append(s); n += len(s)
        else:
            t = L - n
            if t > 0: out.append(s[:t])
            break
    return out

def cv(X, y, seed=0):
    X = np.atleast_2d(np.asarray(X, float))
    if X.shape[0] != len(y): X = X.T
    y = np.asarray(y, int); LR = np.empty(len(y))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(X, y):
        LR[te] = LambdaGCalibrator().fit(X[tr], y[tr]).transform(X[te])
    return roc_auc_score(y, LR), cllr(LR[y == 1], LR[y == 0])

def main():
    man = []
    with open(ROOT / "phase3" / "pairs500.tsv", encoding="utf-8") as f:
        next(f)
        for line in f:
            pid, lab, ka, qa = line.rstrip("\n").split("\t"); man.append((pid, int(lab)))
    pdir = HERE / "pairs500_morph"
    KQ = {pid: (read_tsv(pdir / f"{pid}_known.tsv"), read_tsv(pdir / f"{pid}_q.tsv")) for pid, _ in man}

    ref = []
    for f in sorted((HERE / "bank_morph").glob("*.tsv")): ref += read_tsv(f)
    lg = LambdaG(N=10, r=30, engine="kn", random_state=0); lg.set_reference(ref)

    kn = {}
    for L in LENS:
        fn = HERE / f"kn500morph_L{L}.jsonl"
        if fn.exists():
            d = {}
            for line in open(fn, encoding="utf-8"):
                r = json.loads(line); d[r["id"]] = r["kn"]
            kn[L] = d; continue
        t0 = time.time(); d = {}
        for pid, lab in man:
            k, q = KQ[pid]
            kt, qt = trunc(k, L), trunc(q, L)
            if kt and qt: d[int(pid)] = float(lg.score(qt, kt, with_details=False).lambda_G)
        with open(fn, "w", encoding="utf-8") as f:
            for i, v in d.items(): f.write(json.dumps({"id": i, "kn": v}) + "\n")
        kn[L] = d
        print(f"KN morph L={L or 'full'} scored ({time.time()-t0:.0f}s)", flush=True)

    lab_of = {int(pid): lab for pid, lab in man}
    print(f"\n{'L':>6} {'KNcur':>11} {'KNmorph':>11}")
    for L in LENS:
        base = HERE / ("kn500_L0.jsonl" if L == 0 else f"kn500sym_L{L}.jsonl")
        b = {}
        for line in open(base, encoding="utf-8"):
            r = json.loads(line); b[r["id"]] = r["kn"]
        ids = sorted(set(kn[L]) & set(b) & set(lab_of))
        y = np.array([lab_of[i] for i in ids])
        B = np.array([b[i] for i in ids]); M = np.array([kn[L][i] for i in ids])
        ab, cb = cv(B, y); am, cm = cv(M, y)
        print(f"{L or 'full':>6} {ab:.4f}/{cb:.3f} {am:.4f}/{cm:.3f}")

if __name__ == "__main__":
    main()
