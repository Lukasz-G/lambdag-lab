# KN lambda_G under the SYMMETRIC length protocol: known AND questioned both truncated
# to the same L tokens (L in {full,1200,600,300,150}) -> kn500sym_L{X}.jsonl, then the
# fused-by-length table KN vs KN+TM(sym) if tm_sym*.jsonl exist.
#
#   python phase4/kn_sym.py

import json, sys, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent; ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
from lambdag import LambdaG, LambdaGCalibrator, cllr
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

LENS = [0, 1200, 600, 300, 150]        # 0 = full (identical to kn500_L0: nothing truncated)

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
    lab_of = {int(pid): lab for pid, lab in man}
    pdir = ROOT / "phase3" / "pairs500"
    KQ = {pid: (read_tsv(pdir / f"{pid}_known.tsv"), read_tsv(pdir / f"{pid}_q.tsv")) for pid, _ in man}

    ref = []
    for f in sorted((ROOT / "phase1" / "bank").glob("*.tsv")): ref += read_tsv(f)
    lg = LambdaG(N=10, r=30, engine="kn", random_state=0); lg.set_reference(ref)

    kn = {}
    for L in LENS:
        fn = HERE / ("kn500_L0.jsonl" if L == 0 else f"kn500sym_L{L}.jsonl")
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
        print(f"KN sym L={L or 'full'} scored ({time.time()-t0:.0f}s)", flush=True)

    def jl(p):
        d = {}
        for line in open(p, encoding="utf-8"):
            r = json.loads(line); d[r["id"]] = (r["label"], r["tm"])
        return d
    tm = {}
    for L in LENS:
        p = HERE / ("tm_sym.jsonl" if L == 0 else f"tm_sym_L{L}.jsonl")
        if p.exists(): tm[L] = jl(p)
    if not tm:
        print("tm_sym*.jsonl not found yet - KN caches written, rerun for the table."); return

    print(f"\n{'L':>6} {'KN':>11} {'TMsym':>7} {'KN+TM':>13}")
    for L in LENS:
        if L not in tm: continue
        ids = sorted(set(kn[L]) & set(tm[L]))
        y = np.array([tm[L][i][0] for i in ids])
        KN = np.array([kn[L][i] for i in ids]); T = np.array([tm[L][i][1] for i in ids])
        a_kn, c_kn = cv(KN, y); a_t = roc_auc_score(y, T)
        a_f, c_f = cv(np.c_[KN, T], y)
        print(f"{L or 'full':>6} {a_kn:.3f}/{c_kn:.2f} {a_t:7.3f} {a_f:.4f}/{c_f:.3f}")

if __name__ == "__main__":
    main()
