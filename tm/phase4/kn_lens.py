# Save KN lambda_G at truncated questioned lengths (full/600/300/150/75) -> kn500_L{X}.jsonl,
# then print the final fused-by-length table: KN vs KN + multiscale-TM (+ cosine).
#
#   python phase4/kn_lens.py

import json, sys, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent; ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
from lambdag import LambdaG, LambdaGCalibrator, cllr
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

LENS = [0, 600, 300, 150, 75]          # 0 = full

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
    pdir = ROOT / "phase3" / "pairs500"
    KQ = {pid: (read_tsv(pdir / f"{pid}_known.tsv"), read_tsv(pdir / f"{pid}_q.tsv")) for pid, _ in man}

    ref = []
    for f in sorted((ROOT / "phase1" / "bank").glob("*.tsv")): ref += read_tsv(f)
    lg = LambdaG(N=10, r=30, engine="kn", random_state=0); lg.set_reference(ref)

    kn = {}
    for L in LENS:
        fn = HERE / f"kn500_L{L}.jsonl"
        if fn.exists():
            d = {}
            for line in open(fn, encoding="utf-8"):
                r = json.loads(line); d[r["id"]] = r["kn"]
            kn[L] = d; continue
        t0 = time.time(); d = {}
        for pid, lab in man:
            k, q = KQ[pid]
            qt = trunc(q, L)
            if k and qt: d[int(pid)] = float(lg.score(qt, k, with_details=False).lambda_G)
        with open(fn, "w", encoding="utf-8") as f:
            for i, v in d.items(): f.write(json.dumps({"id": i, "kn": v}) + "\n")
        kn[L] = d
        print(f"KN L={L or 'full'} scored ({time.time()-t0:.0f}s)", flush=True)

    # TM multiscale + cosine per length
    def jl(p):
        d = {}
        for line in open(p, encoding="utf-8"):
            r = json.loads(line); d[r["id"]] = (r["label"], r["tm"])
        return d
    tm = {0: jl(HERE / "tm_multi.jsonl")}
    for L in LENS[1:]: tm[L] = jl(HERE / f"tm_multi_L{L}.jsonl")
    cosd = {}
    for line in open(ROOT / "phase3" / "hdc_len500.jsonl", encoding="utf-8"):
        r = json.loads(line); cosd[r["id"]] = r
    cosL = {0: 10000, 600: 600, 300: 300, 150: 150, 75: 75}

    print(f"\n{'L':>6} {'KN':>7} {'TMms':>7} {'KN+TM':>13} {'KN+TM+cos':>15}")
    for L in LENS:
        ids = sorted(set(kn[L]) & set(tm[L]) & set(cosd))
        y = np.array([tm[L][i][0] for i in ids])
        KN = np.array([kn[L][i] for i in ids]); T = np.array([tm[L][i][1] for i in ids])
        C = np.array([cosd[i][f"cc_{cosL[L]}"] for i in ids])
        a_kn, c_kn = cv(KN, y); a_t = roc_auc_score(y, T)
        a_f, c_f = cv(np.c_[KN, T], y); a_g, c_g = cv(np.c_[KN, T, C], y)
        print(f"{L or 'full':>6} {a_kn:.3f}/{c_kn:.2f} {a_t:7.3f} {a_f:.4f}/{c_f:.3f} {a_g:.4f}/{c_g:.3f}")

if __name__ == "__main__":
    main()
