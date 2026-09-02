# Final head-to-head on the SAME 500 pairs: KN oracle vs HDC direct verifiers
# (unsupervised cosine + author-disjoint-CV TM), RANDOM & SUBWORD atoms. AUC + Cllr.
#
#   python phase3/compare500.py

import json, sys, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
from lambdag import LambdaG, LambdaGCalibrator, cllr, cllr_min
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

def read_tsv(p):
    out = []
    with open(p, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if line:
                out.append(line.split("\t"))
    return out

def cv_cllr(lam, y, seed=0):
    lam = np.asarray(lam, float); y = np.asarray(y, int); Lam = np.empty_like(lam)
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(lam, y):
        Lam[te] = LambdaGCalibrator().fit(lam[tr], y[tr]).transform(lam[te])
    return cllr(Lam[y == 1], Lam[y == 0])

def report(name, lam, y):
    lam = np.asarray(lam, float); y = np.asarray(y, int)
    print(f"  {name:22} AUC={roc_auc_score(y,lam):.3f}  Cllr_min={cllr_min(lam[y==1],lam[y==0]):.3f}  Cllr(cv)={cv_cllr(lam,y):.3f}")

def run_kn(manifest):
    ref = []
    for f in sorted((ROOT / "phase1" / "bank").glob("*.tsv")):
        ref += read_tsv(f)
    lg = LambdaG(N=10, r=30, engine="kn", random_state=0); lg.set_reference(ref)
    print(f"KN: scoring {len(manifest)} pairs (ref {len(ref)} sents) ...")
    out = {}; t0 = time.time()
    for i, (pid, lab) in enumerate(manifest):
        k = read_tsv(HERE / "pairs500" / f"{pid}_known.tsv"); q = read_tsv(HERE / "pairs500" / f"{pid}_q.tsv")
        if not k or not q:
            continue
        out[int(pid)] = float(lg.score(q, k, with_details=False).lambda_G)
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(manifest)} ({time.time()-t0:.0f}s)")
    print(f"KN done ({time.time()-t0:.0f}s)")
    return out

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

    ids = sorted(set(hdc) & set(kn))
    y = [hdc[i]["label"] for i in ids]
    print(f"\n=== {len(ids)} pairs, same/diff = {sum(y)}/{len(y)-sum(y)} ===")
    report("KN oracle",         [kn[i] for i in ids], y)
    report("HDC cosine RANDOM", [hdc[i]["cos_rnd"] for i in ids], y)
    report("HDC cosine SUBWORD",[hdc[i]["cos_sub"] for i in ids], y)
    report("HDC TM(adCV) RANDOM",[hdc[i]["tm_rnd"] for i in ids], y)
    report("HDC TM(adCV) SUBWORD",[hdc[i]["tm_sub"] for i in ids], y)
    print("\n(Cllr: 0=perfect, ~1=uninformative. TM = author-disjoint CV.)")

if __name__ == "__main__":
    main()
