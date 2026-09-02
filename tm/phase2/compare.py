# Phase 2: calibrate the HDC->TM lambda_G and put it head-to-head with the Kneser-Ney
# LambdaG oracle on the SAME 80 av_test pairs (same masked known/questioned docs, same
# reference authors). Reports discrimination (AUC), the calibration-free floor (Cllr_min),
# and the honest cross-validated calibrated cost (Cllr) for both engines.
#
#   python phase2/compare.py
#
# HDC lambda_G comes from phase1/lambdas.jsonl (already computed in Julia). KN lambda_G is
# computed here with lambdag.py's own LambdaG(engine="kn") on the identical inputs.

import json, sys, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
from lambdag import LambdaG, LambdaGCalibrator, cllr, cllr_min
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

P1 = ROOT / "phase1"

def read_tsv(path):
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if line:
                out.append(line.split("\t"))
    return out

def load_manifest():
    rows = []
    with open(P1 / "pairs.tsv", encoding="utf-8") as f:
        next(f)
        for line in f:
            pid, lab, authors = line.rstrip("\n").split("\t")
            rows.append((pid, int(lab)))
    return rows

# ---- cross-validated calibrated Cllr (honest: calibrator never sees its own test case) ----
def cv_cllr(lam, y, n_splits=5, seed=0):
    lam = np.asarray(lam, float); y = np.asarray(y, int)
    Lam = np.full_like(lam, np.nan)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for tr, te in skf.split(lam, y):
        cal = LambdaGCalibrator().fit(lam[tr], y[tr])
        Lam[te] = cal.transform(lam[te])
    return cllr(Lam[y == 1], Lam[y == 0], base=10.0), Lam

def metrics(name, lam, y):
    lam = np.asarray(lam, float); y = np.asarray(y, int)
    auc = roc_auc_score(y, lam)
    cmin = cllr_min(lam[y == 1], lam[y == 0])
    ccal, Lam = cv_cllr(lam, y)
    acc = float(((Lam > 0).astype(int) == y).mean())        # decision at calibrated LR > 0
    print(f"  {name:10}  AUC={auc:.3f}   Cllr_min={cmin:.3f}   Cllr(cv)={ccal:.3f}   acc@0={acc:.3f}")
    return dict(name=name, auc=auc, cllr_min=cmin, cllr=ccal, acc=acc)

def run_kn(manifest):
    # reference pool = all bank authors' masked sentences (same authors HDC's ref used)
    ref = []
    for f in sorted((P1 / "bank").glob("*.tsv")):
        ref += read_tsv(f)
    lg = LambdaG(N=10, r=30, engine="kn", random_state=0)
    lg.set_reference(ref)
    print(f"KN reference: {len(ref)} sentences; scoring {len(manifest)} pairs ...")
    ids, ys, lams = [], [], []
    t0 = time.time()
    for i, (pid, lab) in enumerate(manifest):
        k = read_tsv(P1 / "pairs" / f"{pid}_known.tsv")
        q = read_tsv(P1 / "pairs" / f"{pid}_q.tsv")
        if not k or not q:
            continue
        res = lg.score(q, k, with_details=False)
        ids.append(int(pid)); ys.append(lab); lams.append(float(res.lambda_G))
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(manifest)}  ({time.time()-t0:.0f}s)")
    print(f"KN scoring done in {time.time()-t0:.0f}s")
    return ids, ys, lams

def main():
    manifest = load_manifest()

    # HDC->TM lambda_G (from Julia)
    hdc = {}
    with open(P1 / "lambdas.jsonl", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line); hdc[r["id"]] = (r["label"], r["lambda"])

    # KN lambda_G (computed here on the same pairs)
    kn_ids, kn_y, kn_lam = run_kn(manifest)

    # align both methods on the SAME pair ids
    ids = [i for i in kn_ids if i in hdc]
    y   = np.array([hdc[i][0] for i in ids])
    hdc_lam = np.array([hdc[i][1] for i in ids])
    kn_lam  = np.array([kn_lam[kn_ids.index(i)] for i in ids])

    with open(HERE / "kn_lambdas.jsonl", "w", encoding="utf-8") as f:
        for i, l in zip(kn_ids, kn_lam):
            f.write(json.dumps({"id": i, "label": int(hdc[i][0]), "lambda": float(l)}) + "\n")

    print(f"\n=== {len(ids)} pairs, same/diff = {int(y.sum())}/{int((1-y).sum())} ===")
    print("  method      discrimination + calibrated forensic cost")
    metrics("HDC->TM", hdc_lam, y)
    metrics("KN oracle", kn_lam, y)
    print("\n(Cllr: 0=perfect, ~1=uninformative, >1=misleading. Cllr_min=discrimination floor.)")

if __name__ == "__main__":
    main()
