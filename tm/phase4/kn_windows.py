# KN LambdaG under fixed-window segmentation (10/20/30-token pseudo-sentences), same windowed
# units as the HDC-TM side: reference bank, known and questioned docs all re-chunked identically.
# Writes kn500_w{N}.jsonl per window and prints AUC.
#
#   python phase4/kn_windows.py            (expects rechunked dirs bank_wN / pairs500_wN)

import json, sys, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent; ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
from lambdag import LambdaG
from sklearn.metrics import roc_auc_score

WINDOWS = [10, 20, 30]

def read_tsv(p):
    out = []
    for line in open(p, encoding="utf-8"):
        line = line.rstrip("\n")
        if line: out.append(line.split("\t"))
    return out

def main():
    man = []
    with open(ROOT / "phase3" / "pairs500.tsv", encoding="utf-8") as f:
        next(f)
        for line in f:
            pid, lab, ka, qa = line.rstrip("\n").split("\t"); man.append((pid, int(lab)))

    for w in WINDOWS:
        ref = []
        for f in sorted((HERE / f"bank_w{w}").glob("*.tsv")): ref += read_tsv(f)
        lg = LambdaG(N=10, r=30, engine="kn", random_state=0); lg.set_reference(ref)
        pdir = HERE / f"pairs500_w{w}"
        out = {}; t0 = time.time()
        for pid, lab in man:
            k = read_tsv(pdir / f"{pid}_known.tsv"); q = read_tsv(pdir / f"{pid}_q.tsv")
            if k and q: out[int(pid)] = (lab, float(lg.score(q, k, with_details=False).lambda_G))
        y = np.array([v[0] for v in out.values()]); s = np.array([v[1] for v in out.values()])
        print(f"w={w:2}  KN AUC = {roc_auc_score(y, s):.4f}   ({len(out)} pairs, {time.time()-t0:.0f}s)", flush=True)
        with open(HERE / f"kn500_w{w}.jsonl", "w", encoding="utf-8") as f:
            for i, (lab, lam) in out.items():
                f.write(json.dumps({"id": i, "label": lab, "kn": lam}) + "\n")

if __name__ == "__main__":
    main()
