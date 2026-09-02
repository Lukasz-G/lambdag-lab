# Does windowed segmentation (10/20-token pseudo-sentences) or lower model order (N=3)
# rescue KN LambdaG in the SYMMETRIC short-length regime where sentence-N10 KN collapses?
# Both known and questioned truncated to L tokens (same budget as kn_sym.py), THEN
# re-chunked into w-token windows for the window arms. Caches kn500sym_{tag}_L{L}.jsonl.
#
#   python phase4/kn_sym_win.py

import json, sys, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent; ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
from lambdag import LambdaG
from sklearn.metrics import roc_auc_score

LENS = [1200, 600, 300, 150]

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

def rechunk(sents, w):
    toks = [t for s in sents for t in s]
    return [toks[i:i+w] for i in range(0, len(toks), w) if toks[i:i+w]]

def main():
    man = []
    with open(ROOT / "phase3" / "pairs500.tsv", encoding="utf-8") as f:
        next(f)
        for line in f:
            pid, lab, ka, qa = line.rstrip("\n").split("\t"); man.append((pid, int(lab)))
    pdir = ROOT / "phase3" / "pairs500"
    KQ = {pid: (read_tsv(pdir / f"{pid}_known.tsv"), read_tsv(pdir / f"{pid}_q.tsv")) for pid, _ in man}
    labs = {int(pid): lab for pid, lab in man}

    # (tag, model order, reference bank dir, window size or 0=natural sentences)
    CONFIGS = [("w10", 10, HERE / "bank_w10", 10),
               ("w20", 10, HERE / "bank_w20", 20),
               ("w30", 10, HERE / "bank_w30", 30)]

    scores = {}                                    # (tag, L) -> {id: lambda}
    for tag, N, bank, w in CONFIGS:
        ref = []
        for f in sorted(bank.glob("*.tsv")): ref += read_tsv(f)
        lg = LambdaG(N=N, r=30, engine="kn", random_state=0); lg.set_reference(ref)
        for L in LENS:
            fn = HERE / f"kn500sym_{tag}_L{L}.jsonl"
            if fn.exists():
                d = {}
                for line in open(fn, encoding="utf-8"):
                    r = json.loads(line); d[r["id"]] = r["kn"]
                scores[(tag, L)] = d; continue
            t0 = time.time(); d = {}
            for pid, lab in man:
                k, q = KQ[pid]
                kt, qt = trunc(k, L), trunc(q, L)
                if w > 0: kt, qt = rechunk(kt, w), rechunk(qt, w)
                if kt and qt: d[int(pid)] = float(lg.score(qt, kt, with_details=False).lambda_G)
            with open(fn, "w", encoding="utf-8") as f:
                for i, v in d.items(): f.write(json.dumps({"id": i, "kn": v}) + "\n")
            scores[(tag, L)] = d
            print(f"{tag} L={L} scored ({time.time()-t0:.0f}s)", flush=True)

    base = {}                                      # sentence-N10 baseline from kn_sym.py
    for L in LENS:
        d = {}
        for line in open(HERE / f"kn500sym_L{L}.jsonl", encoding="utf-8"):
            r = json.loads(line); d[r["id"]] = r["kn"]
        base[L] = d

    print(f"\n{'L':>6} {'KNsent':>7} {'KNw10':>7} {'KNw20':>7} {'KNw30':>7}")
    for L in LENS:
        row = [f"{L:>6}"]
        for d in [base[L]] + [scores[(t, L)] for t, _, _, _ in CONFIGS]:
            ids = sorted(d)
            y = np.array([labs[i] for i in ids]); s = np.array([d[i] for i in ids])
            row.append(f"{roc_auc_score(y, s):7.3f}")
        print(" ".join(row))

if __name__ == "__main__":
    main()
