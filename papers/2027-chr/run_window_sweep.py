# Window-size sweep requested by the user (2026-08-14): does a LARGER window
# (w=50, 100) buy anything over w=10/20/30 or sentences, at symmetric evidence
# lengths 500/1000/2000/5000?
#
# Four languages (English, German, Polish, Lithuanian novels), bank-built cases
# exactly as in run_longtexts.py: known = an author's first L tokens, questioned
# = the next L, disjoint; same/different balanced via rotations; per-case
# reference pool excludes both case authors. Known, questioned AND reference
# streams are re-tiled into w-token windows (sentences for seg="sent").
#
#   python experiments/run_window_sweep.py
# Scores: scores/{ds}__kn__wsweep-{seg}__L{L}.jsonl (resumable); summary lines
# to stdout (redirect to window_sweep.log).

import json
import random
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from run_kn_grid import MASKED, SCORES, read_tsv, rechunk  # noqa: E402
from run_longtexts import build_cases  # noqa: E402

sys.path.insert(0, str(HERE.parent))
from lambdag import LambdaG, cllr, cllr_min  # noqa: E402

DATASETS = ["lithuanian_novels", "polish_novels", "german_novels", "english_novels"]
LENGTHS = [500, 1000, 2000, 5000]
SEGS = {"sent": 0, "w10": 10, "w20": 20, "w30": 30, "w50": 50, "w100": 100}


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    from sklearn.metrics import roc_auc_score
    for ds in DATASETS:
        bank = {}
        for f in sorted((MASKED / ds / "bank").glob("*.tsv")):
            s = read_tsv(f)
            bank[f.stem] = {"sents": s, "ntok": sum(len(x) for x in s)}
        for L in LENGTHS:
            cases = build_cases(bank, L)
            by_excl = {}
            for cid, (a, b, k, q) in enumerate(cases):
                by_excl.setdefault(frozenset((a, b)), []).append((cid, a, b, k, q))
            for seg, w in SEGS.items():
                fn = SCORES / f"{ds}__kn__wsweep-{seg}__L{L}.jsonl"
                if fn.exists():
                    recs = [json.loads(l) for l in open(fn, encoding="utf-8")]
                else:
                    t0 = time.time()
                    lg = LambdaG(N=10, r=30, engine="kn", random_state=0)
                    recs = []
                    for excl, group in sorted(by_excl.items(),
                                              key=lambda kv: sorted(kv[0])):
                        pool = [s for n in sorted(bank) if n not in excl
                                for s in bank[n]["sents"]]
                        rng = random.Random(f"{ds}|{L}|{'|'.join(sorted(excl))}")
                        rng.shuffle(pool)
                        tot, kept = 0, []
                        for s in pool:
                            kept.append(s); tot += len(s)
                            if tot >= max(100_000, 60 * L):
                                break
                        lg.set_reference(rechunk(kept, w))
                        for cid, a, b, k, q in group:
                            r = lg.score(rechunk(q, w), rechunk(k, w),
                                         with_details=False)
                            recs.append({"id": cid, "label": int(a == b),
                                         "sqrt": r.lambda_sqrt,
                                         "lambda_G": r.lambda_G})
                    with open(fn, "w", encoding="utf-8") as fh:
                        for rec in recs:
                            fh.write(json.dumps(rec) + "\n")
                s = np.array([r["sqrt"] for r in recs])
                y = np.array([r["label"] for r in recs])
                print(f"  {ds:20s} L={L:>4d} {seg:>5s}  n={len(recs):3d}  "
                      f"AUC {roc_auc_score(y, s):.3f}  "
                      f"sqrt-Cllr {cllr(s[y==1], s[y==0]):.3f} "
                      f"(min {cllr_min(s[y==1], s[y==0]):.3f})", flush=True)


if __name__ == "__main__":
    main()
