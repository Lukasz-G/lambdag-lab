# Bank-manufactured calibration for every Table-4 dataset (generalises lt_zcal.py).
#
# For each dataset: score the FULL ordered pair matrix of bank authors at symmetric
# L=5000 (known = author's first L tokens, questioned = the partner's next-L window),
# with per-known-author reference sets excluding that author. Then, per case, fit a
# logistic calibrator on all pseudo-cases involving NEITHER case author
# (leave-both-out) and convert the sqrt-corrected score into a calibrated log10 LR.
# Reported per dataset: uncalibrated vs bank-calibrated Cllr on the same case set.
#
#   python experiments/run_bankcal.py
#
# Pair scores go to scores/{ds}__kn__bankcal__L5000.jsonl (resumable); summary lines
# to stdout (redirect to bankcal.log; make_paper_tables.py reads the score files).

import json
import random
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from run_kn_grid import MASKED, SCORES, read_tsv  # noqa: E402
from run_longtexts import window  # noqa: E402

sys.path.insert(0, str(HERE.parent))
from lambdag import LambdaG, cllr, cllr_min  # noqa: E402

DATASETS = ["lithuanian_novels", "ukrainian_novels", "polish_novels",
            "german_novels", "english_novels"]   # smallest matrix first
L = 5000


def score_matrix(ds):
    fn = SCORES / f"{ds}__kn__bankcal__L{L}.jsonl"
    if fn.exists():
        return [json.loads(l) for l in open(fn, encoding="utf-8")]
    bank = {}
    for f in sorted((MASKED / ds / "bank").glob("*.tsv")):
        s = read_tsv(f)
        bank[f.stem] = {"sents": s, "ntok": sum(len(x) for x in s)}
    names = sorted(n for n in bank if bank[n]["ntok"] >= 2 * L)
    known = {n: window(bank[n]["sents"], 0, L) for n in names}
    quest = {n: window(bank[n]["sents"], L, L) for n in names}

    lg = LambdaG(N=10, r=30, engine="kn", random_state=0)
    t0, rows = time.time(), []
    for a in names:
        pool = [s for n in names if n != a for s in bank[n]["sents"]]
        rng = random.Random(f"{ds}|{L}|{a}")
        rng.shuffle(pool)
        tot, kept = 0, []
        for s in pool:
            kept.append(s); tot += len(s)
            if tot >= 60 * L:
                break
        lg.set_reference(kept)
        for b in names:
            r = lg.score(quest[b], known[a], with_details=False)
            rows.append({"known": a, "quest": b, "sqrt": r.lambda_sqrt})
    with open(fn, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    print(f"  {ds}: scored {len(rows)} pairs in {time.time()-t0:.0f}s", flush=True)
    return rows


def calibrate(rows):
    from sklearn.linear_model import LogisticRegression
    s = np.array([r["sqrt"] for r in rows])
    y = np.array([int(r["known"] == r["quest"]) for r in rows])
    ka = [r["known"] for r in rows]; qa = [r["quest"] for r in rows]
    llr = np.empty(len(rows))
    for i in range(len(rows)):
        tr = [j for j in range(len(rows)) if {ka[j], qa[j]}.isdisjoint({ka[i], qa[i]})]
        X, yy = s[tr].reshape(-1, 1), y[tr]
        m = LogisticRegression(C=1e6).fit(X, yy)
        prior = np.log(yy.mean() / (1 - yy.mean()))
        llr[i] = (m.coef_[0, 0] * s[i] + m.intercept_[0] - prior) / np.log(10)
    return s, llr, y


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    for ds in DATASETS:
        rows = score_matrix(ds)
        s, llr, y = calibrate(rows)
        print(f"  {ds:22s} bankcal L={L}  n={len(rows)} ({int(y.sum())}/{int((1-y).sum())})  "
              f"uncal Cllr {cllr(s[y==1], s[y==0]):.3f} (min {cllr_min(s[y==1], s[y==0]):.3f})  "
              f"cal Cllr {cllr(llr[y==1], llr[y==0]):.3f}", flush=True)


if __name__ == "__main__":
    main()
