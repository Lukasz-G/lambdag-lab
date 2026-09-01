# Bank calibration for the MAIN grid at full evidence length: can the label-free
# calibrator rescue the failing full-length Cllr rows of Table 2?
#
# Pseudo-cases mirror the main grid's own condition -- known = a bank author's
# first 5,000 tokens, questioned = further disjoint 1,000-token windows -- with
# the known author excluded from the per-case reference pool. Bank and test
# authors are disjoint by construction, so one calibrator per dataset is fitted
# on all pseudo-cases and applied to the existing full-length score file.
#
# Datasets are processed poetry -> novels -> drama (descending expected rescue).
#
#   python experiments/run_bankcal_main.py
# Pseudo scores: scores/{ds}__kn__bankmain__L0.jsonl (resumable); summary lines
# to stdout (redirect to bankcal_main.log; make_paper_tables.py reads both).

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

K, Q = 5000, 1000
SAME_WINDOWS = 2      # same-author questioned windows per author (text permitting)
ROTATIONS = 2         # different-author partners per known author
MAX_AUTHORS = 60      # cap for very large banks (english_poetree has 174)
GENRE_ORDER = {"poetree": 0, "novels": 1, "dracor": 2}


def eligible():
    ds = []
    for d in sorted(MASKED.iterdir()):
        if not (d.is_dir() and (d / "DONE").exists()):
            continue
        info = json.loads((d / "DONE").read_text(encoding="utf-8"))
        if info.get("ref", 0) < 10:
            continue
        if not (SCORES / f"{d.name}__kn__sent__L0.jsonl").exists():
            continue
        ds.append(d.name)
    return sorted(ds, key=lambda n: (GENRE_ORDER[n.rsplit("_", 1)[1]], n))


def pseudo_scores(ds):
    fn = SCORES / f"{ds}__kn__bankmain__L0.jsonl"
    if fn.exists():
        return [json.loads(l) for l in open(fn, encoding="utf-8")]
    bank = {}
    for f in sorted((MASKED / ds / "bank").glob("*.tsv")):
        s = read_tsv(f)
        bank[f.stem] = {"sents": s, "ntok": sum(len(x) for x in s)}
    names = sorted(n for n in bank if bank[n]["ntok"] >= K + Q)
    if len(names) < 5:
        return []
    if len(names) > MAX_AUTHORS:
        names = random.Random(f"{ds}|authors").sample(names, MAX_AUTHORS)
        names.sort()
    lg = LambdaG(N=10, r=30, engine="kn", random_state=0)
    rows = []
    for i, a in enumerate(names):
        known = window(bank[a]["sents"], 0, K)
        pool = [s for n in names if n != a for s in bank[n]["sents"]]
        rng = random.Random(f"{ds}|main|{a}")
        rng.shuffle(pool)
        tot, kept = 0, []
        for s in pool:
            kept.append(s); tot += len(s)
            if tot >= 60 * K:
                break
        lg.set_reference(kept)
        nwin = min(SAME_WINDOWS, (bank[a]["ntok"] - K) // Q)
        for w in range(nwin):
            r = lg.score(window(bank[a]["sents"], K + w * Q, Q), known,
                         with_details=False)
            rows.append({"known": a, "quest": a, "sqrt": r.lambda_sqrt})
        for step in range(1, ROTATIONS + 1):
            b = names[(i + step) % len(names)]
            r = lg.score(window(bank[b]["sents"], K, Q), known, with_details=False)
            rows.append({"known": a, "quest": b, "sqrt": r.lambda_sqrt})
    with open(fn, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    return rows


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    from sklearn.linear_model import LogisticRegression
    for ds in eligible():
        t0 = time.time()
        rows = pseudo_scores(ds)
        if not rows:
            print(f"  {ds:24s} SKIPPED (too few bank authors with {K+Q} tokens)",
                  flush=True)
            continue
        X = np.array([r["sqrt"] for r in rows]).reshape(-1, 1)
        y = np.array([int(r["known"] == r["quest"]) for r in rows])
        if len(set(y.tolist())) < 2:
            print(f"  {ds:24s} SKIPPED (single-class pseudo-cases)", flush=True)
            continue
        m = LogisticRegression(C=1e6).fit(X, y)
        prior = np.log(y.mean() / (1 - y.mean()))
        recs = [json.loads(l) for l in
                open(SCORES / f"{ds}__kn__sent__L0.jsonl", encoding="utf-8")]
        s = np.array([r["sqrt"] for r in recs])
        yy = np.array([r["label"] for r in recs])
        llr = (m.coef_[0, 0] * s + m.intercept_[0] - prior) / np.log(10)
        print(f"  {ds:24s} bankmain  pseudo n={len(rows):4d}  "
              f"uncal Cllr {cllr(s[yy==1], s[yy==0]):.3f} "
              f"(min {cllr_min(s[yy==1], s[yy==0]):.3f})  ->  "
              f"bank-cal Cllr {cllr(llr[yy==1], llr[yy==0]):.3f}  "
              f"({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
