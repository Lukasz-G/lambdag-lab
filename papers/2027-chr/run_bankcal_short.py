# Can the label-free bank calibration rescue the main grid's SHORTER-length Cllr?
#
# For each (dataset, L) with L in the truncation grid {1200, 600, 300, 150}:
# build bank pseudo-cases at the same symmetric length (known = a bank author's
# first L tokens, questioned = further disjoint L-token windows, known author
# excluded from the per-case reference pool), fit one logistic calibrator on
# them, and apply it to the existing main-grid score file. Bank and test authors
# are disjoint by construction, so no per-case exclusion is needed on the
# evaluation side.
#
#   python experiments/run_bankcal_short.py
# Pseudo scores: scores/{ds}__kn__bankshort__L{L}.jsonl (resumable);
# summary lines to stdout (redirect to bankcal_short.log);
# make_paper_tables.py rebuilds the paper table from the persisted files.

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

# genre-spanning subset: five distant novel corpora + drama + poetry
DATASETS = ["lithuanian_novels", "polish_novels", "german_novels",
            "english_novels", "ukrainian_novels",
            "english_dracor", "german_dracor",
            "english_poetree", "german_poetree"]
LENGTHS = [1200, 600, 300, 150]
SAME_WINDOWS = 4      # same-author questioned windows per author (text permitting)
ROTATIONS = 4         # different-author partners per known author
MAX_AUTHORS = 60      # cap for very large banks (english_poetree has 174)


def pseudo_scores(ds, L):
    fn = SCORES / f"{ds}__kn__bankshort__L{L}.jsonl"
    if fn.exists():
        return [json.loads(l) for l in open(fn, encoding="utf-8")]
    bank = {}
    for f in sorted((MASKED / ds / "bank").glob("*.tsv")):
        s = read_tsv(f)
        bank[f.stem] = {"sents": s, "ntok": sum(len(x) for x in s)}
    names = sorted(n for n in bank if bank[n]["ntok"] >= 2 * L)
    if len(names) > MAX_AUTHORS:
        names = random.Random(f"{ds}|{L}|authors").sample(names, MAX_AUTHORS)
        names.sort()
    lg = LambdaG(N=10, r=30, engine="kn", random_state=0)
    rows = []
    for i, a in enumerate(names):
        known = window(bank[a]["sents"], 0, L)
        pool = [s for n in names if n != a for s in bank[n]["sents"]]
        rng = random.Random(f"{ds}|short{L}|{a}")
        rng.shuffle(pool)
        tot, kept = 0, []
        for s in pool:
            kept.append(s); tot += len(s)
            if tot >= max(100_000, 60 * L):
                break
        lg.set_reference(kept)
        nwin = min(SAME_WINDOWS, bank[a]["ntok"] // L - 1)
        for w in range(1, nwin + 1):
            r = lg.score(window(bank[a]["sents"], w * L, L), known,
                         with_details=False)
            rows.append({"known": a, "quest": a, "sqrt": r.lambda_sqrt})
        for step in range(1, ROTATIONS + 1):
            b = names[(i + step) % len(names)]
            r = lg.score(window(bank[b]["sents"], L, L), known, with_details=False)
            rows.append({"known": a, "quest": b, "sqrt": r.lambda_sqrt})
    with open(fn, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    return rows


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    from sklearn.linear_model import LogisticRegression
    for ds in DATASETS:
        for L in LENGTHS:
            main_fn = SCORES / f"{ds}__kn__sent__L{L}.jsonl"
            if not main_fn.exists():
                print(f"  {ds:22s} L={L:>4d}  SKIPPED (no main-grid scores)", flush=True)
                continue
            t0 = time.time()
            rows = pseudo_scores(ds, L)
            X = np.array([r["sqrt"] for r in rows]).reshape(-1, 1)
            y = np.array([int(r["known"] == r["quest"]) for r in rows])
            m = LogisticRegression(C=1e6).fit(X, y)
            prior = np.log(y.mean() / (1 - y.mean()))
            recs = [json.loads(l) for l in open(main_fn, encoding="utf-8")]
            s = np.array([r["sqrt"] for r in recs])
            yy = np.array([r["label"] for r in recs])
            llr = (m.coef_[0, 0] * s + m.intercept_[0] - prior) / np.log(10)
            print(f"  {ds:22s} L={L:>4d}  pseudo n={len(rows):4d}  "
                  f"uncal Cllr {cllr(s[yy==1], s[yy==0]):.3f} "
                  f"(min {cllr_min(s[yy==1], s[yy==0]):.3f})  ->  "
                  f"bank-cal Cllr {cllr(llr[yy==1], llr[yy==0]):.3f}  "
                  f"({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
