# The corpus-adequacy gauge: symmetrised LambdaG with PER-AUTHOR, size-matched
# reference grammars instead of pooled samples (journal paper, calibration chapter).
#
# Theory: with single-author reference models the known author is exchangeable
# with the reference cohort under H_d, so E[per-token lambda | H_d] = 0 by
# construction -- and any RESIDUAL offset measures how badly the reference
# population fits the case population (the objective corpus-adequacy statistic).
# Two arms per configuration:
#   matched:    knowns and reference donors from the SAME bank      -> b_sym ~ 0?
#   mismatched: knowns from one corpus, donors from another (genre) -> b_sym != 0?
#
# Implementation: lambda_sym = mean_j over R donor authors of
# score(q, k, ref_sentences=donor_j, r=1).lambda_G  (algebraically identical to
# the symmetrised estimator; each donor model is size-matched to S_A by the
# sampler). Per-donor lambdas are persisted -- they double as the case-internal
# cohort for the studentisation analysis.
#
#   python experiments/run_symmeter.py --dataset german_novels
#   python experiments/run_symmeter.py --dataset german_novels --refdataset german_poetree
#
# Output: scores/{ds}__symref-{refds}__L{L}.jsonl, one row per pair:
#   {"known","kw","quest","qw","within","lambda_G","n_q","lam_j":[per-donor]}

import argparse
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
from lambdag import LambdaG  # noqa: E402

L = 2000
MINW = 3          # known windows per author (>= 6 ordered within pairs)
NCROSS = 4        # cross-questioned windows per known window
R_DONORS = 15     # per-author reference models per score
MAX_KNOWNS = 24


def load_bank(ds):
    bank = {}
    for f in sorted((MASKED / ds / "bank").glob("*.tsv")):
        s = read_tsv(f)
        bank[f.stem] = {"sents": s, "ntok": sum(len(x) for x in s)}
    return bank


def run(ds, refds, args):
    tag = f"{ds}__symref-{refds}__L{args.L}{args.tag}"
    fn = SCORES / f"{tag}.jsonl"
    if fn.exists():
        print(f"  {tag}: exists, skipped", flush=True)
        return
    bank = load_bank(ds)
    refbank = bank if refds == ds else load_bank(refds)
    donors = {n: v["sents"] for n, v in refbank.items() if v["ntok"] >= args.L}
    elig = sorted(n for n in bank if bank[n]["ntok"] >= MINW * args.L)
    if len(elig) > args.max_knowns:
        rng = random.Random(f"{ds}|{refds}|knowns")
        elig = sorted(rng.sample(elig, args.max_knowns))
    if len(elig) < 3 or len(donors) < 8:
        print(f"  {tag}: too thin ({len(elig)} knowns, {len(donors)} donors)",
              flush=True)
        fn.write_text("")
        return
    wins = {a: [window(bank[a]["sents"], k * args.L, args.L)
                for k in range(min(bank[a]["ntok"] // args.L, MINW))]
            for a in elig}
    lg = LambdaG(N=10, r=1, engine="kn", random_state=0)
    t0, rows = time.time(), []

    def sym_score(q, k, ka, qa):
        pool = [n for n in donors if n not in (ka, qa)]
        n_pick = min(R_DONORS, len(pool))     # thin banks: fewer donor models
        prng = random.Random(f"{ds}|{refds}|{ka}|{qa}|donors")
        picks = prng.sample(pool, n_pick)
        lams, n_q = [], 0
        for dn in picks:
            r = lg.score(q, k, ref_sentences=donors[dn], r=1,
                         with_details=False)
            lams.append(r.lambda_G); n_q = r.n_query_tokens
        return float(np.mean(lams)), n_q, lams

    for a in elig:
        m = len(wins[a])
        others = [n for n in elig if n != a]
        for i in range(m):
            for j in range(m):
                if j == i:
                    continue
                lam, n_q, lams = sym_score(wins[a][j], wins[a][i], a, a)
                rows.append({"known": a, "kw": i, "quest": a, "qw": j,
                             "within": 1, "lambda_G": lam, "n_q": n_q,
                             "lam_j": [round(x, 3) for x in lams]})
            prng = random.Random(f"{ds}|{refds}|{a}|{i}|cross")
            partners = (prng.sample(others, NCROSS) if len(others) >= NCROSS
                        else [prng.choice(others) for _ in range(NCROSS)])
            for b in partners:
                jq = prng.randrange(len(wins[b]))
                lam, n_q, lams = sym_score(wins[b][jq], wins[a][i], a, b)
                rows.append({"known": a, "kw": i, "quest": b, "qw": jq,
                             "within": 0, "lambda_G": lam, "n_q": n_q,
                             "lam_j": [round(x, 3) for x in lams]})
        print(f"    {tag} {a}: {len(rows)} pairs, {time.time()-t0:.0f}s",
              flush=True)
    with open(fn, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    print(f"  {tag}: {len(rows)} pairs in {time.time()-t0:.0f}s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--refdataset", default="")
    ap.add_argument("--L", type=int, default=L)
    ap.add_argument("--max-knowns", type=int, default=MAX_KNOWNS)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    run(args.dataset, args.refdataset or args.dataset, args)


if __name__ == "__main__":
    main()
