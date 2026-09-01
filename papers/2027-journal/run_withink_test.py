# Within-K statistics for the TEST-CASE known authors (component B of the matched-L
# decisive test; component A = run_withink.py on the banks at the same lengths).
#
# The primary grid's ill-calibrated cells (sentences, symmetric L in {600, 1200}) use
# test pairs whose known authors are NOT bank authors, so the case-internal statistics
# must come from each case's own known document -- which is also the honest casework
# protocol: everything below uses only (a) the known author's document and (b) the
# reference bank the score already requires. Per distinct known author:
#   windows = disjoint L-token cuts of their (full, untruncated) known doc, m<=6, m>=3
#   WITHIN pairs: every ordered window pair            -> within_mean / within_sd
#   CROSS  pairs: 6 bank-author windows per known window -> measured b (oracle only)
# Reference pool: bank minus the known author (if present), capped 60*L, seeded.
#
#   python experiments/run_withink_test.py --datasets german_novels --lengths 1200
#
# Output: scores/{ds}__kn__withinkT__L{L}.jsonl, rows as in run_withink.py plus the
# known author's name; resumable per (ds, L).

import argparse
import json
import random
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from run_kn_grid import MASKED, SCORES, read_tsv  # noqa: E402
from run_longtexts import window  # noqa: E402

sys.path.insert(0, str(HERE.parent))
from lambdag import LambdaG  # noqa: E402

NCROSS = 6
MAXW = 6
REF_TOKENS = 60


def known_docs(ds):
    """{known_author: sentence list of their (deduplicated) known document}."""
    d = MASKED / ds
    man = [l.rstrip("\n").split("\t") for l in open(d / "pairs.tsv", encoding="utf-8")][1:]
    first_pid = {}
    for pid, _lab, ka, _qa in man:
        first_pid.setdefault(ka, pid)
    return {ka: read_tsv(d / "pairs" / f"{pid}_known.tsv")
            for ka, pid in first_pid.items()}


def run_dataset(ds, L, args):
    fn = SCORES / f"{ds}__kn__withinkT__L{L}{args.tag}.jsonl"
    if fn.exists():
        print(f"  {ds} L={L}: exists, skipped", flush=True)
        return
    bank = {}
    for f in sorted((MASKED / ds / "bank").glob("*.tsv")):
        s = read_tsv(f)
        bank[f.stem] = {"sents": s, "ntok": sum(len(x) for x in s)}
    bank_wins = {n: [window(v["sents"], k * L, L)
                     for k in range(min(v["ntok"] // L, MAXW))]
                 for n, v in bank.items() if v["ntok"] >= L}

    docs = known_docs(ds)
    lg = LambdaG(N=10, r=30, engine="kn", random_state=0)
    t0, rows, kept_a, skipped_a = time.time(), [], 0, 0
    for ka in sorted(docs):
        sents = docs[ka]
        ntok = sum(len(s) for s in sents)
        m = min(ntok // L, MAXW)
        if m < 3:                                      # need >= 6 ordered within pairs
            skipped_a += 1
            continue
        wins = [window(sents, k * L, L) for k in range(m)]
        pool = [s for n in bank if n != ka for s in bank[n]["sents"]]
        rng = random.Random(f"{ds}|{L}|T|{ka}")
        rng.shuffle(pool)
        tot, kept = 0, []
        for s in pool:
            kept.append(s); tot += len(s)
            if tot >= REF_TOKENS * L:
                break
        lg.set_reference(kept)
        names = sorted(bank_wins)
        for i in range(m):
            for j in range(m):
                if j == i:
                    continue
                r = lg.score(wins[j], wins[i], with_details=False)
                rows.append({"known": ka, "kw": i, "quest": ka, "qw": j, "within": 1,
                             "lambda_G": r.lambda_G, "n_q": r.n_query_tokens,
                             "v1_q": r.n_query_hapax, "sqrt": r.lambda_sqrt})
            prng = random.Random(f"{ds}|{L}|T|{ka}|{i}|cross")
            partners = (prng.sample(names, NCROSS) if len(names) >= NCROSS
                        else [prng.choice(names) for _ in range(NCROSS)])
            for b in partners:
                jq = prng.randrange(len(bank_wins[b]))
                r = lg.score(bank_wins[b][jq], wins[i], with_details=False)
                rows.append({"known": ka, "kw": i, "quest": b, "qw": jq, "within": 0,
                             "lambda_G": r.lambda_G, "n_q": r.n_query_tokens,
                             "v1_q": r.n_query_hapax, "sqrt": r.lambda_sqrt})
        kept_a += 1
    with open(fn, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    print(f"  {ds} L={L}: {kept_a} known authors ({skipped_a} too short), "
          f"{len(rows)} scores in {time.time()-t0:.0f}s -> {fn.name}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", required=True)
    ap.add_argument("--lengths", default="600,1200")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    for ds in args.datasets.split(","):
        for L in [int(x) for x in args.lengths.split(",")]:
            run_dataset(ds.strip(), L, args)


if __name__ == "__main__":
    main()
