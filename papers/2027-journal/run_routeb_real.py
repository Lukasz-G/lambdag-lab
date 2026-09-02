# Route B on REAL grid test pairs (defence against the pseudo-case objection):
# per-donor lambdas for the actual evaluation cases (masked/<ds>/pairs) at
# symmetric L=1200 -- the ill-calibrated grid regime -- with R per-author donor
# models from the reference bank. Stratified case subsample for cost control.
#
#   python experiments/run_routeb_real.py --dataset german_novels
#
# Output: scores/{ds}__routebreal__L{L}.jsonl:
#   {"id","label","n_q","lam_j":[per-donor lambda]}

import argparse
import json
import random
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from run_kn_grid import MASKED, SCORES, read_tsv, trunc  # noqa: E402

sys.path.insert(0, str(HERE.parent))
from lambdag import LambdaG  # noqa: E402

L = 1200
R_DONORS = 12
MAX_SAME, MAX_DIFF = 200, 400


def run(ds, args):
    fn = SCORES / f"{ds}__routebreal__L{args.L}{args.tag}.jsonl"
    if fn.exists():
        print(f"  {ds}: exists, skipped", flush=True)
        return
    d = MASKED / ds
    bank = {}
    for f in sorted((d / "bank").glob("*.tsv")):
        s = read_tsv(f)
        if sum(len(x) for x in s) >= args.L:
            bank[f.stem] = s
    man = [l.rstrip("\n").split("\t") for l in open(d / "pairs.tsv", encoding="utf-8")][1:]
    rng = random.Random(f"{ds}|routebreal")
    same = [m for m in man if m[1] == "1"]
    diff = [m for m in man if m[1] == "0"]
    rng.shuffle(same); rng.shuffle(diff)
    picked = same[:MAX_SAME] + diff[:MAX_DIFF]
    print(f"  {ds}: {len(picked)} cases ({min(len(same),MAX_SAME)} same / "
          f"{min(len(diff),MAX_DIFF)} diff), {len(bank)} donors", flush=True)

    lg = LambdaG(N=10, r=1, engine="kn", random_state=0)
    t0, rows = time.time(), []
    for pid, lab, ka, qa in picked:
        k = trunc(read_tsv(d / "pairs" / f"{pid}_known.tsv"), args.L)
        q = trunc(read_tsv(d / "pairs" / f"{pid}_q.tsv"), args.L)
        if not k or not q:
            continue
        pool = [n for n in bank if n not in (ka, qa)]
        prng = random.Random(f"{ds}|{pid}|donors")
        picks = prng.sample(pool, min(R_DONORS, len(pool)))
        lams, n_q = [], 0
        for dn in picks:
            r = lg.score(q, k, ref_sentences=bank[dn], r=1, with_details=False)
            lams.append(round(r.lambda_G, 3)); n_q = r.n_query_tokens
        rows.append({"id": int(pid), "label": int(lab), "n_q": n_q,
                     "lam_j": lams})
        if len(rows) % 100 == 0:
            print(f"    {ds}: {len(rows)} cases, {time.time()-t0:.0f}s",
                  flush=True)
    with open(fn, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    print(f"  {ds}: {len(rows)} cases in {time.time()-t0:.0f}s -> {fn.name}",
          flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--L", type=int, default=L)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    run(args.dataset, args)


if __name__ == "__main__":
    main()
