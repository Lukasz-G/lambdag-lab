# Per-token detail dump for the aggregation-functional sweep (Layer 3 of the
# improvement map): re-score selected grid cells with with_details=True and
# persist, per case, the full per-token contribution vector aligned with its
# tokens, plus per-sentence sums. The functional sweep itself (sign-fraction,
# trimmed sums, tail mass, habit consistency, Wasserstein ceiling) then runs
# offline on these files -- no further scoring needed.
#
#   python experiments/run_details.py --datasets lithuanian_novels --segs sent --lengths 600
#
# Output: scores/{ds}__kn__details-{seg}__L{L}.jsonl.gz, one row per case:
#   {"id","label","lambda_G","n_q","v1_q","sqrt",
#    "slam":[per-sentence sums], "toks":[flat tokens incl <EOS>],
#    "tlam":[flat per-token contributions, 4dp]}
# Resumable per (ds, seg, L).

import argparse
import gzip
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from run_kn_grid import MASKED, SCORES, SEGMENTS, read_tsv, rechunk, trunc  # noqa: E402

sys.path.insert(0, str(HERE.parent))
from lambdag import LambdaG  # noqa: E402


def run_cell(ds, seg, L, limit=None):
    fn = SCORES / f"{ds}__kn__details-{seg}__L{L}.jsonl.gz"
    if fn.exists():
        print(f"  {ds} {seg} L={L}: exists, skipped", flush=True)
        return
    d = MASKED / ds
    w = SEGMENTS[seg]
    man = [l.rstrip("\n").split("\t") for l in open(d / "pairs.tsv", encoding="utf-8")][1:]
    if limit:
        man = man[:limit]
    ref = []
    for f in sorted((d / "bank").glob("*.tsv")):
        ref += rechunk(read_tsv(f), w)
    lg = LambdaG(N=10, r=30, engine="kn", random_state=0)
    lg.set_reference(ref)

    t0, n_done = time.time(), 0
    with gzip.open(fn, "wt", encoding="utf-8") as fh:
        for pid, lab, _ka, _qa in man:
            k = rechunk(read_tsv(d / "pairs" / f"{pid}_known.tsv"), w)
            q = rechunk(read_tsv(d / "pairs" / f"{pid}_q.tsv"), w)
            kt, qt = trunc(k, L), trunc(q, L)
            if not kt or not qt:
                continue
            r = lg.score(qt, kt, with_details=True)
            row = {"id": int(pid), "label": int(lab), "lambda_G": r.lambda_G,
                   "n_q": r.n_query_tokens, "v1_q": r.n_query_hapax,
                   "sqrt": r.lambda_sqrt,
                   "slam": [round(float(x), 4) for x in r.sentence_lambda],
                   "toks": [t for s in r.tokens for t in s],
                   "tlam": [round(float(x), 4) for s in r.token_lambda for x in s]}
            assert len(row["toks"]) == len(row["tlam"])
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            n_done += 1
            if n_done % 200 == 0:
                print(f"    {ds} {seg} L={L}: {n_done} cases, "
                      f"{time.time()-t0:.0f}s", flush=True)
    print(f"  {ds} {seg} L={L}: {n_done} cases in {time.time()-t0:.0f}s -> {fn.name}",
          flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", required=True)
    ap.add_argument("--segs", default="sent")
    ap.add_argument("--lengths", default="0,1200,600")
    ap.add_argument("--limit-pairs", type=int, default=0)
    args = ap.parse_args()
    for ds in args.datasets.split(","):
        for seg in args.segs.split(","):
            for L in [int(x) for x in args.lengths.split(",")]:
                run_cell(ds.strip(), seg.strip(), L, args.limit_pairs or None)


if __name__ == "__main__":
    main()
