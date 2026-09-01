# Within-K variance vs H_d band -- the "range of variation around K" experiment
# (Nini's option 2, 2026-08-26 correspondence; see memory b-elimination-theory).
#
# SYMMETRIC design (no 1000-vs-5000 asymmetry): per dataset and per length L, each
# eligible bank author is cut into m <= MAXW disjoint L-token windows (sentence units,
# as in run_longtexts.window). For every author a and every window i used as the KNOWN
# side we score
#   WITHIN pairs: every other window j of a as questioned  (m-1 scores)  -> within-K
#   CROSS  pairs: NCROSS windows sampled from other eligible authors     -> H_d band
# with the reference pool excluding the known author (run_bankcal.py precedent; the
# questioned author stays in the pool -- mild conservative bias, as in the paper).
#
# The later analysis regresses, per author: spread of within per-token lambda_G
# against location/scale of cross per-token lambda_G. Outcome map: A = within-K
# predicts location (reference-free b exists); B = predicts scale only ("the case
# knows its own noise but not the population's position"); C = null; D = dataset-level
# only (ecological confound).
#
#   python experiments/run_withink.py --datasets lithuanian_novels --lengths 2000
#   python experiments/run_withink.py                    # all analysed datasets, both L
#   python experiments/run_withink.py --list             # eligibility/cost report only
#
# Output: scores/{ds}__kn__withink__L{L}{tag}.jsonl, one row per scored pair:
#   {"known", "kw", "quest", "qw", "within", "lambda_G", "n_q", "v1_q", "sqrt"}
# Resumable: a (ds, L) whose output file exists is skipped.

import argparse
import json
import random
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from run_kn_grid import MASKED, SCORES, read_tsv, MIN_REF_AUTHORS  # noqa: E402
from run_longtexts import window  # noqa: E402

sys.path.insert(0, str(HERE.parent))
from lambdag import LambdaG  # noqa: E402

LENGTHS = [2000, 5000]
MINW = 4          # an author needs >= MINW windows to yield a usable within-K variance
MAXW = 6          # windows materialised per author (caps the O(m^2) within block)
NCROSS = 6        # cross-questioned windows per known window
MIN_AUTHORS = 8   # a (ds, L) below this yields too few regression points -> skipped
MAX_AUTHORS = 40  # cap on eligible authors per (ds, L)
REF_TOKENS = 60   # reference pool cap, in multiples of L (run_bankcal.py precedent)


def datasets():
    for d in sorted(MASKED.iterdir()):
        if d.is_dir() and (d / "DONE").exists():
            info = json.loads((d / "DONE").read_text(encoding="utf-8"))
            if info.get("ref", 0) >= MIN_REF_AUTHORS:
                yield d.name


def load_bank(ds):
    bank = {}
    for f in sorted((MASKED / ds / "bank").glob("*.tsv")):
        s = read_tsv(f)
        bank[f.stem] = {"sents": s, "ntok": sum(len(x) for x in s)}
    return bank


def plan(ds, bank, L, args):
    """Eligible authors, their window counts, and the score budget for one (ds, L)."""
    elig = sorted(n for n in bank if bank[n]["ntok"] >= args.minw * L)
    if len(elig) > args.max_authors:
        rng = random.Random(f"{ds}|{L}|authors")
        elig = sorted(rng.sample(elig, args.max_authors))
    m = {n: min(bank[n]["ntok"] // L, args.maxw) for n in elig}
    n_scores = sum(m[a] * ((m[a] - 1) + args.ncross) for a in elig)
    return elig, m, n_scores


def run_dataset(ds, L, args):
    fn = SCORES / f"{ds}__kn__withink__L{L}{args.tag}.jsonl"
    if fn.exists():
        print(f"  {ds} L={L}: exists, skipped", flush=True)
        return
    bank = load_bank(ds)
    elig, m, n_scores = plan(ds, bank, L, args)
    if len(elig) < args.min_authors:
        print(f"  {ds} L={L}: only {len(elig)} authors with >= {args.minw}x{L} tokens "
              f"-> skipped", flush=True)
        fn.write_text("")            # empty file = done-marker, keeps resume simple
        return
    print(f"  {ds} L={L}: {len(elig)} authors, {n_scores} scores ...", flush=True)

    wins = {a: [window(bank[a]["sents"], k * L, L) for k in range(m[a])] for a in elig}
    lg = LambdaG(N=10, r=30, engine="kn", random_state=0)
    t0, rows = time.time(), []
    for a in elig:
        # reference pool: every bank author's sentences except the known author's
        pool = [s for n in bank if n != a for s in bank[n]["sents"]]
        rng = random.Random(f"{ds}|{L}|{a}")
        rng.shuffle(pool)
        tot, kept = 0, []
        for s in pool:
            kept.append(s); tot += len(s)
            if tot >= REF_TOKENS * L:
                break
        lg.set_reference(kept)
        others = [n for n in elig if n != a]
        for i in range(m[a]):
            for j in range(m[a]):                       # within: every other window of a
                if j == i:
                    continue
                r = lg.score(wins[a][j], wins[a][i], with_details=False)
                rows.append({"known": a, "kw": i, "quest": a, "qw": j, "within": 1,
                             "lambda_G": r.lambda_G, "n_q": r.n_query_tokens,
                             "v1_q": r.n_query_hapax, "sqrt": r.lambda_sqrt})
            prng = random.Random(f"{ds}|{L}|{a}|{i}|cross")
            partners = (prng.sample(others, args.ncross) if len(others) >= args.ncross
                        else [prng.choice(others) for _ in range(args.ncross)])
            for b in partners:                          # cross: the H_d band around K
                jq = prng.randrange(m[b])
                r = lg.score(wins[b][jq], wins[a][i], with_details=False)
                rows.append({"known": a, "kw": i, "quest": b, "qw": jq, "within": 0,
                             "lambda_G": r.lambda_G, "n_q": r.n_query_tokens,
                             "v1_q": r.n_query_hapax, "sqrt": r.lambda_sqrt})
        done = len(rows)
        print(f"    {ds} L={L} {a}: {done}/{n_scores} scores, "
              f"{time.time()-t0:.0f}s elapsed", flush=True)
    with open(fn, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    print(f"  {ds} L={L}: {len(rows)} scores in {time.time()-t0:.0f}s -> {fn.name}",
          flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="", help="comma list; default = all analysed")
    ap.add_argument("--lengths", default="", help="comma list; default = 2000,5000")
    ap.add_argument("--max-authors", type=int, default=MAX_AUTHORS)
    ap.add_argument("--min-authors", type=int, default=MIN_AUTHORS)
    ap.add_argument("--minw", type=int, default=MINW)
    ap.add_argument("--maxw", type=int, default=MAXW)
    ap.add_argument("--ncross", type=int, default=NCROSS)
    ap.add_argument("--tag", default="", help="suffix for output files (tests)")
    ap.add_argument("--list", action="store_true", help="print the plan and exit")
    args = ap.parse_args()

    ds_list = ([d.strip() for d in args.datasets.split(",") if d.strip()]
               or list(datasets()))
    lengths = ([int(x) for x in args.lengths.split(",") if x.strip()]
               or LENGTHS)

    if args.list:
        grand = 0
        for ds in ds_list:
            bank = load_bank(ds)
            for L in lengths:
                elig, m, n_scores = plan(ds, bank, L, args)
                ok = len(elig) >= args.min_authors
                grand += n_scores if ok else 0
                print(f"  {ds:28s} L={L}  authors {len(elig):3d}  "
                      f"scores {n_scores:6d}  {'' if ok else 'SKIP (<min authors)'}")
        print(f"\n  total scheduled scores: {grand}")
        return

    for ds in ds_list:
        for L in lengths:
            run_dataset(ds, L, args)


if __name__ == "__main__":
    main()
