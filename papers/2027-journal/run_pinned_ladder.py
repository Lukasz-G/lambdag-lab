# Pinned-known length ladder: the accumulation exponent of the authorial signal.
#
# WHY A SEPARATE DRIVER. Every length grid already in this directory is
# SYMMETRIC -- run_kn_grid.py and run_longtexts.py truncate the known AND the
# questioned text to the same L, because that is the honest "how little text do
# we need" question. A symmetric ladder cannot, however, be read as an exponent.
# Growing L simultaneously (a) lengthens the questioned block and (b) shrinks
# G_A's estimation error, so the measured growth is
#
#       signal(L)  ~  L * rate(L)  ~  L^(1+gamma)
#
# and comes out SUPER-linear (beta ~ 1.3-2.0 across the multilingual grid) -- an
# artefact of the plug-in estimator, not an information law. Read against
# Hilberg's law the setup is superficially right (two blocks of length n) but
# actually wrong, because the model of the first block is estimated, not known.
#
# THIS DESIGN pins the known text at K tokens, so G_A and the size-matched
# reference grammars are held CONSTANT across rungs and only the questioned
# block grows:
#
#       signal(N_q) ~ N_q^beta
#         beta = 1   constant per-token evidence rate (LambdaG's own premise:
#                    lambda_G is a plain sum of per-token log-ratios)
#         beta < 1   sub-linear accumulation, i.e. a Hilberg-type drag, under
#                    which the text-length threshold is n* = (E/c)^(1/beta)
#
# Questioned windows are NESTED -- rung Q is the first Q tokens of the longest
# rung -- so the rungs are paired within a case and between-rung sampling noise
# does not enter the slope. Cases are built from the masked reference bank
# exactly as in run_longtexts.py (known = the author's first K tokens,
# questioned = disjoint material beyond it, reference pool excludes both case
# authors), so this is a self-contained side experiment.
#
# The companion H_d readout is beta_drift, the exponent of |mean lambda| on the
# different-author cases: the offset b is defined as a constant per-token rate,
# so beta_drift = 1 is a direct check of that definition.
#
#   python experiments/run_pinned_ladder.py
#   python experiments/run_pinned_ladder.py --datasets german_novels --k 10000
#
# Output: experiments/scores/{dataset}__kn__pinned__K{K}.jsonl (resumable), one
# record per (case, rung); read with analyze_pinned_ladder.py.

import argparse
import json
import random
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
# read_tsv/rechunk and the case builder live with the grid driver, which sits in
# a sibling paper directory once the scripts are filed by paper; add it so this
# runs from either layout.
for _sib in (HERE, HERE.parent / "2027-chr"):
    if (_sib / "run_kn_grid.py").exists():
        sys.path.insert(0, str(_sib))
        break
from run_kn_grid import MASKED, SCORES, read_tsv  # noqa: E402
from run_longtexts import window  # noqa: E402

sys.path.insert(0, str(HERE.parent))
from lambdag import LambdaG  # noqa: E402

DATASETS = ["german_novels", "english_novels", "german_dracor", "german_poetree"]
K = 5000                                  # pinned known length, in tokens
QS = [150, 300, 600, 1200, 2400]          # nested questioned rungs (16x span)
MAXAUTH = 40                              # cap per dataset, for runtime
ROT = 2                                   # different-author partners per author


def build_cases(bank, names, k, qmax, rot=ROT):
    """(known_author, questioned_author, known sents, questioned sents at qmax).

    Same-author: known = tokens[0:k], questioned = tokens[k:k+qmax], disjoint.
    Different-author: the SAME known sample against the next authors' questioned
    windows, so both classes reuse identical known material and the contrast is
    the questioned side alone.
    """
    cases = []
    for i, a in enumerate(names):
        known = window(bank[a]["sents"], 0, k)
        cases.append((a, a, known, window(bank[a]["sents"], k, qmax)))
        for step in range(1, rot + 1):
            b = names[(i + step) % len(names)]
            if b != a:
                cases.append((a, b, known, window(bank[b]["sents"], k, qmax)))
    return cases


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default=",".join(DATASETS))
    ap.add_argument("--k", type=int, default=K)
    ap.add_argument("--qs", default=",".join(str(q) for q in QS))
    ap.add_argument("--order", type=int, default=10)
    ap.add_argument("--r", type=int, default=30)
    ap.add_argument("--max-authors", type=int, default=MAXAUTH)
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    qs = sorted(int(q) for q in args.qs.split(","))
    qmax, k = max(qs), args.k

    for ds in args.datasets.split(","):
        fn = SCORES / f"{ds}__kn__pinned__K{k}.jsonl"
        if fn.exists():
            print(f"{ds}: {fn.name} exists -- skipped", flush=True)
            continue
        t0 = time.time()

        # only authors who afford the known sample AND a disjoint longest rung
        bank = {}
        for f in sorted((MASKED / ds / "bank").glob("*.tsv")):
            s = read_tsv(f)
            n = sum(len(x) for x in s)
            if n >= k + qmax:
                bank[f.stem] = {"sents": s, "ntok": n}
        names = sorted(bank)
        if len(names) > args.max_authors:
            names = sorted(random.Random(ds).sample(names, args.max_authors))
        if len(names) < 12:
            print(f"{ds}: only {len(names)} authors afford {k + qmax} tokens "
                  f"-- skipped", flush=True)
            continue

        cases = build_cases(bank, names, k, qmax)
        print(f"{ds}: {len(names)} authors, {len(cases)} cases "
              f"({sum(1 for c in cases if c[0] == c[1])} same-author)", flush=True)

        by_excl = {}
        for cid, c in enumerate(cases):
            by_excl.setdefault(frozenset((c[0], c[1])), []).append((cid,) + c)

        lg = LambdaG(N=args.order, r=args.r, engine="kn", random_state=0)
        recs = []
        for excl, group in sorted(by_excl.items(), key=lambda kv: sorted(kv[0])):
            pool = [s for n in names if n not in excl for s in bank[n]["sents"]]
            rng = random.Random(f"{ds}|pinned|{'|'.join(sorted(excl))}")
            rng.shuffle(pool)
            tot, kept = 0, []
            for s in pool:
                kept.append(s)
                tot += len(s)
                if tot >= max(100_000, 60 * k):
                    break
            # constant across rungs, because K is pinned: the r reference
            # grammars are size-matched to the known text, not to the questioned
            lg.set_reference(kept)
            for cid, a, b, known, qfull in group:
                for q in qs:
                    quest = window(qfull, 0, q)
                    res = lg.score(quest, known, with_details=False)
                    recs.append({"id": cid, "label": int(a == b), "Q": q,
                                 "lambda_G": res.lambda_G,
                                 "n_q": sum(len(x) for x in quest)})
        with open(fn, "w", encoding="utf-8") as fh:
            for rec in recs:
                fh.write(json.dumps(rec) + "\n")
        print(f"{ds}: {len(recs)} records in {time.time() - t0:.0f}s "
              f"-> {fn.name}", flush=True)


if __name__ == "__main__":
    main()
