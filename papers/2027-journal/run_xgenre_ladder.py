# Cross-genre accumulation exponent: is genre a RATE or a BOUNDARY?
#
# Within genre the exponent is already measured (run_pinned_ladder.py): with the
# known side pinned, author evidence accumulates linearly, beta ~= 0.95 and
# beta = 1/2 is excluded, in prose, verse and drama alike. That number is the
# baseline this experiment is read against.
#
# THE FORK. If crossing genre only changes WHICH profile is sampled, the rate
# falls but accumulation stays linear: beta ~= 1 with a smaller constant, the
# text-length threshold n* scales linearly, and the boundary is crossable by
# supplying more known text. If instead the author's component must be recovered
# from a genre-contaminated stream, accumulation turns sub-linear, beta < 1, n*
# explodes and the boundary is effectively closed. Exponent versus constant.
#
# DESIGN, and why it is pinned. A symmetric ladder cannot answer this: growing
# both sides at once improves the questioned block AND shrinks G_A's estimation
# error, which reads as super-linear growth (measured at 1.3-2.0) and is an
# artefact of the plug-in estimator. So the known side is fixed and only the
# questioned block grows, exactly as in the within-genre measurement, and the two
# numbers are then comparable.
#
# Six directions, every ordered pair of the three genres. Questioned windows are
# NESTED -- rung Q is the first Q tokens of the longest rung -- so the rungs are
# paired within a case and between-rung sampling noise does not enter the slope.
#
#   python experiments/run_xgenre_ladder.py
#   python experiments/run_xgenre_ladder.py --known 10000 --directions prose2verse
#
# Output: experiments/scores/xgenre_ladder/{known}2{quest}__K{K}.jsonl, one record
# per (case, rung); read with analyze_pinned_ladder.py's slope fitting.

import argparse
import json
import random
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
for _sib in (HERE, HERE.parent / "2027-chr"):
    if (_sib / "run_kn_grid.py").exists():
        sys.path.insert(0, str(_sib))
        break
from run_kn_grid import MASKED, SCORES, read_tsv, rechunk  # noqa: E402
from run_longtexts import window  # noqa: E402

sys.path.insert(0, str(HERE.parent))
from lambdag import LambdaG  # noqa: E402

GENRES = {"prose": "german_tgproseall", "verse": "german_tgverseall",
          "drama": "german_tgdramaall"}
QS = [150, 300, 600, 1200, 2400]
OUT = SCORES / "xgenre_ladder"


def banks():
    out = {}
    for g, ds in GENRES.items():
        d = MASKED / ds / "bank"
        out[g] = {}
        for f in sorted(d.glob("*.tsv")):
            stem = f.stem.split("_", 1)[1] if f.stem[:3].isdigit() else f.stem
            out[g][stem] = read_tsv(f)
    return out


def ntok(s):
    return sum(len(x) for x in s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--known", type=int, default=8000,
                    help="pinned known length; only the questioned block grows")
    ap.add_argument("--qs", default=",".join(str(q) for q in QS))
    ap.add_argument("--directions", default="",
                    help="comma-separated known2quest, e.g. prose2verse")
    ap.add_argument("--rot", type=int, default=2)
    ap.add_argument("--seg", type=int, default=100)
    ap.add_argument("--order", type=int, default=10)
    ap.add_argument("--r", type=int, default=30)
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    OUT.mkdir(parents=True, exist_ok=True)

    qs = sorted(int(q) for q in args.qs.split(","))
    qmax, K = max(qs), args.known
    B = banks()

    dirs = ([tuple(d.split("2")) for d in args.directions.split(",") if d]
            or [(a, b) for a in GENRES for b in GENRES if a != b])

    lg = LambdaG(N=args.order, r=args.r, engine="kn", random_state=0)

    for kg, qg in dirs:
        fn = OUT / f"{kg}2{qg}__K{K}.jsonl"
        if fn.exists():
            print(f"{kg}->{qg}: exists, skipped", flush=True)
            continue
        # an author needs the pinned known block in kg and the longest rung in qg
        elig = sorted(a for a in B[kg]
                      if a in B[qg] and ntok(B[kg][a]) >= K
                      and ntok(B[qg][a]) >= qmax)
        if len(elig) < 6:
            print(f"{kg}->{qg}: only {len(elig)} authors, skipped", flush=True)
            continue
        print(f"\n=== known {kg} -> questioned {qg}: {len(elig)} authors ===",
              flush=True)

        cases = []
        for a in elig:
            cases.append((a, a))
            rng = random.Random(f"{kg}2{qg}|{a}")
            others = [x for x in elig if x != a]
            for b in rng.sample(others, min(args.rot, len(others))):
                cases.append((a, b))

        t0, recs = time.time(), []
        for ci, (qa, ka) in enumerate(cases):
            known = window(B[kg][ka], 0, K)
            qfull = window(B[qg][qa], 0, qmax)
            pool = [s for n in sorted(B[qg]) if n not in (qa, ka)
                    for s in B[qg][n]]
            rng = random.Random(f"{kg}2{qg}|{qa}|{ka}")
            rng.shuffle(pool)
            kept, tot = [], 0
            for s in pool:
                kept.append(s); tot += len(s)
                if tot >= max(100_000, 60 * K):
                    break
            # constant across rungs, because K is pinned: the r reference
            # grammars are size-matched to the known text, not the questioned
            lg.set_reference(rechunk(kept, args.seg))
            for q in qs:
                quest = window(qfull, 0, q)
                res = lg.score(rechunk(quest, args.seg),
                               rechunk(known, args.seg), with_details=False)
                recs.append({"id": ci, "label": int(qa == ka), "Q": q,
                             "known_genre": kg, "quest_genre": qg,
                             "lambda_G": res.lambda_G, "n_q": ntok(quest)})
        with open(fn, "w", encoding="utf-8") as fh:
            for r in recs:
                fh.write(json.dumps(r) + "\n")
        print(f"  {len(cases)} cases x {len(qs)} rungs = {len(recs)} records "
              f"in {time.time() - t0:.0f}s -> {fn.name}", flush=True)


if __name__ == "__main__":
    main()
