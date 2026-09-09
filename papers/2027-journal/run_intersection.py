# Does a genre-invariant component of an author's grammar exist at all?
#
# THE QUESTION, and why it is not another representation. Three alphabets have
# now been tested on the cross-genre ladder -- POSNoise, class-by-rank, and two
# dependency encodings -- and not one moves the cross-genre ranking. Every one of
# them asks the same thing in a different vocabulary: can a single model of the
# author, fitted on one genre, recognise him in another? The repeated null says
# the limit is not in the vocabulary. So this asks a different question: given
# TWO of an author's genres, is there a component they share which generalises to
# a third?
#
# INTERSECTION VERSUS UNION, which is the whole design. Pooling an author's
# genres as TEXT and fitting one model gives an arithmetic mixture -- the UNION
# of his grammars, dominated by whichever genre supplied more tokens. That was
# tested (mixed-vs-single known) and moved lambda's magnitude without moving AUC.
# The untested object is the INTERSECTION: what his prose model and his drama
# model BOTH consider likely. At the level of per-token evidence that is a
# minimum, not a mean -- the author is credited only where both of his grammars
# agree the text fits. If an invariant component exists, the minimum is where it
# survives and the mean is where it is diluted.
#
# WHY THE COMBINATION IS AT THE SCORE LEVEL AND NOT THE MODEL LEVEL. A geometric
# mean of two conditional distributions is not normalised, and renormalising it
# costs a sum over the vocabulary at every context. Sub-normalised models break
# the likelihood-ratio semantics the method rests on, which is not a trade worth
# making. Each component here is instead a COMPLETE lambda_G with its own
# size-matched reference models, so every per-token contribution being combined
# is already a valid log-LR increment; the combination is a decision rule over
# valid statistics, and is calibrated like any other score.
#
# THE ARMS, all matched at the same TOTAL known budget so the comparison is about
# how the budget is SPENT, not how large it is:
#   single_a / single_b   N tokens from one known genre. The baseline.
#   pooled                N/2 + N/2 fitted as ONE model. The union, and the arm
#                         already known not to help.
#   mean2                 mean of the two components' per-token evidence.
#   min2                  MINIMUM of the two -- the intersection, the arm this
#                         experiment exists for.
#   max2                  maximum, the mirror control. Without it a win for min2
#                         cannot be told from any two-model combination winning.
#
# Questioned text always comes from the author's THIRD genre, held out entirely.
#
#   python experiments/run_intersection.py
#   python experiments/run_intersection.py --known 10000 --quest 10000 --imp 4
#
# Output: experiments/scores/intersection/intersection__K{N}__Q{Q}.jsonl
#         one record per (case, arm).

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

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
OUT = SCORES / "intersection"


def banks():
    out = {}
    for g, ds in GENRES.items():
        out[g] = {}
        for f in sorted((MASKED / ds / "bank").glob("*.tsv")):
            stem = f.stem.split("_", 1)[1] if f.stem[:3].isdigit() else f.stem
            out[g][stem] = [[sys.intern(t) for t in s] for s in read_tsv(f)]
    return out


def ntok(s):
    return sum(len(x) for x in s)


def refpool(B, genre, exclude, seg, need, rng):
    """Reference sentences from ONE genre, excluding the case's authors.

    Genre-matched to the KNOWN side, following the placement correction: the
    numerator and the denominator are then fitted on the same genre, so the
    handicap of explaining an out-of-genre questioned text is common to both and
    cancels in the difference.
    """
    pool = [s for a in sorted(B[genre]) if a not in exclude for s in B[genre][a]]
    rng.shuffle(pool)
    kept, tot = [], 0
    for s in pool:
        kept.append(s)
        tot += len(s)
        if tot >= need:
            break
    return rechunk(kept, seg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--known", type=int, default=20000,
                    help="TOTAL known tokens, identical in every arm")
    ap.add_argument("--quest", type=int, default=20000)
    ap.add_argument("--imp", type=int, default=4, help="impostors per case")
    ap.add_argument("--seg", type=int, default=100)
    ap.add_argument("--order", type=int, default=10)
    ap.add_argument("--r", type=int, default=30)
    ap.add_argument("--tag", default="")
    ap.add_argument("--authors", default="", help="restrict to these authors; "
                    "the eligible set is computed first and unchanged, so a "
                    "shard sees exactly the cases the full run would give it")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    OUT.mkdir(parents=True, exist_ok=True)

    B = banks()
    half = args.known // 2
    # An author qualifies only if EVERY genre can supply both roles, so the same
    # authors appear in every arm and every held-out direction. Selecting per arm
    # would let the arms differ by author composition.
    elig = sorted(a for a in B["prose"]
                  if all(a in B[g] and ntok(B[g][a]) >= max(args.known, args.quest)
                         for g in GENRES))
    print(f"{len(elig)} authors with >= {max(args.known, args.quest)} tokens in "
          f"all three genres:\n  {', '.join(elig)}\n", flush=True)
    if len(elig) < 6:
        print("too few authors; lower --known/--quest")
        return
    if args.authors:
        want = {x.strip() for x in args.authors.split(",") if x.strip()}
        elig = [a for a in elig if a in want]
        print(f"shard: {len(elig)} of them -> {', '.join(elig)}", flush=True)

    lg = LambdaG(N=args.order, r=args.r, engine="kn", random_state=0)
    fn = OUT / f"intersection__K{args.known}__Q{args.quest}{args.tag}.jsonl"
    recs, t0 = [], time.time()
    cid = 0

    for a in elig:
        for h in GENRES:                       # the HELD-OUT questioned genre
            g1, g2 = sorted(g for g in GENRES if g != h)
            rng = random.Random(f"{a}|{h}")
            imps = [x for x in sorted(B[h]) if x != a
                    and ntok(B[h][x]) >= args.quest]
            picks = rng.sample(imps, min(args.imp, len(imps)))
            for qa in [a] + picks:
                quest = rechunk(window(B[h][qa], 0, args.quest), args.seg)
                if ntok(quest) < args.quest:
                    continue
                k1 = rechunk(window(B[g1][a], 0, half), args.seg)
                k2 = rechunk(window(B[g2][a], 0, half), args.seg)
                full1 = rechunk(window(B[g1][a], 0, args.known), args.seg)
                full2 = rechunk(window(B[g2][a], 0, args.known), args.seg)
                # Pooled uses the SAME halves as the two components, so the arms
                # differ only in how the halves are combined, never in the text.
                pooled = rechunk(window(B[g1][a], 0, half)
                                 + window(B[g2][a], 0, half), args.seg)
                ex = {a, qa}
                need = max(100_000, 60 * args.known)
                r1 = refpool(B, g1, ex, args.seg, need, random.Random(f"{a}|{h}|1"))
                r2 = refpool(B, g2, ex, args.seg, need, random.Random(f"{a}|{h}|2"))

                out = {}
                lg.set_reference(r1)
                out["single_a"] = lg.score(quest, full1, with_details=False).lambda_G
                d1 = lg.score(quest, k1, with_details=True)
                lg.set_reference(r2)
                out["single_b"] = lg.score(quest, full2, with_details=False).lambda_G
                d2 = lg.score(quest, k2, with_details=True)
                # pooled spans both genres, so its reference population does too
                lg.set_reference(refpool(B, g1, ex, args.seg, need // 2,
                                         random.Random(f"{a}|{h}|p1"))
                                 + refpool(B, g2, ex, args.seg, need // 2,
                                           random.Random(f"{a}|{h}|p2")))
                out["pooled"] = lg.score(quest, pooled, with_details=False).lambda_G

                t1 = np.concatenate(d1.token_lambda)
                t2 = np.concatenate(d2.token_lambda)
                if len(t1) != len(t2):
                    raise RuntimeError(f"token misalignment {len(t1)} vs {len(t2)}")
                out["mean2"] = float(0.5 * (t1 + t2).sum())
                out["min2"] = float(np.minimum(t1, t2).sum())
                out["max2"] = float(np.maximum(t1, t2).sum())

                for arm, v in out.items():
                    recs.append({"id": cid, "label": int(qa == a), "author": a,
                                 "quest_author": qa, "held_out": h,
                                 "known_genres": f"{g1}+{g2}", "arm": arm,
                                 "lambda_G": v, "n_q": ntok(quest),
                                 "n_k": args.known})
                cid += 1
            print(f"  {a[:26]:28s} held-out {h:6s} "
                  f"{len(picks) + 1:2d} cases  {time.time() - t0:6.0f}s",
                  flush=True)

    with open(fn, "w", encoding="utf-8") as fh:
        for r in recs:
            fh.write(json.dumps(r) + "\n")
    print(f"\n{cid} cases x 6 arms = {len(recs)} records -> {fn.name}")


if __name__ == "__main__":
    main()
