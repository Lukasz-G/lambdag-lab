# Does VARIED known text beat MORE known text of one genre?
#
# THE QUESTION. Cross-genre verification is hard, and the obvious remedy is to
# supply more of the author's known writing. That remedy assumes the genre effect
# is a rate: the same grammar sampled more slowly, so more text closes the gap.
# The alternative is that an author's stationary process INCLUDES his genre
# switching, in which case any amount of prose converges to his prose
# conditional and never to his marginal -- and then no quantity of single-genre
# text closes the gap, while a smaller quantity spanning two genres does.
#
# The two make opposite predictions about a token-matched comparison, and they
# give a forensic examiner opposite instructions: get more known text, or get
# more VARIED known text.
#
# DESIGN. For each questioned genre C, the known side is fixed at K tokens and
# only its composition changes:
#
#   arm single:A   K tokens of genre A          (cross-genre w.r.t. C)
#   arm single:B   K tokens of genre B          (cross-genre w.r.t. C)
#   arm mixed      K/2 of genre A + K/2 of B    (cross-genre w.r.t. C)
#
# Every arm is cross-genre, so the contrast is composition alone and not distance
# from the questioned genre. Arms are PAIRED: the same questioned window, the
# same donor pool and the same author are scored under all three, so the
# comparison is within-case and the between-author variance cancels.
#
# THE DECISION RULE is mixed vs the BETTER single arm, never the average. Against
# the average, mixing could win merely by hedging which genre it drew from, which
# is not the claim under test.
#
#   python experiments/run_mixedknown.py
#   python experiments/run_mixedknown.py --known 6000 --windows 2
#
# Output: experiments/scores/mixedknown/{questioned}__{arm}.jsonl, one record per
# case, read by analyze_mixedknown.py.

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
OUT = SCORES / "mixedknown"


def banks():
    """{genre: {author_slug: sentences}} over the full-author banks."""
    out = {}
    for g, ds in GENRES.items():
        d = MASKED / ds / "bank"
        out[g] = {}
        for f in sorted(d.glob("*.tsv")):
            stem = f.stem.split("_", 1)[1] if f.stem[:3].isdigit() else f.stem
            out[g][stem] = read_tsv(f)
    return out


def ntok(sents):
    return sum(len(s) for s in sents)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--known", type=int, default=10000,
                    help="total known tokens, identical in every arm")
    ap.add_argument("--quest", type=int, default=2000)
    ap.add_argument("--windows", type=int, default=3,
                    help="questioned windows per author per direction")
    ap.add_argument("--rot", type=int, default=2,
                    help="different-author partners per questioned window")
    ap.add_argument("--seg", type=int, default=100, help="w-unit size")
    ap.add_argument("--order", type=int, default=10)
    ap.add_argument("--r", type=int, default=30)
    ap.add_argument("--max-donors", type=int, default=40)
    ap.add_argument("--split-control", action="store_true",
                    help="add same-genre split arms, which separate the cost of "
                         "FRAGMENTING the known text from the cost of mixing "
                         "genres; raises the per-author requirement to 3*K/2")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    OUT.mkdir(parents=True, exist_ok=True)

    B = banks()
    half = args.known // 2
    # authors present in all three genres with enough on every side
    three = [a for a in B["prose"]
             if all(a in B[g] for g in GENRES)]
    three = [a for a in three
             if all(ntok(B[g][a]) >= half for g in GENRES)]
    print(f"{len(three)} three-genre authors with >= {half:,} tokens everywhere",
          flush=True)

    lg = LambdaG(N=args.order, r=args.r, engine="kn", random_state=0)

    for C in GENRES:
        A, Bg = sorted(g for g in GENRES if g != C)
        # Eligibility demands that EVERY arm be runnable for this author, not
        # merely the one being scored. A single arm needs the full K from one
        # genre while the mixed arm needs only K/2 from each, so a laxer test
        # admits thin authors to the mixed arm alone -- the arms would then rest
        # on different case sets and mixed would be flattered by authors the
        # single arms could not attempt. The comparison is paired or it is
        # nothing.
        # The split control needs two NON-ADJACENT blocks from one genre, so it
        # reaches to 3*half; requiring that of both non-questioned genres keeps
        # every arm runnable for every eligible author.
        need = 3 * half if args.split_control else args.known
        elig = [a for a in three
                if ntok(B[C][a]) >= args.quest
                and min(ntok(B[A][a]), ntok(B[Bg][a])) >= need]
        print(f"\n=== questioned {C} (known from {A}+{Bg}): {len(elig)} authors ===",
              flush=True)

        # (genre, start, length). The split arms are the CONTROL: two blocks of
        # the same size as the mixed arm's, drawn from ONE genre but from
        # non-adjacent spans. If splitting alone costs what mixing costs, the
        # mixed arm's loss is fragmentation of the known text and says nothing
        # about genre; if splitting is free, genre is what the loss measures.
        arms = {f"single:{A}": [(A, 0, args.known)],
                f"single:{Bg}": [(Bg, 0, args.known)],
                "mixed": [(A, 0, half), (Bg, 0, half)]}
        if args.split_control:
            arms[f"split:{A}"] = [(A, 0, half), (A, 2 * half, half)]
            arms[f"split:{Bg}"] = [(Bg, 0, half), (Bg, 2 * half, half)]
        recs = {k: [] for k in arms}

        # questioned windows, disjoint from nothing: known comes from OTHER genres
        cases = []
        for a in elig:
            for w in range(args.windows):
                q = window(B[C][a], w * args.quest, args.quest)
                if ntok(q) < args.quest:
                    break
                cases.append((a, a, q, w))              # same-author
                rng = random.Random(f"{C}|{a}|{w}")
                partners = [x for x in elig if x != a]
                for b in rng.sample(partners, min(args.rot, len(partners))):
                    cases.append((a, b, q, w))          # different-author
        print(f"  {len(cases)} cases "
              f"({sum(1 for c in cases if c[0] == c[1])} same-author)", flush=True)

        t0 = time.time()
        for arm, spec in arms.items():
            # donor pool: C's own bank, size-matched, case authors excluded per case
            for ci, (qa, ka, q, w) in enumerate(cases):
                known = []
                ok = True
                for g, start, n in spec:
                    part = window(B[g][ka], start, n)
                    if ntok(part) < n:
                        ok = False
                        break
                    known.extend(part)
                if not ok:
                    continue
                pool = [s for n2 in sorted(B[C]) if n2 not in (qa, ka)
                        for s in B[C][n2]]
                rng = random.Random(f"{C}|{arm}|{qa}|{ka}|{w}")
                rng.shuffle(pool)
                kept, tot = [], 0
                for s in pool:
                    kept.append(s); tot += len(s)
                    if tot >= max(100_000, 60 * args.known):
                        break
                lg.set_reference(rechunk(kept, args.seg))
                res = lg.score(rechunk(q, args.seg), rechunk(known, args.seg),
                               with_details=False)
                recs[arm].append({"id": ci, "label": int(qa == ka),
                                  "q_author": qa, "k_author": ka, "window": w,
                                  "arm": arm, "questioned": C,
                                  "lambda_G": res.lambda_G,
                                  "n_q": ntok(q), "n_k": ntok(known)})
            fn = OUT / f"{C}__{arm.replace(':', '-')}.jsonl"
            with open(fn, "w", encoding="utf-8") as fh:
                for r in recs[arm]:
                    fh.write(json.dumps(r) + "\n")
            print(f"  {arm:16s} {len(recs[arm]):4d} cases -> {fn.name}", flush=True)
        print(f"  ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
