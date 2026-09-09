# What is genre WORTH, in tokens?
#
# WHY THIS AND NOT ANOTHER LEVER. Five interventions have now failed to move the
# cross-genre ranking: four alphabets (POSNoise, class-by-rank, and two
# dependency encodings), and the model intersection, which was not merely null
# but actively harmful. The unit was ruled out on arithmetic. At that point a
# sixth lever is a worse investment than a measurement, so this stops asking how
# to remove the genre penalty and asks how large it is in the only currency an
# examiner can act on: how much text.
#
# THE QUESTION. Take an author known through 20,000 tokens of the WRONG genre,
# and add M tokens of the RIGHT one. How large must M be before the case is worth
# what a same-genre case is worth? That number is the exchange rate, and it is
# also the n* of the accumulation argument, expressed as a quantity rather than
# as an asymptotic claim.
#
# THE CONTROL, without which the curve means nothing. Adding text improves G_A
# whatever genre it comes from -- more data is more data -- so a rising curve
# proves nothing on its own. Every rung is therefore run twice:
#   add_h   20,000 tokens of genre g, plus M tokens of the QUESTIONED genre h
#   add_g   20,000 tokens of genre g, plus M tokens of MORE genre g
# Identical token budgets, identical everything else. The exchange rate is the
# separation between the two curves; their common component is just sample size.
#
# DISJOINTNESS, which is the trap here. The M tokens of genre h are training
# text, and the questioned text is also genre h from the same author. If they
# overlap the result is worthless in the most flattering possible direction. So a
# fixed reserve of MMAX tokens at the head of every author's h-genre bank is set
# aside for training, and questioned text is drawn only from beyond it -- at the
# SAME offset for impostors, so no arm gains an easier questioned block.
#
# The reference population is drawn from genre g and held FIXED across M: the
# curve is the object of study, so nothing else may vary along it.
#
#   python experiments/run_exchange_rate.py
#   python experiments/run_exchange_rate.py --authors tieck_ludwig --tag _s0
#
# Output: experiments/scores/exchange/exchange__K{K}__Q{Q}{tag}.jsonl
#         one record per (case, M, arm).

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
OUT = SCORES / "exchange"
MS = [0, 250, 500, 1000, 2000, 5000]


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--known", type=int, default=20000, help="cross-genre base")
    ap.add_argument("--quest", type=int, default=20000)
    ap.add_argument("--imp", type=int, default=4)
    ap.add_argument("--seg", type=int, default=100)
    ap.add_argument("--order", type=int, default=10)
    ap.add_argument("--r", type=int, default=30)
    ap.add_argument("--ms", default="")
    ap.add_argument("--authors", default="")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    OUT.mkdir(parents=True, exist_ok=True)

    ms = sorted(int(x) for x in args.ms.split(",")) if args.ms else MS
    mmax = max(ms)
    need = max(mmax + args.quest, args.known + mmax)
    B = banks()
    elig = sorted(a for a in B["prose"]
                  if all(a in B[g] and ntok(B[g][a]) >= need for g in GENRES))
    print(f"{len(elig)} authors with >= {need} tokens in all three genres",
          flush=True)
    if args.authors:
        want = {x.strip() for x in args.authors.split(",") if x.strip()}
        elig = [a for a in elig if a in want]
        print(f"shard: {', '.join(elig)}", flush=True)
    if not elig:
        return

    lg = LambdaG(N=args.order, r=args.r, engine="kn", random_state=0)
    fn = OUT / f"exchange__K{args.known}__Q{args.quest}{args.tag}.jsonl"
    recs, t0, cid = [], time.time(), 0

    for a in elig:
        for h in GENRES:                      # questioned genre
            for g in sorted(x for x in GENRES if x != h):   # known base genre
                rng = random.Random(f"{a}|{h}|{g}")
                imps = [x for x in sorted(B[h]) if x != a
                        and ntok(B[h][x]) >= mmax + args.quest]
                picks = rng.sample(imps, min(args.imp, len(imps)))
                # Fixed reference population from the base genre, drawn ONCE per
                # case and reused at every rung: the curve must move because of M
                # and nothing else.
                pool = [s for n in sorted(B[g]) if n not in (a,)
                        for s in B[g][n]]
                rng2 = random.Random(f"{a}|{h}|{g}|ref")
                rng2.shuffle(pool)
                kept, tot = [], 0
                for s in pool:
                    kept.append(s)
                    tot += len(s)
                    if tot >= max(100_000, 60 * (args.known + mmax)):
                        break
                ref = rechunk(kept, args.seg)
                lg.set_reference(ref)

                base = window(B[g][a], 0, args.known)
                for qa in [a] + picks:
                    # questioned text starts BEYOND the training reserve, for
                    # every case alike, so impostors face the same block
                    quest = rechunk(window(B[h][qa], mmax, args.quest), args.seg)
                    if ntok(quest) < args.quest:
                        continue
                    for m in ms:
                        for arm in (("add_h", "add_g") if m else ("add_h",)):
                            extra = (window(B[h][a], 0, m) if arm == "add_h"
                                     else window(B[g][a], args.known, m))
                            if ntok(extra) < m:
                                continue
                            known = rechunk(base + extra, args.seg)
                            v = lg.score(quest, known, with_details=False)
                            recs.append({
                                "id": cid, "label": int(qa == a), "author": a,
                                "quest_author": qa, "quest_genre": h,
                                "known_genre": g, "M": m, "arm": arm,
                                "lambda_G": v.lambda_G, "n_q": ntok(quest),
                                "n_k": ntok(known)})
                    cid += 1
                print(f"  {a[:24]:26s} {g}->{h:6s} {len(picks) + 1:2d} cases "
                      f"{time.time() - t0:6.0f}s", flush=True)

    with open(fn, "w", encoding="utf-8") as fh:
        for r in recs:
            fh.write(json.dumps(r) + "\n")
    print(f"\n{cid} cases -> {len(recs)} records -> {fn.name}")


if __name__ == "__main__":
    main()
