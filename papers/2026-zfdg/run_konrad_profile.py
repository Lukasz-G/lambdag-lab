# The Kaiserchronik read against the Rolandslied: is any of it Pfaffe Konrad's?
#
# THE QUESTION. Pfaffe Konrad names himself in the Rolandslied's epilogue, and
# an old and long-argued position holds that he wrote a large part of the
# Kaiserchronik as well; current handbooks treat that attribution as discredited
# rather than as settled either way, and both works belong to the same Regensburg
# milieu of the middle of the twelfth century. The question is therefore open in
# the only sense that matters here: it has never been put to a verifier that
# reports the weight of its own evidence.
#
# WHAT IS COMPARED. Known passages of the Rolandslied, in the fullest witness,
# against windows rolled along a whole witness of the Kaiserchronik, with the
# reference grammars drawn from other texts, one per donor, averaged. Three
# further sets of passages are scored against the SAME known passages so that
# the chronicle's windows can be read on a scale and not in the air:
#
#   the CEILING, passages of the Rolandslied's other witnesses. They carry the
#     same work in another hand, so they show what the top of this scale looks
#     like when author, work and everything else agree and only the scribe does
#     not. No window of the chronicle can be expected to reach it.
#   the COHORT, forty passages by other authors. Under the hypothesis that
#     Konrad did not write the chronicle, a window is exchangeable with them, so
#     its RANK among them is uniform -- a denominator that needs no fitting.
#   the ALEXANDERLIED, by Pfaffe Lambrecht, which the codex test uses as its
#     different author and which stands here as a named point on the floor.
#
# THE MASK IS OPTIONAL HERE, AND THAT IS THE POINT. The reuse mask was built on
# the premise that wording shared with another work is borrowed. For the
# Annolied that is documented. For the Rolandslied it is the very thing in
# question: shared wording is what borrowing produces AND what one author writing
# both works produces, and no count of shingles can separate the two. So the
# profile is computed both ways and both are reported; a signal that survives the
# mask cannot be the shared wording, and a signal that only appears without it
# may be either.
#
# THE TOLL RUNS AGAINST A POSITIVE. The chronicle and the Rolandslied are in
# different manuscripts, and the measured cost of crossing hands at this length
# is AUC 0.926 -> 0.824. A stretch of the chronicle that rises into Konrad's
# range in spite of that is worth something; a null result is the weaker
# direction and has to be read with the toll in mind.
#
#   python experiments/run_konrad_profile.py --mask none
#   python experiments/run_konrad_profile.py --mask reuse
#
# Output: scores/konrad_profile/{stream}__L{L}__w{seg}__{witness}__{mask}.jsonl

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
import build_kc_reuse_mask as M  # noqa: E402
import run_witness_cells as W  # noqa: E402
from lambdag import LambdaG  # noqa: E402

OUT = ROOT / "scores" / "konrad_profile"
REUSE = ROOT / "medieval" / "kc_reuse"
KNOWN_WIT = "rem__M205P"                 # Rolandslied, the fullest witness
CEILING_WITS = ["rem__M205A", "rem__M205S"]
ALEX = "rem__M009"
# texts whose relation to the chronicle is itself under discussion have no place
# in the denominator: the two Alexander poems, every Rolandslied witness, the
# Annolied, and the chronicle's own witnesses
EXCLUDE_PREFIX = ("rem__M205", "rem__M013")
EXCLUDE = {ALEX, "rem__M008"}


def toks(entry):
    return [t for s in entry["sents"] for t in s]


def spread(n_tokens, n, L):
    """n disjoint passages of L tokens, evenly spaced over a text"""
    if n_tokens < n * L:
        n = max(n_tokens // L, 1)
    room = n_tokens - n * L
    gap = room // max(n, 1)
    return [i * (L + gap) for i in range(n)]


def chronicle_tokens(bank, name, mask):
    """the witness's tokens, with the reuse-masked units dropped if asked

    The mask names bank LINES, which is what the verifier reads, so the two need
    no coordinate mapping; dropping them shortens the text and the profile's
    x-axis is therefore in tokens of what was scored, not of the witness.
    """
    sents = bank[name]["sents"]
    if mask == "none":
        return [t for s in sents for t in s], 0
    rec = json.loads((REUSE / f"{name}.json").read_text(encoding="utf-8"))
    drop = set(rec["masked_units"])
    kept = [t for i, s in enumerate(sents) if i not in drop for t in s]
    return kept, sum(len(s) for i, s in enumerate(sents) if i in drop)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stream", choices=["lemma", "surface"], default="lemma")
    ap.add_argument("--witness", default="kcd__anon__kaiserchronik__A1")
    ap.add_argument("--mask", choices=["none", "reuse"], default="none")
    ap.add_argument("--known-witness", default=KNOWN_WIT,
                    help="whose known passages the chronicle is read against; "
                         "the control run puts another author of the same "
                         "milieu here, and the chronicle should then fall to "
                         "the level the Alexanderlied reaches against Konrad")
    ap.add_argument("--ceiling", default=",".join(CEILING_WITS),
                    help="witnesses of the known text's own work, the top of "
                         "the scale")
    ap.add_argument("--named", default=ALEX,
                    help="further texts scored as named points")
    ap.add_argument("--tag", default="", help="suffix for the output file")
    ap.add_argument("--known-length", type=int, default=0,
                    help="tokens per known passage; 0 means the window length. "
                         "Five passages of two thousand tokens gave profiles "
                         "that contradicted one another, so the known side is "
                         "the lever: with 0 passages of the whole text the "
                         "model is built on everything the author left.")
    ap.add_argument("--donor-length", type=int, default=0,
                    help="tokens per reference grammar; 0 means the window "
                         "length. Raising it with the known keeps the "
                         "reference models comparable to the known model, at "
                         "the price of a smaller pool of texts long enough.")
    ap.add_argument("--donor-per-author", action="store_true",
                    help="one donor text per author, longest first. Sixteen of "
                         "the twenty-three texts above 43,000 tokens are "
                         "witnesses of one work, so a long donor pool drawn at "
                         "random is a population of one identity.")
    ap.add_argument("--L", type=int, default=2000)
    ap.add_argument("--step", type=int, default=500)
    ap.add_argument("--seg", type=int, default=100)
    ap.add_argument("--kpassages", type=int, default=5)
    ap.add_argument("--donors", type=int, default=15)
    ap.add_argument("--cohort", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    L, seg = args.L, args.seg
    rng = random.Random(args.seed)
    bank = W.load(args.stream)

    kw = args.known_witness
    ceiling_wits = [w for w in args.ceiling.split(",") if w]
    named_wits = [w for w in args.named.split(",") if w]
    known_toks = toks(bank[kw])
    klen = args.known_length or L
    dlen = args.donor_length or L
    kstarts = spread(len(known_toks), args.kpassages, klen)
    known = [known_toks[s:s + klen] for s in kstarts]
    chron, dropped = chronicle_tokens(bank, args.witness, args.mask)
    print(f"known: {kw} ({bank[kw]['author']}), {len(known)} passages of "
          f"{len(known[0]):,} tokens from {len(known_toks):,}")
    print(f"questioned: {args.witness}, {len(chron):,} tokens"
          + (f" after dropping {dropped:,} masked by the reuse mask"
             if args.mask != "none" else ""))

    pool = [n for n in bank
            if not M.is_kc(n, bank) and n not in EXCLUDE
            and not n.startswith(EXCLUDE_PREFIX)
            and bank[n]["ntok"] >= 2 * L]
    pool.sort()
    rng.shuffle(pool)
    long_enough = [n for n in pool if bank[n]["ntok"] >= dlen]
    if args.donor_per_author:
        best = {}
        for n in sorted(long_enough, key=lambda n: -bank[n]["ntok"]):
            best.setdefault(bank[n]["author"], n)
        long_enough = sorted(best.values())
        rng.shuffle(long_enough)
    dn = long_enough[:args.donors]
    cn = [n for n in pool if n not in dn][:args.cohort]
    refs, donor_names = [], []
    for n in dn:
        t = toks(bank[n])
        st = rng.randrange(0, len(t) - dlen + 1)
        refs.append(W.rechunk(t[st:st + dlen], seg))
        donor_names.append(f"{n}@{st}")
    print(f"reference: {len(refs)} grammars of {dlen:,} tokens from "
          f"{len(set(bank[n]['author'] for n in dn))} authors; "
          f"cohort: {len(cn)} passages by other authors")

    # everything that is scored besides the windows
    extra = []
    for w in ceiling_wits:
        if w not in bank:
            continue
        t = toks(bank[w])
        for st in spread(len(t), 4, L):
            if st + L <= len(t):
                extra.append(("ceiling", w, st, t[st:st + L]))
    for w in named_wits:
        t = toks(bank[w])
        for st in spread(len(t), 4, L):
            if st + L <= len(t):
                extra.append(("named", w, st, t[st:st + L]))
    for n in cn:
        t = toks(bank[n])
        st = rng.randrange(0, len(t) - L + 1)
        extra.append(("cohort", n, st, t[st:st + L]))

    wins = [(i, chron[i:i + L]) for i in range(0, len(chron) - L + 1, args.step)]
    print(f"{len(wins)} windows + {len(extra)} reference passages, "
          f"x {len(known)} known x {len(refs)} donors = "
          f"{(len(wins) + len(extra)) * len(known) * len(refs):,} scores")
    if args.dry_run:
        return

    lg = LambdaG(N=10, r=1, engine="kn", random_state=0)
    kunits = [W.rechunk(k, seg) for k in known]
    t0, rows = time.time(), []

    def score(q):
        return [float(np.mean([lg.score(q, ku, ref_sentences=ref, r=1,
                                        with_details=False).lambda_G
                               for ref in refs])) for ku in kunits]

    for kind, name, st, t in extra:
        lam = score(W.rechunk(t, seg))
        rows.append(dict(kind=kind, text=name, start=st,
                         author=bank[name]["author"],
                         lambda_G=round(float(np.mean(lam)), 4),
                         per_known=[round(x, 4) for x in lam]))
    print(f"  reference passages done in {time.time() - t0:.0f}s", flush=True)
    for w, (start, t) in enumerate(wins):
        lam = score(W.rechunk(t, seg))
        rows.append(dict(kind="window", w=w, start=start,
                         lambda_G=round(float(np.mean(lam)), 4),
                         per_known=[round(x, 4) for x in lam]))
        if (w + 1) % 20 == 0:
            el = time.time() - t0
            print(f"    {w + 1}/{len(wins)} windows, {el:.0f}s", flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / (f"{args.stream}__L{L}__w{seg}__"
               f"{args.witness.split('__')[-1]}__{args.mask}"
               f"{args.tag}.jsonl")
    with p.open("w", encoding="utf-8") as fh:
        fh.write(json.dumps(dict(kind="meta", stream=args.stream, L=L, seg=seg,
                                 step=args.step, witness=args.witness,
                                 mask=args.mask, known=kw,
                                 known_starts=kstarts, donors=donor_names,
                                 n_tokens=len(chron), dropped=dropped)) + "\n")
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    win = np.array([r["lambda_G"] for r in rows if r["kind"] == "window"])
    coh = np.array([r["lambda_G"] for r in rows if r["kind"] == "cohort"])
    ceil_ = np.array([r["lambda_G"] for r in rows if r["kind"] == "ceiling"])
    named = {w: np.array([r["lambda_G"] for r in rows
                          if r["kind"] == "named" and r["text"] == w])
             for w in named_wits}
    # the rank of each window among the cohort: under different authorship it is
    # uniform, so the share of windows in the cohort's top decile should be 10 %
    rank = np.array([(coh < v).sum() / len(coh) for v in win])
    print(f"\n  the known text's other witnesses (same work, other hand): "
          f"median {np.median(ceil_):8.1f}")
    print(f"  the chronicle's windows:                                 "
          f"median {np.median(win):8.1f}   best {win.max():8.1f}")
    for w, v in named.items():
        lab = f"{w} ({bank[w]['author']})"
        print(f"  {lab:<55} median {np.median(v):8.1f}")
    print(f"  passages by other authors (the cohort):                  "
          f"median {np.median(coh):8.1f}")
    print(f"  windows above the cohort's 90th percentile: "
          f"{np.mean(rank >= 0.9):.1%} (uniform would be 10%)")
    print(f"  windows above EVERY cohort passage: {np.mean(rank == 1):.1%}")
    top = np.argsort(-win)[:8]
    print("  the highest windows, by position in what was scored:")
    for i in top:
        print(f"    token {wins[i][0]:7,}  lambda {win[i]:8.1f}  "
              f"rank {rank[i]:.3f}")
    print(f"  wrote {p.relative_to(ROOT)} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
