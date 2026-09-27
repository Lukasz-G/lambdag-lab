# The other direction: each region of the Kaiserchronik as the known text, and
# Konrad's poem as the questioned one.
#
# The first attempt at the Konrad question put his poem on the known side in
# passages of two thousand tokens, and those passages disagreed with one another
# about which parts of the chronicle they liked -- so the known side was too
# thin to carry an attribution. This turns the comparison around and uses the
# chronicle's own length: it is cut into regions of several thousand tokens,
# each region is made the known text in turn, and the Rolandslied is scored
# against it along with the controls.
#
# The question a region answers is therefore: does THIS stretch of the chronicle
# claim the Rolandslied more strongly than a stretch of it should? And because
# the hypothesis is that Konrad wrote a PART, the answer is allowed to differ
# from region to region -- which is the whole point of asking it this way.
#
# Scored against every region, on the same reference grammars:
#   the Rolandslied, in its two long witnesses, as the candidate;
#   the Alexanderlied, a different author of the same milieu, as the control
#     that says how high the period alone reaches;
#   passages by other authors, as the cohort that sets the scale;
#   passages of the chronicle from OTHER regions, which are the same work and
#     therefore the top of the scale, not a comparison.
#
#   python experiments/run_kc_regions.py
#
# Output: scores/konrad_profile/regions__{witness}__r{regions}__q{L}.jsonl

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
CANDIDATES = ["rem__M205P", "rem__M205A"]
CONTROL = ["rem__M009"]
EXCLUDE_PREFIX = ("rem__M205", "rem__M013")
EXCLUDE = {"rem__M009", "rem__M008"}


def toks(entry):
    return [t for s in entry["sents"] for t in s]


def cut(t, L, n=None):
    out = [t[i:i + L] for i in range(0, len(t) - L + 1, L)]
    return out if n is None else out[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stream", choices=["lemma", "surface"], default="lemma")
    ap.add_argument("--encoding", default="posnoise",
                    help="posnoise for the masked grammatical stream, catranks "
                         "for the shared alphabet a borrowed reference "
                         "population needs")
    ap.add_argument("--donor-bank", default="",
                    help="draw the reference grammars from ANOTHER bank -- the "
                         "borrowed population. It requires the shared alphabet "
                         "on both sides, so it is used with --encoding "
                         "catranks; the impostor cohort stays inside the case "
                         "corpus, since only the denominator is borrowed.")
    ap.add_argument("--witness", default="kcd__anon__kaiserchronik__A1")
    ap.add_argument("--regions", type=int, default=20)
    ap.add_argument("--L", type=int, default=2000, help="questioned length")
    ap.add_argument("--seg", type=int, default=100)
    ap.add_argument("--donors", type=int, default=12)
    ap.add_argument("--cohort", type=int, default=40)
    ap.add_argument("--cohort-period", default="",
                    help="restrict the cohort of strangers to witnesses of one "
                         "period. The masked stream still carries the age of "
                         "the copy -- on this corpus a cohort of early "
                         "witnesses sits two thirds of a deviation above a "
                         "cohort of late ones, and the Ambraser's sixteenth-"
                         "century copies more than two below -- so a cohort of "
                         "the wrong date lifts or sinks every candidate that "
                         "survives in a manuscript of the right one.")
    ap.add_argument("--cohort-source", default="",
                    help="restrict the cohort of strangers to one corpus, so "
                         "that the scale is set by texts annotated as the "
                         "candidate was. A cohort drawn mostly from other "
                         "corpora raises every text of the candidate's corpus "
                         "against it, and that offset would be read as milieu.")
    ap.add_argument("--candidates", default=",".join(CANDIDATES),
                    help="the texts whose authorship against the known regions "
                         "is in question")
    ap.add_argument("--control", default=",".join(CONTROL),
                    help="the texts that say how high a poem of the period "
                         "reaches without being the candidate. One control is "
                         "a point and cannot separate the author from the "
                         "genre; several make a distribution.")
    ap.add_argument("--tag", default="")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    L, seg = args.L, args.seg
    rng = random.Random(args.seed)
    bank = W.load(args.stream, args.encoding)

    kc = toks(bank[args.witness])
    rlen = len(kc) // args.regions
    regions = [(i * rlen, kc[i * rlen:(i + 1) * rlen])
               for i in range(args.regions)]
    print(f"{args.witness}: {len(kc):,} tokens in {args.regions} regions of "
          f"{rlen:,}")

    pool = [n for n in bank
            if not M.is_kc(n, bank) and n not in EXCLUDE
            and not n.startswith(EXCLUDE_PREFIX)
            and bank[n]["ntok"] >= max(2 * L, rlen)]
    pool.sort()
    rng.shuffle(pool)
    best = {}
    for n in sorted(pool, key=lambda n: -bank[n]["ntok"]):
        best.setdefault(bank[n]["author"], n)
    dn = sorted(best.values())[:args.donors]
    cn = [n for n in pool if n not in dn
          and (not args.cohort_source or n.startswith(args.cohort_source))
          and (not args.cohort_period
               or bank[n]["period"] == args.cohort_period)]
    cn = cn[:args.cohort]
    dbank = W.load_donors(args.donor_bank) if args.donor_bank else bank
    if args.donor_bank:
        dn = sorted(dbank, key=lambda n: -len(toks(dbank[n])))[:args.donors]
    refs, donor_names = [], []
    for n in dn:
        t = toks(dbank[n])
        st = rng.randrange(0, len(t) - rlen + 1)
        refs.append(W.rechunk(t[st:st + rlen], seg))
        donor_names.append(f"{n}@{st}")
    print(f"reference: {len(refs)} grammars of {rlen:,} tokens from "
          f"{len(set(dbank[n].get('author', n) for n in dn))} authors")

    # everything that is scored against every region
    quest = []
    for w in [x for x in args.candidates.split(",") if x]:
        for j, q in enumerate(cut(toks(bank[w]), L, 8)):
            quest.append(("candidate", w, j, q))
    for w in [x for x in args.control.split(",") if x]:
        for j, q in enumerate(cut(toks(bank[w]), L, 4)):
            quest.append(("control", w, j, q))
    for n in cn:
        t = toks(bank[n])
        st = rng.randrange(0, len(t) - L + 1)
        quest.append(("cohort", n, st, t[st:st + L]))
    print(f"{len(quest)} passages x {len(regions)} regions x {len(refs)} "
          f"donors = {len(quest) * len(regions) * len(refs):,} scores")
    if args.dry_run:
        return

    lg = LambdaG(N=10, r=1, engine="kn", random_state=0)
    t0, rows = time.time(), []
    for ri, (start, reg) in enumerate(regions):
        ku = W.rechunk(reg, seg)
        # the chronicle's own other regions, cut to the questioned length, are
        # the top of this region's scale: same work, so not a comparison
        own = [q for oi, (_, r2) in enumerate(regions) if oi != ri
               for q in cut(r2, L, 1)]
        rng.shuffle(own)
        todo = quest + [("own", args.witness, -1, q) for q in own[:6]]
        for kind, name, j, q in todo:
            lam = float(np.mean([lg.score(W.rechunk(q, seg), ku,
                                          ref_sentences=ref, r=1,
                                          with_details=False).lambda_G
                                 for ref in refs]))
            rows.append(dict(kind=kind, region=ri, region_start=start,
                             text=name, j=j, lambda_G=round(lam, 4)))
        el = time.time() - t0
        print(f"  region {ri + 1}/{len(regions)} at token {start:7,}, "
              f"{el:.0f}s", flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / (f"regions__{args.witness.split('__')[-1]}__"
               f"r{args.regions}__q{L}{args.tag}.jsonl")
    with p.open("w", encoding="utf-8") as fh:
        fh.write(json.dumps(dict(kind="meta", stream=args.stream, L=L, seg=seg,
                                 witness=args.witness, regions=args.regions,
                                 region_len=rlen, donors=donor_names)) + "\n")
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\n  per region, in robust deviations of that region's own cohort")
    print(f"  {'region':>8} {'tokens':>9} {'Rolandslied':>12} "
          f"{'Alexanderlied':>14} {'the chronicle':>14}")
    for ri, (start, _) in enumerate(regions):
        sel = [r for r in rows if r["region"] == ri]
        coh = np.array([r["lambda_G"] for r in sel if r["kind"] == "cohort"])
        m = float(np.median(coh))
        sd = max(1.4826 * float(np.median(np.abs(coh - m))),
                 (float(np.quantile(coh, 0.9)) - m) / 1.2816, 1e-9)
        def z(kind):
            v = [r["lambda_G"] for r in sel if r["kind"] == kind]
            return (float(np.median(v)) - m) / sd if v else float("nan")
        print(f"  {ri + 1:8} {start:9,} {z('candidate'):12.1f} "
              f"{z('control'):14.1f} {z('own'):14.1f}")
    print(f"  wrote {p.relative_to(ROOT)} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
