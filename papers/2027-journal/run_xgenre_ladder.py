# Cross-genre accumulation exponent: is genre a RATE or a BOUNDARY?
#
# Within genre the exponent is already known to be near 1 -- with the known side
# pinned, author evidence accumulates linearly and beta = 1/2 is excluded. But
# that measurement (run_pinned_ladder.py) was made on ELTeC, DraCor and PoeTree,
# so using it as the baseline here would confound the comparison with edition
# tradition, author set and period. The within-genre directions are therefore
# measured HERE, on these banks, and shipped as part of the same run.
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
# Nine directions: every ordered pair of the three genres, the three same-genre
# pairs included as the baseline. Questioned windows are
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
# The same banks under two encodings. POSNoise keeps the function word itself;
# CatSRank folds it into class x frequency rank. Whether the accumulation result
# is a fact about authorship or about one representation cannot be told from a
# single alphabet, and the trend -- not the gap -- is where the question lives.
# dep_* replace the alphabet with `relation·symbol`, one symbol per token, so
# stream length is unchanged and per-token rates stay comparable. linear keeps
# surface order and is the control; traversal walks the tree head-first, which
# removes surface order and therefore metre.
ALPHABETS = {"posnoise": (None, ""),
             "catsrank": (Path("masked_catsrank"), "_heldout"),
             "dep_linear": (Path("masked_dep"), "_linear"),
             "dep_traversal": (Path("masked_dep"), "_traversal")}
# PINNED: the known side is held fixed and only the questioned block grows, so
# the slope is an accumulation exponent and not the estimator's own convergence.
# 150 -> 20000 is a 133x span; the earlier 16x was too short to fit a slope
# through, and 2400 questioned tokens is in any case too little to see a
# cross-genre boundary, where the signal is far weaker than within genre.
QS_PINNED = [150, 600, 2400, 10000, 20000]
# SYMMETRIC: both sides grow together. This CANNOT be read as an exponent --
# growing the known side improves G_A's estimate at the same time -- but it is
# the operational curve, the one that answers how much text a case needs, and it
# is what an examiner is actually told. The two ladders answer different
# questions and neither substitutes for the other.
QS_SYM = [500, 1000, 2000, 5000, 10000, 20000]
K_DEFAULT = 20000
OUT = SCORES / "xgenre_ladder"


def banks(root, suffix):
    """Load every genre's bank, with the symbols interned.

    Interning is not a micro-optimisation here. The three banks are 40M tokens,
    and without it each job holds 40M distinct str objects -- some 3GB, which a
    box running one job per core cannot survive. The alphabet is at most a few
    thousand symbols, so interning collapses the strings to one copy each and
    leaves an array of pointers. It matters most for the dependency alphabets,
    whose symbols are the longest.
    """
    out = {}
    for g, ds in GENRES.items():
        d = root / f"{ds}{suffix}" / "bank"
        out[g] = {}
        for f in sorted(d.glob("*.tsv")):
            stem = f.stem.split("_", 1)[1] if f.stem[:3].isdigit() else f.stem
            out[g][stem] = [[sys.intern(t) for t in s] for s in read_tsv(f)]
    return out


def ntok(s):
    return sum(len(x) for x in s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alphabet", default="posnoise", choices=sorted(ALPHABETS),
                    help="which encoding of the same banks to score")
    ap.add_argument("--mode", default="both",
                    choices=("pinned", "symmetric", "both"))
    ap.add_argument("--known", type=int, default=K_DEFAULT,
                    help="pinned known length; only the questioned block grows")
    ap.add_argument("--qs", default="", help="override the rungs")
    ap.add_argument("--directions", default="",
                    help="comma-separated known2quest, e.g. prose2verse")
    ap.add_argument("--donor-genre", default="questioned",
                    choices=("questioned", "known", "both"),
                    help="genre the reference population is drawn from. 'known' "
                         "is the genre-CONTROLLED design and should be the "
                         "primary one: G_A and the reference grammars are then "
                         "fitted on the same genre, so the handicap of "
                         "explaining an out-of-genre questioned text is common "
                         "to numerator and denominator and cancels, leaving "
                         "identity. 'questioned' (the default, for continuity "
                         "with earlier runs) hands the denominator a genre "
                         "advantage the numerator lacks. 'both' gives a "
                         "genre-neutral population.")
    ap.add_argument("--rot", type=int, default=2)
    ap.add_argument("--max-authors", type=int, default=40,
                    help="cap per direction; the within-genre directions draw on "
                         "the whole bank and would otherwise dominate the run")
    ap.add_argument("--seg", type=int, default=100)
    ap.add_argument("--order", type=int, default=10)
    ap.add_argument("--r", type=int, default=30)
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    OUT.mkdir(parents=True, exist_ok=True)

    aroot, asuf = ALPHABETS[args.alphabet]
    aroot = MASKED if aroot is None else MASKED.parent / aroot
    B = banks(aroot, asuf)
    print(f"alphabet: {args.alphabet}  banks: {aroot.name}", flush=True)
    modes = (["pinned", "symmetric"] if args.mode == "both" else [args.mode])
    dirs = ([tuple(d.split("2")) for d in args.directions.split(",") if d]
            or [(a, b) for a in GENRES for b in GENRES])

    lg = LambdaG(N=args.order, r=args.r, engine="kn", random_state=0)

    for mode in modes:
        default = QS_PINNED if mode == "pinned" else QS_SYM
        qs = sorted(int(q) for q in args.qs.split(",")) if args.qs else default
        qmax = max(qs)
        for kg, qg in dirs:
            same = (kg == qg)
            tag = f"{kg}2{qg}__{mode}"
            dsuf = ("" if args.donor_genre == "questioned"
                    else f"__d-{args.donor_genre}")
            if args.alphabet != "posnoise":
                dsuf += f"__{args.alphabet}"
            fn = OUT / (f"{tag}__K{args.known}{dsuf}.jsonl" if mode == "pinned"
                        else f"{tag}{dsuf}.jsonl")
            if fn.exists():
                print(f"{tag}: exists, skipped", flush=True)
                continue

            # Budgets differ by mode. Pinned: K in the known genre, qmax in the
            # questioned one. Symmetric: qmax on BOTH sides, since the known side
            # grows with the questioned. Where the genres coincide the questioned
            # block starts beyond the known one, so they never overlap.
            need_k = args.known if mode == "pinned" else qmax
            elig = sorted(a for a in B[kg]
                          if a in B[qg] and ntok(B[kg][a]) >= need_k
                          and ntok(B[qg][a]) >= (need_k + qmax if same else qmax))
            if len(elig) < 6:
                print(f"{tag}: only {len(elig)} authors, skipped", flush=True)
                continue
            if len(elig) > args.max_authors:
                elig = sorted(random.Random(f"{tag}|cap").sample(
                    elig, args.max_authors))
            print(f"\n=== {mode}: known {kg} -> questioned {qg}: "
                  f"{len(elig)} authors ===", flush=True)

            cases = []
            for a in elig:
                cases.append((a, a))
                rng = random.Random(f"{tag}|{a}")
                others = [x for x in elig if x != a]
                for b in rng.sample(others, min(args.rot, len(others))):
                    cases.append((a, b))

            t0, recs = time.time(), []
            for ci, (qa, ka) in enumerate(cases):
                qfull = window(B[qg][qa], need_k if same else 0, qmax)
                # Which genre the reference population comes from decides what
                # the difference measures. Drawn from the QUESTIONED genre, the
                # denominator is genre-matched to the questioned text while G_A
                # is not, so the reference explains it better for reasons of
                # genre alone and the score is depressed by the design. Drawn
                # from the KNOWN genre, both models are fitted on the same genre
                # and the handicap is common to numerator and denominator, so it
                # cancels and identity is what remains.
                if args.donor_genre == "known":
                    src = [kg]
                elif args.donor_genre == "both":
                    src = sorted({kg, qg})
                else:
                    src = [qg]
                pool = [s for g in src for n in sorted(B[g])
                        if n not in (qa, ka) for s in B[g][n]]
                rng = random.Random(f"{tag}|{qa}|{ka}")
                rng.shuffle(pool)
                kept, tot = [], 0
                for s in pool:
                    kept.append(s); tot += len(s)
                    if tot >= max(100_000, 60 * max(args.known, qmax)):
                        break
                ref = rechunk(kept, args.seg)

                if mode == "pinned":
                    # set once: K is fixed, so the r size-matched reference
                    # grammars are identical across rungs and the slope reflects
                    # the questioned block alone
                    known = window(B[kg][ka], 0, args.known)
                    lg.set_reference(ref)
                for q in qs:
                    if mode == "symmetric":
                        known = window(B[kg][ka], 0, q)
                        if ntok(known) < q:
                            continue
                        lg.set_reference(ref)
                    quest = window(qfull, 0, q)
                    if ntok(quest) < q:
                        continue
                    res = lg.score(rechunk(quest, args.seg),
                                   rechunk(known, args.seg), with_details=False)
                    recs.append({"id": ci, "label": int(qa == ka), "Q": q,
                                 "mode": mode, "known_genre": kg,
                                 "quest_genre": qg, "within_genre": int(same),
                                 "donor_genre": args.donor_genre,
                                 "lambda_G": res.lambda_G,
                                 "n_q": ntok(quest), "n_k": ntok(known)})
            with open(fn, "w", encoding="utf-8") as fh:
                for r in recs:
                    fh.write(json.dumps(r) + "\n")
            print(f"  {len(cases)} cases x {len(qs)} rungs = {len(recs)} records "
                  f"in {time.time() - t0:.0f}s -> {fn.name}", flush=True)


if __name__ == "__main__":
    main()
