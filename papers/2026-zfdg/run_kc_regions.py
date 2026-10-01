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
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
import build_kc_reuse_mask as M  # noqa: E402
import run_calibration_cases as C  # noqa: E402  (its edition-bank loader)
import run_witness_cells as W  # noqa: E402
from lambdag import LambdaG  # noqa: E402

OUT = ROOT / "scores" / "konrad_profile"
CANDIDATES = ["rem__M205P", "rem__M205A"]
CONTROL = ["rem__M009"]
EXCLUDE_PREFIX = ("rem__M205", "rem__M013")
EXCLUDE = {"rem__M009", "rem__M008"}
# The defaults above keep the candidate, the control and their sister witnesses
# out of the grammars and out of the scale for the Kaiserchronik case. Another
# case has other texts to keep out -- a work whose authorship is the question may
# not set the scale it is measured against -- so --exclude supplies them, and a
# name ending in a star excludes everything beginning with it.


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
    ap.add_argument("--donor-years", default="",
                    help="the period of the edition grammars, by the poem's "
                         "date; defaults to the cohort's")
    ap.add_argument("--witness", default="kcd__anon__kaiserchronik__A1")
    ap.add_argument("--case-bank", choices=["witnesses", "editions"],
                    default="witnesses",
                    help="which layer the CASE is in -- the known text, the "
                         "candidates and the controls. Every case so far has "
                         "been manuscript text, with the scale borrowed from the "
                         "editions and the layer offset named as a residual. "
                         "editions puts the whole case in the editors' layer "
                         "instead, so that a relational result can be repeated "
                         "with no layer boundary anywhere in it: a verdict that "
                         "holds on a scribe's copy and on an editor's text is a "
                         "property of the poems and not of either hand.")
    ap.add_argument("--case-years", default="",
                    help="the period filter for an edition case; defaults to "
                         "the cohort's")
    ap.add_argument("--regions", type=int, default=20)
    ap.add_argument("--L", type=int, default=2000, help="questioned length")
    ap.add_argument("--seg", type=int, default=100)
    ap.add_argument("--donors", type=int, default=12)
    ap.add_argument("--cohort", type=int, default=40)
    ap.add_argument("--donor-period", default="",
                    help="restrict the reference population to witnesses of one "
                         "period. The denominator is what the case is measured "
                         "against, so a population of the case's own century "
                         "absorbs the milieu into it and leaves only what is "
                         "more specific; a broad population leaves the milieu "
                         "unexplained and lifts every text of that century.")
    ap.add_argument("--cohort-period", default="",
                    help="restrict the cohort of strangers to witnesses of one "
                         "period. The masked stream still carries the age of "
                         "the copy -- on this corpus a cohort of early "
                         "witnesses sits two thirds of a deviation above a "
                         "cohort of late ones, and the Ambraser's sixteenth-"
                         "century copies more than two below -- so a cohort of "
                         "the wrong date lifts or sinks every candidate that "
                         "survives in a manuscript of the right one.")
    ap.add_argument("--cohort-bank", choices=["witnesses", "editions", "both"],
                    default="witnesses",
                    help="where the scale comes from. The witness corpus holds "
                         "only fourteen authors whose copies are early enough, "
                         "and twelve of them are needed for the grammars, so one "
                         "text per author leaves six; the tagged editions add a "
                         "dozen more authors of the poem's own period at the "
                         "price of a different layer -- an editor's normalised "
                         "text against a scribe's. both scores each cohort "
                         "separately so that the offset between the two layers "
                         "is measured instead of assumed.")
    ap.add_argument("--cohort-years", default="1150-1250",
                    help="the period of the edition cohort, by the poem's date; "
                         "an edition has no date of its own")
    ap.add_argument("--cohort-source", default="",
                    help="restrict the cohort of strangers to one corpus, so "
                         "that the scale is set by texts annotated as the "
                         "candidate was. A cohort drawn mostly from other "
                         "corpora raises every text of the candidate's corpus "
                         "against it, and that offset would be read as milieu.")
    ap.add_argument("--cohort-per-author", type=int, default=1,
                    help="how many texts one author may contribute to the "
                         "cohort. The cohort is not a sample of texts but the "
                         "scale every score is expressed in, so an author with "
                         "many surviving witnesses sets that scale for all the "
                         "others: of the 33 early texts this corpus can offer, "
                         "17 are witnesses of one poem. One text per author is "
                         "the rule; 0 lifts the cap and reproduces the arms "
                         "measured before it was introduced.")
    ap.add_argument("--candidates", default=",".join(CANDIDATES),
                    help="the texts whose authorship against the known regions "
                         "is in question")
    ap.add_argument("--control", default=",".join(CONTROL),
                    help="the texts that say how high a poem of the period "
                         "reaches without being the candidate. One control is "
                         "a point and cannot separate the author from the "
                         "genre; several make a distribution.")
    ap.add_argument("--exclude", default="",
                    help="further texts to keep out of the reference grammars "
                         "and out of the cohort, comma separated; a name ending "
                         "in * excludes every text beginning with it. The known "
                         "text, the candidates and the controls are excluded "
                         "automatically.")
    ap.add_argument("--tag", default="")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    L, seg = args.L, args.seg
    rng = random.Random(args.seed)
    if args.case_bank == "editions":
        bank = C.load_editions(args.stream, args.case_years or args.cohort_years)
        # the witness bank carries `period` and the edition bank `year`; the
        # period filters read the former, so it is supplied here rather than
        # special-cased at each of the three places that reads it
        for e in bank.values():
            e["period"] = str(e["year"])
    else:
        bank = W.load(args.stream, args.encoding)

    kc = toks(bank[args.witness])
    rlen = len(kc) // args.regions
    regions = [(i * rlen, kc[i * rlen:(i + 1) * rlen])
               for i in range(args.regions)]
    print(f"{args.witness}: {len(kc):,} tokens in {args.regions} regions of "
          f"{rlen:,}")

    # the texts of the case itself, which may be neither grammars nor strangers
    case = {args.witness}
    case |= {x for x in args.candidates.split(",") if x}
    case |= {x for x in args.control.split(",") if x}
    pre = tuple(x[:-1] for x in args.exclude.split(",") if x.endswith("*"))
    case |= {x for x in args.exclude.split(",") if x and not x.endswith("*")}

    def out_of_the_case(n, src=None):
        """may this text stand in the population that judges the case?

        The rule is general: no text the case is about -- the known text, a
        candidate, a control, or anything named by --exclude -- may be a reference
        grammar or a member of the cohort. It was written for the witness bank,
        where the case lives, and the edition pools were left outside it because
        no edition could be a case; with --case-bank editions one can, so `src`
        names the bank the pool is being drawn from.
        """
        src = bank if src is None else src
        return (not M.is_kc(n, src) and n not in EXCLUDE and n not in case
                and not n.startswith(EXCLUDE_PREFIX)
                and not (pre and n.startswith(pre)))

    # the grammars' pool: a grammar is fitted on the known text's length, so that
    # is the whole requirement. It also had to hold two questioned passages while
    # the cohort was read out of it, and at a long questioned length that dropped
    # two of the early authors for no reason.
    pool = [n for n in bank
            if out_of_the_case(n) and bank[n]["ntok"] >= rlen]
    pool.sort()
    rng.shuffle(pool)
    best = {}
    for n in sorted(pool, key=lambda n: -bank[n]["ntok"]):
        if args.donor_period and bank[n]["period"] != args.donor_period:
            continue
        best.setdefault(bank[n]["author"], n)
    dn = sorted(best.values())[:args.donors]
    dbank = bank
    if args.donor_bank == "editions":
        # The witness corpus cannot supply grammars of a long known text: at 8,636
        # tokens only three authors survive in copies early enough, so a run that
        # doubles the known side has to take its denominator from the editions.
        dbank = C.load_editions(args.stream, args.donor_years or args.cohort_years)
        best = {}
        for n in sorted(dbank, key=lambda n: -dbank[n]["ntok"]):
            if dbank[n]["ntok"] >= rlen and out_of_the_case(n, dbank):
                best.setdefault(dbank[n]["author"], n)
        dn = sorted(best.values())[:args.donors]
    elif args.donor_bank:
        dbank = W.load_donors(args.donor_bank)
        dn = sorted(dbank, key=lambda n: -len(toks(dbank[n])))[:args.donors]
    dwork = {dbank[n]["author"]: dbank[n].get("work", n) for n in dn
             if "author" in dbank[n]}

    def one_per_author(src, names):
        """the cohort as a scale rather than a sample of texts

        An author with many surviving copies would otherwise set the scale for
        everyone else: of the 33 early texts this corpus can offer, 17 are
        witnesses of one poem. And an author whose grammar is in the denominator
        is predicted by it, so his cohort text scores low and sinks the scale
        under him -- with fourteen early authors and twelve grammars that overlap
        cannot be avoided, but it can be kept off the grammar's own poem, where
        the effect is strongest, and counted in the report.
        """
        names = sorted(names)
        rng.shuffle(names)
        if not args.cohort_per_author:
            return names
        names.sort(key=lambda n: src[n]["work"] == dwork.get(src[n]["author"]))
        kept, out = Counter(), []
        for n in names:
            a = src[n]["author"]
            if kept[a] < args.cohort_per_author:
                kept[a] += 1
                out.append(n)
        return out

    # A cohort text has to supply ONE passage of the questioned length, while a
    # reference grammar has to supply the whole known length. Reading the cohort
    # out of the grammars' pool imposed the grammars' threshold on it and cost
    # this corpus two of its fourteen early authors.
    coh = []
    if args.cohort_bank in ("witnesses", "both"):
        names = [n for n in bank
                 if out_of_the_case(n) and n not in dn
                 and bank[n]["ntok"] >= L
                 and (not args.cohort_source or n.startswith(args.cohort_source))
                 and (not args.cohort_period
                      or bank[n]["period"] == args.cohort_period)]
        coh += [("witness", n, bank) for n in one_per_author(bank, names)]
    if args.cohort_bank in ("editions", "both"):
        ed = (bank if args.case_bank == "editions"
              else C.load_editions(args.stream, args.cohort_years))
        names = [n for n in ed if ed[n]["ntok"] >= L and n not in dn
                 and out_of_the_case(n, ed)]
        coh += [("edition", n, ed) for n in one_per_author(ed, names)]
    coh = coh[:args.cohort] if args.cohort_bank != "both" else coh
    cn = [n for _, n, _ in coh]
    refs, donor_names = [], []
    for n in dn:
        t = toks(dbank[n])
        st = rng.randrange(0, len(t) - rlen + 1)
        refs.append(W.rechunk(t[st:st + rlen], seg))
        donor_names.append(f"{n}@{st}")
    da = {dbank[n].get("author", n) for n in dn}
    print(f"reference: {len(refs)} grammars of {rlen:,} tokens from "
          f"{len(da)} authors")
    for layer in ("witness", "edition"):
        sub = [(n, src) for lay, n, src in coh if lay == layer]
        if not sub:
            continue
        ca = {src[n]["author"] for n, src in sub}
        same = sum(src[n]["work"] == dwork.get(src[n]["author"]) for n, src in sub)
        print(f"cohort ({layer}): {len(sub)} passages from {len(ca)} authors, at "
              f"most {args.cohort_per_author or len(sub)} each; {len(ca & da)} of "
              f"those authors also supply a reference grammar, {same} of the "
              f"passages from the same work as that grammar")

    # everything that is scored against every region
    quest = []
    for w in [x for x in args.candidates.split(",") if x]:
        for j, q in enumerate(cut(toks(bank[w]), L, 8)):
            quest.append(("candidate", w, j, q))
    for w in [x for x in args.control.split(",") if x]:
        for j, q in enumerate(cut(toks(bank[w]), L, 4)):
            quest.append(("control", w, j, q))
    for layer, n, src in coh:
        t = toks(src[n])
        st = rng.randrange(0, len(t) - L + 1)
        quest.append((f"cohort_{layer}" if args.cohort_bank == "both"
                      else "cohort", n, st, t[st:st + L]))
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
        # a run that scores two cohorts to measure the offset between their layers
        # has no single scale; the summary then pools them and the comparison of
        # the layers is left to compare_cohort_layers.py
        coh = np.array([r["lambda_G"] for r in sel
                        if r["kind"].startswith("cohort")])
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
