# The class the chronicle's calibration is missing: one author, two works.
#
# WHY. The verdict per fragment is a likelihood ratio, and a likelihood ratio is
# only as good as the two classes it was fitted on. The region run supplies a
# different-author class of the right kind -- passages by strangers -- but its
# same-author class is the chronicle's OWN other regions, which are the same work
# as well as the same author. Fitted on those, the calibrator answers "is this
# passage as close to the region as another stretch of the chronicle is", which is
# a harder question than the attribution asks, and a low ratio therefore means
# nothing. What the attribution needs is the class the question is about: the same
# man writing a DIFFERENT poem.
#
# WHY IT CANNOT SIMPLY BE SCORED SEPARATELY. lambda_G is not comparable from one
# known text to another -- how well a known passage models anything at all is a
# property of that passage -- so a calibrator trained on cases with one known text
# and applied to cases with another is fitted on the wrong scale. Two things are
# therefore held fixed here: every case is built exactly as a chronicle region is
# (a known text of the region's length, questioned passages of the run's length,
# the SAME reference grammars at the same offsets, and the SAME cohort passages),
# and every score is expressed in its own case's units -- deviations from that
# case's cohort median in that cohort's robust spread -- before any calibrator
# sees it. That statistic is comparable across known texts by construction, which
# is the whole reason the project's gauge is written the way it is.
#
# THE CASES. Only authors with two genuinely different works can supply one, and
# in a manuscript corpus that is a short list. Witness variants of a single work
# are NOT two works, however the catalogue titles them: the Rolandslied's seven
# copies, the Annolied's two, Tristan's two, the Nibelungenlied's two and
# Priester Wernher's Maria beside his Driu liet are all one work each, and
# counting them would reintroduce exactly the defect this run exists to remove.
# The poems whose relation to the chronicle is under discussion are excluded
# altogether, on either side.
#
#   python experiments/run_calibration_cases.py --dry-run
#   python experiments/run_calibration_cases.py
#
# Output: scores/konrad_profile/calib__crosswork__q{L}.jsonl

import argparse
import csv
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
import run_witness_cells as W  # noqa: E402
from lambdag import LambdaG  # noqa: E402

OUT = ROOT / "scores" / "konrad_profile"

# (author, the known work's witness, the witnesses of his OTHER works). Each
# entry is one case; a case is a known text and the passages scored against it.
CASES = [
    ("hartmann_von_aue", "ambraser__hartmann_von_aue__erec__d",
     ["hvad__hartmann_von_aue__iwein__B", "hvad__hartmann_von_aue__gregorius__E",
      "ambraser__hartmann_von_aue__klage__d",
      "hvad__hartmann_von_aue__der_arme_heinrich__Bb"]),
    ("hartmann_von_aue", "hvad__hartmann_von_aue__iwein__B",
     ["ambraser__hartmann_von_aue__erec__d",
      "hvad__hartmann_von_aue__gregorius__E",
      "ambraser__hartmann_von_aue__klage__d",
      "hvad__hartmann_von_aue__der_arme_heinrich__Bb"]),
    ("hartmann_von_aue", "hvad__hartmann_von_aue__gregorius__E",
     ["ambraser__hartmann_von_aue__erec__d",
      "hvad__hartmann_von_aue__iwein__B",
      "ambraser__hartmann_von_aue__klage__d",
      "hvad__hartmann_von_aue__der_arme_heinrich__Bb"]),
    ("hartmann_von_aue", "ambraser__hartmann_von_aue__klage__d",
     ["ambraser__hartmann_von_aue__erec__d",
      "hvad__hartmann_von_aue__iwein__B",
      "hvad__hartmann_von_aue__gregorius__E"]),
    ("der_stricker", "ambraser__stricker__pfaffe_amis__d",
     ["ambraser__stricker__frauenehre__d"]),
    ("ulrich_von_liechtenstein", "rem__M314",
     ["ambraser__ulrich_von_liechtenstein__frauenbuch__d"]),
    ("ulrich_von_liechtenstein",
     "ambraser__ulrich_von_liechtenstein__frauenbuch__d", ["rem__M314"]),
    ("rudolf_von_ems", "rem__M336", ["rem__M359"]),
    ("rudolf_von_ems", "rem__M359", ["rem__M336"]),
    ("wolfram_von_eschenbach", "parzival__wolfram_von_eschenbach__parzival__d",
     ["ambraser__wolfram_von_eschenbach__titurel__d"]),
]


def load_named(stream, names):
    """only the witnesses this run actually reads

    A run needs the reference grammars and the cohort passages the region run
    used -- some fifty texts -- and the bank loader reads all 247, several of them
    over a hundred thousand tokens. Reading the named files keeps the run inside
    a gigabyte instead of tens of them.
    """
    # A region run may set its scale with the tagged editions rather than the
    # witnesses -- the witness corpus has only fourteen authors whose copies are
    # early enough and the grammars need twelve of them -- so a name that is not a
    # witness is looked for there before it is given up.
    bases = [ROOT / "masked" / f"mhg_witnesses_{stream}",
             ROOT / "masked" / f"mhdbdb_{stream}"]
    man = {}
    for b in bases:
        man.update({r["file"]: r for r in csv.DictReader(
            (b / "MANIFEST.tsv").open(encoding="utf-8"), delimiter="	")})
    out = {}
    for n in sorted(set(names)):
        f = next((b / "bank" / f"{n}.tsv" for b in bases
                  if (b / "bank" / f"{n}.tsv").exists()), None)
        if f is None:
            continue
        lines = [l.split("	") for l in
                 f.read_text(encoding="utf-8").splitlines() if l]
        r = man.get(n, {})
        out[n] = dict(sents=lines, author=W.author_id(r) if r else n,
                      work=W.norm(r.get("work", n)), period=r.get("date", "-"),
                      ntok=sum(len(u) for u in lines))
    return out


def load_editions(stream, years):
    """the tagged editions as a bank, with a period filter

    They are masked by the same masker and tagged by the same model as the
    witnesses, which is why their passages may stand in one calibration class
    with a case built on witnesses. What still differs is the layer of the KNOWN
    side -- an editor's normalised text against a scribe's -- and that residual is
    named in the report rather than assumed away.
    """
    base = ROOT / "masked" / f"mhdbdb_{stream}"
    lo, hi = ((int(x) for x in years.split("-")) if years else (0, 9999))
    out = {}
    with (base / "MANIFEST.tsv").open(encoding="utf-8") as fh:
        for r in csv.DictReader(fh, delimiter="	"):
            y = int(r["date"]) if r["date"].isdigit() else 0
            if not (lo <= y <= hi):
                continue
            lines = [l.split("	") for l in
                     (base / "bank" / f"{r['file']}.tsv").read_text(
                         encoding="utf-8").splitlines() if l]
            out[r["file"]] = dict(sents=lines, author=W.norm(r["author"]),
                                  work=W.norm(r["work"]), year=y,
                                  ntok=sum(len(u) for u in lines))
    return out


def edition_cases(bank, klen, L, per_author, min_pos):
    """(author, known work, its other works) for every author with two works

    The known side takes the author's longest work, so that a case is built on as
    much of him as the chronicle's regions are built on of itself; the positives
    come from every other work he has, which is the class the attribution is
    about.
    """
    by = defaultdict(list)
    for n, e in bank.items():
        by[e["author"]].append(n)
    cases = []
    for a, ns in sorted(by.items()):
        works = {}
        for n in ns:
            w = bank[n]["work"]
            if w not in works or bank[n]["ntok"] > bank[works[w]]["ntok"]:
                works[w] = n
        ws = sorted(works.values(), key=lambda n: -bank[n]["ntok"])
        for known in ws[:per_author]:
            others = [n for n in ws if n != known and bank[n]["ntok"] >= L]
            if bank[known]["ntok"] >= klen and len(others) >= min_pos:
                cases.append((a, known, others))
    return cases


def toks(entry):
    return [t for s in entry["sents"] for t in s]


def cut(t, L, n=None):
    out = [t[i:i + L] for i in range(0, len(t) - L + 1, L)]
    return out if n is None else out[:n]


def spread(v):
    m = float(np.median(v))
    return m, max(1.4826 * float(np.median(np.abs(v - m))),
                  (float(np.quantile(v, 0.9)) - m) / 1.2816, 1e-9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-run",
                    default="regions__A1__r20__q2000__controls.jsonl",
                    help="the region run whose reference grammars and cohort "
                         "passages this run must share, so that the two are on "
                         "one scale")
    ap.add_argument("--positives", type=int, default=6,
                    help="passages of the author's other works per case")
    ap.add_argument("--case-bank", default="",
                    help="mhdbdb draws the cases from the tagged editions, where "
                         "an author has several works; the default uses the "
                         "witness bank and the case list written into this file")
    ap.add_argument("--years", default="",
                    help="with --case-bank, keep only the authors writing inside "
                         "this band, so that the class is of the case's own "
                         "milieu and not of another century")
    ap.add_argument("--per-author", type=int, default=2,
                    help="with --case-bank, how many of an author's works may "
                         "serve as the known text; more than one or two lets a "
                         "prolific author dominate the class")
    ap.add_argument("--tag", default="crosswork")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    rows = [json.loads(l) for l in
            (OUT / args.from_run).open(encoding="utf-8")]
    meta = rows[0]
    L, seg = int(meta["L"]), int(meta["seg"])
    klen = int(meta["region_len"])
    need = [d.rpartition("@")[0] for d in meta["donors"]]
    need += [r["text"] for r in rows[1:] if r.get("kind") == "cohort"]
    bank = load_named(meta["stream"], need)
    print(f"{len(bank)} witnesses read: {len(set(d.rpartition('@')[0] for d in meta['donors']))} "
          f"for the reference grammars, the rest for the cohort")
    rng = random.Random(args.seed)

    # the same reference grammars, at the same offsets, as the region run
    refs, donors = [], []
    for d in meta["donors"]:
        n, _, st = d.rpartition("@")
        t = toks(bank[n])
        refs.append(W.rechunk(t[int(st):int(st) + klen], seg))
        donors.append(d)
    # and the same cohort passages
    coh = sorted({(r["text"], int(r["j"])) for r in rows[1:]
                  if r.get("kind") == "cohort"})
    print(f"from {args.from_run}: {len(refs)} reference grammars of {klen:,} "
          f"tokens, {len(coh)} cohort passages of {L:,}, stream "
          f"{meta['stream']}")

    case_list = CASES
    if args.case_bank:
        eb = load_editions(meta["stream"], args.years)
        case_list = edition_cases(eb, klen, L, args.per_author, 1)
        bank = {**bank, **eb}
        print(f"cases from the tagged editions: {len(eb)} texts"
              + (f" written {args.years}" if args.years else "")
              + f", {len(case_list)} cases from "
              f"{len({a for a, _, _ in case_list})} authors")
    cases = []
    for author, known, others in case_list:
        if known not in bank or any(o not in bank for o in others):
            print(f"  skipped {known}: not in the bank")
            continue
        kt = toks(bank[known])
        if len(kt) < klen:
            print(f"  skipped {known}: {len(kt):,} tokens, needs {klen:,}")
            continue
        st = rng.randrange(0, len(kt) - klen + 1)
        pos = []
        for o in others:
            for q in cut(toks(bank[o]), L, 3):
                pos.append((o, q))
        rng.shuffle(pos)
        pos = pos[:args.positives]
        # a passage by the case's own author cannot stand in the class of
        # strangers, so those cohort passages are dropped and the loss reported
        mine = [c for c in coh if bank[c[0]]["author"] == author]
        cases.append(dict(author=author, known=known, start=st,
                          ku=kt[st:st + klen], pos=pos,
                          coh=[c for c in coh if c not in mine]))
        print(f"  {author:26} known {known.split('__')[-2:][0][:18]:18} "
              f"{len(pos)} passages of other works, cohort "
              f"{len(cases[-1]['coh'])}"
              + (f" ({len(mine)} of the author's own removed)" if mine else ""))
    n = sum(len(c["pos"]) + len(c["coh"]) for c in cases)
    print(f"\n{len(cases)} cases, {n} passages x {len(refs)} grammars = "
          f"{n * len(refs):,} scores")
    if args.dry_run:
        return

    lg = LambdaG(N=10, r=1, engine="kn", random_state=0)
    t0, out = time.time(), []
    for ci, c in enumerate(cases):
        ku = W.rechunk(c["ku"], seg)

        def score(q):
            return float(np.mean([lg.score(W.rechunk(q, seg), ku,
                                           ref_sentences=ref, r=1,
                                           with_details=False).lambda_G
                                  for ref in refs]))

        for name, q in c["pos"]:
            out.append(dict(kind="same_author", case=ci, author=c["author"],
                            known=c["known"], text=name,
                            lambda_G=round(score(q), 4)))
        for name, st in c["coh"]:
            t = toks(bank[name])
            out.append(dict(kind="cohort", case=ci, author=c["author"],
                            known=c["known"], text=name,
                            lambda_G=round(score(t[st:st + L]), 4)))
        el = time.time() - t0
        print(f"  case {ci + 1}/{len(cases)} {c['known'].split('__')[-1]:>10} "
              f"{el:.0f}s", flush=True)

    p = OUT / f"calib__{args.tag}__q{L}.jsonl"
    with p.open("w", encoding="utf-8") as fh:
        fh.write(json.dumps(dict(kind="meta", stream=meta["stream"], L=L,
                                 seg=seg, known_len=klen, donors=donors,
                                 from_run=args.from_run,
                                 cases=[dict(author=c["author"],
                                             known=c["known"],
                                             start=c["start"],
                                             positives=[x[0] for x in c["pos"]])
                                        for c in cases])) + "\n")
        for r in out:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\n  {'case':>5} {'known':>34} {'other works':>12} "
          f"{'cohort':>9}   in the case's own units")
    for ci, c in enumerate(cases):
        sel = [r for r in out if r["case"] == ci]
        cv = np.array([r["lambda_G"] for r in sel if r["kind"] == "cohort"])
        m, sd = spread(cv)
        pv = [r["lambda_G"] for r in sel if r["kind"] == "same_author"]
        print(f"  {ci:>5} {c['known'].split('__')[-2] + ' ' + c['known'].split('__')[-1]:>34} "
              f"{(float(np.median(pv)) - m) / sd:>12.1f} {0.0:>9.1f}")
    print(f"  wrote {p.relative_to(ROOT)} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
