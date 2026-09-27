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
    bank = W.load(meta["stream"])
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

    cases = []
    for author, known, others in CASES:
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
