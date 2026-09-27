# The gauge inside the case: an anchor from the known material and a cohort of
# impostors, so that the score of a witness pair can be uttered as odds.
#
# The four cells discriminate (experiments/run_witness_cells.py) but do not
# calibrate: a logistic calibrator fitted on other identities costs more than
# it saves, because each known passage puts its scores in a place of its own --
# how well a hand's language is modelled at all is a property of that passage,
# not of the hypothesis under test. The journal paper's remedy is to measure
# that place inside the case, from material an examiner has:
#
#   the ANCHOR.  The known material is by one author by definition, so a second
#     known passage of the same author, scored against the first, is a
#     same-author score in the case's own geometry. Its mean over the spare
#     passages locates the same-author hypothesis; their spread scales it.
#     Here the known passages of an identity come from several witnesses, so
#     the anchor is emitted twice over: in the SAME hand as the known passage
#     and in ANOTHER hand, which is what tells us whether an anchor has to be
#     hand-matched to the questioned text for the scale to hold.
#
#   the COHORT.  A set of impostor passages, by identities the known author is
#     not, scored against the same known passage. Under the different-author
#     hypothesis the questioned passage is exchangeable with them, so its rank
#     among them is uniform over the cohort's values -- a denominator that is
#     known without fitting any population at all.
#
# Both require every score of one known passage to stand against the SAME
# reference grammars, which the per-case donor draw of the cells run does not
# give. So the donors are drawn once per known passage, from identities that
# are neither the known's, nor any questioned identity paired with it, nor any
# identity in its cohort; and the cases themselves are rescored against them,
# so that cases, anchors and cohort share one denominator exactly.
#
# The cases are read from the cells run, so the two are the same cases.
#
#   python experiments/run_witness_gauge.py --stream lemma --L 1000
#   python experiments/run_witness_gauge.py --stream lemma --L 1000 --dry-run
#
# Output: scores/witness_gauge/{stream}__L{L}__w{seg}{tag}.jsonl, one row per
# scored passage pair:
#   {"kind","cell","known","kw","quest","qw","from_pool","known_author",
#    "quest_author","known_scribe","quest_scribe","known_period","quest_period",
#    "lambda_G","n_q","lam_j","n_donors"}
# where kind is "case", "anchor" or "cohort" and from_pool says which half of
# the work the questioned passage was cut from.

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

CELLS_DIR = ROOT / "scores" / "witness"
OUT = ROOT / "scores" / "witness_gauge"


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--stream", choices=["lemma", "surface"], default="lemma")
    ap.add_argument("--encoding", default="posnoise",
                    help="posnoise, or the suffix of an encoded bank "
                         "(catranks, catranks_leaky)")
    ap.add_argument("--donor-bank", default="",
                    help="draw the reference GRAMMARS from another bank, the "
                         "borrowed condition. The impostor cohort stays inside "
                         "the case bank, being the gauge rather than the "
                         "population.")
    ap.add_argument("--donor-keep-punct", action="store_true")
    ap.add_argument("--L", type=int, default=1000)
    ap.add_argument("--seg", type=int, default=100)
    ap.add_argument("--kper", type=int, default=2, help="known passages per witness, as in the cells run")
    ap.add_argument("--same", type=int, default=3, help="questioned passages per witness, as in the cells run")
    ap.add_argument("--max-anchors", type=int, default=8)
    ap.add_argument("--anchors-same-witness", type=int, default=4)
    ap.add_argument("--anchor-extra", type=int, default=3,
                    help="further known passages per witness, drawn for the anchors alone, so that "
                         "an identity with a single witness still has a same-hand anchor")
    ap.add_argument("--cohort-authors", type=int, default=10)
    ap.add_argument("--cohort-per-author", type=int, default=2)
    ap.add_argument("--r-donors", type=int, default=15)
    ap.add_argument("--min-donors", type=int, default=8)
    ap.add_argument("--tag", default="")
    ap.add_argument("--limit", type=int, default=0, help="score only the first N known passages (a smoke test)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    L, seg = args.L, args.seg
    # the seed never takes the encoding or the donor bank, so arms differing
    # only in those score the very same passages
    seed = f"{args.stream}__L{L}__w{seg}"
    enc = "" if args.encoding == "posnoise" else f"__{args.encoding}"
    dbk = "__borrowed" if args.donor_bank else ""
    tag = seed + enc + dbk + args.tag
    OUT.mkdir(parents=True, exist_ok=True)
    fn = OUT / (tag + ".jsonl")
    if fn.exists() and not args.dry_run:
        print(f"{tag}: exists, skipped"); return

    banks = W.load(args.stream, args.encoding)
    known_of = {n: dict(W.passages(b["first"], L, seg, args.kper, random.Random(seed + n + "k")))
                for n, b in banks.items()}
    quest_of = {n: dict(W.passages(b["last"], L, seg, args.same, random.Random(seed + n + "q")))
                for n, b in banks.items()}
    # further known passages for the anchors only, on their own draw so that the
    # cases of the cells run are untouched, and disjoint from them by block
    anch_of = {n: dict(W.passages(b["first"], L, seg, args.kper + args.anchor_extra,
                                  random.Random(seed + n + "a"))) for n, b in banks.items()}
    for n in anch_of:
        anch_of[n] = {j: u for j, u in anch_of[n].items() if j not in known_of[n]}
        anch_of[n].update(known_of[n])
    quest_wits = [n for n in banks if quest_of[n]]
    donor_src = (W.load_donors(args.donor_bank, not args.donor_keep_punct)
                 if args.donor_bank else banks)
    donors_all = {n: b for n, b in donor_src.items() if b["ntok"] >= L + seg}

    cells = CELLS_DIR / (seed + ".jsonl")
    if not cells.exists():
        print(f"{tag}: no cells run at {cells}"); return
    cases_by_k = defaultdict(list)
    for line in cells.open(encoding="utf-8"):
        r = json.loads(line)
        c = (r["cell"], r["quest"], r["qw"])
        if c not in cases_by_k[(r["known"], r["kw"])]:
            cases_by_k[(r["known"], r["kw"])].append(c)

    # the known passages of every identity, for the anchors
    known_by_author = defaultdict(list)
    for n, ps in anch_of.items():
        for j in ps:
            known_by_author[banks[n]["author"]].append((n, j))
    for v in known_by_author.values():
        v.sort()

    plan = []
    for (kn, kj), cs in sorted(cases_by_k.items()):
        a = banks[kn]["author"]
        rng = random.Random(seed + "|gauge|" + kn + "|" + str(kj))
        cand = {a} | {banks[qn]["author"] for _, qn, _ in cs}

        # anchors: spare known passages of the same identity, in this hand first
        spare = [(sn, sj) for (sn, sj) in known_by_author[a] if (sn, sj) != (kn, kj)]
        own = [p for p in spare if p[0] == kn]
        other = [p for p in spare if p[0] != kn]
        take_own = own[:args.anchors_same_witness]
        room = args.max_anchors - len(take_own)
        take_other = sorted(rng.sample(other, min(room, len(other)))) if room > 0 else []
        anchors = take_own + take_other

        # cohort: impostor identities, their passages mirroring the periods the
        # questioned passages of this known passage have
        periods = [banks[qn]["period"] for _, qn, _ in cs] or [banks[kn]["period"]]
        pool_auth = sorted({banks[n]["author"] for n in quest_wits if banks[n]["author"] not in cand})
        coh_auth = sorted(rng.sample(pool_auth, min(args.cohort_authors, len(pool_auth))))
        cohort, t = [], 0
        for ca in coh_auth:
            wits = [n for n in quest_wits if banks[n]["author"] == ca]
            for _ in range(args.cohort_per_author):
                per = periods[t % len(periods)]; t += 1
                w = [n for n in wits if banks[n]["period"] == per] or wits
                qn = rng.choice(w)
                qj = rng.choice(sorted(quest_of[qn]))
                if (qn, qj) not in cohort:
                    cohort.append((qn, qj))

        # donors: identities that are neither candidate nor impostor here
        excl = cand | set(coh_auth)
        pool = sorted(n for n, b in donors_all.items() if b["author"] not in excl)
        if len(pool) < args.min_donors:      # too tight: let the impostors' identities back in
            pool = sorted(n for n, b in donors_all.items() if b["author"] not in cand)
        picks = sorted(rng.sample(pool, min(args.r_donors, len(pool))))
        starts = [random.Random(seed + "|d|" + kn + str(kj) + dn).randrange(
            0, max(1, donors_all[dn]["ntok"] - L)) for dn in picks]
        plan.append(((kn, kj), cs, anchors, cohort, picks, starts))

    if args.limit:
        plan = plan[:args.limit]
    ncase = sum(len(p[1]) for p in plan)
    nanch = sum(len(p[2]) for p in plan)
    ncoh = sum(len(p[3]) for p in plan)
    nd = [len(p[4]) for p in plan]
    print(f"{tag}: {len(plan)} known passages; {ncase} cases, {nanch} anchors, {ncoh} cohort; "
          f"donors per known passage min {min(nd)} median {int(np.median(nd))}; "
          f"{(ncase + nanch + ncoh)} passages to score", flush=True)
    if args.dry_run:
        for (kp, cs, anchors, cohort, picks, _) in plan[:3]:
            print("  ", kp, "cases", len(cs), "anchors", [f"{n}#{j}" for n, j in anchors],
                  "cohort", len(cohort), "donors", len(picks))
        nsw = sum(1 for p in plan for n, j in p[2] if n == p[0][0])
        print(f"  anchors in the known passage's own witness: {nsw} of {nanch}")
        return

    lg = LambdaG(N=10, r=1, engine="kn", random_state=0)
    t0, rows, done = time.time(), [], 0
    for (kn, kj), cs, anchors, cohort, picks, starts in plan:
        kunits = known_of[kn][kj]
        refs = []
        for dn, st in zip(picks, starts):
            toks = [t for s in donors_all[dn]["sents"] for t in s]
            refs.append(W.rechunk(toks[st:st + L], seg))

        def score(qunits):
            lams, n_q = [], 0
            for ref in refs:
                r = lg.score(qunits, kunits, ref_sentences=ref, r=1, with_details=False)
                lams.append(r.lambda_G); n_q = r.n_query_tokens
            return float(np.mean(lams)), n_q, lams

        def row(kind, cell, qn, qj, qunits, pool_name):
            lam, n_q, lams = score(qunits)
            return dict(kind=kind, cell=cell, known=kn, kw=kj, quest=qn, qw=qj, from_pool=pool_name,
                        known_author=banks[kn]["author"], quest_author=banks[qn]["author"],
                        known_scribe=banks[kn]["scribe"], quest_scribe=banks[qn]["scribe"],
                        known_period=banks[kn]["period"], quest_period=banks[qn]["period"],
                        lambda_G=lam, n_q=n_q, lam_j=[round(x, 3) for x in lams], n_donors=len(refs))

        for cell, qn, qj in cs:
            rows.append(row("case", cell, qn, qj, quest_of[qn][qj], "quest"))
        for sn, sj in anchors:
            cell = "AN-SS" if banks[sn]["scribe"] == banks[kn]["scribe"] else "AN-DS"
            rows.append(row("anchor", cell, sn, sj, anch_of[sn][sj], "known"))
        for qn, qj in cohort:
            cell = "IM-SS" if banks[qn]["scribe"] == banks[kn]["scribe"] else "IM-DS"
            rows.append(row("cohort", cell, qn, qj, quest_of[qn][qj], "quest"))
        done += 1
        if done % 10 == 0:
            print(f"  {tag}: {done}/{len(plan)} known passages, {len(rows)} rows, "
                  f"{time.time() - t0:.0f}s", flush=True)
    with open(fn, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{tag}: {len(rows)} rows in {time.time() - t0:.0f}s -> {fn}", flush=True)


if __name__ == "__main__":
    main()
