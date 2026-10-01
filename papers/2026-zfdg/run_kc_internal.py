# Is the chronicle consistent with itself? Every region against every other.
#
# The attribution arms ask whether a stretch of the chronicle claims the
# Rolandslied more strongly than a poem of the milieu. That comparison has no
# meaning until a second question is answered: how strongly does a stretch of the
# chronicle claim the REST OF THE CHRONICLE? If the work is of one hand throughout,
# its regions should claim one another highly and evenly, and a region that was
# reworked from another poem or added late should stand apart -- which is a
# statement about the chronicle's composition, made with the same instrument and
# on the same scale as the attribution.
#
# THE SCALE IS TAKEN FROM AN EXISTING ARM AND NOTHING IS REFITTED. The run named
# by --from-run supplies the reference grammars at their own offsets and, for each
# region, the cohort passages that set that region's location and spread. The only
# new scores are the region-against-region ones, so an internal score and the
# candidate's score for the same region are in the same units and may be read
# against each other.
#
# Each region gives one questioned passage of the run's length, taken from its
# start, and is scored against every OTHER region as known text. The diagonal is
# left out: a text against itself is not a comparison.
#
#   python experiments/run_kc_internal.py --from-run regions__A1__r20__q4000__ed4000.jsonl
#
# Output: scores/konrad_profile/internal__{witness}__r{regions}__q{L}.jsonl
#         medieval/kc_reuse/kc_internal.tsv

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
import run_calibration_cases as C  # noqa: E402  (its light bank reader)
import run_witness_cells as W  # noqa: E402
from lambdag import LambdaG  # noqa: E402

SCORES = ROOT / "scores" / "konrad_profile"
OUT = ROOT / "medieval" / "kc_reuse" / "kc_internal.tsv"


def spread(v):
    """the cohort's own location and scale, robust to its tail"""
    m = float(np.median(v))
    return m, max(1.4826 * float(np.median(np.abs(v - m))),
                  (float(np.quantile(v, 0.9)) - m) / 1.2816, 1e-9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-run", default="regions__A1__r20__q4000__ed4000.jsonl",
                    help="the arm whose grammars, cohort and lengths are reused, "
                         "so that an internal score and the candidate's score for "
                         "one region are in the same units")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    rows = [json.loads(l) for l in (SCORES / args.from_run).open(encoding="utf-8")]
    meta, rows = rows[0], rows[1:]
    L, seg, klen = int(meta["L"]), int(meta["seg"]), int(meta["region_len"])
    nreg, wit = int(meta["regions"]), meta["witness"]

    # the chronicle itself, and the reference grammars at the run's own offsets
    need = [d.rpartition("@")[0] for d in meta["donors"]] + [wit]
    bank = C.load_named(meta["stream"], need)
    kc = [t for s in bank[wit]["sents"] for t in s]
    regions = [kc[i * klen:(i + 1) * klen] for i in range(nreg)]
    refs = []
    for d in meta["donors"]:
        n, _, st = d.rpartition("@")
        t = [x for s in bank[n]["sents"] for x in s]
        refs.append(W.rechunk(t[int(st):int(st) + klen], seg))
    print(f"{wit}: {len(kc):,} tokens in {nreg} regions of {klen:,}; "
          f"{len(refs)} grammars from {args.from_run}")

    # each region's own scale, as that arm measured it, and its candidate score
    scale, cand = {}, {}
    for ri in range(nreg):
        coh = [r["lambda_G"] for r in rows
               if r["region"] == ri and r["kind"].startswith("cohort")]
        scale[ri] = spread(np.array(coh))
        cv = [r["lambda_G"] for r in rows
              if r["region"] == ri and r["kind"] == "candidate"]
        cand[ri] = float(np.median([(v - scale[ri][0]) / scale[ri][1]
                                    for v in cv])) if cv else float("nan")
    quest = [r[:L] for r in regions]
    print(f"{nreg} known regions x {nreg - 1} questioned passages of {L:,} x "
          f"{len(refs)} grammars = {nreg * (nreg - 1) * len(refs):,} scores")

    lg = LambdaG(N=10, r=1, engine="kn", random_state=0)
    t0, out = time.time(), []
    z = np.full((nreg, nreg), np.nan)
    for ki in range(nreg):
        ku = W.rechunk(regions[ki], seg)
        for qj in range(nreg):
            if qj == ki:
                continue
            lam = float(np.mean([lg.score(W.rechunk(quest[qj], seg), ku,
                                          ref_sentences=ref, r=1,
                                          with_details=False).lambda_G
                                 for ref in refs]))
            m, sd = scale[ki]
            z[ki, qj] = (lam - m) / sd
            out.append(dict(known=ki, quest=qj, lambda_G=round(lam, 4),
                            z=round(z[ki, qj], 4)))
        print(f"  known region {ki + 1}/{nreg}, {time.time() - t0:.0f}s",
              flush=True)

    p = SCORES / (f"internal__{wit.split('__')[-1]}__r{nreg}__q{L}"
                  f"{args.tag}.jsonl")
    with p.open("w", encoding="utf-8") as fh:
        fh.write(json.dumps(dict(from_run=args.from_run, witness=wit,
                                 regions=nreg, region_len=klen, L=L, seg=seg,
                                 stream=meta["stream"],
                                 donors=meta["donors"])) + "\n")
        for r in out:
            fh.write(json.dumps(r) + "\n")

    # A region is read twice: as a known text claiming the others, and as a
    # questioned passage the others claim. The second is the one the composition
    # question asks about, so it is the one reported first.
    claimed = [float(np.nanmedian(z[:, j])) for j in range(nreg)]
    claims = [float(np.nanmedian(z[i, :])) for i in range(nreg)]
    m, sd = spread(np.array(claimed))
    print(f"\n  a region of this chronicle is claimed by the rest at a median "
          f"{np.median(claimed):.2f} deviations of the strangers' scale")
    print(f"  {'region':>6} {'verses':>14} {'claimed by':>11} {'claims':>8} "
          f"{'the candidate':>14}   apart")
    for j in range(nreg):
        v0 = j * (16571 // nreg) + 1
        v1 = (j + 1) * (16571 // nreg) if j < nreg - 1 else 16571
        flag = "yes" if (claimed[j] - m) / sd < -1.5 else ""
        print(f"  {j:>6} {f'{v0}-{v1}':>14} {claimed[j]:>11.2f} "
              f"{claims[j]:>8.2f} {cand[j]:>14.2f}   {flag}")
    inside = sum(cand[j] < claimed[j] for j in range(nreg)
                 if not np.isnan(cand[j]))
    print(f"\n  the candidate stands BELOW the chronicle's own regions in "
          f"{inside} of {nreg}")
    name = OUT if not args.tag else OUT.with_name(f"{OUT.stem}{args.tag}.tsv")
    with name.open("w", encoding="utf-8") as fh:
        fh.write("region\tclaimed_by_rest\tclaims_rest\tcandidate\n")
        for j in range(nreg):
            fh.write(f"{j}\t{claimed[j]:.4f}\t{claims[j]:.4f}\t{cand[j]:.4f}\n")
    print(f"  wrote {p.relative_to(ROOT)} and {name.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
