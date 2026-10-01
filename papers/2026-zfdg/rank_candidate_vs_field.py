# Does the candidate beat the field, and how often would a poem of the milieu?
#
# The calibrated verdict rests on two things the reader may doubt: the cohort that
# sets the scale, and the class that turns a score into odds. This test rests on
# neither. Within one region the candidate and the controls are compared with each
# other alone -- the same known text, the same grammars, the same questioned
# length, and all of them witnesses, so the layer of the cohort cannot enter.
#
# THE NULL is that the candidate is just another poem of the milieu. The candidate
# enters as ONE value -- the median over its witnesses' passages -- and each control
# as one, so the field has 1 + (number of controls) members; under the null they are
# exchangeable and the candidate's value is the highest with probability 1/(1+c).
# Over the regions the count is binomial, and that is an exact test with no
# distributional assumption and nothing fitted. Pooling the candidate's witnesses
# lowers its variance, which under the null makes it LESS likely to be the maximum,
# so the probability is conservative rather than the reverse.
#
# A region is the unit of replication, as everywhere in this design. Regions share
# the grammars and the questioned passages, so they are not independent replicates
# of a corpus -- they are independent only as known texts, which is what the
# hypothesis is about: whether a PART of the chronicle claims the candidate.
#
#   python experiments/rank_candidate_vs_field.py --run regions__A1__r20__q4000__ed4000.jsonl
#
# Output: printed, plus medieval/kc_reuse/kc_regions_field.tsv

import argparse
import json
import sys
from collections import defaultdict
from math import comb
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SCORES = ROOT / "scores" / "konrad_profile"
OUT = ROOT / "medieval" / "kc_reuse" / "kc_regions_field.tsv"


def binom_tail(k, n, p):
    """P(X >= k) for X ~ Bin(n, p)"""
    return float(sum(comb(n, i) * p ** i * (1 - p) ** (n - i)
                     for i in range(k, n + 1)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="regions__A1__r20__q4000__ed4000.jsonl")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    rows = [json.loads(l) for l in (SCORES / args.run).open(encoding="utf-8")]
    meta, rows = rows[0], rows[1:]
    # one value per text per region: passages of one text are not independent of
    # each other, so the text is the unit and its median is the value
    per = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r["kind"] in ("candidate", "control"):
            per[r["region"]][(r["kind"], r["text"])].append(r["lambda_G"])
    regions = sorted(per)
    texts = sorted({t for ri in regions for t in per[ri]})
    cands = [t for t in texts if t[0] == "candidate"]
    ctrls = [t for t in texts if t[0] == "control"]
    field = 1 + len(ctrls)   # the candidate counts once, not per witness
    print(f"{args.run}: {len(regions)} regions of {meta['region_len']:,} tokens, "
          f"questioned {meta['L']:,}, {len(meta['donors'])} grammars")
    print(f"the field: the candidate ({len(cands)} witness(es), pooled to one "
          f"value) against {len(ctrls)} controls, so a poem of the milieu tops a "
          f"field of {field} with probability 1/{field} = {1 / field:.3f}")

    out, wins = [], 0
    print(f"\n  {'region':>6} {'candidate':>10} {'best control':>13} "
          f"{'margin':>8}   tops the field")
    for ri in regions:
        d = per[ri]
        c = float(np.median([np.median(d[t]) for t in cands if t in d]))
        others = [float(np.median(d[t])) for t in ctrls if t in d]
        best = max(others)
        top = c > best
        wins += top
        out.append((ri, c, best, c - best, top))
        print(f"  {ri:>6} {c:>10.1f} {best:>13.1f} {c - best:>8.1f}   "
              f"{'yes' if top else 'no'}")

    p = binom_tail(wins, len(regions), 1 / field)
    print(f"\n  the candidate's median is the highest of the {field} in {wins} of "
          f"{len(regions)} regions; under the null that it is one poem of the "
          f"milieu among them, P = {p:.2e}")
    # and the same test against each control on its own, which asks whether the
    # ordering is carried by one weak comparison
    print("\n  against each control separately, and the margin over it:")
    for t in ctrls:
        k = sum(1 for ri in regions if t in per[ri]
                and float(np.median([np.median(per[ri][x]) for x in cands
                                     if x in per[ri]])) > np.median(per[ri][t]))
        n = sum(1 for ri in regions if t in per[ri])
        marg = float(np.median([float(np.median([np.median(per[ri][x])
                                                 for x in cands if x in per[ri]]))
                                - float(np.median(per[ri][t]))
                                for ri in regions if t in per[ri]]))
        print(f"    {t[1]:<12} above it in {k:>2} of {n} regions, median margin "
              f"{marg:>+8.1f}, P = {binom_tail(k, n, 0.5):.2e}")

    name = OUT if not args.tag else OUT.with_name(f"{OUT.stem}__{args.tag}.tsv")
    name.parent.mkdir(parents=True, exist_ok=True)
    with name.open("w", encoding="utf-8") as fh:
        fh.write("region\tcandidate\tbest_control\tmargin\ttops_field\n")
        for ri, c, b, m, top in out:
            fh.write(f"{ri}\t{c:.4f}\t{b:.4f}\t{m:.4f}\t{int(top)}\n")
    print(f"\n  wrote {name.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
