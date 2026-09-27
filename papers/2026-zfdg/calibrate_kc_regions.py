# A verdict per fragment of the Kaiserchronik, with the error rate stated.
#
# The region run leaves a score for every passage against every region of the
# chronicle, and a score is not an answer. Two ways of turning one into an answer
# are taken here, because they fail differently.
#
# THE RANK. For each region the cohort of passages by other authors is the null
# distribution: if the candidate is a stranger like them, its rank among them is
# uniform, and the probability of landing at or above its observed rank is
# (r + 1) / (n + 1). That needs no training at all, which is its virtue. Twenty
# regions are twenty tests, so the p-values are corrected for multiplicity by
# the Benjamini-Hochberg procedure and the verdict is read off the corrected
# value.
#
# THE LIKELIHOOD RATIO. The forensic answer wants a ratio, not a tail: how much
# more probable is this score if the same man wrote both texts than if he did
# not. That needs training pairs, and the training pairs have to be scored inside
# THIS run, since a score is only comparable to others made against the same
# reference grammars. What the run holds is passages of the chronicle from OTHER
# regions -- the same author by construction, but also the same work -- and the
# cohort. A calibrator fitted on those two classes therefore answers a narrower
# question than the attribution does: is this passage as close to the region as
# another stretch of the chronicle is? A poem by the same author but of another
# kind would fall below that, so an LR near zero is evidence of nothing, while an
# LR above it is strong evidence indeed. The bound is reported as a bound.
#
# The region whose verdict is being computed never contributes to its own
# calibration: the fit uses the other regions' passages only.
#
#   python experiments/calibrate_kc_regions.py
#   python experiments/calibrate_kc_regions.py --alpha 0.05 --candidate rem__M205P
#
# Output: medieval/kc_reuse/kc_regions_verdicts.tsv and the table on stdout

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
from lambdag import PAVCalibrator, cllr, cllr_min  # noqa: E402

SCORES = ROOT / "scores" / "konrad_profile"
OUT = ROOT / "medieval" / "kc_reuse" / "kc_regions_verdicts.tsv"


def load(path):
    rows = [json.loads(l) for l in path.open(encoding="utf-8")]
    meta = rows[0]
    by = defaultdict(lambda: defaultdict(list))
    for r in rows[1:]:
        if "region" not in r:
            continue
        by[int(r["region"])][r["kind"]].append(
            (r.get("text", ""), float(r["lambda_G"])))
    return meta, by


def verse_ranges(witness, region_len, n_regions):
    """each region's first and last verse of the edition, for citation

    The regions are cut on the masked token stream, and a reader wants verses.
    The bank keeps the verse identifier of every unit, so the boundaries are
    carried back through the units the tokens came from.
    """
    base = ROOT / "masked" / "mhg_witnesses_lemma"
    units = [l.split("	") for l in
             (base / "bank" / f"{witness}.tsv").read_text(
                 encoding="utf-8").splitlines() if l]
    vids = [v for v in (base / "verse_ids" / f"{witness}.txt").read_text(
        encoding="utf-8").splitlines() if v]
    if len(vids) != len(units):
        return [("", "")] * n_regions
    pos, at = [], 0
    for u, v in zip(units, vids):
        pos.append((at, v))
        at += len(u)
    out = []
    for i in range(n_regions):
        lo, hi = i * region_len, (i + 1) * region_len
        inside = [v for a, v in pos if lo <= a < hi]
        out.append((inside[0], inside[-1]) if inside else ("", ""))
    return out


def bh(p):
    """Benjamini-Hochberg: the corrected values, in the input's order"""
    p = np.asarray(p, dtype=float)
    n = len(p)
    o = np.argsort(p)
    q = np.empty(n)
    run = 1.0
    for k in range(n - 1, -1, -1):
        run = min(run, p[o[k]] * n / (k + 1))
        q[o[k]] = run
    return q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default="regions__A1__r20__q2000__controls.jsonl")
    ap.add_argument("--candidate", default="",
                    help="one witness of the candidate's work; the default "
                         "pools every candidate passage in the run")
    ap.add_argument("--alpha", type=float, default=0.05,
                    help="the rate of false verdicts allowed over the whole set "
                         "of regions, not per region")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    meta, by = load(SCORES / args.file)
    regions = sorted(by)
    vr = verse_ranges(meta["witness"], int(meta["region_len"]), len(regions))
    print(f"{args.file}: {len(regions)} regions of "
          f"{int(meta['region_len']):,} tokens, questioned passages of "
          f"{int(meta['L']):,}, {len(meta['donors'])} reference grammars\n")

    def pick(region, kind):
        v = [s for t, s in by[region][kind]
             if not (kind == "candidate" and args.candidate
                     and t != args.candidate)]
        return np.array(v, dtype=float)

    rows = []
    for i in regions:
        coh = pick(i, "cohort")
        cand = pick(i, "candidate")
        ctrl = pick(i, "control")
        own = pick(i, "own")
        # the rank test, on the median passage of each text against the cohort
        def rank_p(v):
            if not len(v):
                return float("nan"), float("nan")
            m = float(np.median(v))
            return m, (int(np.sum(coh >= m)) + 1) / (len(coh) + 1)

        m, p = rank_p(cand)
        m_ctrl, p_ctrl = rank_p(ctrl)
        # the calibrator, fitted on every OTHER region's own passages and cohort
        tr_s, tr_y = [], []
        for j in regions:
            if j == i:
                continue
            tr_s += list(pick(j, "own")) + list(pick(j, "cohort"))
            tr_y += [1] * len(pick(j, "own")) + [0] * len(pick(j, "cohort"))
        cal = PAVCalibrator().fit(np.array(tr_s), np.array(tr_y))
        llr = float(np.median(cal.transform(cand)))
        llr_ctrl = float(np.median(cal.transform(ctrl))) if len(ctrl)             else float("nan")
        llr_own = float(np.median(cal.transform(own)))
        rows.append(dict(region=i, verses=f"{vr[i][0]}-{vr[i][1]}",
                         cand=m, control=m_ctrl,
                         cohort_med=float(np.median(coh)),
                         cohort_max=float(np.max(coh)), n_cohort=len(coh),
                         p=p, p_control=p_ctrl, elub=cal.hi,
                         llr=llr, llr_control=llr_ctrl, llr_own=llr_own))
    for key, out in (("p", "q"), ("p_control", "q_control")):
        for r, qq in zip(rows, bh([r[key] for r in rows])):
            r[out] = float(qq)
    for r in rows:
        r["verdict"] = "SAME" if (r["q"] <= args.alpha and r["llr"] > 0) else "no"
        r["verdict_control"] = "SAME" if (r["q_control"] <= args.alpha
                                         and r["llr_control"] > 0) else "no"

    # how well the calibrator separates the classes it was trained on, measured
    # on the held-out region: the honest cost of the answers above
    same, diff = [], []
    for i in regions:
        tr_s, tr_y = [], []
        for j in regions:
            if j == i:
                continue
            tr_s += list(pick(j, "own")) + list(pick(j, "cohort"))
            tr_y += [1] * len(pick(j, "own")) + [0] * len(pick(j, "cohort"))
        cal = PAVCalibrator().fit(np.array(tr_s), np.array(tr_y))
        same += list(cal.transform(pick(i, "own")))
        diff += list(cal.transform(pick(i, "cohort")))
    same, diff = np.array(same), np.array(diff)
    print(f"  {'':6}{'the candidate: the Rolandslied':>36}"
          f"{'a different author of the same milieu':>42}")
    print(f"  {'region':>6}{'verses':>14}{'lambda':>9}{'q':>8}"
          f"{'LR':>7}{'verdict':>10}   "
          f"{'lambda':>9}{'q':>8}{'LR':>7}{'verdict':>10}   {'own LR':>7}")
    for r in rows:
        print(f"  {r['region']:>6}{r['verses']:>14}{r['cand']:>9.1f}"
              f"{r['q']:>8.3f}"
              f"{r['llr']:>7.2f}{r['verdict']:>10}   {r['control']:>9.1f}"
              f"{r['q_control']:>8.3f}{r['llr_control']:>7.2f}"
              f"{r['verdict_control']:>10}   {r['llr_own']:>7.2f}")
    n_same = sum(r["verdict"] == "SAME" for r in rows)
    n_ctrl = sum(r["verdict_control"] == "SAME" for r in rows)
    print(f"\n  the candidate: {n_same} of {len(rows)} regions called SAME at "
          f"a false-discovery rate of {args.alpha:.0%}; a different author of "
          f"the same milieu, through the same test: {n_ctrl}")
    print(f"  the calibration's ELUB ceiling is {rows[0]['elub']:+.2f} in "
          f"log10: no passage may be given stronger odds than that")
    print(f"  calibration held out by region, on the classes it was trained on: "
          f"Cllr {cllr(same, diff):.3f}, Cllr_min {cllr_min(same, diff):.3f}, "
          f"{len(same)} same-work and {len(diff)} different-author passages")
    print(f"  the candidate's LR is bounded ABOVE by this class: its positives "
          f"are passages of the chronicle itself")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as fh:
        fh.write("region\tverses\tcandidate_lambda\tcontrol_lambda\t"
                 "cohort_median\t"
                 "cohort_max\tn_cohort\tp\tq\tllr\tverdict\tp_control\t"
                 "q_control\tllr_control\tverdict_control\tllr_own\telub\n")
        for r in rows:
            fh.write(f"{r['region']}	{r['verses']}\t{r['cand']:.3f}\t{r['control']:.3f}\t"
                     f"{r['cohort_med']:.3f}\t{r['cohort_max']:.3f}\t"
                     f"{r['n_cohort']}\t{r['p']:.4f}\t{r['q']:.4f}\t"
                     f"{r['llr']:.4f}\t{r['verdict']}\t{r['p_control']:.4f}\t"
                     f"{r['q_control']:.4f}\t{r['llr_control']:.4f}\t"
                     f"{r['verdict_control']}\t{r['llr_own']:.4f}\t"
                     f"{r['elub']:.4f}\n")
    print(f"  wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
