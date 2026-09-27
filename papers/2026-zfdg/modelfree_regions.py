# Odds per fragment with no calibration class at all.
#
# The verdicts of experiments/calibrate_crosswork.py are bounded by the set they
# were fitted on: 38 passages support odds of at most 25 to one, however extreme a
# score may be, and that bound is a property of the calibration set rather than of
# the evidence. The model-free corrections (Barlow, Nini & Manino 2026,
# arXiv:2607.09501, implemented in lambdag.py) offer a way round it: lambda_G is
# already sign-aligned with the truth and merely inflated in magnitude, so scaling
# it by a factor read off the questioned document itself -- its length, or its
# proportion of hapax legomena -- yields a log-LR without any labelled data.
#
# WHETHER THAT LR IS HONEST HAS TO BE MEASURED, not assumed, and this corpus can
# measure it, because the region runs carry two labelled classes: the chronicle's
# own other regions against the cohort of strangers (same work), and, from the
# cross-work run, one author's second work against the same cohort. The cost of the
# corrected scores on those two classes is reported FIRST. If the correction is not
# calibrated on the class the attribution is about, its verdicts about the
# chronicle are not evidence, whatever their magnitude.
#
#   python experiments/modelfree_regions.py
#   python experiments/modelfree_regions.py --regions regions__A1__r20__q2000__early.jsonl
#
# Output: medieval/kc_reuse/kc_regions_modelfree.tsv and the table on stdout

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
import run_witness_cells as W  # noqa: E402
from lambdag import cllr, cllr_min, hapax_correction, sqrt_correction  # noqa: E402
from calibrate_kc_regions import verse_ranges  # noqa: E402

SCORES = ROOT / "scores" / "konrad_profile"
OUT = ROOT / "medieval" / "kc_reuse" / "kc_regions_modelfree.tsv"
CORR = {"hapax": "the hapax correction", "sqrt": "the square-root correction"}


def toks(entry):
    return [t for s in entry["sents"] for t in s]


def cut(t, L, n=None):
    out = [t[i:i + L] for i in range(0, len(t) - L + 1, L)]
    return out if n is None else out[:n]


def passage(bank, kind, text, j, L, witness):
    """the very tokens that were scored, rebuilt as the runner cut them"""
    t = toks(bank[text]) if text in bank else []
    if not t:
        return []
    if kind == "candidate":
        c = cut(t, L, 8)
    elif kind == "control":
        c = cut(t, L, 4)
    elif kind == "cohort":
        return t[int(j):int(j) + L]
    else:                      # the chronicle's own regions, cut region by region
        return []
    return c[int(j)] if 0 <= int(j) < len(c) else []


def corrected(lam, q, how):
    n = len(q)
    if not n:
        return float("nan")
    if how == "sqrt":
        return float(sqrt_correction(lam, n))
    v1 = sum(1 for _, c in Counter(q).items() if c == 1)
    return float(hapax_correction(lam, n, v1))


def classes(bank, rows, L, witness, how):
    """the corrected scores of the two labelled classes a region run carries"""
    same, diff = [], []
    for r in rows:
        if r["kind"] not in ("own", "cohort"):
            continue
        q = passage(bank, r["kind"], r["text"], r.get("j", -1), L, witness)
        if r["kind"] == "own" or not q:
            continue
        diff.append(corrected(float(r["lambda_G"]), q, how))
    return same, diff


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--regions", default="regions__A1__r20__q2000__early.jsonl")
    ap.add_argument("--calibration", default="calib__earlycase__q2000.jsonl",
                    help="the cross-work run whose classes say whether these "
                         "corrected scores are calibrated at all")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    rows = [json.loads(l) for l in (SCORES / args.regions).open(
        encoding="utf-8")]
    meta, rows = rows[0], rows[1:]
    L = int(meta["L"])
    bank = W.load(meta["stream"])

    # the class the attribution is about, corrected the same way
    crows = [json.loads(l) for l in (SCORES / args.calibration).open(
        encoding="utf-8")][1:]
    for how in ("hapax", "sqrt"):
        same, diff = [], []
        for r in crows:
            if r["kind"] == "same_author":
                q = None
                for o in cut(toks(bank[r["text"]]), L, 3):
                    q = o
                    break
                v = corrected(float(r["lambda_G"]), q or [], how)
            else:
                v = corrected(float(r["lambda_G"]),
                              toks(bank[r["text"]])[:L], how)
            (same if r["kind"] == "same_author" else diff).append(v)
        same, diff = np.array(same), np.array(diff)
        print(f"{CORR[how]}, on one author's two different works: "
              f"Cllr {cllr(same, diff):.3f}, Cllr_min {cllr_min(same, diff):.3f}"
              f", median {np.median(same):+.2f} against {np.median(diff):+.2f}")
    print()

    by = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if "region" not in r:
            continue
        by[int(r["region"])][r["kind"]].append(r)
    regions = sorted(by)
    vr = verse_ranges(meta["witness"], int(meta["region_len"]), len(regions))
    out = []
    print(f"  {'region':>6}{'verses':>14}{'candidate':>11}{'control':>9}"
          f"{'cohort':>9}   (hapax-corrected log-LR)")
    for i in regions:
        v = {}
        for kind in ("candidate", "control", "cohort"):
            vals = [corrected(float(r["lambda_G"]),
                              passage(bank, kind, r["text"], r.get("j", -1), L,
                                      meta["witness"]), "hapax")
                    for r in by[i][kind]]
            vals = [x for x in vals if not np.isnan(x)]
            v[kind] = float(np.median(vals)) if vals else float("nan")
        out.append(dict(region=i, verses=f"{vr[i][0]}-{vr[i][1]}", **v))
        print(f"  {i:>6}{out[-1]['verses']:>14}{v['candidate']:>11.2f}"
              f"{v['control']:>9.2f}{v['cohort']:>9.2f}")
    n = sum(r["candidate"] > 0 for r in out)
    nc = sum(r["control"] > 0 for r in out)
    print(f"\n  positive odds for the candidate in {n} of {len(out)} regions, "
          f"for the controls in {nc}; strangers at "
          f"{np.median([r['cohort'] for r in out]):+.2f}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as fh:
        fh.write("region\tverses\tcandidate\tcontrol\tcohort\n")
        for r in out:
            fh.write(f"{r['region']}\t{r['verses']}\t{r['candidate']:.4f}\t"
                     f"{r['control']:.4f}\t{r['cohort']:.4f}\n")
    print(f"  wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
