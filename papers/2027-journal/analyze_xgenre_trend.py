# Does evidence CONVERGE on the author as text grows -- within genre and across?
#
# THE QUANTITY, and why it is not the exponent. analyze_pinned_ladder.py fits a
# log-log slope and answers how fast the signal grows. That slope is near 1 both
# within and across genre, so accumulation is linear on both sides of the genre
# boundary and the pessimistic "boundary, n* explodes" branch is dead. It does
# not follow that the two behave alike: a linear law with a constant that shrinks
# as the window widens still accumulates, but never converges on the author.
#
# So the readout here is the PER-TOKEN rate lambda_G / n_q -- evidence per token,
# the quantity LambdaG's own premise says should be constant -- and the statistic
# is its CHANGE from the shortest rung to the longest. Positive means each extra
# token is worth more than the ones before it: the case sharpens. Zero means
# evidence merely accumulates at a fixed price. Negative means the rate decays
# and more text buys proportionally less, which is what non-convergence looks
# like at finite N.
#
# PAIRING. The rungs are nested windows of the same case, so the change is taken
# WITHIN a case and only then aggregated; the bootstrap resamples cases, never
# rungs. Comparing rung means across independent draws would let case
# composition move the answer, which at these sample sizes it easily can.
#
# TWO RATES ARE REPORTED, and they answer different questions:
#   same    - rate on the same-author cases alone. Does the true case get
#             stronger per token?
#   signal  - same-author mean minus different-author mean, per token. Does the
#             SEPARATION get cheaper per token? Robust to any drift common to
#             both classes (period, edition, genre offset b), so it is the
#             conservative reading and the one to quote if the two disagree.
#
#   python experiments/analyze_xgenre_trend.py
#   python experiments/analyze_xgenre_trend.py --alphabet catsrank --mode symmetric
#   python experiments/analyze_xgenre_trend.py --compare      # both alphabets
#
# Reads experiments/scores/xgenre_ladder/*.jsonl written by run_xgenre_ladder.py.

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
LAD = HERE / "scores" / "xgenre_ladder"
GENRES = ("prose", "verse", "drama")


def fname(kg, qg, mode, donor, alphabet, known):
    tag = f"{kg}2{qg}__{mode}"
    suf = "" if donor == "questioned" else f"__d-{donor}"
    if alphabet != "posnoise":
        suf += f"__{alphabet}"
    return LAD / (f"{tag}__K{known}{suf}.jsonl" if mode == "pinned"
                  else f"{tag}{suf}.jsonl")


def load(fn):
    """{id: {Q: lambda}}, {id: label}, sorted rungs."""
    lam, lab, qs = defaultdict(dict), {}, set()
    with open(fn, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            d = json.loads(line)
            lam[d["id"]][d["Q"]] = d["lambda_G"]
            lab[d["id"]] = d["label"]
            qs.add(d["Q"])
    return lam, lab, sorted(qs)


def deltas(lam, lab, qlo, qhi):
    """Per-case change in per-token rate, split by class.

    A case counts only if it reached BOTH rungs; a case that dropped out at the
    top rung for want of text would otherwise bias the top of the ladder towards
    the long -- and therefore the well-represented -- authors.
    """
    one, zero = [], []
    for i, d in lam.items():
        if qlo not in d or qhi not in d:
            continue
        (one if lab[i] else zero).append(d[qhi] / qhi - d[qlo] / qlo)
    return np.array(one), np.array(zero)


def rates(lam, lab, q):
    one = np.array([d[q] / q for i, d in lam.items() if lab[i] and q in d])
    zero = np.array([d[q] / q for i, d in lam.items() if not lab[i] and q in d])
    return one, zero


def auc(one, zero):
    if not len(one) or not len(zero):
        return float("nan")
    a = np.concatenate([one, zero])
    r = np.argsort(np.argsort(a)) + 1.0
    # average ranks over ties, else a discrete score inflates or deflates AUC
    for v in np.unique(a):
        m = a == v
        if m.sum() > 1:
            r[m] = r[m].mean()
    n1 = len(one)
    return (r[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * len(zero))


def boot_ci(vals, boot, seed, stat=np.mean):
    """Case-level bootstrap of a paired statistic.

    The MEAN is the headline, and deliberately: the change is already a
    within-case difference, so the heavy right tail that makes the mean the
    wrong summary for lambda itself has largely been differenced away. The
    median is printed beside it and is the check -- where the two disagree the
    direction is being carried by a few authors, which is worth knowing.
    """
    if len(vals) < 3:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(vals), size=(boot, len(vals)))
    s = stat(vals[idx], axis=1)
    return float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))


def boot_ci_signal(one, zero, boot, seed):
    """The two classes are independent draws, so resample them independently."""
    if len(one) < 3 or len(zero) < 3:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    a = one[rng.integers(0, len(one), size=(boot, len(one)))].mean(axis=1)
    b = zero[rng.integers(0, len(zero), size=(boot, len(zero)))].mean(axis=1)
    s = a - b
    return float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))


def one_direction(kg, qg, args):
    fn = fname(kg, qg, args.mode, args.donor, args.alphabet, args.known)
    if not fn.exists():
        return None
    lam, lab, qs = load(fn)
    if len(qs) < 2:
        return None
    qlo, qhi = qs[0], qs[-1]
    one, zero = deltas(lam, lab, qlo, qhi)
    if len(one) < 3:
        return None
    d_same = float(one.mean())
    med_same = float(np.median(one))
    lo_s, hi_s = boot_ci(one, args.boot, hash((kg, qg, "same")) & 0xFFFF)
    d_sig = float(one.mean() - zero.mean()) if len(zero) else float("nan")
    lo_g, hi_g = boot_ci_signal(one, zero, args.boot,
                                hash((kg, qg, "sig")) & 0xFFFF)
    a_lo = auc(*rates(lam, lab, qlo))
    a_hi = auc(*rates(lam, lab, qhi))
    return {"kg": kg, "qg": qg, "within": kg == qg, "n": len(one),
            "qlo": qlo, "qhi": qhi, "d_same": d_same, "med_same": med_same,
            "ci_same": (lo_s, hi_s), "d_sig": d_sig, "ci_sig": (lo_g, hi_g),
            "auc_lo": a_lo, "auc_hi": a_hi}


def table(rows, title):
    print(f"\n{title}")
    print(f"  {'direction':16s} {'n':>3s}  {'d(same)':>8s} {'95% CI':>17s} "
          f"{'med':>8s}  {'d(signal)':>9s} {'95% CI':>17s}   {'AUC lo->hi':>12s}")
    for r in rows:
        sig = "*" if (r["ci_same"][0] > 0 or r["ci_same"][1] < 0) else " "
        print(f"  {r['kg']}2{r['qg']:<10s} {r['n']:3d}  {r['d_same']:+8.4f}"
              f" [{r['ci_same'][0]:+.4f},{r['ci_same'][1]:+.4f}]{sig} "
              f"{r['med_same']:+8.4f}  "
              f"{r['d_sig']:+9.4f} [{r['ci_sig'][0]:+.4f},{r['ci_sig'][1]:+.4f}]"
              f"   {r['auc_lo']:.3f}->{r['auc_hi']:.3f}")


def summarise(rows, label):
    win = [r for r in rows if r["within"]]
    cro = [r for r in rows if not r["within"]]
    out = {}
    for name, sub in (("within", win), ("cross", cro)):
        if not sub:
            continue
        d = np.array([r["d_same"] for r in sub])
        imp = sum(1 for r in sub if r["ci_same"][0] > 0)
        deg = sum(1 for r in sub if r["ci_same"][1] < 0)
        out[name] = {"median": float(np.median(d)), "n_dir": len(sub),
                     "improve": imp, "degrade": deg}
    if "within" in out and "cross" in out:
        w, c = out["within"], out["cross"]
        print(f"\n  {label:10s} within  median {w['median']:+.4f}  "
              f"({w['improve']}/{w['n_dir']} improve, {w['degrade']} degrade)")
        print(f"  {'':10s} cross   median {c['median']:+.4f}  "
              f"({c['improve']}/{c['n_dir']} improve, {c['degrade']} degrade)")
    return out


def run(args, quiet=False):
    rows = [r for kg in GENRES for qg in GENRES
            if (r := one_direction(kg, qg, args))]
    if not rows:
        print(f"  no files for alphabet={args.alphabet} mode={args.mode} "
              f"donor={args.donor}")
        return None, []
    if not quiet:
        within = [r for r in rows if r["within"]]
        cross = [r for r in rows if not r["within"]]
        rungs = f"{rows[0]['qlo']} -> {rows[0]['qhi']} tokens"
        table(within, f"WITHIN GENRE  ({rungs}, {args.alphabet}, "
                      f"{args.mode}, donors={args.donor})")
        table(cross, f"CROSS GENRE   ({rungs}, {args.alphabet}, "
                     f"{args.mode}, donors={args.donor})")
    return summarise(rows, args.alphabet), rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alphabet", default="posnoise")
    ap.add_argument("--mode", default="symmetric",
                    choices=("symmetric", "pinned"))
    ap.add_argument("--donor", default="known",
                    choices=("known", "both", "questioned"))
    ap.add_argument("--known", type=int, default=20000)
    ap.add_argument("--boot", type=int, default=10000)
    ap.add_argument("--compare", action="store_true",
                    help="run both alphabets and print them side by side")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    if not args.compare:
        run(args)
        return

    # The comparison the run exists for. POSNoise and CatSRank encode the SAME
    # banks, the same cases and the same rungs, so a difference between the two
    # columns is a fact about the representation and nothing else.
    both = {}
    for alpha in ("posnoise", "catsrank"):
        args.alphabet = alpha
        print(f"\n{'=' * 72}\n{alpha.upper()}\n{'=' * 72}")
        s, rows = run(args)
        if s:
            both[alpha] = (s, rows)
    if len(both) < 2:
        return
    print(f"\n{'=' * 72}\nSIDE BY SIDE  (median change in per-token rate, "
          f"same-author cases)\n{'=' * 72}")
    print(f"  {'':10s} {'POSNoise':>12s} {'CatSRank':>12s}")
    for k in ("within", "cross"):
        a = both["posnoise"][0].get(k, {})
        b = both["catsrank"][0].get(k, {})
        print(f"  {k:10s} {a.get('median', float('nan')):+12.4f} "
              f"{b.get('median', float('nan')):+12.4f}")
    print(f"  {'improve':10s} "
          f"{both['posnoise'][0]['cross']['improve']:>7d}/6 cross "
          f"{both['catsrank'][0]['cross']['improve']:>4d}/6 cross")
    # Per-direction, so that a change in the median cannot hide six directions
    # moving in different ways.
    pa = {(r["kg"], r["qg"]): r["d_same"] for r in both["posnoise"][1]}
    pb = {(r["kg"], r["qg"]): r["d_same"] for r in both["catsrank"][1]}
    print(f"\n  {'direction':16s} {'POSNoise':>10s} {'CatSRank':>10s} "
          f"{'delta':>9s}")
    for k in sorted(set(pa) & set(pb), key=lambda x: (x[0] == x[1], x)):
        mark = "  (within)" if k[0] == k[1] else ""
        print(f"  {k[0]}2{k[1]:<10s} {pa[k]:+10.4f} {pb[k]:+10.4f} "
              f"{pb[k] - pa[k]:+9.4f}{mark}")


if __name__ == "__main__":
    main()
