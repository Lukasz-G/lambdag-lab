# Read the mixed-vs-single known-text comparison (run_mixedknown.py).
#
# THE COMPARISON IS PAIRED and must stay that way: every arm scored the same
# questioned windows, under the same donor pool, for the same authors, so the
# difference is taken WITHIN a case and the large between-author variance
# cancels. An unpaired comparison of these arms would be dominated by which
# authors happened to be eligible.
#
# THE DECISION RULE is mixed against the BETTER single arm, never the mean of
# the two. Mixing draws half its known text from each genre, so against the
# average it could win merely by hedging which genre it drew from -- an
# insurance effect, not evidence that variety per se carries authorial signal.
# Beating the better single arm is the only result that discriminates.
#
# Confidence intervals bootstrap over AUTHORS, not over cases: the windows of one
# author are not independent observations of the effect.
#
#   python experiments/analyze_mixedknown.py
#
# Reads experiments/scores/mixedknown/*.jsonl.

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
IN = HERE / "scores" / "mixedknown"
B = 5000


def load():
    by = defaultdict(dict)                       # questioned -> arm -> [rec]
    for f in sorted(IN.glob("*.jsonl")):
        C, arm = f.stem.split("__")
        by[C][arm.replace("-", ":", 1)] = [json.loads(l) for l in
                                           open(f, encoding="utf-8") if l.strip()]
    return by


def auc(lam, y):
    from sklearn.metrics import roc_auc_score
    return roc_auc_score(y, lam) if 0 < sum(y) < len(y) else float("nan")


def per_author_signal(recs):
    """signal = mean lambda(same) - mean lambda(diff), per QUESTIONED author."""
    same, diff = defaultdict(list), defaultdict(list)
    for r in recs:
        (same if r["label"] else diff)[r["q_author"]].append(r["lambda_G"])
    out = {}
    for a in same:
        if a in diff and same[a] and diff[a]:
            out[a] = float(np.mean(same[a]) - np.mean(diff[a]))
    return out


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    by = load()
    if not by:
        print("no score files; run run_mixedknown.py first")
        return

    print("Known side fixed in size; only its COMPOSITION differs. Every arm is "
          "cross-genre\nwith respect to the questioned text.\n")
    rows = []
    for C in sorted(by):
        arms = by[C]
        if "mixed" not in arms:
            continue
        print(f"=== questioned: {C} ===")
        print(f"  {'arm':16s} {'n':>4s} {'AUC':>6s} {'mean λ same':>12s} "
              f"{'mean λ diff':>12s} {'signal':>9s}")
        stats = {}
        for arm in sorted(arms, key=lambda a: (a != "mixed", a)):
            recs = arms[arm]
            lam = np.array([r["lambda_G"] for r in recs])
            y = np.array([r["label"] for r in recs])
            s1, s0 = lam[y == 1].mean(), lam[y == 0].mean()
            stats[arm] = per_author_signal(recs)
            print(f"  {arm:16s} {len(recs):4d} {auc(lam, y):6.3f} {s1:12.1f} "
                  f"{s0:12.1f} {s1 - s0:9.1f}")

        singles = [a for a in stats if a != "mixed"]
        common = set(stats["mixed"])
        for a in singles:
            common &= set(stats[a])
        common = sorted(common)
        if not common:
            print("  (no authors common to all arms)\n")
            continue

        mixed = np.array([stats["mixed"][a] for a in common])
        best = np.array([max(stats[s][a] for s in singles) for a in common])
        d = mixed - best
        rng = np.random.default_rng(0)
        boot = [np.mean(d[rng.integers(0, len(d), len(d))]) for _ in range(B)]
        lo, hi = np.percentile(boot, [2.5, 97.5])
        win = int((d > 0).sum())
        print(f"\n  PAIRED mixed - BETTER single, over {len(common)} authors:")
        print(f"    mean {d.mean():+.2f}  95% CI [{lo:+.2f}, {hi:+.2f}]  "
              f"mixed ahead in {win}/{len(d)} authors")
        print(f"    -> {'MIXED WINS' if lo > 0 else 'SINGLE WINS' if hi < 0 else 'no difference resolved'}\n")
        rows.append((C, d.mean(), lo, hi, win, len(d)))

    if rows:
        print("SUMMARY (paired mixed - better single, per questioned genre)")
        print(f"  {'questioned':12s} {'mean':>8s} {'95% CI':>20s} {'authors ahead':>15s}")
        for C, m, lo, hi, win, n in rows:
            print(f"  {C:12s} {m:+8.2f}  [{lo:+7.2f}, {hi:+7.2f}] {win:>10d}/{n}")
        print("\nA positive interval excluding zero is the signature of an author's"
              "\nstationary process including his genre switching: single-genre text"
              "\nconverges to a genre conditional, and only varied text reaches him."
              "\nA null result leaves genre as a rate, and more text of any one genre"
              "\nremains the right advice.")


if __name__ == "__main__":
    main()
