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

        singles = sorted(a for a in stats if a.startswith("single:"))
        splits = sorted(a for a in stats if a.startswith("split:"))
        common = set(stats["mixed"])
        for a in singles + splits:
            common &= set(stats[a])
        common = sorted(common)
        if not common:
            print("  (no authors common to all arms)\n")
            continue

        def paired(lhs, rhs, label):
            d = np.array(lhs) - np.array(rhs)
            rng = np.random.default_rng(0)
            boot = [np.mean(d[rng.integers(0, len(d), len(d))]) for _ in range(B)]
            lo, hi = np.percentile(boot, [2.5, 97.5])
            verdict = ("first" if lo > 0 else "second" if hi < 0 else "neither")
            print(f"    {label:34s} {d.mean():+8.2f}  [{lo:+7.2f},{hi:+7.2f}]  "
                  f"{int((d > 0).sum()):2d}/{len(d)}  "
                  f"{'^' if verdict == 'first' else 'v' if verdict == 'second' else '='}")
            return d.mean(), lo, hi, verdict

        mixed = [stats["mixed"][a] for a in common]
        best_single = [max(stats[s][a] for s in singles) for a in common]
        print(f"\n  PAIRED differences over {len(common)} authors "
              f"(^ = first arm wins, v = second, = unresolved):")
        r_ms = paired(mixed, best_single, "mixed - better single")

        if splits:
            # THE CONTROL. split:X is the same genre as single:X and the same
            # total, differing only in being drawn from two non-adjacent blocks.
            # If splitting alone costs what mixing costs, the mixed arm's deficit
            # is fragmentation of the known text and carries no claim about
            # genre; only a mixed-minus-split gap isolates genre.
            for s in singles:
                g = s.split(":", 1)[1]
                sp = f"split:{g}"
                if sp in stats:
                    paired([stats[sp][a] for a in common],
                           [stats[s][a] for a in common],
                           f"split:{g} - single:{g}  (fragmentation)")
            best_split = [max(stats[s][a] for s in splits) for a in common]
            r_msp = paired(mixed, best_split,
                           "mixed - better split  (genre, net)")
            rows.append((C, r_ms, r_msp))
        else:
            rows.append((C, r_ms, None))
        print()

    if rows:
        print("SUMMARY")
        print(f"  {'questioned':12s} {'mixed-single':>22s} {'mixed-split':>22s}")
        for C, ms, msp in rows:
            a = f"{ms[0]:+7.2f} [{ms[1]:+6.1f},{ms[2]:+6.1f}]"
            b = (f"{msp[0]:+7.2f} [{msp[1]:+6.1f},{msp[2]:+6.1f}]"
                 if msp else "n/a")
            print(f"  {C:12s} {a:>22s} {b:>22s}")
        print()
        if all(r[2] for r in rows):
            print("READING. mixed-single is the headline; mixed-split is what it")
            print("means. If split arms lose as much as mixed, the cost is")
            print("FRAGMENTING the known text and the comparison says nothing")
            print("about genre. Only a mixed-split gap that excludes zero is")
            print("evidence that mixing GENRES costs beyond mixing blocks.")


if __name__ == "__main__":
    main()
