# Pooled analysis of the genre-absent grid: every (target language, known
# genre) cell run under all four reference arms, pooled so that the arm
# comparison has enough independent authors to be worth making.
#
# German alone yields 10 candidate authors for drama-to-poetry and 4 for
# novels-to-poetry; author-resampled confidence intervals at that size span
# roughly 0.2 AUC, which cannot separate the arms. Pooling the nine languages
# that have cross-genre-with-poetry authors raises the pool to ~36 and is the
# only honest route to an arm comparison.
#
# Reported per arm: pooled AUC of the symmetrised lambda (the PRIMARY readout,
# because the donor pool is its denominator and therefore actually responds to
# the arm) and of the cohort statistic mean_j/sd_j (secondary); confidence
# intervals resample (language, author) units, never rows, since the three
# windows of one author are not independent evidence; the paired per-cell
# record against the oracle; and the cross-arm agreement of each statistic --
# a paired comparison that does not depend on author counts at all.
#
#   python experiments/analyze_genreabsent_pooled.py

import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
XS = HERE.parent / "scores" / "xabsent2"
from scipy import stats as sps  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

ARMS = ["oracle", "borrowed", "native-wrong", "pooled"]


def load_all():
    """(cell, arm) -> rows, with the standardised statistic attached."""
    out = {}
    for fn in sorted(XS.glob("*__K10000w100.jsonl")):
        stem = fn.name[:-len("__K10000w100.jsonl")]
        cell, arm = stem.split("__")
        rows = [json.loads(l) for l in open(fn, encoding="utf-8")]
        if not rows:
            continue
        for r in rows:
            lam = np.array(r["lam_j"], dtype=float)
            r["_t"] = float(lam.mean() / (lam.std() + 1e-9))
            r["_sym"] = float(r["lam_sym"])
            r["_unit"] = f"{cell}|{r['cand']}"
            r["_key"] = (cell, r["cand"], r["quest"], r["qw"])
        out[(cell, arm)] = rows
    return out


def auc(rows, key):
    y = np.array([r["within"] for r in rows])
    s = np.array([r[key] for r in rows], dtype=float)
    return float(roc_auc_score(y, s))


def boot(rows, key, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    units = sorted({r["_unit"] for r in rows})
    by = {u: [r for r in rows if r["_unit"] == u] for u in units}
    vals = []
    for _ in range(n):
        pick = rng.choice(len(units), size=len(units), replace=True)
        samp = [r for i in pick for r in by[units[i]]]
        if len({r["within"] for r in samp}) < 2:
            continue
        vals.append(auc(samp, key))
    if not vals:
        return float("nan"), float("nan")
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    data = load_all()
    cells = sorted({c for c, _ in data})
    if not cells:
        raise SystemExit("no score files under scores/xabsent")
    print(f"{len(cells)} cells: {', '.join(cells)}\n")

    print("PER-CELL AUC of the symmetrised lambda (primary readout)")
    print(f"{'cell':30s} {'auth':>5s} " + "".join(f"{a:>14s}" for a in ARMS))
    for c in cells:
        row = f"{c:30s}"
        any_rows = next((data[(c, a)] for a in ARMS if (c, a) in data), None)
        row += f" {len({r['_unit'] for r in any_rows}):5d} "
        for a in ARMS:
            row += (f"{auc(data[(c, a)], '_sym'):14.3f}" if (c, a) in data
                    else f"{'--':>14s}")
        print(row)

    print("\nPOOLED across cells (CI resamples language-author units)")
    print(f"{'arm':14s} {'legal':>6s} {'units':>6s} {'cases':>6s} "
          f"{'AUC sym':>8s} {'95% CI':>16s} {'AUC t':>7s}")
    for a in ARMS:
        rows = [r for c in cells if (c, a) in data for r in data[(c, a)]]
        if not rows:
            continue
        lo, hi = boot(rows, "_sym")
        print(f"{a:14s} {'no' if a == 'oracle' else 'yes':>6s} "
              f"{len({r['_unit'] for r in rows}):6d} {len(rows):6d} "
              f"{auc(rows, '_sym'):8.3f} {f'[{lo:.3f},{hi:.3f}]':>16s} "
              f"{auc(rows, '_t'):7.3f}")

    print("\nPAIRED per-cell record against the oracle (symmetrised lambda)")
    for a in ARMS:
        if a == "oracle":
            continue
        wins = losses = 0
        diffs = []
        for c in cells:
            if (c, a) not in data or (c, "oracle") not in data:
                continue
            d = auc(data[(c, a)], "_sym") - auc(data[(c, "oracle")], "_sym")
            diffs.append(d)
            wins += d > 0
            losses += d < 0
        if diffs:
            n = wins + losses
            # exact two-sided sign test -- with a handful of cells the floor on
            # the attainable p-value is itself worth printing
            from math import comb
            p = min(1.0, 2 * sum(comb(n, i) for i in range(min(wins, losses) + 1))
                    / 2 ** n) if n else float("nan")
            print(f"  {a:14s} beats oracle in {wins}/{n} cells, "
                  f"median difference {np.median(diffs):+.3f}, sign-test p={p:.3f} "
                  f"(floor at n={n}: {2 / 2 ** n if n else float('nan'):.3f})")

    print("\nPAIRED ARM CONTRASTS (author-level; only arms sharing a case list)")
    for a, b in combinations(ARMS, 2):
        per_auth, shared_ok = [], True
        for c in cells:
            if (c, a) not in data or (c, b) not in data:
                continue
            ka = {r["_key"]: r for r in data[(c, a)]}
            kb = {r["_key"]: r for r in data[(c, b)]}
            common = [k for k in ka if k in kb]
            if len(common) < len(ka):
                shared_ok = False
            for u in sorted({ka[k]["_unit"] for k in common}):
                sub = [k for k in common if ka[k]["_unit"] == u]
                ys = {ka[k]["within"] for k in sub}
                if len(ys) < 2:
                    continue
                ra = [ka[k] for k in sub]
                rb = [kb[k] for k in sub]
                per_auth.append(auc(ra, "_sym") - auc(rb, "_sym"))
        if len(per_auth) >= 4:
            w = sum(d > 0 for d in per_auth)
            n = sum(d != 0 for d in per_auth)
            from math import comb
            p = min(1.0, 2 * sum(comb(n, i) for i in range(min(w, n - w) + 1))
                    / 2 ** n) if n else float("nan")
            flag = "" if shared_ok else "  [NOT fully paired -- interpret with care]"
            print(f"  {a:13s} - {b:13s}: median {np.median(per_auth):+.3f} over "
                  f"{len(per_auth)} authors, {w}/{n} positive, sign p={p:.3f}{flag}")

    print("\nREFERENCE-DEPENDENCE: paired cross-arm agreement, pooled over cells")
    print("  (this comparison is paired case-by-case and does not depend on "
          "author counts)")
    for stat, label in (("_t", "cohort t"), ("_sym", "symmetrised lambda")):
        rs = []
        for a, b in combinations(ARMS, 2):
            va, vb = [], []
            for c in cells:
                if (c, a) not in data or (c, b) not in data:
                    continue
                ka = {r["_key"]: r[stat] for r in data[(c, a)]}
                kb = {r["_key"]: r[stat] for r in data[(c, b)]}
                for k in ka:
                    if k in kb:
                        va.append(ka[k]); vb.append(kb[k])
            if len(va) > 10:
                rs.append(sps.pearsonr(np.array(va), np.array(vb))[0])
        if rs:
            print(f"  {label:13s} cross-arm r: median {np.median(rs):.3f}, "
                  f"range [{min(rs):.3f}, {max(rs):.3f}]  (n={len(rs)} arm pairs)")


if __name__ == "__main__":
    main()
