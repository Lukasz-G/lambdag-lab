# Part 1: fit b_hat(K) = f(within_mean, within_sd) per L from the within-K run
#         (run_withink.py); report pooled R^2, within-dataset R^2, LODO transfer.
# Part 2: the decisive test -- recalibrate the long-text case sets (run_longtexts.py)
#         with the CASE-INTERNAL location estimate:  z = (lambda - b_hat*N) / sqrt(N)
#         against (a) the raw sqrt rendering and (b) the measured-b oracle.
#
# Leakage discipline: the long-text cases use bank tokens [0,L) as known and [L,2L) /
# [2L,3L) as questioned; within-K statistics used here EXCLUDE windows 0 and 1
# (CLEAN stats, >= 6 within pairs required), and the b_hat regression applied to a
# dataset is fitted LEAVE-ONE-DATASET-OUT on the other datasets' clean stats.
# Residual caveat: texts >= 3L contribute a second same-author case whose questioned
# span is window 2, which clean stats retain (~1/6 of within pairs; per-author
# constant effect, noted not fixed).
#
#   python experiments/fit_bhat.py

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
SCORES = HERE / "scores"
MASKED = HERE.parent / "masked"
ROTATIONS = 2
MIN_CLEAN_WITHIN = 6
sys.path.insert(0, str(HERE.parent))
from lambdag import cllr, cllr_min  # noqa: E402


def stats_from_file(fn, min_kw=0):
    """Per-author clean stats: within pairs with kw,qw >= min_kw; cross with kw >= min_kw."""
    within, cross = defaultdict(list), defaultdict(list)
    for line in open(fn, encoding="utf-8"):
        r = json.loads(line)
        w = r["lambda_G"] / r["n_q"]
        if r["within"]:
            if r["kw"] >= min_kw and r["qw"] >= min_kw:
                within[r["known"]].append(w)
        elif r["kw"] >= min_kw:
            cross[r["known"]].append(w)
    out = {}
    for a in within:
        if len(within[a]) >= MIN_CLEAN_WITHIN and len(cross.get(a, [])) >= 4:
            wi, cr = np.array(within[a]), np.array(cross[a])
            out[a] = (wi.mean(), wi.std(ddof=1), cr.mean())
    return out


def all_stats(L, min_kw):
    """{ds: {author: (wmean, wsd, cmean)}} for one length."""
    out = {}
    for fn in sorted(SCORES.glob(f"*__kn__withink__L{L}.jsonl")):
        st = stats_from_file(fn, min_kw)
        if len(st) >= 8:
            out[fn.name.split("__")[0]] = st
    return out


def design(stats_by_ds, exclude=None):
    X, y, ds_lab = [], [], []
    for ds, st in stats_by_ds.items():
        if ds == exclude:
            continue
        for a, (wm, wsd, cm) in st.items():
            X.append([1.0, wm, wsd]); y.append(cm); ds_lab.append(ds)
    return np.array(X), np.array(y), ds_lab


def ols(X, y):
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return beta


def r2(y, yhat):
    ss = float(((y - yhat) ** 2).sum()); st = float(((y - y.mean()) ** 2).sum())
    return 1 - ss / st if st > 0 else float("nan")


# --------------------------------------------------------------------------- #
# Part 1: the regression
# --------------------------------------------------------------------------- #

def part1():
    print("=== PART 1: b_hat(K) = beta0 + beta1*within_mean + beta2*within_sd ===")
    fitted = {}
    for L in (2000, 5000):
        stats_by_ds = all_stats(L, min_kw=0)          # full stats for the headline fit
        X, y, ds_lab = design(stats_by_ds)
        beta = ols(X, y)
        pooled = r2(y, X @ beta)
        # within-dataset R^2: demean per dataset, refit
        Xd, yd = X.copy().astype(float), y.copy().astype(float)
        for ds in set(ds_lab):
            m = np.array([d == ds for d in ds_lab])
            Xd[m, 1:] -= Xd[m, 1:].mean(axis=0); yd[m] -= yd[m].mean()
        bw = ols(Xd, yd)
        within_r2 = r2(yd, Xd @ bw)
        # LODO transfer: per dataset, Spearman(predicted, actual)
        lodo_rs = []
        for ds in stats_by_ds:
            Xt, yt, _ = design(stats_by_ds, exclude=ds)
            b = ols(Xt, yt)
            st = stats_by_ds[ds]
            Xe = np.array([[1.0, v[0], v[1]] for v in st.values()])
            ye = np.array([v[2] for v in st.values()])
            lodo_rs.append(stats.spearmanr(Xe @ b, ye).statistic)
        fitted[L] = (stats_by_ds, beta)
        print(f"  L={L}: n={len(y)} authors  beta=[{beta[0]:+.4f}, {beta[1]:+.3f}, "
              f"{beta[2]:+.3f}]  pooled R2 {pooled:.3f}  within-dataset R2 {within_r2:.3f}  "
              f"LODO Spearman median {np.nanmedian(lodo_rs):+.2f} "
              f"({sum(1 for r in lodo_rs if r > 0)}/{len(lodo_rs)} positive)")
    return fitted


# --------------------------------------------------------------------------- #
# Part 2: Cllr rescue on the long-text case sets
# --------------------------------------------------------------------------- #

def rebuild_case_meta(ds, L):
    """Replicate run_longtexts.build_cases pairing (known, quest author) per cid."""
    bank = {}
    for f in sorted((MASKED / ds / "bank").glob("*.tsv")):
        n = 0
        for line in open(f, encoding="utf-8"):
            line = line.rstrip("\n")
            if line:
                n += len(line.split("\t"))
        bank[f.stem] = n
    names = sorted(bank)
    meta = []
    for i, a in enumerate(names):
        if bank[a] < 2 * L:
            continue
        meta.append((a, a))
        if bank[a] >= 3 * L:
            meta.append((a, a))
        hits = 0
        for step in range(1, len(names)):
            b = names[(i + step) % len(names)]
            if bank[b] < 2 * L:
                continue
            meta.append((a, b)); hits += 1
            if hits == ROTATIONS:
                break
    return meta


def part2():
    print("\n=== PART 2: Cllr with case-internal b_hat on the long-text cases ===")
    print(f"{'dataset':20s} {'L':>5s} {'n':>4s} {'cov':>5s}   "
          f"{'AUC raw/bhat':>14s}   {'Cllr raw':>8s} {'bhat':>7s} {'oracle':>7s} "
          f"{'pooled':>7s}   {'Cllrmin raw/bhat':>17s}")
    for L in (2000, 5000):
        clean = all_stats(L, min_kw=2)                # CLEAN stats for application
        for fn in sorted(SCORES.glob(f"*__kn__long__L{L}.jsonl")):
            ds = fn.name.split("__")[0]
            if ds not in clean:
                continue
            recs = [json.loads(l) for l in open(fn, encoding="utf-8")]
            meta = rebuild_case_meta(ds, L)
            assert len(meta) == len(recs), f"{ds} L{L}: case-count mismatch"
            # recs are written grouped by exclusion set, NOT in case order -> join on id
            for r in recs:
                a, b = meta[r["id"]]
                assert r["label"] == int(a == b), f"{ds} L{L} id {r['id']}: label mismatch"
            # LODO fit on clean stats of every other dataset
            Xt, yt, _ = design(clean, exclude=ds)
            beta = ols(Xt, yt)
            yt_sd = [v[1] for d2, s2 in clean.items() if d2 != ds for v in s2.values()]
            st = clean[ds]
            lam, n, y, bhat, borac, wmean, wsd = [], [], [], [], [], [], []
            for r in recs:
                a, b = meta[r["id"]]
                if a not in st:
                    continue
                wm, ws, cm = st[a]
                lam.append(r["lambda_G"]); n.append(r["n_q"]); y.append(r["label"])
                bhat.append(beta[0] + beta[1] * wm + beta[2] * ws)
                borac.append(cm); wmean.append(wm); wsd.append(ws)
            lam, n, y = np.array(lam), np.array(n, float), np.array(y)
            bhat, borac = np.array(bhat), np.array(borac)
            wmean, wsd = np.array(wmean), np.array(wsd)
            if len(set(y.tolist())) < 2:
                continue
            raw = lam / np.sqrt(n)
            # Gaussian two-class LLR, all parameters case-internal:
            #   mu_s = within_mean*N (measured from K), mu_d = b_hat*N (LODO regression),
            #   sigma = within_sd*N shrunk 50/50 toward the LODO-median within_sd.
            # LLR = (mu_s - mu_d)/sigma^2 * (lambda - (mu_s + mu_d)/2) / ln(10)
            med_sd = float(np.median(yt_sd))
            sig = np.sqrt(0.5 * (wsd ** 2 + med_sd ** 2)) * n
            def gauss(mu_d_tok):
                mu_s, mu_d = wmean * n, mu_d_tok * n
                return (mu_s - mu_d) / sig ** 2 * (lam - (mu_s + mu_d) / 2) / np.log(10)
            zh, zo = gauss(bhat), gauss(borac)
            # pooled variant: dataset-median parameters -> affine per dataset, same
            # ranking as raw, only the magnitudes re-anchored (still label-free).
            pm_s = float(np.median(wmean)) * n
            pm_d = float(np.median(bhat)) * n
            psig = float(np.median(np.sqrt(0.5 * (wsd ** 2 + med_sd ** 2)))) * n
            zp = (pm_s - pm_d) / psig ** 2 * (lam - (pm_s + pm_d) / 2) / np.log(10)
            from sklearn.metrics import roc_auc_score
            out = {}
            for name, s in [("raw", raw), ("bhat", zh), ("orac", zo), ("pool", zp)]:
                out[name] = (roc_auc_score(y, s), cllr(s[y == 1], s[y == 0]),
                             cllr_min(s[y == 1], s[y == 0]))
            cov = len(lam) / len(recs)
            print(f"{ds:20s} {L:5d} {len(lam):4d} {cov:5.0%}   "
                  f"{out['raw'][0]:.3f}/{out['bhat'][0]:.3f}   "
                  f"{out['raw'][1]:8.3f} {out['bhat'][1]:7.3f} {out['orac'][1]:7.3f} "
                  f"{out['pool'][1]:7.3f}   "
                  f"{out['raw'][2]:.3f}/{out['bhat'][2]:.3f}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    part1()
    part2()
