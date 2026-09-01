# The DECISIVE TEST: does the case-internal location estimate rescue the primary
# grid's ill-calibrated cells (sentences, symmetric L in {600, 1200})?
#
# Ingredients (matched length everywhere):
#   component A  {ds}__kn__withink__L{L}.jsonl   bank authors  -> LODO b_hat regression
#   component B  {ds}__kn__withinkT__L{L}.jsonl  TEST known authors -> case-internal
#                within_mean / within_sd (their own document) + measured cross_mean (oracle)
#   cases        {ds}__kn__sent__L{L}.jsonl      the ill-calibrated grid cells,
#                joined to known authors via pairs.tsv on the stored id
# Variants per case set (identical case subset):
#   raw    lambda / sqrt(N)                       (the paper's sqrt rendering)
#   bhat   Gaussian two-class LLR, mu_s = within_mean*N (measured, case-internal),
#          mu_d = b_hat*N (LODO regression on bank stats), sigma = shrunk within_sd*N
#   orac   same with mu_d = measured cross_mean (ceiling, not case-internal)
#   pool   dataset-median parameters (affine, ranking-preserving, label-free)
#
#   python experiments/analyze_decisive.py

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SCORES = HERE / "scores"
MASKED = HERE.parent / "masked"
sys.path.insert(0, str(HERE.parent))
from lambdag import cllr, cllr_min  # noqa: E402

DS = ["romanian_dracor", "english_novels", "italian_dracor", "german_novels",
      "slovenian_novels", "french_novels", "spanish_novels", "hungarian_dracor",
      "italian_novels", "latvian_novels", "swedish_novels", "polish_novels"]


def stats_from(fn):
    within, cross = defaultdict(list), defaultdict(list)
    for line in open(fn, encoding="utf-8"):
        r = json.loads(line)
        (within if r["within"] else cross)[r["known"]].append(r["lambda_G"] / r["n_q"])
    return {a: (float(np.mean(within[a])), float(np.std(within[a], ddof=1)),
                float(np.mean(cross[a])))
            for a in within if len(within[a]) >= 6 and len(cross.get(a, [])) >= 4}


def ols(X, y):
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return beta


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    from sklearn.metrics import roc_auc_score
    print(f"{'dataset':20s} {'L':>5s} {'n':>4s} {'cov':>5s}  "
          f"{'AUC raw/bhat':>13s}  {'Cllr raw':>8s} {'bhat':>7s} {'orac':>7s} "
          f"{'pool':>7s}  {'floor raw/bhat':>14s}")
    agg = defaultdict(list)
    for L in (1200, 600):
        bank_stats = {ds: stats_from(SCORES / f"{ds}__kn__withink__L{L}.jsonl")
                      for ds in DS if (SCORES / f"{ds}__kn__withink__L{L}.jsonl").exists()}
        bank_stats = {ds: st for ds, st in bank_stats.items() if len(st) >= 8}
        for ds in DS:
            fT = SCORES / f"{ds}__kn__withinkT__L{L}.jsonl"
            fC = SCORES / f"{ds}__kn__sent__L{L}.jsonl"
            if not (fT.exists() and fC.exists()):
                continue
            test_st = stats_from(fT)
            if len(test_st) < 4:
                continue
            # LODO regression on the OTHER datasets' bank stats at this L
            X, y, sds = [], [], []
            for d2, st in bank_stats.items():
                if d2 == ds:
                    continue
                for wm, wsd, cm in st.values():
                    X.append([1.0, wm, wsd]); y.append(cm); sds.append(wsd)
            beta = ols(np.array(X), np.array(y))
            med_sd = float(np.median(sds))

            man = [l.rstrip("\n").split("\t")
                   for l in open(MASKED / ds / "pairs.tsv", encoding="utf-8")][1:]
            ka_of = {int(pid): ka for pid, _l, ka, _q in man}
            lab_of = {int(pid): int(lab) for pid, lab, _k, _q in man}
            recs = [json.loads(l) for l in open(fC, encoding="utf-8")]
            lam, n, yy, wmean, wsd_, bhat, borac = [], [], [], [], [], [], []
            for r in recs:
                assert r["label"] == lab_of[r["id"]], f"{ds} L{L} id {r['id']} label"
                a = ka_of[r["id"]]
                if a not in test_st:
                    continue
                wm, ws, cm = test_st[a]
                lam.append(r["lambda_G"]); n.append(r["n_q"]); yy.append(r["label"])
                wmean.append(wm); wsd_.append(ws)
                bhat.append(beta[0] + beta[1] * wm + beta[2] * ws)
                borac.append(cm)
            lam, n, yy = np.array(lam), np.array(n, float), np.array(yy)
            wmean, wsd_ = np.array(wmean), np.array(wsd_)
            bhat, borac = np.array(bhat), np.array(borac)
            if len(set(yy.tolist())) < 2:
                continue
            raw = lam / np.sqrt(n)
            sig = np.sqrt(0.5 * (wsd_ ** 2 + med_sd ** 2)) * n

            def gauss(mu_d_tok):
                mu_s, mu_d = wmean * n, mu_d_tok * n
                return (mu_s - mu_d) / sig ** 2 * (lam - (mu_s + mu_d) / 2) / np.log(10)

            zh, zo = gauss(bhat), gauss(borac)
            pm_s, pm_d = float(np.median(wmean)) * n, float(np.median(bhat)) * n
            psig = float(np.median(np.sqrt(0.5 * (wsd_ ** 2 + med_sd ** 2)))) * n
            zp = (pm_s - pm_d) / psig ** 2 * (lam - (pm_s + pm_d) / 2) / np.log(10)

            out = {}
            for name, s in [("raw", raw), ("bhat", zh), ("orac", zo), ("pool", zp)]:
                out[name] = (roc_auc_score(yy, s), cllr(s[yy == 1], s[yy == 0]),
                             cllr_min(s[yy == 1], s[yy == 0]))
                agg[(L, name)].append(out[name][1])
            cov = len(lam) / len(recs)
            print(f"{ds:20s} {L:5d} {len(lam):4d} {cov:5.0%}  "
                  f"{out['raw'][0]:.3f}/{out['bhat'][0]:.3f}  "
                  f"{out['raw'][1]:8.3f} {out['bhat'][1]:7.3f} {out['orac'][1]:7.3f} "
                  f"{out['pool'][1]:7.3f}  {out['raw'][2]:.3f}/{out['bhat'][2]:.3f}")
    print()
    for L in (1200, 600):
        if (L, "raw") in agg:
            line = f"L={L} median Cllr:  " + "  ".join(
                f"{k} {np.median(agg[(L, k)]):.3f}" for k in ("raw", "bhat", "orac", "pool"))
            wins = sum(1 for a, b in zip(agg[(L, 'bhat')], agg[(L, 'raw')]) if a < b)
            line += f"   bhat<raw in {wins}/{len(agg[(L, 'raw')])}"
            print(line)


if __name__ == "__main__":
    main()
