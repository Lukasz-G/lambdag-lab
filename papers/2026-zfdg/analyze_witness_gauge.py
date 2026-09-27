# The gauge read out: what the case's own material buys in calibrated cost.
#
# Reads scores/witness_gauge/{stream}__L{L}__w{seg}.jsonl, where every score of
# one known passage -- the cases, the anchors from the known material, the
# impostor cohort -- stands against the same reference grammars, and reports
# for each route the discrimination and the cost of uttering it as odds:
#
#   raw            lambda_G, the mean over donors: the cells run's statistic
#   donor z        lambda_G divided by the spread of the per-donor lambdas
#   anchor         lambda_G minus the mean of the anchor scores, which locates
#                  the same-author hypothesis inside the case
#   anchor z       that difference divided by the anchor spread, with the
#                  sqrt(1 + 1/n) correction for estimating the location from n
#                  anchors
#   cohort z       lambda_G standardised by the impostor cohort, which locates
#                  and scales the different-author hypothesis instead
#   two-point      (lambda_G - cohort mean) / (anchor mean - cohort mean): both
#                  hypotheses located, so the units are the case's own gap
#   rank           the share of the cohort the questioned passage outscores.
#                  Under the different-author hypothesis it is exchangeable
#                  with the cohort, so this is uniform by construction and the
#                  denominator of the ratio needs no fitting at all; the
#                  numerator is the same rank taken by the anchors.
#
# Everything on the left of that list is label-free inside a case: the anchors
# are known material, the cohort is a set of texts the known author did not
# write. Only the calibrator that turns a route into odds is fitted on the
# labelled population, in five folds grouped by the known passage's identity --
# and the rank route is reported with nothing fitted at all.
#
# The anchors come in two kinds, in the known passage's own hand and in
# another, so the hand-matched and unmatched anchor are read out separately:
# the question is whether an anchor has to stand in the same scribal condition
# as the questioned text for its scale to hold.
#
#   python experiments/analyze_witness_gauge.py [--seg 100] [--routes ...]
#
# Writes medieval/tables/tab_witness_gauge.tex.

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.special import logsumexp
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
from lambdag import PAVCalibrator, cllr, cllr_min  # noqa: E402

SCORES = ROOT / "scores" / "witness_gauge"
TAB = ROOT / "medieval" / "tables" / "tab_witness_gauge.tex"
LENGTHS = [500, 1000, 2000, 5000]
LN10 = np.log(10.0)
# the console reports every route; the table carries the ones the chapter argues
HEADLINE = ["raw", "marginal denominator", "donor rank in cohort", "composition",
            "fusion: + composition [mean]", "route chosen blind", "rank ratio"]


def calibrated(x, y, groups):
    """monotone calibration bounded by what a set of this size supports, fitted
    in five folds grouped by the known passage's identity"""
    x = np.asarray(x, float); y = np.asarray(y, int); groups = np.asarray(groups)
    X = x[:, None] if x.ndim == 1 else x
    if len(set(y)) < 2 or len(set(groups)) < 5:
        return float("nan"), float("nan")
    LR = np.empty(len(y))
    for tr, te in GroupKFold(5).split(X, y, groups):
        if len(set(y[tr])) < 2:
            LR[te] = 0.0; continue
        LR[te] = PAVCalibrator().fit(X[tr], y[tr]).transform(X[te])
    return cllr(LR[y == 1], LR[y == 0]), cllr_min(LR[y == 1], LR[y == 0])


def build(rows):
    """attach the case's own anchor and cohort statistics to every case row"""
    byk = defaultdict(list)
    for r in rows:
        byk[(r["known"], r["kw"])].append(r)
    cases = []
    for _, group in byk.items():
        anch = [r for r in group if r["kind"] == "anchor"]
        coh = [r for r in group if r["kind"] == "cohort"]
        if not anch or len(coh) < 4:
            continue
        for r in group:
            if r["kind"] != "case":
                continue
            ss = r["known_scribe"] == r["quest_scribe"]
            # the anchor and the cohort, taken over all of them and over those
            # standing in the questioned passage's own scribal condition
            am = [a for a in anch if (a["cell"] == "AN-SS") == ss]
            cm = [c for c in coh if (c["cell"] == "IM-SS") == ss]
            r["_a"] = np.array([a["lambda_G"] for a in anch])
            r["_am"] = np.array([a["lambda_G"] for a in am]) if am else r["_a"]
            r["_am_matched"] = bool(am)
            r["_c"] = np.array([c["lambda_G"] for c in coh])
            r["_cm"] = np.array([c["lambda_G"] for c in cm]) if len(cm) >= 4 else r["_c"]
            # the cohort collapsed to one score per impostor identity: the unit
            # the questioned passage is exchangeable with
            per_id = defaultdict(list)
            for c in coh:
                per_id[c["quest_author"]].append(c["lambda_G"])
            r["_cid"] = np.array([np.mean(v) for v in per_id.values()])
            # the cohort donor by donor: every score of this known passage stands
            # against the same grammars in the same order, so a case can be
            # ranked against the impostors within each donor separately and the
            # ranks averaged -- the cohort rank at the resolution of r x m
            # comparisons instead of m
            r["_cmat"] = np.array([c["lam_j"] for c in coh], float)
            r["_arank"] = np.array([rank_of(a["lambda_G"], r["_cid"]) for a in anch])
            r["_arank_m"] = np.array([rank_of(a["lambda_G"], r["_cid"]) for a in am]) if am else r["_arank"]
            cases.append(r)
    return cases


def marginal(lam):
    """lambda with an arithmetic-mean denominator: log10 P(Q | G_A) minus the log
    of the MEAN likelihood over the donor grammars, rather than the mean of the
    logs the method sums. The two differ by Jensen's gap, and the arithmetic mean
    is the marginal likelihood of the questioned text under a uniform prior over
    the reference population -- so it is led by the donor that models the text
    best, where the mean of logs is held back by the donors that model it worst."""
    a = np.asarray(lam, float) * np.log(10.0)
    return float(np.log10(len(a)) - logsumexp(-a) / np.log(10.0))


def rank_of(x, cohort):
    return float(np.mean(x > cohort) + 0.5 * np.mean(x == cohort))


def sd(v, floor=1e-9):
    return float(np.std(v, ddof=1)) + floor if len(v) > 1 else float("nan")


def routes(r, ctx):
    """ctx carries the two constants a single case cannot estimate well on its
    own -- a typical anchor spread and a typical anchor-to-cohort gap -- each
    taken over the arm's OTHER identities, so no case sees its own"""
    t = r["lambda_G"]
    dz = np.std(r["lam_j"], ddof=1) + 1e-9
    a, am, c, cm, cid = r["_a"], r["_am"], r["_c"], r["_cm"], r["_cid"]
    n = len(a)
    sd0, gap0 = ctx["sd0"], ctx["gap0"]
    # the anchor spread, shrunk towards the arm's typical spread: with two or
    # three anchors the case's own estimate is far too noisy to divide by
    nu = 4.0
    s2 = (np.var(a, ddof=1) if n > 1 else 0.0)
    sd_a = np.sqrt((nu * sd0 ** 2 + (n - 1) * s2) / (nu + max(n - 1, 0))) if np.isfinite(sd0) else sd(a)
    gap = a.mean() - c.mean()
    gapm = am.mean() - cm.mean()
    floor = 0.25 * gap0 if np.isfinite(gap0) else np.nan
    out = {
        "raw": t,
        "marginal denominator": marginal(r["lam_j"]),
        "donor z": t / dz,
        "anchor": t - a.mean(),
        "anchor, matched hand": t - am.mean(),
        "anchor z": (t - a.mean()) / (sd_a * np.sqrt(1 + 1 / n)),
        "anchor z, donor sd": (t - a.mean()) / dz,
        "cohort z": (t - c.mean()) / sd(c),
        "cohort z, matched hand": (t - cm.mean()) / sd(cm),
        "two-point": (t - c.mean()) / max(gap, floor) if np.isfinite(floor) else np.nan,
        "two-point, matched hand": (t - cm.mean()) / max(gapm, floor) if np.isfinite(floor) else np.nan,
        "rank": rank_of(t, cid),
        "donor rank in cohort": float(np.mean([rank_of(v, r["_cmat"][:, j])
                                               for j, v in enumerate(r["lam_j"])])),
    }
    return out


def context(sel):
    """per identity, the two constants taken over every other identity"""
    ids = np.array([r["known_author"] for r in sel])
    sdv = np.array([np.std(r["_a"], ddof=1) if len(r["_a"]) > 1 else np.nan for r in sel])
    gapv = np.array([r["_a"].mean() - r["_c"].mean() for r in sel])
    ctx = {}
    for who in set(ids):
        keep = ids != who
        ctx[who] = dict(sd0=float(np.nanmedian(sdv[keep])) if np.isfinite(sdv[keep]).any() else float("nan"),
                        gap0=float(np.nanmedian(gapv[keep])) if keep.any() else float("nan"))
    return ctx


def fused_llr(feats, y, groups, mode="mean"):
    """mode "mean": each score calibrated on its own inside the fold and the
    ratios averaged -- a fusion that needs no weights and cannot be dominated by
    the scale of one score. mode "weighted": the scores weighted by a logistic
    fitted in the fold and the sum calibrated, the standard forensic fusion."""
    y = np.asarray(y, int); groups = np.asarray(groups)
    X = np.column_stack([np.asarray(f, float) for f in feats])
    LR = np.zeros(len(y))
    for tr, te in GroupKFold(5).split(X, y, groups):
        if len(set(y[tr])) < 2:
            continue
        if mode == "weighted":
            LR[te] = PAVCalibrator().fit(X[tr], y[tr]).transform(X[te])
        else:
            LR[te] = np.mean([PAVCalibrator().fit(X[tr, i:i + 1], y[tr]).transform(X[te, i:i + 1])
                              for i in range(X.shape[1])], axis=0)
    return LR


def selected_route(vals, y, groups, names):
    """what an examiner actually gets, with no route chosen on the answer: in
    each outer fold the route is picked by a further split of the TRAINING
    identities alone, on calibrated cost, and only then applied to the held-out
    identities. The reported cost therefore includes the price of choosing."""
    y = np.asarray(y, int); groups = np.asarray(groups)
    LR = np.zeros(len(y)); picked = []
    X = {nm: np.asarray(vals[nm], float) for nm in names}
    for tr, te in GroupKFold(5).split(np.zeros((len(y), 1)), y, groups):
        best, best_c = names[0], np.inf
        for nm in names:
            x = X[nm]
            if not np.isfinite(x[tr]).all():
                continue
            inner = np.zeros(len(tr))
            ok = True
            for itr, ite in GroupKFold(4).split(np.zeros((len(tr), 1)), y[tr], groups[tr]):
                if len(set(y[tr][itr])) < 2:
                    ok = False; break
                inner[ite] = PAVCalibrator().fit(x[tr][itr, None], y[tr][itr]).transform(x[tr][ite, None])
            if not ok:
                continue
            c = cllr(inner[y[tr] == 1], inner[y[tr] == 0])
            if c < best_c:
                best, best_c = nm, c
        picked.append(best)
        x = X[best]
        LR[te] = PAVCalibrator().fit(x[tr, None], y[tr]).transform(x[te, None])
    return LR, picked


def rank_lr(cases, y, ident, key="_arank"):
    """the ratio with nothing fitted on the labels of the arm: the denominator
    is the uniform rank of an exchangeable passage, the numerator the rank
    distribution of the anchors, estimated from every OTHER identity"""
    ident = np.asarray(ident)
    R = float(np.median([len(r["_cid"]) for r in cases]))
    edges = np.linspace(0, 1, int(min(8, max(3, R // 2))) + 1)
    lr = np.empty(len(cases))
    for who in set(ident):
        hold = ident == who
        pool = np.concatenate([r[key] for r, h in zip(cases, hold) if not h]) if (~hold).any() else None
        if pool is None or len(pool) < 20:
            lr[hold] = 0.0; continue
        cnt, _ = np.histogram(pool, bins=edges)
        dens = (cnt + 1.0) / (cnt.sum() + len(cnt))          # Laplace, so no bin is impossible
        dens = dens / np.diff(edges)                          # density against the uniform
        idx = np.clip(np.digitize([r["rank_stat"] for r, h in zip(cases, hold) if h], edges) - 1, 0, len(dens) - 1)
        lr[hold] = np.log10(dens[idx])
    return lr


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--seg", type=int, default=100)
    ap.add_argument("--tag", default="")
    ap.add_argument("--fuse", action="store_true",
                    help="also score the composition statistic of the scribe-direction "
                         "analysis on the same cases and fuse the two")
    ap.add_argument("--k", type=int, default=10, help="scribe directions projected out")
    args = ap.parse_args()
    table = []
    for stream in ("lemma", "surface"):
        for L in LENGTHS:
            fn = SCORES / f"{stream}__L{L}__w{args.seg}{args.tag}.jsonl"
            if not fn.exists():
                continue
            rows = [json.loads(l) for l in fn.open(encoding="utf-8")]
            cases = build(rows)
            comp = {}
            if args.fuse:
                import analyze_scribe_directions as SD
                comp = SD.composition_scores(stream, L, args.seg, args.k, cases, verbose=False)
            for cond, sc, dc in (("one hand", "SA-SS", "DA-SS"), ("two hands", "SA-DS", "DA-DS")):
                sel = [r for r in cases if r["cell"] in (sc, dc)]
                if not sel:
                    continue
                y = np.array([1 if r["cell"] == sc else 0 for r in sel])
                g = np.array([r["known_author"] for r in sel])
                if len(set(y)) < 2:
                    continue
                print("\n%-8s L=%-5d %-10s n=%d (%d same / %d different), anchors med %d, cohort med %d identities"
                      % (stream, L, cond, len(sel), y.sum(), len(y) - y.sum(),
                         int(np.median([len(r["_a"]) for r in sel])), int(np.median([len(r["_cid"]) for r in sel]))))
                print("   %-24s %6s %6s %8s" % ("route", "AUC", "Cllr", "Cllrmin"))
                ctx = context(sel)
                scored = [routes(r, ctx[r["known_author"]]) for r in sel]
                names = list(scored[0])
                vals = {nm: np.array([d[nm] for d in scored], float) for nm in names}
                for nm in names:
                    x = vals[nm]
                    ok = np.isfinite(x)
                    if ok.sum() < 0.8 * len(x) or len(set(y[ok])) < 2:
                        continue
                    auc = roc_auc_score(y[ok], x[ok])
                    c, cm = calibrated(x[ok], y[ok], g[ok])
                    print("   %-24s %6.3f %6.3f %8.3f" % (nm, auc, c, cm))
                    table.append((stream, L, cond, nm, int(y.sum()), int(len(y) - y.sum()), auc, c, cm))
                # the ratio built from the rank with nothing fitted on labels
                for r, v in zip(sel, vals["rank"]):
                    r["rank_stat"] = v
                if comp:
                    cp = np.array([comp.get((r["known"], r["kw"], r["quest"], r["qw"]), (np.nan, np.nan))[1]
                                   for r in sel], float)
                    ok = np.isfinite(cp)
                    if ok.all():
                        vals["composition"] = cp; names.append("composition")
                    if ok.sum() > 0.8 * len(cp):
                        for nm, feats in (
                                ("composition", [cp]),
                                ("fusion: two denominators", [vals["raw"], vals["marginal denominator"]]),
                                ("fusion: + in-case gauge", [vals["raw"], vals["marginal denominator"],
                                                             vals["donor z"], vals["rank"],
                                                             vals["anchor, matched hand"]]),
                                ("fusion: + composition", [vals["raw"], vals["marginal denominator"],
                                                           vals["donor z"], vals["rank"],
                                                           vals["anchor, matched hand"], cp])):
                            for mode in ("mean", "weighted"):
                                if len(feats) == 1 and mode == "weighted":
                                    continue
                                lr = fused_llr([f[ok] for f in feats], y[ok], g[ok], mode)
                                auc = roc_auc_score(y[ok], lr)
                                c = cllr(lr[y[ok] == 1], lr[y[ok] == 0])
                                cm = cllr_min(lr[y[ok] == 1], lr[y[ok] == 0])
                                label = nm if len(feats) == 1 else "%s [%s]" % (nm, mode)
                                print("   %-24s %6.3f %6.3f %8.3f" % (label, auc, c, cm))
                                table.append((stream, L, cond, label, int(y[ok].sum()),
                                              int(len(y[ok]) - y[ok].sum()), auc, c, cm))
                pool = [nm for nm in names if np.isfinite(vals[nm]).all()]
                if len(pool) > 1:
                    lr, picked = selected_route(vals, y, g, pool)
                    c, cm = cllr(lr[y == 1], lr[y == 0]), cllr_min(lr[y == 1], lr[y == 0])
                    print("   %-24s %6.3f %6.3f %8.3f   (chosen per fold: %s)"
                          % ("route chosen blind", roc_auc_score(y, lr), c, cm,
                             ", ".join(sorted(set(picked)))))
                    table.append((stream, L, cond, "route chosen blind", int(y.sum()),
                                  int(len(y) - y.sum()), roc_auc_score(y, lr), c, cm))
                for nm, key in (("rank ratio", "_arank"), ("rank ratio, matched hand", "_arank_m")):
                    lr = rank_lr(sel, y, g, key)
                    c, cm = cllr(lr[y == 1], lr[y == 0]), cllr_min(lr[y == 1], lr[y == 0])
                    auc = roc_auc_score(y, lr)
                    print("   %-24s %6.3f %6.3f %8.3f   (nothing fitted on labels)" % (nm, auc, c, cm))
                    table.append((stream, L, cond, nm, int(y.sum()), int(len(y) - y.sum()), auc, c, cm))
    if not table:
        print("no gauge scores in", SCORES); return
    TAB.parent.mkdir(parents=True, exist_ok=True)
    with TAB.open("w", encoding="utf-8") as f:
        f.write("% Autogenerated by experiments/analyze_witness_gauge.py -- do not edit.\n")
        f.write("\\begin{tabular}{llllrrrrr}\n\\toprule\n")
        f.write("stream & tokens a side & scribe & route & same & different & AUC & "
                "$C_{\\mathrm{llr}}$ & $C_{\\mathrm{llr}}^{\\min}$ \\\\\n\\midrule\n")
        for row in table:
            if row[3] not in HEADLINE:
                continue
            f.write("%s & %d & %s & %s & %d & %d & %.3f & %.3f & %.3f \\\\\n" % row)
        f.write("\\bottomrule\n\\end{tabular}\n")
    print("\n->", TAB)


if __name__ == "__main__":
    main()
