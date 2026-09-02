# Within-K statistics as a FEATURE (not a calibrator): does adding the known
# author's case-internal moments (within_mean, within_sd) to lambda/sqrt(N)
# improve discrimination on the ill-calibrated grid cells? 5-fold CV logistic,
# grouped by known author would be ideal; StratifiedKFold used with the caveat
# that authors repeat across folds (same-direction bias for base and extended
# model, so the DELTA is the meaningful number).
#
#   python experiments/analyze_wkfeature.py

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SCORES = HERE / "scores"
MASKED = HERE.parent / "masked"

DS = ["romanian_dracor", "english_novels", "italian_dracor", "german_novels",
      "slovenian_novels", "french_novels", "spanish_novels", "hungarian_dracor",
      "italian_novels", "latvian_novels", "swedish_novels", "polish_novels"]


def test_stats(fn):
    within = defaultdict(list)
    for line in open(fn, encoding="utf-8"):
        r = json.loads(line)
        if r["within"]:
            within[r["known"]].append(r["lambda_G"] / r["n_q"])
    return {a: (float(np.mean(v)), float(np.std(v, ddof=1)))
            for a, v in within.items() if len(v) >= 6}


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold
    print(f"{'dataset':20s} {'L':>5s} {'n':>5s}  {'AUC base':>8s} {'+wm':>7s} "
          f"{'+wm+wsd':>8s}  {'dAUC':>7s}")
    deltas = []
    for L in (1200, 600):
        for ds in DS:
            fT = SCORES / f"{ds}__kn__withinkT__L{L}.jsonl"
            fC = SCORES / f"{ds}__kn__sent__L{L}.jsonl"
            if not (fT.exists() and fC.exists()):
                continue
            st = test_stats(fT)
            man = [l.rstrip("\n").split("\t")
                   for l in open(MASKED / ds / "pairs.tsv", encoding="utf-8")][1:]
            ka_of = {int(p): ka for p, _l, ka, _q in man}
            X, y = [], []
            for r in (json.loads(l) for l in open(fC, encoding="utf-8")):
                a = ka_of[r["id"]]
                if a in st:
                    wm, wsd = st[a]
                    X.append([r["lambda_G"] / np.sqrt(max(r["n_q"], 1)), wm, wsd])
                    y.append(r["label"])
            X, y = np.asarray(X), np.asarray(y)
            if len(set(y.tolist())) < 2 or len(y) < 80:
                continue
            aucs = {k: [] for k in (1, 2, 3)}
            for tr, te in StratifiedKFold(5, shuffle=True,
                                          random_state=0).split(X, y):
                for k in (1, 2, 3):
                    m = LogisticRegression(max_iter=1000).fit(X[tr, :k], y[tr])
                    aucs[k].append(roc_auc_score(
                        y[te], m.decision_function(X[te, :k])))
            a1, a2, a3 = (float(np.mean(aucs[k])) for k in (1, 2, 3))
            deltas.append(a3 - a1)
            print(f"{ds:20s} {L:5d} {len(y):5d}  {a1:8.3f} {a2:7.3f} "
                  f"{a3:8.3f}  {a3-a1:+7.4f}")
    print(f"\nmedian dAUC (+wm+wsd over base): {np.median(deltas):+.4f}   "
          f"positive in {sum(1 for d in deltas if d > 0)}/{len(deltas)} cells")


if __name__ == "__main__":
    main()
