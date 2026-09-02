# Engine baseline table for the journal paper: KN (primary grid) vs HPY vs
# PPMd (static / adaptive-sentence) on 8 morphology-contrast novels datasets,
# symmetric L in {full, 1200, 600, 300, 150}. Metrics per cell: AUC of the raw
# lambda_G and Cllr / Cllr_min of the sqrt-corrected score.
#
#   python experiments/analyze_engines.py

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SCORES = HERE / "scores"
sys.path.insert(0, str(HERE.parent))
from lambdag import cllr, cllr_min  # noqa: E402

LANGS = ["czech", "english", "french", "hungarian", "latvian", "lithuanian",
         "polish", "ukrainian"]
ENGINES = ["kn", "hpy", "ppmd_static", "ppmd_sent"]
LENGTHS = [0, 1200, 600, 300, 150]


def metrics(ds, eng, L):
    fn = SCORES / f"{ds}__{eng}__sent__L{L}.jsonl"
    if not fn.exists():
        return None
    recs = [json.loads(l) for l in open(fn, encoding="utf-8")]
    if not recs:
        return None
    y = np.array([r["label"] for r in recs])
    if len(set(y.tolist())) < 2:
        return None
    from sklearn.metrics import roc_auc_score
    lam = np.array([r["lambda_G"] for r in recs])
    s = np.array([r["sqrt"] for r in recs])
    return (float(roc_auc_score(y, lam)),
            float(cllr(s[y == 1], s[y == 0])),
            float(cllr_min(s[y == 1], s[y == 0])))


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    agg = defaultdict(list)
    hdr = " ".join(f"{e:>22s}" for e in ENGINES)
    print(f"{'dataset':18s} {'L':>5s} {hdr}   (AUC / Cllr_sqrt / floor)")
    for lang in LANGS:
        ds = f"{lang}_novels"
        for L in LENGTHS:
            cells = []
            for e in ENGINES:
                m = metrics(ds, e, L)
                cells.append(f"{m[0]:.3f}/{m[1]:5.2f}/{m[2]:4.2f}" if m
                             else "         --        ")
                if m:
                    agg[(e, L)].append(m)
            print(f"{ds:18s} {L:5d} " + " ".join(f"{c:>22s}" for c in cells))
    print("\nmedians per engine x L (AUC / Cllr_sqrt / floor, n datasets):")
    for L in LENGTHS:
        row = f"  L={str(L) or 'full':>5s}: "
        for e in ENGINES:
            v = agg.get((e, L))
            if v:
                a = np.median([x[0] for x in v]); c = np.median([x[1] for x in v])
                f = np.median([x[2] for x in v])
                row += f"{e} {a:.3f}/{c:4.2f}/{f:4.2f} (n={len(v)})   "
        print(row)


if __name__ == "__main__":
    main()
