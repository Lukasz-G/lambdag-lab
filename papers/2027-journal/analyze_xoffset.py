# Route 4 -- calibrate the cross-genre gap instead of removing it. Base
# statistic is the sqrt-corrected score (Barlow et al.'s ready-to-use log-LR,
# lambda_G / sqrt(N(Q)) -- the form the calibration chapter evaluates
# directly by Cllr, never raw lambda_G). For each direction, a
# leave-one-candidate-out per-token offset bhat_loo(a) is estimated from the
# OTHER cross-genre candidates' same-author rows and subtracted before the
# sqrt scaling, exactly the b-chapter's Lambda_G_hat = lambda_G - bhat*N(Q)
# move applied to the genre axis instead of the population axis. Payoff
# metric is raw Cllr (not Cllr_min, which a per-case-constant shift cannot
# move) -- does the direction-specific offset make the score closer to a
# usable LLR out of the box? AUC is reported too, since the per-candidate
# (not global) offset is not guaranteed rank-preserving.
#
#   python experiments/analyze_xoffset.py

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
XS = HERE.parent / "scores" / "xgenre"
sys.path.insert(0, str(HERE.parent))
from lambdag import cllr, cllr_min  # noqa: E402

from sklearn.metrics import roc_auc_score  # noqa: E402

WIN = {"novels2dracor": 20, "novels2poetree": 100, "dracor2novels": 100,
       "dracor2poetree": 100, "poetree2novels": 100, "poetree2dracor": 100}


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    print(f"{'direction':16s} {'n':>4s} {'Cllr raw':>9s} {'Cllr loo':>9s} "
          f"{'Cmin raw':>9s} {'AUC raw':>8s} {'AUC loo':>8s}")
    for d, w in WIN.items():
        fn = XS / f"{d}__native__L1000K10000w{w}.jsonl"
        if not fn.exists():
            print(f"{d}: missing {fn.name}"); continue
        rows = [json.loads(l) for l in open(fn, encoding="utf-8")]
        known = sorted({r["known"] for r in rows})
        # per-candidate mean per-token lambda on SAME-author (within=1) rows
        bhat = {}
        for a in known:
            v = [r["lambda_G"] / r["n_q"] for r in rows
                 if r["known"] == a and r["within"] == 1]
            bhat[a] = float(np.mean(v)) if v else 0.0
        if len(known) < 3:
            print(f"{d}: only {len(known)} candidates, offset LOO skipped")
            continue
        y = np.array([r["within"] for r in rows])
        raw = np.array([r["lambda_G"] / np.sqrt(r["n_q"]) for r in rows])
        loo = np.array([
            (r["lambda_G"] - r["n_q"] *
             float(np.mean([bhat[b] for b in known if b != r["known"]])))
            / np.sqrt(r["n_q"])
            for r in rows])
        c_raw = cllr(raw[y == 1], raw[y == 0])
        c_loo = cllr(loo[y == 1], loo[y == 0])
        c_min = cllr_min(raw[y == 1], raw[y == 0])
        a_raw = roc_auc_score(y, raw)
        a_loo = roc_auc_score(y, loo)
        print(f"{d:16s} {len(y):4d} {c_raw:9.3f} {c_loo:9.3f} {c_min:9.3f} "
              f"{a_raw:8.3f} {a_loo:8.3f}")


if __name__ == "__main__":
    main()
