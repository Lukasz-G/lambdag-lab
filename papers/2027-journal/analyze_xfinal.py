# Final, protocol-clean cross-genre picture: for each direction's WINNING
# window (established by the w20/w100 impostor rerun), order-10 vs order-1
# ablation (arrangement vs rate evidence) on the symmetrised design, and the
# impostor design's raw vs standardised-t AUC at that same window.
#
#   python experiments/analyze_xfinal.py

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
XS = HERE.parent / "scores" / "xgenre"
from sklearn.metrics import roc_auc_score  # noqa: E402

WIN = {"novels2dracor": 20, "novels2poetree": 100, "dracor2novels": 100,
       "dracor2poetree": 100, "poetree2novels": 100, "poetree2dracor": 100}


def load(fn):
    if not fn.exists():
        return None
    return [json.loads(l) for l in open(fn, encoding="utf-8")]


def auc_lambda(rows):
    y = np.array([r["within"] for r in rows])
    lam = np.array([r["lambda_G"] for r in rows])
    return float(roc_auc_score(y, lam))


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    print("Protocol-clean summary (each direction at its winning window)\n")
    print(f"{'direction':16s} {'w':>4s} {'budget':>7s} {'AUC O10':>8s} "
          f"{'AUC O1':>7s} {'O10-O1':>7s}  {'imp raw':>8s} {'imp t':>6s}")
    for d, w in WIN.items():
        for budget, ktag in (("3k", ""), ("10k", "K10000")):
            f10 = XS / f"{d}__native__L1000{ktag}w{w}.jsonl"
            f1 = XS / f"{d}__native__L1000{ktag}O1w{w}.jsonl"
            r10, r1 = load(f10), load(f1)
            if r10 is None or r1 is None:
                continue
            k10 = {(r["known"], r["kw"], r["quest"], r["qw"]): r for r in r10}
            k1 = {(r["known"], r["kw"], r["quest"], r["qw"]): r for r in r1}
            ks = [k for k in k10 if k in k1]
            y = np.array([k10[k]["within"] for k in ks])
            l10 = np.array([k10[k]["lambda_G"] for k in ks])
            l1 = np.array([k1[k]["lambda_G"] for k in ks])
            a10, a1 = roc_auc_score(y, l10), roc_auc_score(y, l1)
            adiff = roc_auc_score(y, l10 - l1)
            imp_raw = imp_t = None
            if budget == "10k":
                rows = load(XS / f"{d}__impostor__K10000w{w}.jsonl")
                if rows:
                    yy = np.array([r["within"] for r in rows])
                    raw = np.array([r["lam_cand"] for r in rows])
                    t = np.array([(r["lam_cand"] - np.mean(r["lam_imp"]))
                                 / (np.std(r["lam_imp"]) + 1e-9)
                                 for r in rows])
                    imp_raw, imp_t = roc_auc_score(yy, raw), roc_auc_score(yy, t)
            print(f"{d:16s} w{w:<3d} {budget:>7s} {a10:8.3f} {a1:7.3f} "
                  f"{adiff:7.3f}  " +
                  (f"{imp_raw:8.3f} {imp_t:6.3f}" if imp_raw else ""))


if __name__ == "__main__":
    main()
