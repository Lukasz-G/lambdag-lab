# The XGBoost-vs-logistic ADJUDICATION of the TM's complementary signal.
#
# Question: the KN+TM fusion gain (~+0.015 AUC / -0.08 Cllr over KN alone) is not
# explained by simple shape statistics of the KN score. Two live hypotheses:
#   (a) INTERACTIONS -- conjunctions over rate agreements, which an additive
#       score cannot represent (the TM's clauses; any tree ensemble);
#   (c) SUPERVISED WEIGHTING -- a task-tuned linear metric over the same
#       agreements would do as well.
# Design: identical CONTINUOUS features (the pre-thermometer z products the TM
# consumes, exported by tm_scale.jl P4_EXPORTZ) into a linear model and into
# XGBoost; evaluate standalone and in CV fusion with the cached KN scores.
#   - linear recovers the gain            -> (c) weighting
#   - only XGBoost recovers it           -> (a) interactions
#   - neither reaches the TM             -> the TM's bias (thermometer bands +
#     distributed conjunctions) matters beyond generic nonlinearity.
#
#   python phase4/adjudicate_xgb.py

import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
EXP = HERE / "exportz"
sys.path.insert(0, str(HERE.parent))
from lambdag import cllr  # noqa: E402


def load_split(stem, n_expect=None):
    meta = [json.loads(l) for l in open(EXP / f"{stem}_meta.jsonl", encoding="utf-8")]
    m = json.loads(open(EXP / "meta.json", encoding="utf-8").read())
    X = np.fromfile(EXP / f"{stem}_z.f32", dtype=np.float32)
    X = X.reshape(len(meta), m["row_dim"])
    lens = np.array([[np.log(r["klen"] + 1), np.log(r["qlen"] + 1)] for r in meta],
                    dtype=np.float32)
    X = np.hstack([X, lens])
    y = np.array([r["label"] for r in meta])
    return X, y, meta


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold

    cache = EXP / "margins.npz"
    if cache.exists():
        d = np.load(cache)
        m_lin, m_xgb, yte = d["m_lin"], d["m_xgb"], d["yte"]
        tmeta = [json.loads(l) for l in
                 open(EXP / "test_meta.jsonl", encoding="utf-8")]
        print("margins loaded from cache")
    else:
        Xtr, ytr, _ = load_split("train")
        Xte, yte, tmeta = load_split("test")
        print(f"train {Xtr.shape}, test {Xte.shape} "
              f"({int(ytr.sum())}/{int((1-ytr).sum())} same/diff)")

        # ---- linear model (the weighting hypothesis) ----
        t0 = time.time()
        lin = LogisticRegression(solver="lbfgs", C=1.0, max_iter=1000)
        lin.fit(Xtr, ytr)
        m_lin = lin.decision_function(Xte)
        print(f"linear: {time.time()-t0:.0f}s  standalone AUC "
              f"{roc_auc_score(yte, m_lin):.4f}")

        # ---- XGBoost (the interaction hypothesis) ----
        import xgboost as xgb
        t0 = time.time()
        bst = xgb.XGBClassifier(tree_method="hist", n_estimators=700,
                                max_depth=7, learning_rate=0.08, subsample=0.8,
                                colsample_bytree=0.5, min_child_weight=5,
                                n_jobs=-1, eval_metric="auc")
        bst.fit(Xtr, ytr)
        m_xgb = bst.predict_proba(Xte)[:, 1]
        m_xgb = np.log(np.clip(m_xgb, 1e-6, 1 - 1e-6) /
                       (1 - np.clip(m_xgb, 1e-6, 1 - 1e-6)))
        print(f"xgboost: {time.time()-t0:.0f}s  standalone AUC "
              f"{roc_auc_score(yte, m_xgb):.4f}")
        np.savez(cache, m_lin=m_lin, m_xgb=m_xgb, yte=yte)
    print(f"standalone AUC: linear {roc_auc_score(yte, m_lin):.4f}  "
          f"xgboost {roc_auc_score(yte, m_xgb):.4f}")

    # ---- KN join + author-grouped CV fusion ----
    kn = {}
    for l in open(HERE / "kn500_L0.jsonl", encoding="utf-8"):
        r = json.loads(l)
        kn[r["id"]] = r["kn"]
    qa_of = {}
    p3 = HERE.parent / "phase3"
    for l in list(open(p3 / "pairs500.tsv", encoding="utf-8"))[1:]:
        p = l.rstrip("\n").split("\t")
        qa_of[int(p[0])] = p[3] if len(p) > 3 else p[0]

    ids = [r["id"] for r in tmeta]
    keep = [i for i, pid in enumerate(ids) if pid in kn]
    y = yte[keep]
    knv = np.array([kn[ids[i]] for i in keep])
    groups = np.array([qa_of.get(ids[i], str(ids[i])) for i in keep])
    feats = {"KN alone": knv.reshape(-1, 1),
             "KN + linear": np.column_stack([knv, m_lin[keep]]),
             "KN + xgboost": np.column_stack([knv, m_xgb[keep]]),
             "linear alone": m_lin[keep].reshape(-1, 1),
             "xgboost alone": m_xgb[keep].reshape(-1, 1)}
    print(f"\nfusion (author-grouped 5-fold CV, n={len(y)}):")
    for name, X in feats.items():
        margins = np.zeros(len(y))
        for tr, te in GroupKFold(5).split(X, y, groups):
            f = LogisticRegression(max_iter=1000).fit(X[tr], y[tr])
            prior = np.log(y[tr].mean() / (1 - y[tr].mean()))
            margins[te] = (X[te] @ f.coef_[0] + f.intercept_[0] - prior) / np.log(10)
        print(f"  {name:14s} AUC {roc_auc_score(y, margins):.4f}  "
              f"Cllr {cllr(margins[y == 1], margins[y == 0]):.3f}")
    print("\nanchors (same 500-pair set, earlier campaign): KN 0.935/0.502; "
          "TM standalone ~0.90; KN+TM fusion 0.950/0.42")


if __name__ == "__main__":
    main()
