# XGBoost tuning for the adjudication, tuned-vs-tuned fairness pass.
# Grid with author-grouped early stopping (val = 15% of TRAIN AUTHORS held out),
# plus the scientifically pointed depth-2 probe (pairwise interactions only).
# Writes every config's test margins to exportz/xgb_tune/<name>.npz and a
# summary tsv; fusion analysis happens locally afterwards.
#
#   python phase4/xgb_tune.py [--threads N]

import argparse
import json
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
EXP = HERE / "exportz"
OUT = EXP / "xgb_tune"
OUT.mkdir(exist_ok=True)


def load(stem):
    meta = [json.loads(l) for l in open(EXP / f"{stem}_meta.jsonl", encoding="utf-8")]
    m = json.loads(open(EXP / "meta.json", encoding="utf-8").read())
    X = np.fromfile(EXP / f"{stem}_z.f32", dtype=np.float32).reshape(
        len(meta), m["row_dim"])
    lens = np.array([[np.log(r["klen"] + 1), np.log(r["qlen"] + 1)] for r in meta],
                    dtype=np.float32)
    return np.hstack([X, lens]), np.array([r["label"] for r in meta]), meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=0)
    args = ap.parse_args()
    import xgboost as xgb
    from sklearn.metrics import roc_auc_score

    Xtr, ytr, tmeta = load("train")
    Xte, yte, _ = load("test")
    # author-grouped validation split (both pair authors held out together)
    authors = sorted({r["ka"] for r in tmeta} | {r["qa"] for r in tmeta})
    rng = np.random.default_rng(0)
    val_auth = set(rng.choice(authors, size=max(2, len(authors) // 7),
                              replace=False).tolist())
    is_val = np.array([(r["ka"] in val_auth) or (r["qa"] in val_auth)
                       for r in tmeta])
    print(f"train {int((~is_val).sum())} / val {int(is_val.sum())} pairs "
          f"({len(val_auth)}/{len(authors)} authors held out)")

    grid = [
        ("d2_probe", dict(max_depth=2, learning_rate=0.08)),
        ("d3", dict(max_depth=3, learning_rate=0.08)),
        ("d5", dict(max_depth=5, learning_rate=0.08)),
        ("d7_base", dict(max_depth=7, learning_rate=0.08)),
        ("d9", dict(max_depth=9, learning_rate=0.08)),
        ("d7_slow", dict(max_depth=7, learning_rate=0.03)),
        ("d7_mcw20", dict(max_depth=7, learning_rate=0.08, min_child_weight=20)),
        ("d7_col3", dict(max_depth=7, learning_rate=0.08, colsample_bytree=0.3)),
    ]
    rows = []
    for name, kw in grid:
        p = dict(tree_method="hist", n_estimators=3000, subsample=0.8,
                 colsample_bytree=0.5, min_child_weight=5,
                 n_jobs=args.threads or -1, eval_metric="auc",
                 early_stopping_rounds=60)
        p.update(kw)
        t0 = time.time()
        bst = xgb.XGBClassifier(**p)
        bst.fit(Xtr[~is_val], ytr[~is_val],
                eval_set=[(Xtr[is_val], ytr[is_val])], verbose=False)
        pr = bst.predict_proba(Xte)[:, 1]
        m = np.log(np.clip(pr, 1e-6, 1 - 1e-6) / (1 - np.clip(pr, 1e-6, 1 - 1e-6)))
        auc = roc_auc_score(yte, m)
        best_it = getattr(bst, "best_iteration", None)
        np.savez(OUT / f"{name}.npz", m=m, yte=yte)
        rows.append((name, auc, best_it, time.time() - t0))
        print(f"{name:10s} test AUC {auc:.4f}  best_iter {best_it}  "
              f"{time.time()-t0:.0f}s", flush=True)
    with open(OUT / "summary.tsv", "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write("\t".join(str(x) for x in r) + "\n")
    print("DONE_XGB")


if __name__ == "__main__":
    main()
