"""Recompute the paper's median full-length AUC from the deposited score files alone."""
import glob
import json
import os
import statistics
import sys

SRC = sys.argv[1]  # directory holding *__kn__sent__L0.jsonl


def auc(scores, labels):
    pairs = sorted(zip(scores, labels))
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    if not n_pos or not n_neg:
        return None
    # rank-based (Mann-Whitney), averaging ties
    ranks = [0.0] * len(pairs)
    i = 0
    while i < len(pairs):
        j = i
        while j + 1 < len(pairs) and pairs[j + 1][0] == pairs[i][0]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[k] = avg
        i = j + 1
    s = sum(r for r, (_, lab) in zip(ranks, pairs) if lab == 1)
    return (s - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


out = {}
for path in sorted(glob.glob(os.path.join(SRC, "*__kn__sent__L0.jsonl"))):
    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    if not rows:
        continue
    labels = [int(r["label"]) for r in rows]
    name = os.path.basename(path).split("__")[0]
    out[name] = {
        "n": len(rows),
        "auc_lambda": auc([r["lambda_G"] for r in rows], labels),
        "auc_sqrt": auc([r["sqrt"] for r in rows], labels),
    }

for key in ("auc_lambda", "auc_sqrt"):
    vals = [v[key] for v in out.values() if v[key] is not None]
    print("%s : %d datasets, median %.4f" % (key, len(vals), statistics.median(vals)))

# the paper analyses only datasets with >= 10 reference authors (38 of 44)
vals = sorted((v["auc_sqrt"], k) for k, v in out.items() if v["auc_sqrt"] is not None)
print("\nlowest 3:", [(k, round(a, 3)) for a, k in vals[:3]])
print("highest 3:", [(k, round(a, 3)) for a, k in vals[-3:]])
print("total score files found:", len(out))
