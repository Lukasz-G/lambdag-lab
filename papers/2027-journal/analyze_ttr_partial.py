# Defence against the "entrenchment or just model sharpness?" objection:
# does the within-K -> H_d location correlation survive partialling out crude
# text-repetitiveness properties of the author's material (type-token ratio
# and hapax rate over a fixed-size masked sample)? Partial Spearman = Pearson
# on rank-transformed variables with confound ranks regressed out.
#
#   python experiments/analyze_ttr_partial.py

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats as sps

HERE = Path(__file__).resolve().parent
SCORES = HERE / "scores"
MASKED = HERE.parent / "masked"
FIX = 4000     # fixed-size token sample per author for TTR/hapax comparability


def read_tokens(f, n):
    toks = []
    for line in open(f, encoding="utf-8"):
        line = line.rstrip("\n")
        if line:
            toks += line.split("\t")
            if len(toks) >= n:
                break
    return toks[:n]


def confounds(ds):
    out = {}
    for f in sorted((MASKED / ds / "bank").glob("*.tsv")):
        t = read_tokens(f, FIX)
        if len(t) < FIX:
            continue
        vals, counts = np.unique(t, return_counts=True)
        out[f.stem] = (len(vals) / FIX, float((counts == 1).sum()) / FIX)
    return out


def author_stats(fn):
    within, cross = defaultdict(list), defaultdict(list)
    for line in open(fn, encoding="utf-8"):
        r = json.loads(line)
        (within if r["within"] else cross)[r["known"]].append(
            r["lambda_G"] / r["n_q"])
    return {a: (float(np.mean(within[a])), float(np.mean(cross[a])))
            for a in within if len(within[a]) >= 6 and len(cross.get(a, [])) >= 6}


def partial_spearman(x, y, Z):
    rx = sps.rankdata(x); ry = sps.rankdata(y)
    RZ = np.column_stack([sps.rankdata(z) for z in Z] + [np.ones(len(x))])
    bx, *_ = np.linalg.lstsq(RZ, rx, rcond=None)
    by, *_ = np.linalg.lstsq(RZ, ry, rcond=None)
    return float(sps.pearsonr(rx - RZ @ bx, ry - RZ @ by).statistic)


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    for L in (2000, 5000):
        raw_rs, part_rs = [], []
        for fn in sorted(SCORES.glob(f"*__kn__withink__L{L}.jsonl")):
            ds = fn.name.split("__")[0]
            st = author_stats(fn)
            cf = confounds(ds)
            common = [a for a in st if a in cf]
            if len(common) < 10:
                continue
            wm = np.array([st[a][0] for a in common])
            cm = np.array([st[a][1] for a in common])
            ttr = np.array([cf[a][0] for a in common])
            hap = np.array([cf[a][1] for a in common])
            raw_rs.append(float(sps.spearmanr(wm, cm).statistic))
            part_rs.append(partial_spearman(wm, cm, [ttr, hap]))
        print(f"L={L}: n_datasets={len(raw_rs)}  "
              f"raw median rho {np.median(raw_rs):+.2f} "
              f"({sum(1 for r in raw_rs if r < 0)}/{len(raw_rs)} neg)  "
              f"partial (TTR+hapax) median {np.median(part_rs):+.2f} "
              f"({sum(1 for r in part_rs if r < 0)}/{len(part_rs)} neg)")


if __name__ == "__main__":
    main()
