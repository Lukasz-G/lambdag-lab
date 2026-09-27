# Scribe directions: is the residue that survives lemmatisation low-dimensional?
#
# The cross-genre chapter of the journal paper found genre to be a few
# directions of composition, removable by projection. The same question for
# the scribe: in the standardised symbol-rate space of the masked lemma
# stream, the shift of one witness of a work from the mean of that work's
# witnesses is pure scribal variation (same author, same text, another
# hand). The leading singular vectors of those shifts, fitted on works
# OTHER than the case's own, are the scribe directions; projecting them out
# of both passages of a case and scoring by cosine tests whether the
# different-scribe penalty is a few directions or spread over everything.
#
# The cases are the very ones the grammar model scored
# (experiments/run_witness_cells.py, same seeds), so the two scorers are
# compared pair by pair on identical passages.
#
#   python experiments/analyze_scribe_directions.py --L 2000 [--k 10] [--stream lemma]
#
# Writes medieval/tables/tab_scribe_directions.tex (appends one block per run).

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(HERE))
import run_witness_cells as W  # noqa: E402

TAB = ROOT / "medieval" / "tables" / "tab_scribe_directions.tex"
WIN, FLOOR = 1000, 200


def work_of(name):
    p = name.split("__")
    return W.norm(p[2] if len(p) > 3 else p[1])


def rates(tokens, idx):
    v = np.zeros(len(idx))
    for s, n in Counter(tokens).items():
        if s in idx:
            v[idx[s]] = n / len(tokens)
    return v


def composition_scores(stream, L, seg, k, rows, verbose=True):
    """cos and projected cos for every row of a scores file, on the passages the
    runners cut: the composition space is the standardised rate of every symbol
    of the stream over 1000-token windows, the scribe directions the leading
    singular vectors of the witness-minus-work shifts, fitted on the works other
    than the case's own. Returns {(known, kw, quest, qw): (cos, cos_proj)}."""
    banks = W.load(stream)
    seed = f"{stream}__L{L}__w{seg}"
    known_of = {n: dict(W.passages(b["first"], L, seg, 2, random.Random(seed + n + "k"))) for n, b in banks.items()}
    quest_of = {n: dict(W.passages(b["last"], L, seg, 3, random.Random(seed + n + "q"))) for n, b in banks.items()}

    # symbol space and standardisation over every witness's windows
    total = Counter()
    wins = {}
    for n, b in banks.items():
        toks = [t for s in b["sents"] for t in s]
        wins[n] = [toks[i:i + WIN] for i in range(0, len(toks) - WIN + 1, WIN)]
        total.update(toks)
    vocab = [s for s, c in total.most_common() if c >= FLOOR]
    idx = {s: i for i, s in enumerate(vocab)}
    allw = np.array([rates(w, idx) for ws in wins.values() for w in ws])
    mu, sd = allw.mean(0), allw.std(0) + 1e-12
    z = lambda toks: (rates(toks, idx) - mu) / sd
    wmean = {n: np.mean([z(w) for w in ws], axis=0) for n, ws in wins.items() if ws}

    # scribe shift vectors: witness mean minus its work's mean, works with >= 2 witnesses
    by_work = defaultdict(list)
    for n in wmean:
        by_work[banks[n]["work"]].append(n)
    shifts = {}
    for wk, ns in by_work.items():
        if len(ns) < 2:
            continue
        m = np.mean([wmean[n] for n in ns], axis=0)
        shifts[wk] = [wmean[n] - m for n in ns]
    if verbose:
        print("%d symbols, %d witnesses with windows, %d works with several witnesses (%d shift vectors)"
              % (len(vocab), len(wmean), len(shifts), sum(len(v) for v in shifts.values())))

    def directions(exclude_works):
        vecs = [v for wk, vs in shifts.items() if wk not in exclude_works for v in vs]
        if len(vecs) < k:
            return None
        _, _, vt = np.linalg.svd(np.array(vecs), full_matrices=False)
        return vt[:k]

    cache, out = {}, {}
    for r in rows:
        kn, qn = r["known"], r["quest"]
        ku = known_of[kn].get(r["kw"]); qu = quest_of[qn].get(r["qw"])
        if ku is None or qu is None:
            continue
        kz = z([t for u in ku for t in u]); qz = z([t for u in qu for t in u])
        ex = frozenset({banks[kn]["work"], banks[qn]["work"]})
        if ex not in cache:
            cache[ex] = directions(ex)
        B = cache[ex]
        cos = float(kz @ qz / (np.linalg.norm(kz) * np.linalg.norm(qz) + 1e-12))
        if B is not None:
            kp, qp = kz - (kz @ B.T) @ B, qz - (qz @ B.T) @ B
            cosp = float(kp @ qp / (np.linalg.norm(kp) * np.linalg.norm(qp) + 1e-12))
        else:
            cosp = cos
        out[(kn, r["kw"], qn, r["qw"])] = (cos, cosp)
    return out


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--stream", default="lemma")
    ap.add_argument("--L", type=int, default=2000)
    ap.add_argument("--seg", type=int, default=100)
    ap.add_argument("--k", type=int, default=10)
    args = ap.parse_args()
    L, seg, k = args.L, args.seg, args.k
    seed = f"{args.stream}__L{L}__w{seg}"
    rows = [json.loads(l) for l in (ROOT / "scores" / "witness" / f"{seed}.jsonl").open(encoding="utf-8")]
    comp = composition_scores(args.stream, L, seg, k, rows)
    scored = []
    for r in rows:
        v = comp.get((r["known"], r["kw"], r["quest"], r["qw"]))
        if v is not None:
            scored.append(dict(r, cos=v[0], cos_proj=v[1]))
    print("%d of %d cases rescored" % (len(scored), len(rows)))

    # fusion without fitting: each score standardised over the condition's
    # cases (no labels consulted) and the two summed -- a fitted logistic
    # under author-grouped folds does not transfer across so few identities
    for sc, dc in (("SA-SS", "DA-SS"), ("SA-DS", "DA-DS")):
        sel = [r for r in scored if r["cell"] in (sc, dc)]
        if not sel:
            continue
        a = np.array([r["lambda_G"] / L for r in sel]); b = np.array([r["cos_proj"] for r in sel])
        za = (a - a.mean()) / (a.std() + 1e-12); zb = (b - b.mean()) / (b.std() + 1e-12)
        for r, v in zip(sel, za + zb):
            r["fused"] = float(v)

    def fused(sel, y):
        return roc_auc_score(y, [r.get("fused", 0.0) for r in sel])

    lines = []
    print("%-28s %6s %6s %8s %8s" % ("condition", "same", "diff", "lambda_G", "cosine", ) + " %10s %8s" % ("cos, k=%d" % k, "fusion"))
    for cond, sc, dc, split in (("same scribe", "SA-SS", "DA-SS", None), ("different scribe", "SA-DS", "DA-DS", None),
                                ("  same work, other witness", "SA-DS", "DA-DS", "same"),
                                ("  other work, same author", "SA-DS", "DA-DS", "other")):
        sel = [r for r in scored if r["cell"] in (sc, dc)]
        if split == "same":
            sel = [r for r in sel if r["cell"] == dc or work_of(r["known"]) == work_of(r["quest"])]
        elif split == "other":
            sel = [r for r in sel if r["cell"] == dc or work_of(r["known"]) != work_of(r["quest"])]
        y = [1 if r["cell"] == sc else 0 for r in sel]
        if len(set(y)) < 2:
            continue
        a_l = roc_auc_score(y, [r["lambda_G"] for r in sel])
        a_c = roc_auc_score(y, [r["cos"] for r in sel])
        a_p = roc_auc_score(y, [r["cos_proj"] for r in sel])
        a_f = fused(sel, y)
        print("%-28s %6d %6d %8.3f %8.3f %10.3f %8.3f" % (cond, sum(y), len(y) - sum(y), a_l, a_c, a_p, a_f))
        lines.append((cond, sum(y), len(y) - sum(y), a_l, a_c, a_p, a_f))
    TAB.parent.mkdir(parents=True, exist_ok=True)
    with TAB.open("a", encoding="utf-8") as f:
        f.write("%% %s L=%d k=%d -- autogenerated by experiments/analyze_scribe_directions.py\n" % (args.stream, L, k))
        for cond, ns, nd, a_l, a_c, a_p, a_f in lines:
            f.write("%s & %d & %d & %d & %.3f & %.3f & %.3f & %.3f \\\\\n" % (cond.strip(), L, ns, nd, a_l, a_c, a_p, a_f))
    print("->", TAB)


if __name__ == "__main__":
    main()
