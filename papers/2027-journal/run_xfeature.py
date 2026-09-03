# Route 5 -- feature-stability weighting. Does a symbol's evidential
# contribution for an author replicate across genres, or is it a genre
# artefact? For every cross-genre (author, known-genre, questioned-genre)
# instance we fit the author's grammar on the known genre and, against one
# fixed reference, obtain per-symbol contributions on TWO probes: a
# held-out same-genre ("home") window and the actual cross-genre
# ("away") window. Pooling home/away pairs over authors (z-scored per
# author to remove scale), a symbol's cross-genre STABILITY is the
# leave-one-author-out correlation of its home value against its away
# value across the *other* authors -- a positive, replicated symbol is a
# candidate for a habitual construction that survives genre; nothing in
# this design lets it be a genre artefact of the held-out author itself.
# The payoff: re-score each held-out author's cross-genre verification case
# with tokens weighted by their (LOO-fitted) stability, against the plain
# unweighted sum.
#
#   python experiments/run_xfeature.py
#
# Output: scores/xgenre/feature_stability.json (per-symbol stability, LOO
# folds) and a printed AUC comparison, weighted vs unweighted, per direction.

import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from run_kn_grid import rechunk  # noqa: E402
from run_xgenre import GERMAN, load_bank, namekey, window  # noqa: E402

sys.path.insert(0, str(HERE.parent))
from lambdag import LambdaG  # noqa: E402

SCORES = HERE.parent / "scores" / "xgenre"
KT = 10000
HOMEK = 3000       # held-out same-genre probe size
QK = 3             # questioned-genre windows (as in run_xgenre SAME_W)
NCROSS = 4
L = 1000
SEG = 100          # protocol window unit (segmentation chapter default)
DIRECTIONS = [("novels", "dracor"), ("novels", "poetree"),
              ("dracor", "novels"), ("dracor", "poetree"),
              ("poetree", "novels"), ("poetree", "dracor")]


def build_instances():
    """One instance per (author, g1, g2) with enough text for both probes."""
    banks = {g: load_bank(GERMAN[g]) for g in ("novels", "dracor", "poetree")}
    keys = {g: {namekey(a): a for a in b} for g, b in banks.items()}
    insts = []
    for g1, g2 in DIRECTIONS:
        shared = sorted(k for k in keys[g1] if k in keys[g2])
        for k in shared:
            a1, a2 = keys[g1][k], keys[g2][k]
            b1, b2 = banks[g1][a1], banks[g2][a2]
            if b1["ntok"] < KT + HOMEK or b2["ntok"] < L:
                continue
            insts.append({"key": k, "g1": g1, "g2": g2, "a1": a1, "a2": a2,
                          "b1": b1, "b2": b2, "banks": banks})
    return insts, banks


def symbol_contribs(lg, query_sents, known_sents, ref_sents, seg):
    r = lg.score(rechunk(query_sents, seg), rechunk(known_sents, seg),
                ref_sentences=rechunk(ref_sents, seg), r=1,
                with_details=True)
    out = defaultdict(list)
    for toks, tls in zip(r.tokens, r.token_lambda):
        for t, v in zip(toks, tls):
            if t != "<EOS>":
                out[t].append(float(v))
    return {s: float(np.mean(v)) for s, v in out.items()}


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    insts, banks = build_instances()
    print(f"{len(insts)} (author, g1, g2) instances", flush=True)
    lg = LambdaG(N=10, r=1, engine="kn", random_state=0)
    t0 = time.time()

    # one fixed reference donor per genre (largest author not used as known
    # side anywhere in this experiment)
    used_known = {(i["g1"], i["a1"]) for i in insts}
    refdonor = {}
    for g, bank in banks.items():
        cand = sorted((a for a in bank if (g, a) not in used_known),
                     key=lambda a: -bank[a]["ntok"])
        refdonor[g] = cand[0]
    print("reference donors:", refdonor, flush=True)

    home, away, neg = {}, {}, {}   # inst_idx -> {symbol: contrib} / list of such
    for idx, inst in enumerate(insts):
        g1, g2, a1, a2 = inst["g1"], inst["g2"], inst["a1"], inst["a2"]
        b1, b2 = inst["b1"], inst["b2"]
        kdoc = window(b1["sents"], 0, KT)
        homedoc = window(b1["sents"], KT, KT + HOMEK)
        awaydoc = window(b2["sents"], 0, min(b2["ntok"], QK * L))
        ref1 = window(banks[g1][refdonor[g1]]["sents"], 0, KT)
        ref2 = window(banks[g2][refdonor[g2]]["sents"], 0, KT)
        home[idx] = symbol_contribs(lg, homedoc, kdoc, ref1, SEG)
        away[idx] = symbol_contribs(lg, awaydoc, kdoc, ref2, SEG)
        # genuine negatives: OTHER g2 authors' text scored under THIS
        # candidate's g1-fitted grammar -- a real impostor case, not a
        # relabelled positive from a different instance
        distract = sorted(a for a in banks[g2]
                          if a != a2 and banks[g2][a]["ntok"] >= L)
        import random as _r
        prng = _r.Random(f"{inst['key']}|{g1}|{g2}|neg")
        picks = prng.sample(distract, min(NCROSS, len(distract)))
        neg[idx] = []
        for b in picks:
            bdoc = window(banks[g2][b]["sents"], 0, min(
                banks[g2][b]["ntok"], QK * L))
            neg[idx].append(symbol_contribs(lg, bdoc, kdoc, ref2, SEG))
        if idx % 10 == 0:
            print(f"  probe {idx}/{len(insts)}, {time.time()-t0:.0f}s",
                  flush=True)

    # z-score per instance (author scale), pool by symbol across instances
    def zscore(d):
        v = np.array(list(d.values()))
        m, s = v.mean(), v.std() + 1e-9
        return {k: (x - m) / s for k, x in d.items()}
    homez = {i: zscore(d) for i, d in home.items()}
    awayz = {i: zscore(d) for i, d in away.items()}

    by_author = defaultdict(list)
    for i, inst in enumerate(insts):
        by_author[inst["key"]].append(i)

    def stability(symbol, exclude_author):
        hs, as_ = [], []
        for i, inst in enumerate(insts):
            if inst["key"] == exclude_author:
                continue
            if symbol in homez[i] and symbol in awayz[i]:
                hs.append(homez[i][symbol]); as_.append(awayz[i][symbol])
        if len(hs) < 4:
            return 0.0
        c = np.corrcoef(hs, as_)[0, 1]
        return 0.0 if np.isnan(c) else float(np.clip(c, 0, 1))

    all_symbols = sorted({s for d in away.values() for s in d})
    authors = sorted(by_author)
    stab = {a: {s: stability(s, a) for s in all_symbols} for a in authors}
    (SCORES / "feature_stability.json").write_text(
        json.dumps({"|".join(a): {s: round(v, 3) for s, v in d.items() if v > 0}
                    for a, d in stab.items()}, indent=1), encoding="utf-8")
    print("wrote feature_stability.json", flush=True)

    # payoff: weighted vs unweighted verification AUC per direction, using
    # the LOO-fitted (author held out of its own stability estimate) weights
    from sklearn.metrics import roc_auc_score
    for g1, g2 in DIRECTIONS:
        cases_y, cases_raw, cases_w = [], [], []
        pool = [i for i, inst in enumerate(insts)
               if inst["g1"] == g1 and inst["g2"] == g2]
        if len(pool) < 3:
            continue
        for i in pool:
            key = insts[i]["key"]
            w = stab[key]
            cases_y.append(1)
            cases_raw.append(sum(away[i].values()))
            cases_w.append(sum(v * w.get(s, 0.0) for s, v in away[i].items()))
            for negprof in neg[i]:
                cases_y.append(0)
                cases_raw.append(sum(negprof.values()))
                cases_w.append(sum(v * w.get(s, 0.0)
                                   for s, v in negprof.items()))
        y = np.array(cases_y)
        if len(set(y.tolist())) < 2:
            continue
        auc_raw = roc_auc_score(y, cases_raw)
        auc_w = roc_auc_score(y, cases_w)
        print(f"{g1}2{g2}: n={len(y):3d}  AUC raw {auc_raw:.3f}  "
              f"weighted {auc_w:.3f}", flush=True)


if __name__ == "__main__":
    main()
