# Cross-genre, cross-language pairwise verification pilot (journal paper,
# cross-lingual/cross-genre section): can a supervised verifier LEARN which
# agreement dimensions survive a genre change, and does that skill transfer to
# a language it has never seen?
#
# Pairs: known-genre window vs questioned-genre window of the SAME language;
# positives = the same cross-genre author on both sides, negatives = two
# different authors, same genre pair -- so neither language nor genre separates
# the labels and the learner must find agreement-beyond-genre. Windows are
# n-gram profiles over the language's SHARED class-rank symbol map
# (masked_catsrank/{ds}_shared), centred per (language, genre) and
# unit-normalised; a pair is the elementwise product of its two profiles.
#
# Protocols:
#   LOLO  leave-one-language-out over the six languages with cross-genre
#         authors (de 16, fr 7, cs 6, it 6, en 3, hu 3)
#   LOGO  leave-one-genre-pair-out (train on the other genre pairs, all
#         languages pooled)
#
#   python experiments/xgenre_pair_pilot.py
#
# The gradient-boosted learner stands in for the Tsetlin machine, as in the
# adjudication experiments.

import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
MASKED = HERE.parent / "masked_catsrank"

GENRES = {"de": ["novels", "dracor", "poetree"],
          "en": ["novels", "dracor", "poetree"],
          "fr": ["novels", "dracor", "poetree"],
          "cs": ["novels", "poetree"],
          "hu": ["novels", "dracor", "poetree"],
          "it": ["novels", "dracor", "poetree"]}
LANGNAME = {"de": "german", "en": "english", "fr": "french",
            "cs": "czech", "hu": "hungarian", "it": "italian"}
EXCLUDE = {("en", ("john", "wilson"))}   # name-collision risk
import os

W = int(os.environ.get("XGP_W", 1000))
MAXWIN = int(os.environ.get("XGP_MAXWIN", 6))
POS_PER_AUTHOR = int(os.environ.get("XGP_POS", 6))
NEG_X = 2          # negatives per positive
MINTOK = 2 * W     # a side needs >= 2 windows' worth to enter


def namekey(stem):
    return tuple(sorted(p for p in re.sub(r"^\d+_", "", stem).split("_") if p))


def read_tsv(f):
    return [line.split("\t") for line in
            f.read_text(encoding="utf-8").splitlines() if line]


def load_genre(lang, genre):
    ds = f"{LANGNAME[lang]}_{genre}_shared"
    bank = {}
    for f in sorted((MASKED / ds / "bank").glob("*.tsv")):
        sents = read_tsv(f)
        toks = [t for s in sents for t in s]
        if len(toks) >= MINTOK:
            bank[f.stem] = toks
    return bank


def windows(toks):
    m = min(len(toks) // W, MAXWIN)
    return [toks[k * W:(k + 1) * W] for k in range(m)]


def ngram_counts(sym):
    c = Counter()
    for i in range(len(sym)):
        c[sym[i]] += 1
        if i + 1 < len(sym):
            c[(sym[i], sym[i + 1])] += 1
        if i + 2 < len(sym):
            c[(sym[i], sym[i + 1], sym[i + 2])] += 1
    return c


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    from sklearn.metrics import roc_auc_score
    import xgboost as xgb

    # windows per (lang, genre, author); n-gram counts
    wins, keys = {}, []
    for lang, genres in GENRES.items():
        for g in genres:
            for a, toks in load_genre(lang, g).items():
                ws = [ngram_counts(w) for w in windows(toks)]
                if ws:
                    wins[(lang, g, a)] = ws
    # vocabulary over everything
    tot = Counter()
    for ws in wins.values():
        for c in ws:
            tot.update(c)
    vocab = [g for g, n in tot.most_common(20000) if n >= 10]
    idx = {g: i for i, g in enumerate(vocab)}
    print(f"{len(wins)} (lang,genre,author) streams, vocab {len(vocab)}")

    # centred unit-norm profiles, centring per (lang, genre)
    feats = {}
    for lang, genres in GENRES.items():
        for g in genres:
            rows, rk = [], []
            for (l2, g2, a), ws in wins.items():
                if (l2, g2) != (lang, g):
                    continue
                for j, c in enumerate(ws):
                    v = np.zeros(len(vocab), dtype=np.float32)
                    for gr, n in c.items():
                        if gr in idx:
                            v[idx[gr]] = n
                    v /= max(v.sum(), 1)
                    rows.append(v); rk.append((l2, g2, a, j))
            if not rows:
                continue
            M = np.vstack(rows)
            M -= M.mean(axis=0, keepdims=True)
            M /= (np.linalg.norm(M, axis=1, keepdims=True) + 1e-9)
            for k, key in enumerate(rk):
                feats[key] = M[k]

    # cross-genre authors per language, genre pair
    pairs = []   # (lang, gpair, x_feature, label)
    for lang, genres in GENRES.items():
        gsets = {}
        for (l2, g, a) in wins:
            if l2 == lang:
                gsets.setdefault(namekey(a), {})[g] = a
        xga = {k: d for k, d in gsets.items()
               if len(d) >= 2 and (lang, k) not in EXCLUDE}
        for g1 in genres:
            for g2 in genres:
                if g1 >= g2:
                    continue
                gp = f"{g1}-{g2}"
                have = [k for k, d in xga.items() if g1 in d and g2 in d]
                if len(have) < 2:
                    continue
                for k in have:
                    a1, a2 = xga[k][g1], xga[k][g2]
                    rng = random.Random(f"{lang}|{gp}|{k}")
                    cand = [(i, j) for i in range(len(wins[(lang, g1, a1)]))
                            for j in range(len(wins[(lang, g2, a2)]))]
                    rng.shuffle(cand)
                    for i, j in cand[:POS_PER_AUTHOR]:
                        x = feats[(lang, g1, a1, i)] * feats[(lang, g2, a2, j)]
                        pairs.append((lang, gp, x, 1))
                        # negatives: same genre pair, different author on side 2
                        others = [(l2, g, a) for (l2, g, a) in wins
                                  if l2 == lang and g == g2
                                  and namekey(a) != k]
                        for _ in range(NEG_X):
                            l2, g, b = rng.choice(others)
                            jb = rng.randrange(len(wins[(lang, g2, b)]))
                            xn = feats[(lang, g1, a1, i)] * feats[(lang, g2, b, jb)]
                            pairs.append((lang, gp, xn, 0))
    langs = sorted({p[0] for p in pairs})
    gps = sorted({p[1] for p in pairs})
    cnt = Counter((p[0], p[3]) for p in pairs)
    print("pairs per language (pos/neg):",
          {l: (cnt[(l, 1)], cnt[(l, 0)]) for l in langs})

    def fit_eval(train, test):
        Xtr = np.vstack([p[2] for p in train]); ytr = [p[3] for p in train]
        Xte = np.vstack([p[2] for p in test]); yte = [p[3] for p in test]
        bst = xgb.XGBClassifier(tree_method="hist", n_estimators=400,
                                max_depth=6, learning_rate=0.1,
                                subsample=0.8, colsample_bytree=0.5,
                                n_jobs=-1).fit(Xtr, ytr)
        return roc_auc_score(yte, bst.predict_proba(Xte)[:, 1])

    print("\nLOLO (held-out language):")
    for H in langs:
        tr = [p for p in pairs if p[0] != H]
        te = [p for p in pairs if p[0] == H]
        if sum(p[3] for p in te) < 6:
            print(f"  {H}: too few positives, skipped"); continue
        print(f"  {H:3s} n={len(te):4d}  AUC {fit_eval(tr, te):.3f}",
              flush=True)

    print("\nLOGO (held-out genre pair, languages pooled):")
    for G in gps:
        tr = [p for p in pairs if p[1] != G]
        te = [p for p in pairs if p[1] == G]
        if sum(p[3] for p in te) < 6:
            print(f"  {G}: too few positives, skipped"); continue
        print(f"  {G:16s} n={len(te):4d}  AUC {fit_eval(tr, te):.3f}",
              flush=True)


if __name__ == "__main__":
    main()
