# Cross-lingual verification plausibility pilot (journal paper, TM chapter:
# "Beyond one language"). Question: does grammatical-habit structure transfer
# across languages when every language is reduced to a shared alphabet?
#
# Two universal representations of the masked stream:
#   tags   placeholders (#, section-sign, O-slash, @, (c), mu, $, yen) kept as-is;
#          punctuation collapsed to a small class set; every kept function word
#          collapsed to a single symbol W.  (pure grammar-tag stream)
#   ranks  as tags, but function words mapped to frequency-rank buckets within
#          their own language (R1, R2, ..., R10, R20, R50, R100, RX) --
#          Zipfian alignment, computable from the banks alone.
#   cats   as tags, but function words mapped to their functional class (AUX,
#          LVERB, ADV, DET, ADP, PRON, CCONJ, SCONJ, PART, INTJ, NUM) from the
#          category-annotated companion lists (posnoise_lists/aligned/);
#          unlisted or classless tokens fall back to W, so cats is a strict
#          refinement of tags.
#
# Protocol: leave-one-language-out over six novel corpora. Per language, bank
# authors are cut into 1000-token windows; same-author and (within-language)
# different-author window pairs are formed; features are centred n-gram-profile
# products (uni+bi+tri over the universal alphabet, per-language centring,
# fitted on training languages); learners: logistic (weighting control) and
# XGBoost (the interaction learner that stood in for the TM in the
# adjudication). Within-language 5-fold author-grouped CV gives the ceiling.
#
#   python experiments/xling_pilot.py [tags,ranks,cats]

import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from aligned_utils import LANG_CODE, class_of, load_aligned  # noqa: E402
MASKED = HERE.parent / "masked"

LANGS = ["german", "english", "french", "polish", "czech", "hungarian"]
W = 1000          # window tokens
MAXWIN = 6        # windows per author
MAX_AUTHORS = 30
PAIRS_PER_AUTHOR = 6   # same pairs (diff matched 1:1)
PLACEHOLDERS = set("#§Ø@©µ$¥")
PUNCT_KEEP = set(".,;:!?()—–-'\"»«")
RANK_EDGES = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 50, 100]


def read_tokens(f):
    toks = []
    for line in open(f, encoding="utf-8"):
        line = line.rstrip("\n")
        if line:
            toks += line.split("\t")
    return toks


def load_lang(lang):
    bank = {}
    for f in sorted((MASKED / f"{lang}_novels" / "bank").glob("*.tsv")):
        t = read_tokens(f)
        if len(t) >= 3 * W:
            bank[f.stem] = t
    names = sorted(bank)
    if len(names) > MAX_AUTHORS:
        names = sorted(random.Random(lang).sample(names, MAX_AUTHORS))
    return {n: bank[n] for n in names}


def rank_map(bank):
    c = Counter(t for toks in bank.values() for t in toks
                if t not in PLACEHOLDERS and any(ch.isalpha() for ch in t))
    ranks = {}
    for i, (tok, _) in enumerate(c.most_common(), start=1):
        b = next((f"R{e}" for e in RANK_EDGES if i <= e), "RX")
        ranks[tok] = b
    return ranks


def encode(toks, mode, ranks, table=None):
    out = []
    for t in toks:
        if t in PLACEHOLDERS:
            out.append(t)
        elif any(ch.isalpha() for ch in t):
            if mode == "tags":
                out.append("W")
            elif mode == "ranks":
                out.append(ranks.get(t, "RX"))
            else:  # cats
                out.append(class_of(t, table))
        else:
            out.append(t if t in PUNCT_KEEP else "P")
    return out


def ngram_counts(sym):
    c = Counter()
    for i in range(len(sym)):
        c[sym[i]] += 1
        if i + 1 < len(sym):
            c[(sym[i], sym[i + 1])] += 1
        if i + 2 < len(sym):
            c[(sym[i], sym[i + 1], sym[i + 2])] += 1
    return c


def build_lang(lang, mode):
    bank = load_lang(lang)
    ranks = rank_map(bank) if mode == "ranks" else {}
    table = load_aligned(LANG_CODE[lang]) if mode == "cats" else None
    if mode == "cats" and not table:
        raise SystemExit(f"no aligned companion file for {lang}; run "
                         f"data_prep/build_aligned_lists.py first")
    wins, n_alpha, n_class = {}, 0, 0
    for a, toks in bank.items():
        sym = encode(toks, mode, ranks, table)
        if mode == "cats":
            for s, t in zip(sym, toks):
                if any(ch.isalpha() for ch in t):
                    n_alpha += 1
                    n_class += s != "W"
        m = min(len(sym) // W, MAXWIN)
        wins[a] = [ngram_counts(sym[k * W:(k + 1) * W]) for k in range(m)]
    if mode == "cats":
        print(f"  [{lang}: {n_class / max(n_alpha, 1):.1%} of kept words "
              f"carry a class]", flush=True)
    return wins


def featurise(wins_by_lang, vocab):
    """centred, unit-norm profile per window; centring per language."""
    idx = {g: i for i, g in enumerate(vocab)}
    out = {}
    for lang, wins in wins_by_lang.items():
        mats = {}
        rows, keys = [], []
        for a, ws in wins.items():
            for j, c in enumerate(ws):
                v = np.zeros(len(vocab), dtype=np.float32)
                for g, n in c.items():
                    if g in idx:
                        v[idx[g]] = n
                v /= max(v.sum(), 1)
                rows.append(v); keys.append((a, j))
        M = np.vstack(rows)
        M -= M.mean(axis=0, keepdims=True)
        M /= (np.linalg.norm(M, axis=1, keepdims=True) + 1e-9)
        for k, key in enumerate(keys):
            mats[key] = M[k]
        out[lang] = mats
    return out


def make_pairs(wins, lang, seed=0):
    rng = random.Random(f"{lang}|{seed}")
    same, diff = [], []
    authors = sorted(wins)
    for a in authors:
        m = len(wins[a])
        cand = [(i, j) for i in range(m) for j in range(i + 1, m)]
        rng.shuffle(cand)
        for i, j in cand[:PAIRS_PER_AUTHOR]:
            same.append((a, i, a, j))
    while len(diff) < len(same):
        a, b = rng.sample(authors, 2)
        diff.append((a, rng.randrange(len(wins[a])),
                     b, rng.randrange(len(wins[b]))))
    return same, diff


def pair_X(pairs, feats):
    return np.vstack([feats[(a, i)] * feats[(b, j)] for a, i, b, j in pairs])


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold
    import xgboost as xgb

    modes = (sys.argv[1].split(",") if len(sys.argv) > 1
             else ["tags", "ranks", "cats"])
    for mode in modes:
        wins_by_lang = {l: build_lang(l, mode) for l in LANGS}
        counts = Counter()
        for wins in wins_by_lang.values():
            for ws in wins.values():
                for c in ws:
                    counts.update(c)
        vocab = [g for g, n in counts.most_common(20000) if n >= 10]
        feats = featurise(wins_by_lang, vocab)
        pairs = {l: make_pairs(wins_by_lang[l], l) for l in LANGS}
        print(f"\n=== mode={mode}  vocab={len(vocab)}  "
              f"pairs/lang ~{len(pairs[LANGS[0]][0]) * 2} ===")
        print(f"{'held-out':10s} {'n':>5s} {'LOLO log':>9s} {'LOLO xgb':>9s} "
              f"{'within xgb':>10s}")
        for H in LANGS:
            Xtr = np.vstack([pair_X(pairs[l][0] + pairs[l][1], feats[l])
                             for l in LANGS if l != H])
            ytr = np.concatenate([[1] * len(pairs[l][0]) + [0] * len(pairs[l][1])
                                  for l in LANGS if l != H])
            Xte = pair_X(pairs[H][0] + pairs[H][1], feats[H])
            yte = np.array([1] * len(pairs[H][0]) + [0] * len(pairs[H][1]))
            lin = LogisticRegression(max_iter=2000).fit(Xtr, ytr)
            a_lin = roc_auc_score(yte, lin.decision_function(Xte))
            bst = xgb.XGBClassifier(tree_method="hist", n_estimators=400,
                                    max_depth=6, learning_rate=0.1,
                                    subsample=0.8, colsample_bytree=0.5,
                                    n_jobs=-1).fit(Xtr, ytr)
            a_xgb = roc_auc_score(yte, bst.predict_proba(Xte)[:, 1])
            # within-language ceiling: author-grouped CV on H alone
            allp = pairs[H][0] + pairs[H][1]
            groups = np.array([p[0] for p in allp])
            m = np.zeros(len(yte))
            for tr, te in GroupKFold(4).split(Xte, yte, groups):
                b2 = xgb.XGBClassifier(tree_method="hist", n_estimators=300,
                                       max_depth=6, learning_rate=0.1,
                                       n_jobs=-1).fit(Xte[tr], yte[tr])
                m[te] = b2.predict_proba(Xte[te])[:, 1]
            a_within = roc_auc_score(yte, m)
            print(f"{H:10s} {len(yte):5d} {a_lin:9.3f} {a_xgb:9.3f} "
                  f"{a_within:10.3f}", flush=True)


if __name__ == "__main__":
    main()
