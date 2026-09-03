# Genre-absent verification: the questioned genre exists in NO corpus of the
# target language, so the reference population must be borrowed -- from a
# foreign language's questioned-genre corpora, or from the target language's
# wrong-genre corpus.
#
# HARD CONSTRAINT: no target-language questioned-genre text may influence any
# FITTED quantity. It appears only as questioned material. The class-rank map
# is fitted on the target language's other genres alone (the *_nopoe banks
# from encode_catsrank.py --fit-on) and applied unchanged to the questioned
# genre -- fit on train, apply to test.
#
# ARCHITECTURE. The reference population must be the thing the arms vary, so
# this uses the SYMMETRISED estimator (per-author donor grammars, r donors per
# case, the arm's pool supplying the donors) rather than a single fixed
# reference: with one shared reference the impostor-standardised statistic
# subtracts it out of candidate and cohort alike, and the manipulation
# cancels. Here each donor j gives lambda_j = logP(Q|G_A) - logP(Q|G_j); the
# primary readout is their mean, which responds to the donor pool, and the
# secondary readout is the cohort statistic mean_j/sd_j.
#
# Arms differ ONLY in where the r donor grammars come from:
#   oracle        target-language questioned-genre  (VIOLATES the constraint;
#                 upper bound only)
#   borrowed      one foreign questioned-genre corpus (right genre, wrong lang)
#   native-wrong  target-language known-genre        (right lang, wrong genre)
#   pooled        all foreign questioned-genre corpora, as separate donor
#                 models (never concatenated: documents in different rank maps
#                 must not be merged into one grammar)
#
# CONTROLLED COMPARISON. The case list, the distractor sample and the donor
# COUNT are identical across arms: no seed string contains the arm, and donors
# are excluded from the candidate and distractor pools so no case is ever
# scored against its own text.
#
# CORPUS HYGIENE. Poetry banks contain translations (the masked function-word
# grammar of a translated poem is the translator's), anonymous files and
# editors' anthologies; these are excluded from donor and distractor pools by
# the patterns below. The filter is deliberately conservative and certainly
# partial -- it is a floor on hygiene, not a guarantee.
#
#   python experiments/run_genreabsent.py --lang german --known-genre dracor \
#          --ref-arm borrowed
#
# Output: scores/xabsent2/{lang}_{g1}2{g2}__{arm}__K{KT}w{seg}.jsonl, one row
#   per case: {"cand","quest","qw","within","lam_sym","n_q","lam_j":[...],
#              "donors":[...]}

import argparse
import json
import random
import re
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from run_kn_grid import rechunk  # noqa: E402

sys.path.insert(0, str(HERE.parent))
from lambdag import LambdaG  # noqa: E402

MASKED = HERE.parent / "masked_catsrank"
SCORES = HERE.parent / "scores" / "xabsent2"

GENRE_BANKS = {"poetree": ["czech_poetree", "english_poetree", "french_poetree",
                           "german_poetree", "hungarian_poetree",
                           "italian_poetree", "russian_poetree"],
               "dracor": ["english_dracor", "french_dracor", "german_dracor",
                          "hungarian_dracor", "italian_dracor",
                          "polish_dracor", "russian_dracor"],
               "novels": ["czech_novels", "english_novels", "french_novels",
                          "german_novels", "hungarian_novels",
                          "italian_novels", "polish_novels"]}
# not the named author's own language-original single-author text
EXCLUDE_PAT = re.compile(
    r"anonym|_hg$|baudelaire|petrarca|shakespeare|poe_edgar|byron|shelley|"
    r"verlaine|dante|homer|villon|burns|whitman|moli_re|racine", re.I)

KT = 10000        # known-document tokens, and the donor size floor
R_DONORS = 15     # donor grammars per case -- the reference POPULATION
MAX_DONORS = 60
SAME_W = 3
NCROSS = 4
L = 1000


def read_tsv(f):
    return [line.split("\t") for line in
            f.read_text(encoding="utf-8").splitlines() if line]


def window(sents, start_tok, ln):
    out, pos = [], 0
    for s in sents:
        if pos + len(s) > start_tok:
            out.append(s)
            if sum(len(x) for x in out) >= ln:
                break
        pos += len(s)
    return out


def load_bank(ds):
    d = MASKED / ds / "bank"
    if not d.exists():
        raise SystemExit(f"missing encoded bank: {d}")
    bank = {}
    for f in sorted(d.glob("*.tsv")):
        s = read_tsv(f)
        bank[f.stem] = {"sents": s, "ntok": sum(len(x) for x in s)}
    return bank


def namekey(stem):
    return tuple(sorted(p for p in re.sub(r"^\d+_", "", stem).split("_") if p))


def foreign_banks(lang, genre):
    """Questioned-genre corpora of OTHER languages that are actually encoded.
    Derived from the inventory rather than hardcoded, so the constraint holds
    for any target language; missing encodings are skipped rather than
    silently changing the pool's language composition mid-run."""
    out = []
    for d in GENRE_BANKS[genre]:
        if d.startswith(lang):
            continue
        if (MASKED / d / "bank").exists():
            out.append(d)
        else:
            print(f"    [donor bank {d} not encoded, skipped]", flush=True)
    return out


def donor_pool(arm, lang, g1, g2, exclude_keys):
    """The arm's reference population: name -> sentences, all >= KT tokens."""
    if arm == "oracle":
        srcs = [f"{lang}_{g2}_nopoe"]
    elif arm == "native-wrong":
        srcs = [f"{lang}_{g1}_nopoe"]
    elif arm == "borrowed":
        srcs = foreign_banks(lang, g2)[:1]
    elif arm == "pooled":
        srcs = foreign_banks(lang, g2)
    else:
        raise SystemExit(f"unknown arm {arm}")
    assert all(not s.startswith(lang) for s in srcs) or arm in ("oracle",
                                                                "native-wrong")
    pool = {}
    for src in srcs:
        for a, v in load_bank(src).items():
            if v["ntok"] < KT or EXCLUDE_PAT.search(a):
                continue
            if src.startswith(lang) and namekey(a) in exclude_keys:
                continue          # never a case author's own text
            pool[f"{src.split('_')[0][:2]}:{a}"] = v["sents"]
    if len(pool) > MAX_DONORS:
        rng = random.Random(f"{lang}|{g1}|{g2}|donorcap")   # arm-free seed
        pool = {n: pool[n] for n in sorted(rng.sample(sorted(pool),
                                                      MAX_DONORS))}
    return pool, srcs


def run(args):
    lang, g1, g2, arm, seg = (args.lang, args.known_genre, args.quest_genre,
                              args.ref_arm, args.seg)
    cell = f"{lang}_{g1}2{g2}"
    tag = f"{cell}__{arm}__K{KT}w{seg}"
    SCORES.mkdir(parents=True, exist_ok=True)
    fn = SCORES / f"{tag}.jsonl"
    if fn.exists():
        print(f"{tag}: exists, skipped", flush=True)
        return
    b1 = load_bank(f"{lang}_{g1}_nopoe")
    b2 = load_bank(f"{lang}_{g2}_nopoe")
    k1 = {namekey(a): a for a in b1}
    k2 = {namekey(a): a for a in b2}
    cands = sorted(k for k in k1 if k in k2
                   if b1[k1[k]]["ntok"] >= KT and b2[k2[k]]["ntok"] >= L
                   and not EXCLUDE_PAT.search(k1[k])
                   and not EXCLUDE_PAT.search(k2[k]))
    if len(cands) < 3:
        print(f"{tag}: only {len(cands)} candidates, skipped", flush=True)
        return
    donors, srcs = donor_pool(arm, lang, g1, g2, set(cands))
    if len(donors) < R_DONORS:
        print(f"{tag}: only {len(donors)} donors, skipped", flush=True)
        return
    # Distractors: every questioned-genre author of the target language, minus
    # the candidates and the hygiene exclusions. This pool depends only on the
    # cell, never on the arm, so all four arms score an identical case list.
    # The "never score a case against its own donor" guarantee is enforced on
    # the DONOR side instead, per case (see below): excluding donors from the
    # distractor pool would make the pool arm-dependent -- the oracle arm's
    # donors live in this very bank -- and, applied to every donor-eligible
    # author, would leave a handful of distractors and a degenerate negative
    # class.
    distract = sorted(a for a in b2
                      if b2[a]["ntok"] >= L and namekey(a) not in cands
                      and not EXCLUDE_PAT.search(a))
    print(f"{tag}: {len(cands)} candidates, {len(donors)} donors from "
          f"{'+'.join(srcs)}, {len(distract)} distractors", flush=True)

    kdoc = {k1[k]: rechunk(window(b1[k1[k]]["sents"], 0, KT), seg)
            for k in cands}
    qwins = {a: [rechunk(window(b2[a]["sents"], i * L, L), seg)
                 for i in range(min(b2[a]["ntok"] // L, SAME_W))]
             for a in set(distract) | {k2[k] for k in cands}}

    lg = LambdaG(N=10, r=1, engine="kn", random_state=0)
    t0, rows = time.time(), []
    for k in cands:
        a1, a2 = k1[k], k2[k]
        cases = [(a2, j, 1) for j in range(len(qwins.get(a2, [])))]
        # arm-FREE seed: every arm scores the identical case list
        prng = random.Random(f"{cell}|K{KT}|{a1}|cross")
        pool_d = [b for b in distract if b != a2]
        for b in prng.sample(pool_d, min(NCROSS, len(pool_d))):
            cases.append((b, prng.randrange(len(qwins[b])), 0))
        drng = random.Random(f"{cell}|K{KT}|{a1}|donors")   # arm-free
        picks = drng.sample(sorted(donors), min(R_DONORS, len(donors)))
        for (qa, qw, within) in cases:
            q = qwins[qa][qw]
            # per-case donor guard: never score a questioned document against a
            # donor grammar fitted on that same author's text
            use = [d for d in picks
                   if namekey(d.split(":", 1)[1]) != namekey(qa)]
            assert len(use) >= R_DONORS - 3, f"donor pool collapsed for {qa}"
            lams, n_q = [], 0
            for dn in use:
                r = lg.score(q, kdoc[a1],
                             ref_sentences=rechunk(donors[dn], seg), r=1,
                             with_details=False)
                lams.append(r.lambda_G); n_q = r.n_query_tokens
            assert not np.allclose(lams, 0), f"degenerate donor set for {a1}"
            rows.append({"cand": a1, "quest": qa, "qw": qw, "within": within,
                         "lam_sym": round(float(np.mean(lams)), 3),
                         "n_q": n_q,
                         "lam_j": [round(x, 3) for x in lams],
                         "donors": use})
        print(f"  {tag} {a1}: {len(rows)} cases, {time.time()-t0:.0f}s",
              flush=True)
    with open(fn, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    print(f"{tag}: {len(rows)} cases in {time.time()-t0:.0f}s", flush=True)


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", default="german")
    ap.add_argument("--known-genre", default="dracor")
    ap.add_argument("--quest-genre", default="poetree")
    ap.add_argument("--ref-arm", required=True,
                    choices=["oracle", "borrowed", "native-wrong", "pooled"])
    ap.add_argument("--seg", type=int, default=100)
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
