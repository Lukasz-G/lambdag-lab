# Cross-genre verification in German under the universal alphabet (journal
# paper, cross-lingual/cross-genre section): the known documents come from one
# genre, the questioned windows from ANOTHER, and the case author must be told
# apart from the questioned genre's other German authors.
#
#   Can a German poet be picked out among German poets, given only that
#   poet's novels or plays? Likewise for playwrights and novelists.
#
# Six directed genre pairs (known -> questioned) over the 16 verified German
# cross-genre authors. The symmetrised per-author-donor protocol scores each
# case under R_DONORS single-donor grammars; two donor-population options:
#   native    German donors from the QUESTIONED genre's bank
#   foreign   donors pooled from five foreign banks of the questioned genre,
#             German held out entirely
# All streams are pre-encoded class-conditioned rank symbols (masked_catsrank/),
# so the native-vs-foreign contrast is donor provenance and nothing else.
# Per-donor lambdas are persisted (cohort statistic + adequacy reading).
#
#   python experiments/run_xgenre.py --arm native
#   python experiments/run_xgenre.py --arm foreign
#   python experiments/run_xgenre.py --arm native --known-tokens 10000
#
# --known-tokens K replaces the three 1000-token known windows with ONE known
# document of K tokens (all the known-genre evidence the author affords, up to
# K), and restricts the donor pool to authors with >= K tokens so the
# size-matched donor grammars keep the estimator's exchangeability. Output
# files carry a K{K} tag.
#
# Output: scores/xgenre/{g1}2{g2}__{arm}__L{L}.jsonl, one row per case:
#   {"known","kw","quest","qw","within","lambda_G","n_q","lam_j":[...]}

import argparse
import json
import random
import re
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from lambdag import LambdaG  # noqa: E402

MASKED = HERE.parent / "masked_catsrank"
SCORES = HERE.parent / "scores" / "xgenre"

# _shared: one pooled German rank map across the three genres, so a symbol
# denotes the same word in the known and the questioned genre
GERMAN = {"novels": "german_novels_shared", "dracor": "german_dracor_shared",
          "poetree": "german_poetree_shared"}
FOREIGN = {"novels": ["english_novels", "french_novels", "polish_novels",
                      "czech_novels", "hungarian_novels"],
           "dracor": ["english_dracor", "french_dracor", "hungarian_dracor",
                      "italian_dracor", "polish_dracor"],
           "poetree": ["czech_poetree", "english_poetree", "french_poetree",
                       "hungarian_poetree", "italian_poetree"]}
DIRECTIONS = [("novels", "poetree"), ("dracor", "poetree"),
              ("novels", "dracor"), ("poetree", "dracor"),
              ("dracor", "novels"), ("poetree", "novels")]
L = 1000
KNOWN_W = 3       # known windows per case author (from genre G1)
SAME_W = 3        # own questioned windows per case author (from genre G2)
NCROSS = 4        # distractor questioned windows per known window
R_DONORS = 15
MAX_DONORS = 60


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
    bank = {}
    for f in sorted((MASKED / ds / "bank").glob("*.tsv")):
        s = read_tsv(f)
        bank[f.stem] = {"sents": s, "ntok": sum(len(x) for x in s)}
    return bank


def namekey(stem):
    return tuple(sorted(p for p in re.sub(r"^\d+_", "", stem).split("_") if p))


def run(g1, g2, arm, args):
    kt = args.known_tokens
    ktag = f"K{kt}" if kt != KNOWN_W * L else ""
    tag = f"{g1}2{g2}__{arm}__L{L}{ktag}"
    SCORES.mkdir(parents=True, exist_ok=True)
    fn = SCORES / f"{tag}.jsonl"
    if fn.exists():
        print(f"{tag}: exists, skipped", flush=True)
        return
    b1, b2 = load_bank(GERMAN[g1]), load_bank(GERMAN[g2])
    k1 = {namekey(a): a for a in b1}
    k2 = {namekey(a): a for a in b2}
    shared = sorted(k for k in k1 if k in k2
                    if b1[k1[k]]["ntok"] >= L and b2[k2[k]]["ntok"] >= L)
    if len(shared) < 3:
        print(f"{tag}: only {len(shared)} cross-genre authors, skipped",
              flush=True)
        return
    # donor pool: questioned-genre banks, native or pooled-foreign; donors must
    # afford the known-document size so donor grammars stay size-matched
    min_donor = kt if kt != KNOWN_W * L else L
    donors = {}
    srcs = [GERMAN[g2]] if arm == "native" else FOREIGN[g2]
    for src in srcs:
        for n, v in load_bank(src).items():
            if v["ntok"] >= min_donor:
                donors[f"{src[:2]}:{n}"] = v["sents"]
    if len(donors) < 8:
        print(f"{tag}: only {len(donors)} donors afford {min_donor} tokens, "
              f"skipped", flush=True)
        return
    if len(donors) > MAX_DONORS:
        rng = random.Random(f"{tag}|donorcap")
        donors = {n: donors[n] for n in sorted(rng.sample(sorted(donors),
                                                          MAX_DONORS))}
    # distractors: questioned-genre German authors who are NOT the case author
    distract = sorted(a for a in b2 if b2[a]["ntok"] >= L)

    if kt != KNOWN_W * L:
        # one known document per author: all his known-genre text up to kt
        kwins = {k: [window(b1[k1[k]]["sents"], 0, min(b1[k1[k]]["ntok"], kt))]
                 for k in shared}
    else:
        kwins = {k: [window(b1[k1[k]]["sents"], i * L, L)
                     for i in range(min(b1[k1[k]]["ntok"] // L, KNOWN_W))]
                 for k in shared}
    qwins = {a: [window(b2[a]["sents"], i * L, L)
                 for i in range(min(b2[a]["ntok"] // L, SAME_W))]
             for a in distract}
    for k in shared:                       # own questioned windows
        a2 = k2[k]
        if a2 not in qwins:
            qwins[a2] = [window(b2[a2]["sents"], i * L, L)
                         for i in range(min(b2[a2]["ntok"] // L, SAME_W))]

    lg = LambdaG(N=10, r=1, engine="kn", random_state=0)
    t0, rows = time.time(), []

    def sym_score(q, k, names):
        pool = [n for n in donors if n.split(":", 1)[-1] not in names]
        prng = random.Random(f"{tag}|{'|'.join(sorted(names))}|donors")
        picks = prng.sample(pool, min(R_DONORS, len(pool)))
        lams, n_q = [], 0
        for dn in picks:
            r = lg.score(q, k, ref_sentences=donors[dn], r=1,
                         with_details=False)
            lams.append(r.lambda_G); n_q = r.n_query_tokens
        return float(np.mean(lams)), n_q, lams

    for k in shared:
        a1, a2 = k1[k], k2[k]
        for i, kw in enumerate(kwins[k]):
            for j, qw in enumerate(qwins[a2]):          # same author, other genre
                lam, n_q, lams = sym_score(qw, kw, {a1, a2})
                rows.append({"known": a1, "kw": i, "quest": a2, "qw": j,
                             "within": 1, "lambda_G": lam, "n_q": n_q,
                             "lam_j": [round(x, 3) for x in lams]})
            prng = random.Random(f"{tag}|{a1}|{i}|cross")
            cand = [b for b in distract if b != a2]
            partners = prng.sample(cand, min(NCROSS, len(cand)))
            for b in partners:                          # other authors, that genre
                jq = prng.randrange(len(qwins[b]))
                lam, n_q, lams = sym_score(qwins[b][jq], kw, {a1, a2, b})
                rows.append({"known": a1, "kw": i, "quest": b, "qw": jq,
                             "within": 0, "lambda_G": lam, "n_q": n_q,
                             "lam_j": [round(x, 3) for x in lams]})
        print(f"  {tag} {a1}: {len(rows)} rows, {time.time()-t0:.0f}s",
              flush=True)
    with open(fn, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    print(f"{tag}: {len(rows)} rows ({len(shared)} authors) in "
          f"{time.time()-t0:.0f}s", flush=True)


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["native", "foreign"])
    ap.add_argument("--directions", default="",
                    help="comma list like novels2poetree; default all six")
    ap.add_argument("--known-tokens", type=int, default=KNOWN_W * L,
                    help="one known doc of this many tokens instead of "
                         "three 1000-token windows")
    args = ap.parse_args()
    dirs = ([tuple(d.split("2")) for d in args.directions.split(",") if d]
            or DIRECTIONS)
    for g1, g2 in dirs:
        run(g1, g2, args.arm, args)


if __name__ == "__main__":
    main()
