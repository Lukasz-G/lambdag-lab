# Borrowed reference populations under the universal alphabet: the symmetrised
# LambdaG estimator with per-author donor grammars drawn from a FOREIGN bank
# (journal paper, cross-lingual section). One (case-language, donor) cell per
# invocation, so the full grid fans out process-per-cell.
#
# All streams are pre-encoded class-conditioned rank symbols
# (masked_catsrank/, from encode_catsrank.py); each language is encoded with
# its own within-class frequency ranks, which is precisely the cross-language
# correspondence under test. Donor arms per case language X:
#   native   donors from X's own bank (baseline)
#   <lang>   donors from one foreign bank
#   pooled   donors drawn from the union of the five foreign banks
#
# Per case we persist the per-donor lambdas (the case-internal cohort), so the
# analysis gets the symmetrised score, the cohort-studentised statistic AND the
# corpus-adequacy reading (mean per-token lambda on different-author rows) from
# one file.
#
#   python experiments/run_xref.py --dataset german_novels --ref french_novels
#   python experiments/run_xref.py --dataset german_novels --ref pooled
#
# Output: scores/xref/{ds}__xref-{ref}__L{L}.jsonl, one row per case:
#   {"known","kw","quest","qw","within","lambda_G","n_q","lam_j":[...]}

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from lambdag import LambdaG  # noqa: E402

MASKED = HERE.parent / "masked_catsrank"
SCORES = HERE.parent / "scores" / "xref"

DATASETS = ["german_novels", "english_novels", "french_novels",
            "polish_novels", "czech_novels", "hungarian_novels"]
L = 1000
MINW = 3          # known windows per author
NCROSS = 4        # cross-questioned windows per known window
R_DONORS = 15     # per-author donor grammars per score
MAX_KNOWNS = 24
MAX_DONORS = 60   # donor-author cap per arm (pooled stays comparable)


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


def donor_bank(ds, ref):
    """donor name -> sentences, from one bank or the pooled foreign banks."""
    srcs = ([d for d in DATASETS if d != ds] if ref == "pooled" else [ref])
    donors = {}
    for src in srcs:
        for n, v in load_bank(src).items():
            if v["ntok"] >= L:
                donors[f"{src[:2]}:{n}"] = v["sents"]
    if len(donors) > MAX_DONORS:
        rng = random.Random(f"{ds}|{ref}|donorcap")
        donors = {n: donors[n] for n in sorted(rng.sample(sorted(donors),
                                                          MAX_DONORS))}
    return donors


def run(ds, ref, args):
    tag = f"{ds}__xref-{'native' if ref == ds else ref}__L{L}"
    SCORES.mkdir(parents=True, exist_ok=True)
    fn = SCORES / f"{tag}.jsonl"
    if fn.exists():
        print(f"{tag}: exists, skipped", flush=True)
        return
    bank = load_bank(ds)
    donors = donor_bank(ds, ds if ref == "native" else ref)
    elig = sorted(n for n in bank if bank[n]["ntok"] >= MINW * L)
    if len(elig) > args.max_knowns:
        rng = random.Random(f"{ds}|knowns")
        elig = sorted(rng.sample(elig, args.max_knowns))
    wins = {a: [window(bank[a]["sents"], k * L, L)
                for k in range(min(bank[a]["ntok"] // L, MINW))]
            for a in elig}
    lg = LambdaG(N=10, r=1, engine="kn", random_state=0)
    t0, rows = time.time(), []

    def sym_score(q, k, ka, qa):
        # donor names are bank-prefixed, so same-bank knowns never coincide
        pool = [n for n in donors if n.split(":", 1)[-1] not in (ka, qa)]
        prng = random.Random(f"{ds}|{ref}|{ka}|{qa}|donors")
        picks = prng.sample(pool, min(R_DONORS, len(pool)))
        lams, n_q = [], 0
        for dn in picks:
            r = lg.score(q, k, ref_sentences=donors[dn], r=1,
                         with_details=False)
            lams.append(r.lambda_G); n_q = r.n_query_tokens
        return float(np.mean(lams)), n_q, lams

    for a in elig:
        m = len(wins[a])
        others = [n for n in elig if n != a]
        for i in range(m):
            for j in range(m):
                if j == i:
                    continue
                lam, n_q, lams = sym_score(wins[a][j], wins[a][i], a, a)
                rows.append({"known": a, "kw": i, "quest": a, "qw": j,
                             "within": 1, "lambda_G": lam, "n_q": n_q,
                             "lam_j": [round(x, 3) for x in lams]})
            prng = random.Random(f"{ds}|{a}|{i}|cross")
            partners = (prng.sample(others, NCROSS) if len(others) >= NCROSS
                        else [prng.choice(others) for _ in range(NCROSS)])
            for b in partners:
                jq = prng.randrange(len(wins[b]))
                lam, n_q, lams = sym_score(wins[b][jq], wins[a][i], a, b)
                rows.append({"known": a, "kw": i, "quest": b, "qw": jq,
                             "within": 0, "lambda_G": lam, "n_q": n_q,
                             "lam_j": [round(x, 3) for x in lams]})
        print(f"  {tag} {a}: {len(rows)} rows, {time.time()-t0:.0f}s",
              flush=True)
    with open(fn, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    print(f"{tag}: {len(rows)} rows in {time.time()-t0:.0f}s", flush=True)


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--ref", required=True,
                    help="'native', 'pooled', or a donor dataset name")
    ap.add_argument("--max-knowns", type=int, default=MAX_KNOWNS)
    args = ap.parse_args()
    run(args.dataset, args.ref, args)


if __name__ == "__main__":
    main()
