# LambdaG on the universal catsrank alphabet vs surface tokens (journal paper,
# cross-lingual chapter) -- the prerequisite for borrowing reference populations
# across languages: does an
# order-10 Kneser-Ney grammar model over ~90 class-conditioned rank symbols
# still discriminate authors WITHIN a language, and at what cost against the
# surface-token baseline?
#
# Protocol (mirrors the window-pair pilot): per dataset, eligible bank authors
# (>= MINW x L tokens, capped at MAX_AUTHORS) are cut into <= MAXW disjoint
# L-token windows on sentence boundaries; per author, PAIRS same-author cases
# (questioned window j vs known window i) plus an equal number of cross-author
# cases. Both arms score the IDENTICAL case list; the catsrank arm re-encodes
# every sentence through the class-conditioned rank map built from the
# language's own bank and its aligned companion list. Reference pool: bank
# authors minus the known author, capped at REF_TOKENS x L tokens (native
# references; the borrowed foreign-reference arm is the follow-up experiment).
#
#   python experiments/run_lambdag_catsrank.py --datasets german_novels
#   python experiments/run_lambdag_catsrank.py --arms catsrank --max-authors 4
#
# Output: scores/{ds}__{arm}__xalpha__L{L}.jsonl, one row per case:
#   {"quest", "qw", "known", "kw", "within", "lambda_G", "sqrt", "n_q"}
# Resumable: existing output files are skipped.

import argparse
import json
import random
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from aligned_utils import LANG_CODE, load_aligned  # noqa: E402
from xling_pilot import class_rank_map, encode  # noqa: E402

sys.path.insert(0, str(HERE.parent))
from lambdag import LambdaG  # noqa: E402

MASKED = HERE.parent / "masked"
SCORES = HERE.parent / "scores"

DATASETS = ["german_novels", "english_novels", "hungarian_novels"]
L = 1000
MINW = 4
MAXW = 6
MAX_AUTHORS = 30
PAIRS = 6          # same-author cases per author (cross matched 1:1 overall)
REF_TOKENS = 60    # reference pool cap, in multiples of L


def read_tsv(f):
    sents = []
    for line in open(f, encoding="utf-8"):
        line = line.rstrip("\n")
        if line:
            sents.append(line.split("\t"))
    return sents


def window(sents, start_tok, ln):
    """Whole sentences covering tokens [start_tok, start_tok + ln)."""
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
        bank[f.stem] = s
    return bank


def make_cases(elig, m, ds):
    same, diff = [], []
    for a in elig:
        rng = random.Random(f"{ds}|{a}|pairs")
        cand = [(i, j) for i in range(m[a]) for j in range(m[a]) if i != j]
        rng.shuffle(cand)
        for i, j in cand[:PAIRS]:
            same.append({"quest": a, "qw": j, "known": a, "kw": i, "within": 1})
    rng = random.Random(f"{ds}|cross")
    while len(diff) < len(same):
        a, b = rng.sample(elig, 2)
        diff.append({"quest": a, "qw": rng.randrange(m[a]),
                     "known": b, "kw": rng.randrange(m[b]), "within": 0})
    return same + diff


def encode_bank(bank, ds):
    """Re-encode every sentence through the class-conditioned rank map."""
    lang = ds.split("_")[0]
    table = load_aligned(LANG_CODE[lang])
    if not table:
        raise SystemExit(f"no aligned companion file for {lang}")
    flat = {a: [t for s in sents for t in s] for a, sents in bank.items()}
    crmap = class_rank_map(flat, table)
    return {a: [encode(s, "catsrank", {}, table, crmap) for s in sents]
            for a, sents in bank.items()}


def run(ds, arm, args):
    fn = SCORES / f"{ds}__{arm}__xalpha__L{L}.jsonl"
    if fn.exists():
        print(f"{ds}/{arm}: exists, skipped", flush=True)
        return
    bank = load_bank(ds)
    ntok = {a: sum(len(s) for s in sents) for a, sents in bank.items()}
    elig = sorted(a for a in bank if ntok[a] >= MINW * L)
    if len(elig) > args.max_authors:
        rng = random.Random(ds.split("_")[0])
        elig = sorted(rng.sample(elig, args.max_authors))
    m = {a: min(ntok[a] // L, MAXW) for a in elig}
    cases = make_cases(elig, m, ds)

    use = bank if arm == "surface" else encode_bank(bank, ds)
    wins = {a: [window(use[a], k * L, L) for k in range(m[a])] for a in elig}

    lg = LambdaG(N=10, r=30, engine="kn", random_state=0)
    by_known = {}
    for c in cases:
        by_known.setdefault(c["known"], []).append(c)
    t0, rows = time.time(), []
    for a in sorted(by_known):
        pool = [s for n in bank if n != a for s in use[n]] if arm == "catsrank" \
            else [s for n in bank if n != a for s in bank[n]]
        rng = random.Random(f"{ds}|{a}|ref")
        rng.shuffle(pool)
        tot, kept = 0, []
        for s in pool:
            kept.append(s); tot += len(s)
            if tot >= REF_TOKENS * L:
                break
        lg.set_reference(kept)
        for c in by_known[a]:
            r = lg.score(wins[c["quest"]][c["qw"]], wins[a][c["kw"]],
                         with_details=False)
            rows.append({**c, "lambda_G": r.lambda_G, "sqrt": r.lambda_sqrt,
                         "n_q": r.n_query_tokens})
        print(f"  {ds}/{arm} {a}: {len(rows)}/{len(cases)}, "
              f"{time.time()-t0:.0f}s", flush=True)
    SCORES.mkdir(exist_ok=True)
    with open(fn, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    print(f"{ds}/{arm}: {len(rows)} scores in {time.time()-t0:.0f}s -> {fn.name}",
          flush=True)


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default=",".join(DATASETS))
    ap.add_argument("--arms", default="surface,catsrank")
    ap.add_argument("--max-authors", type=int, default=MAX_AUTHORS)
    args = ap.parse_args()
    for ds in args.datasets.split(","):
        for arm in args.arms.split(","):
            run(ds.strip(), arm.strip(), args)


if __name__ == "__main__":
    main()
