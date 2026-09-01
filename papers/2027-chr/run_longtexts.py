# Long-text control for the CHR 2027 paper: symmetric 1,000 / 2,000 / 5,000 tokens
# on five linguistically distant novel corpora (EN, DE, PL, UK, LT).
#
# The main grid caps the questioned side at 1,000 tokens by dataset construction,
# so it cannot say whether the calibration-free corrections come right when both
# sides are long. This experiment can, WITHOUT re-masking: the masked reference
# banks hold each author's full masked text, and every bank author in the chosen
# datasets carries >= 11k tokens. Cases are therefore built FROM the bank --
# known sample = the author's first L tokens, questioned = the next L, disjoint --
# and per case the reference pool excludes BOTH case authors. Because the bank
# doubles as case material, this is a self-contained side experiment, not extra
# rows of the main grid; the paper says so in the table caption.
#
#   python experiments/run_longtexts.py            # all five datasets, smallest first
#
# Output: experiments/scores/{dataset}__kn__long__L{len}.jsonl (resumable) and
# grid-format log lines, parsed by make_paper_tables.py from longtexts.log.

import json
import random
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from run_kn_grid import MASKED, SCORES, read_tsv, summarise  # noqa: E402

sys.path.insert(0, str(HERE.parent))
from lambdag import LambdaG  # noqa: E402

# smallest first, so the first log lines arrive within minutes
DATASETS = ["lithuanian_novels", "ukrainian_novels", "polish_novels",
            "german_novels", "english_novels"]
LENGTHS = [1000, 2000, 5000]   # symmetric: both sides L tokens
ROTATIONS = 2                  # different-author partners per known author


def window(sents, start, L):
    """Tokens [start, start+L) of a sentence list, kept as (possibly split) sentences."""
    out, seen = [], 0
    for s in sents:
        lo, hi = max(start - seen, 0), min(start + L - seen, len(s))
        if lo < hi:
            out.append(s[lo:hi])
        seen += len(s)
        if seen >= start + L:
            break
    return out


def build_cases(bank, L):
    """Balanced same/different-author cases at symmetric length L.

    Same-author: known = tokens[0:L], questioned = tokens[L:2L] (and [2L:3L] where
    the text allows). Different-author: the same known sample against the next
    authors' questioned windows, so both classes reuse identical text spans.
    """
    names = sorted(bank)
    cases = []
    for i, a in enumerate(names):
        if bank[a]["ntok"] < 2 * L:
            continue
        known = window(bank[a]["sents"], 0, L)
        cases.append((a, a, known, window(bank[a]["sents"], L, L)))
        if bank[a]["ntok"] >= 3 * L:
            cases.append((a, a, known, window(bank[a]["sents"], 2 * L, L)))
        hits = 0
        for step in range(1, len(names)):
            b = names[(i + step) % len(names)]
            if bank[b]["ntok"] < 2 * L:
                continue
            cases.append((a, b, known, window(bank[b]["sents"], L, L)))
            hits += 1
            if hits == ROTATIONS:
                break
    return cases


def score_dataset(ds):
    d = MASKED / ds
    bank = {}
    for f in sorted((d / "bank").glob("*.tsv")):
        sents = read_tsv(f)
        bank[f.stem] = {"sents": sents, "ntok": sum(len(s) for s in sents)}

    lg = LambdaG(N=10, r=30, engine="kn", random_state=0)
    rows = []
    for L in LENGTHS:
        fn = SCORES / f"{ds}__kn__long__L{L}.jsonl"
        if fn.exists():
            recs = [json.loads(l) for l in open(fn, encoding="utf-8")]
            rows.append(summarise(ds, "kn", "long", L, recs))
            continue
        t0 = time.time()
        cases = build_cases(bank, L)
        # one set_reference per exclusion set; the pool is capped so that encoding
        # stays cheap while still dwarfing the r=30 size-matched samples drawn from it
        cap = max(100_000, 60 * L)
        by_excl = {}
        for cid, (a, b, k, q) in enumerate(cases):
            by_excl.setdefault(frozenset((a, b)), []).append((cid, a, b, k, q))
        recs = []
        for excl, group in sorted(by_excl.items(), key=lambda kv: sorted(kv[0])):
            pool = [s for n in sorted(bank) if n not in excl for s in bank[n]["sents"]]
            rng = random.Random(f"{ds}|{L}|{'|'.join(sorted(excl))}")
            rng.shuffle(pool)
            tot, kept = 0, []
            for s in pool:
                kept.append(s); tot += len(s)
                if tot >= cap:
                    break
            lg.set_reference(kept)
            for cid, a, b, k, q in group:
                r = lg.score(q, k, with_details=False)
                recs.append({"id": cid, "label": int(a == b), "lambda_G": r.lambda_G,
                             "n_q": r.n_query_tokens, "v1_q": r.n_query_hapax,
                             "sqrt": r.lambda_sqrt, "hapax": r.lambda_hapax})
        with open(fn, "w", encoding="utf-8") as fh:
            for rec in recs:
                fh.write(json.dumps(rec) + "\n")
        rows.append(summarise(ds, "kn", "long", L, recs, time.time() - t0))
    return rows


def main():
    for ds in DATASETS:
        for r in score_dataset(ds):
            if r.get("auc_lambda_G") is not None:
                print(f"  {r['dataset']:24s} {r['engine']:11s} {r['seg']:4s} "
                      f"L={r['L']:>4d}  AUC {r['auc_lambda_G']:.3f}  "
                      f"Cllr {r['cllr_lambda_G']:8.3f}  sqrt {r['cllr_sqrt']:.3f}  "
                      f"hapax {r['cllr_hapax']:.3f}  ({r['secs']}s)  n={r['n']}", flush=True)


if __name__ == "__main__":
    main()
