# Gate B for the borrowed-population chapter: does the shared alphabet actually
# bring the Middle High German and the modern German symbol distributions
# together?
#
# The whole mechanism of borrowing a reference population is marginal alignment.
# A donor grammar fitted on modern German verse can only serve a medieval case if
# the two texts are written in the same symbols and use them at comparable rates.
# Under the plain masked stream they are not: the kept tokens are word forms, and
# seven centuries leave almost none of them in common. Under the class-conditioned
# rank alphabet they should be, because a form is replaced by its functional class
# and its rank within that class.
#
# THE MEASUREMENT. Three numbers per encoding, on the unigram symbol
# distributions of the two banks:
#   shared mass   the share of medieval tokens whose symbol occurs at all in the
#                 modern bank -- the necessary condition; a symbol the donor never
#                 saw contributes nothing but smoothing
#   JS divergence Jensen-Shannon, bounded in [0, 1] with base 2, so the two
#                 encodings are directly comparable
#   perplexity    of the medieval stream under the modern bank's unigram
#                 distribution, which is what a borrowed model pays before any
#                 context is used at all
#
# Bigrams are reported as well: lambda_G reads conditionals, so alignment of the
# marginals is necessary and not sufficient, and the bigram figures say how much
# of the alignment survives one step of context.
#
#   python experiments/run_gmh_alignment_gate.py
#   python experiments/run_gmh_alignment_gate.py --modern german_poetree
#
# Reads the banks directly and encodes in memory; nothing is written except the
# report.

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from aligned_utils import LANG_CODE, load_aligned  # noqa: E402
from xling_pilot import PLACEHOLDERS, class_rank_map, encode  # noqa: E402

MASKED = ROOT / "masked"


def read_bank(ds, limit=None):
    """author/text -> token list, from a masked bank."""
    bank = MASKED / ds / "bank"
    if not bank.exists():
        sys.exit(f"no bank at {bank}")
    files = sorted(bank.glob("*.tsv"))
    if limit:
        files = files[:limit]
    out = {}
    for f in files:
        toks = []
        for line in f.read_text(encoding="utf-8").splitlines():
            if line:
                toks += [t for t in line.split("\t") if t]
        out[f.stem] = toks
    return out


def symbols(bank, mode, table, crmap):
    """Flat symbol stream for a whole bank under one encoding."""
    out = []
    for toks in bank.values():
        out += encode(toks, mode, {}, table, crmap)
    return out


def distributions(a, b, order):
    """Aligned probability vectors over the union of the two vocabularies."""
    def grams(s):
        if order == 1:
            return Counter(s)
        return Counter(zip(*[s[i:] for i in range(order)]))
    ca, cb = grams(a), grams(b)
    keys = sorted(set(ca) | set(cb), key=str)
    pa = np.array([ca.get(k, 0) for k in keys], float)
    pb = np.array([cb.get(k, 0) for k in keys], float)
    return pa / pa.sum(), pb / pb.sum(), ca, cb


def js(p, q):
    """Jensen-Shannon divergence, base 2, so the result lies in [0, 1]."""
    m = 0.5 * (p + q)
    def kl(x, y):
        mask = x > 0
        return float(np.sum(x[mask] * np.log2(x[mask] / y[mask])))
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def report_pair(med, mod, order):
    p, q, cmed, cmod = distributions(med, mod, order)
    # shared mass: medieval token occurrences whose symbol the modern bank has
    if order == 1:
        total = sum(cmed.values())
        shared = sum(n for g, n in cmed.items() if g in cmod)
    else:
        total = sum(cmed.values())
        shared = sum(n for g, n in cmed.items() if g in cmod)
    # perplexity of the medieval stream under the modern distribution, with
    # add-one smoothing over the union so an unseen symbol is finite
    n_mod = sum(cmod.values())
    v = len(set(cmed) | set(cmod))
    logp = 0.0
    for g, n in cmed.items():
        pr = (cmod.get(g, 0) + 1) / (n_mod + v)
        logp += n * np.log2(pr)
    return {
        "order": order,
        "symbols_medieval": len(cmed),
        "symbols_modern": len(cmod),
        "shared_mass": round(shared / max(total, 1), 4),
        "js_divergence": round(js(p, q), 4),
        "perplexity_under_modern": round(float(2 ** (-logp / max(total, 1))), 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--medieval", default="mhg_witnesses_lemma")
    ap.add_argument("--modern", default="german_tgverseall")
    ap.add_argument("--limit", type=int, default=None,
                    help="read only the first N files of each bank")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    print(f"gate B: alignment of {args.medieval} and {args.modern}\n")

    med_bank = read_bank(args.medieval, args.limit)
    mod_bank = read_bank(args.modern, args.limit)
    print(f"  medieval {len(med_bank)} texts, "
          f"{sum(len(v) for v in med_bank.values()):,} tokens")
    print(f"  modern   {len(mod_bank)} texts, "
          f"{sum(len(v) for v in mod_bank.values()):,} tokens")

    med_table = load_aligned(LANG_CODE[args.medieval.split("_")[0]])
    mod_table = load_aligned(LANG_CODE[args.modern.split("_")[0]])
    for name, t in (("medieval", med_table), ("modern", mod_table)):
        if not t:
            sys.exit(f"no aligned companion file for the {name} bank")
    print(f"  companion files: {len(med_table):,} medieval entries, "
          f"{len(mod_table):,} modern\n")

    # The rank map is fitted WITHIN each stage, which is what makes the alphabet
    # comparable: rank 1 of ADP means "the commonest preposition of this stage".
    # Fitting one map over both stages would let the modern frequencies decide
    # what a medieval symbol denotes.
    med_cr = class_rank_map(med_bank, med_table)
    mod_cr = class_rank_map(mod_bank, mod_table)

    out = {"medieval": args.medieval, "modern": args.modern, "encodings": {}}
    for mode, table_pair, cr_pair in (
            ("posnoise", (None, None), (None, None)),
            ("cats", (med_table, mod_table), (None, None)),
            ("catranks", (med_table, mod_table), (med_cr, mod_cr))):
        if mode == "posnoise":
            med_sym = [t for toks in med_bank.values() for t in toks]
            mod_sym = [t for toks in mod_bank.values() for t in toks]
        else:
            med_sym = symbols(med_bank, mode, table_pair[0], cr_pair[0])
            mod_sym = symbols(mod_bank, mode, table_pair[1], cr_pair[1])
        rows = [report_pair(med_sym, mod_sym, o) for o in (1, 2)]
        out["encodings"][mode] = rows
        print(f"  {mode}")
        for r in rows:
            print(f"    order {r['order']}: "
                  f"symbols {r['symbols_medieval']:>7,} / "
                  f"{r['symbols_modern']:>7,}   "
                  f"shared mass {r['shared_mass']:.3f}   "
                  f"JS {r['js_divergence']:.3f}   "
                  f"ppl {r['perplexity_under_modern']:>8.1f}")
        print()

    dest = ROOT / "medieval" / "gmh_alignment_gate.json"
    dest.write_text(json.dumps(out, indent=2, ensure_ascii=False),
                    encoding="utf-8")
    print(f"  wrote {dest.relative_to(ROOT)}")
    print("\n  the gate asks whether shared mass rises and JS falls from "
          "posnoise to CatRanks;\n  an unchanged JS says the coarsening does "
          "not align the stages and borrowing\n  will fail for that reason")


if __name__ == "__main__":
    main()
