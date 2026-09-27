# How long a shared run has to be before it means something, on each stream.
#
# The detector compares texts by the share of their k-grams that occur in the
# other, and k decides everything: too short and the formulaic stock of Middle
# High German verse fills the table, too long and nothing is shared at all. The
# right k is not the same on the two streams, because they do not have the same
# alphabet:
#
#   the POSNoise-masked stream keeps the function words and writes every content
#   word as a placeholder, so its alphabet is small and it repeats; eight tokens
#   is where the formula stops reaching.
#
#   the lemma stream keeps every word, so its alphabet is the language's, and
#   eight identical lemmas in a row essentially never happen between unrelated
#   texts -- the baseline is all zeros and the statistic has no spread left.
#
# This measures both, for one focal text against the whole corpus: the formulaic
# baseline at each k, and where the documented partners stand against it. The
# criterion is the one the mask already uses, so the k chosen here is the k the
# mask should run with.
#
#   python experiments/kc_reuse_k_sweep.py
#   python experiments/kc_reuse_k_sweep.py --focal kcd__anon__kaiserchronik__H
#
# Output: medieval/kc_reuse/k_sweep_{focal}.tsv and the table on stdout.

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "medieval"))
import kc_reuse_pairs as P  # noqa: E402
import run_witness_cells as W  # noqa: E402
from kc_dates import display_name  # noqa: E402

OUT = ROOT / "medieval" / "kc_reuse"
WATCH = ["rem__M013O", "rem__M205P", "rem__M205A", "rem__M009", "rem__M008"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--focal", default="kcd__anon__kaiserchronik__A1")
    ap.add_argument("--streams", default="lemma,lemma_full")
    ap.add_argument("--ks", default="3,4,5,6,7,8")
    ap.add_argument("--baseline-min-tokens", type=int, default=2000)
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    ks = [int(x) for x in args.ks.split(",")]
    rows = []

    for stream in args.streams.split(","):
        bank = W.load(stream)
        with (ROOT / "masked" / f"mhg_witnesses_{stream}"
              / "MANIFEST.tsv").open(encoding="utf-8") as fh:
            man = {r["file"]: r for r in csv.DictReader(fh, delimiter="\t")}
        f = args.focal
        pool = [m for m in bank
                if bank[m]["ntok"] >= args.baseline_min_tokens
                and not P.same_work(bank, f, m)]
        ft = P.toks(bank[f])
        print(f"\n{stream}: {display_name(f, man.get(f))}, {len(ft):,} tokens, "
              f"against {len(pool)} texts\n")
        print(f"{'k':>3} {'baseline median':>16} {'90th pct':>10} "
              f"{'sharing':>8}  " +
              "  ".join(f"{display_name(m, man.get(m))[:18]:>26}"
                        for m in WATCH))
        for k in ks:
            fs = set(P.grams(ft, k))
            cover, hits, nz = {}, {}, 0
            for m in pool:
                g = P.grams(P.toks(bank[m]), k)
                h = sum(1 for x in g if x in fs)
                hits[m] = h
                c = h / max(len(g), 1)
                cover[m] = c
                nz += c > 0
            vals = np.array(list(cover.values()))
            med = float(np.median(vals))
            p90 = float(np.quantile(vals, 0.9))
            sd = max(1.4826 * float(np.median(np.abs(vals - med))),
                     (p90 - med) / 1.2816)
            order = sorted(cover, key=lambda m: -cover[m])
            rank = {m: i + 1 for i, m in enumerate(order)}
            cells = []
            for m in WATCH:
                c = cover.get(m)
                if c is None:
                    cells.append(f"{'not in pool':>18}")
                    continue
                z = (c - med) / sd if sd > 0 else float("nan")
                # the absolute number of matches as well, because a z of six on
                # three matched k-grams is a statement about a baseline of
                # zeros and not about the two texts
                cells.append(f"{hits[m]:5d} {c * 1000:6.2f} #{rank[m]:<3} "
                             f"z{z:6.1f}")
                rows.append((stream, k, display_name(m, man.get(m)), hits[m],
                             round(c, 6), rank[m], len(pool), round(med, 6),
                             round(sd, 6), round(z, 2)))
            print(f"{k:>3} {med * 1000:14.3f}/k {p90 * 1000:9.3f} "
                  f"{nz / len(pool):7.0%}  " + "  ".join(cells))
        print("    per text: matched k-grams, then per thousand of its own, "
              "then its rank, then z. 'sharing' is the share of the corpus "
              "that shares anything at all")

    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / f"k_sweep_{args.focal.split('__')[-1]}.tsv"
    with p.open("w", encoding="utf-8") as fh:
        fh.write("stream\tk\tother\tshare_of_other\trank\tof_n\t"
                 "baseline_median\tbaseline_robust_sd\tz\n")
        for r in rows:
            fh.write("\t".join(str(x) for x in r) + "\n")
    print(f"\nwrote {p.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
