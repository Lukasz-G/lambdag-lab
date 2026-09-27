# Text against text: who shares with whom among the early Middle High German
# narrative poems, and how far above the formulaic baseline.
#
# The reuse mask asks one question -- what does the Kaiserchronik share? -- and
# that leaves a second one open: whether the texts it shares with also share
# with each other. The Rolandslied and the Alexanderlied both stand above the
# chronicle's baseline, so either they stand in some relation to one another, or
# each stands separately in a relation to the chronicle, or what they share is
# the common stock of early narrative verse. Only a measurement between them
# decides it, and this is that measurement.
#
# WHAT IS COMPUTED. For every focal text F, the share of each other text's
# k-grams that also occur in F, which is normalised by the OTHER text's length
# and so is not won by length alone; then the median and a robust spread of that
# share over every text of the corpus at 2,000 tokens or more, which is the
# formulaic baseline of the form; then, for each pair, how many robust standard
# deviations above that baseline the two texts stand. The statistic is the
# mask's, so the numbers here and the mask's agree.
#
# It is asymmetric by construction: a short text can have most of its k-grams in
# a long one while the long one has few of its own in the short. Both directions
# are therefore printed, and a pair counts as a relation only if it stands clear
# of the baseline in the direction that is being read.
#
# Witnesses of one work are marked rather than dropped: they are transmission,
# not reuse, and they show what the top of the scale looks like.
#
#   python experiments/kc_reuse_pairs.py --stream lemma_full --k 5
#   python experiments/kc_reuse_pairs.py --stream lemma --k 8
#
# Output: medieval/kc_reuse/pairs_{stream}_k{k}.tsv, and the matrix on stdout.

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "medieval"))
import build_kc_reuse_mask as M  # noqa: E402
import run_witness_cells as W  # noqa: E402
from kc_dates import display_name  # noqa: E402

OUT = ROOT / "medieval" / "kc_reuse"

# the texts the question is about: the chronicle's oldest and youngest long
# witnesses, the two poems the mask names, the two Alexander poems, and three
# texts of the same form and period that the mask does not name, as controls
FOCAL = ["kcd__anon__kaiserchronik__A1",
         "kcd__anon__kaiserchronik__H",
         "rem__M013O", "rem__M013B",
         "rem__M205P", "rem__M205A",
         "rem__M009", "rem__M008",
         "rem__M541H", "rem__M241", "rem__M064R"]


def same_work(bank, a, b):
    """two witnesses of one work, whatever their sigla say

    The reference corpus's Kaiserchronik fragments normalise to work strings of
    their own ("kaiserchronik_a_fragment_w" and the like), so a test on the work
    string alone leaves them in the baseline, where they sit at the level of
    transmission and outrank every real partner.
    """
    return (bank[a]["work"] == bank[b]["work"]
            or (M.is_kc(a, bank) and M.is_kc(b, bank)))


def toks(entry):
    return [t for s in entry["sents"] for t in s]


def grams(t, k):
    return [tuple(t[i:i + k]) for i in range(len(t) - k + 1)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stream", default="lemma_full")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--baseline-min-tokens", type=int, default=2000)
    ap.add_argument("--min-tokens", type=int, default=2000,
                    help="a focal text shorter than this is left out")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    bank = W.load(args.stream)
    with (ROOT / "masked" / f"mhg_witnesses_{args.stream}"
          / "MANIFEST.tsv").open(encoding="utf-8") as fh:
        man = {r["file"]: r for r in csv.DictReader(fh, delimiter="\t")}
    # a text of a few hundred tokens can share a large share of its own k-grams
    # with anything by accident, and the baseline it is compared against is then
    # mostly zeros, so neither its share nor a spread computed from it means
    # anything. Short texts stay out of this table.
    focal = [f for f in FOCAL
             if f in bank and bank[f]["ntok"] >= args.min_tokens]
    dropped = [f for f in FOCAL if f in bank and f not in focal]
    missing = [f for f in FOCAL if f not in bank]
    if missing:
        print("not in this bank:", ", ".join(missing))
    if dropped:
        print("below %d tokens, left out: %s"
              % (args.min_tokens,
                 ", ".join(f"{f} ({bank[f]['ntok']:,})" for f in dropped)))
    names = {f: display_name(f, man.get(f)) for f in focal}

    pool = [m for m in bank if bank[m]["ntok"] >= args.baseline_min_tokens]
    print(f"{args.stream} stream, k={args.k}: {len(focal)} focal texts against "
          f"a baseline of {len(pool)} texts at "
          f"{args.baseline_min_tokens:,} tokens or more")

    gr = {m: grams(toks(bank[m]), args.k) for m in set(pool) | set(focal)}
    rows, zs = [], {}
    for f in focal:
        fs = set(gr[f])
        # The baseline answers "how much does an UNRELATED text of this form
        # share", so the focal text's own other witnesses have no business in
        # it: twelve witnesses of the chronicle in the pool lift its ninetieth
        # percentile to the level of transmission, and every real partner's z
        # then collapses towards zero. They are scored, and marked, separately.
        cover = {m: sum(1 for g in gr[m] if g in fs) / max(len(gr[m]), 1)
                 for m in pool
                 if m != f and not same_work(bank, f, m)}
        vals = np.array(list(cover.values()))
        med = float(np.median(vals))
        # The median absolute deviation is zero whenever more than half the
        # corpus shares nothing at all with this text, which is the ordinary
        # case at k=8, and a zero spread turns every z into a meaningless
        # thousand. The upper half of the distribution still carries a scale,
        # so the spread is read from the ninetieth percentile where the MAD
        # fails.
        mad = 1.4826 * float(np.median(np.abs(vals - med)))
        sd = max(mad, (float(np.quantile(vals, 0.9)) - med) / 1.2816)
        order = sorted(cover, key=lambda m: -cover[m])
        rank = {m: i + 1 for i, m in enumerate(order)}
        for m in focal:
            if m == f:
                continue
            c = cover.get(m)
            if c is None:
                c = sum(1 for g in gr[m] if g in fs) / max(len(gr[m]), 1)
            z = (c - med) / sd if sd > 0 else float("nan")
            zs[(f, m)] = (rank.get(m, 0), z, c)
            rows.append((names[f], names[m], len(gr[f]), len(gr[m]),
                         round(c, 5), rank.get(m, 0), len(cover),
                         round(med, 6), round(sd, 6), round(z, 1),
                         "same work" if same_work(bank, f, m) else ""))
        print(f"  {names[f]:26} {bank[f]['ntok']:7,} tokens  baseline median "
              f"{med * 1000:6.3f}  robust sd {sd * 1000:6.3f} per thousand"
              f"{'  (spread from the 90th percentile)' if sd > mad else ''}")

    w = max(len(names[f]) for f in focal) + 1
    npool = len(pool) - 1
    views = (("the COLUMN text's rank among the %d texts of the corpus, as a "
              "partner of the ROW text" % npool,
              lambda r, z, c: f"{r:9d}"),
             ("how far the COLUMN text stands above the ROW text's formulaic "
              "baseline, in robust standard deviations",
              lambda r, z, c: f"{z:9.1f}"))
    for what, cell in views:
        print("\n  " + what + "\n")
        print(" " * w + "".join(f"{names[m][:9]:>10}" for m in focal))
        for f in focal:
            line = "".join(
                ("         -" if m == f else
                 cell(*zs[(f, m)])
                 + ("*" if same_work(bank, f, m) else " "))
                for m in focal)
            print(f"{names[f]:{w}}" + line)
    print("\n  * the two are witnesses of one work: transmission, not reuse")

    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / f"pairs_{args.stream}_k{args.k}.tsv"
    with p.open("w", encoding="utf-8") as fh:
        fh.write("focal\tother\tfocal_kgrams\tother_kgrams\tshare_of_other\t"
                 "rank\tof_n\tbaseline_median\tbaseline_robust_sd\tz\t"
                 "same_work\n")
        for r in rows:
            fh.write("\t".join(str(x) for x in r) + "\n")
    print(f"  wrote {p.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
