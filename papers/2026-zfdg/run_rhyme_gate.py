# Does the last word of a printed line stand at the rhyme? The gate before the
# rhyme stream is used for anything.
#
# The rhyme stream assumes that one printed line is one metrical verse, so that
# its last word is the rhyme word. That holds for verse printed in couplets and
# for strophic verse printed in long lines; it FAILS silently for an edition
# that prints the two half-lines of a long line separately, where every second
# word taken would be a caesura word instead. Nothing in the stream itself shows
# the difference, so it is measured here.
#
# THE TEST NEEDS SURFACES. Only inflected forms rhyme: the canonical lemmas of a
# rhyming pair need share no ending, since `sprach : gesach` lemmatise to
# `sprechen : gesehen`. The stream is built from lemmas, by design, but the gate
# reads the unmasked surface bank.
#
# WHAT IS MEASURED, per text: the length of the longest common ending of the
# line-final words at lag one, split by parity (line 1 against 2, then 2 against
# 3), and at lag two -- each against the same statistic on the same words
# shuffled, which is the chance level for that text's own spelling. The printing
# scheme then reads off the pattern:
#
#   couplets, or strophic long lines   lag 1 high on one parity, low on the
#                                      other, lag 2 at chance
#   half-lines printed separately      lag 1 at chance, lag 2 high
#   not rhymed verse, or a broken
#   extraction                         everything at chance
#
#   python experiments/run_rhyme_gate.py
#
# Output: medieval/rhyme_gate.tsv and the summary on stdout

import argparse
import csv
import random
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = ROOT / "medieval" / "rhyme_gate.tsv"   # _<bank> when not the witnesses

FOLD = str.maketrans({"y": "i", "v": "u", "w": "u", "j": "i", "ſ": "s",
                      "ë": "e", "ê": "e", "é": "e", "è": "e", "â": "a",
                      "á": "a", "î": "i", "í": "i", "ô": "o", "ó": "o",
                      "û": "u", "ú": "u", "æ": "e", "œ": "o", "ā": "a",
                      "ē": "e", "ī": "i", "ō": "o", "ū": "u", "ÿ": "i"})


def fold(w):
    w = w.lower().translate(FOLD)
    out = []
    for c in w:
        if not c.isalpha():
            continue
        if out and out[-1] == c:      # doubled consonants are spelling
            continue
        out.append(c)
    return "".join(out)


def tail(a, b):
    """the length of the longest common ending of two words, folded"""
    a, b = fold(a), fold(b)
    if not a or not b or a == b:
        return 0.0 if a != b else np.nan   # identical words are not a rhyme
    n = 0
    while n < min(len(a), len(b)) and a[-1 - n] == b[-1 - n]:
        n += 1
    return float(n)


def rate(words, lag, parity=None):
    v = [tail(words[i], words[i + lag]) for i in range(len(words) - lag)
         if parity is None or i % 2 == parity]
    v = [x for x in v if not np.isnan(x)]
    return (float(np.mean(v)), float(np.mean([x >= 3 for x in v])), len(v)) \
        if v else (np.nan, np.nan, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stream", default="surface_full")
    ap.add_argument("--bank", default="mhg_witnesses",
                    help="mhg_witnesses for the manuscripts, mhdbdb for the "
                         "normalised editions, whose lines are an editor's "
                         "verses rather than a scribe's")
    ap.add_argument("--min-verses", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    rng = random.Random(args.seed)
    src = ROOT / "masked" / f"{args.bank}_{args.stream}"
    with (src / "MANIFEST.tsv").open(encoding="utf-8") as fh:
        man = list(csv.DictReader(fh, delimiter="\t"))

    rows, verdicts = [], {}
    for r in man:
        name = r["file"]
        if args.bank == "mhg_witnesses" and                 not name.startswith(("kcd", "hvad", "parzival", "ambraser")):
            continue
        lines = [l.split("\t") for l in
                 (src / "bank" / f"{name}.tsv").read_text(
                     encoding="utf-8").splitlines() if l]
        w = [u[-1] for u in lines if u and u[-1]]
        if len(w) < args.min_verses:
            continue
        sh = w[:]
        rng.shuffle(sh)
        m0, r0, _ = rate(sh, 1)
        m1a, r1a, n = rate(w, 1, 0)      # line 1-2, 3-4, ...
        m1b, r1b, _ = rate(w, 1, 1)      # line 2-3, 4-5, ...
        m2, r2, _ = rate(w, 2)
        best, other = max(m1a, m1b), min(m1a, m1b)
        # the verdict: a parity that stands clear of chance at lag one is a
        # verse per line; lag two clear while lag one is not is half-lines
        if best - m0 > 0.5 and best - other > 0.3:
            v = "verse per line"
        elif m2 - m0 > 0.5 and best - m0 < 0.3:
            v = "HALF-LINES"
        elif best - m0 > 0.5:
            v = "verse per line (no parity)"
        else:
            v = "NO RHYME FOUND"
        verdicts[v] = verdicts.get(v, 0) + 1
        rows.append((name, len(w), round(m0, 2), round(m1a, 2), round(m1b, 2),
                     round(m2, 2), round(r1a, 3), round(r1b, 3), round(r0, 3),
                     v))

    src_of = lambda n: n.split("__")[0]
    print(f"{len(rows)} witnesses of {args.min_verses}+ verses\n")
    print(f"  {'source':10} {'texts':>6} {'chance':>7} {'lag1 odd':>9} "
          f"{'lag1 even':>10} {'lag2':>7}   rhyming pairs at lag 1")
    for s in ("kcd", "hvad", "parzival", "ambraser"):
        sel = [x for x in rows if src_of(x[0]) == s]
        if not sel:
            continue
        a = np.array([[x[2], x[3], x[4], x[5], x[6], x[8]] for x in sel])
        print(f"  {s:10} {len(sel):6} {a[:, 0].mean():7.2f} "
              f"{a[:, 1].mean():9.2f} {a[:, 2].mean():10.2f} "
              f"{a[:, 3].mean():7.2f}   {a[:, 4].mean():.0%} "
              f"against {a[:, 5].mean():.0%} by chance")
    print("\n  verdicts: " + ", ".join(f"{k}: {v}" for k, v in verdicts.items()))
    bad = [x for x in rows if x[-1] != "verse per line"]
    if bad:
        print("  not a plain verse per line:")
        for x in bad[:20]:
            print(f"    {x[0]:56} {x[-1]:26} lag1 {x[3]:.2f}/{x[4]:.2f} "
                  f"lag2 {x[5]:.2f} chance {x[2]:.2f}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out = OUT if args.bank == "mhg_witnesses" else         OUT.with_name(f"rhyme_gate_{args.bank}.tsv")
    with out.open("w", encoding="utf-8") as fh:
        fh.write("file\tverses\tchance_tail\tlag1_odd\tlag1_even\tlag2\t"
                 "lag1_odd_rhyme_rate\tlag1_even_rhyme_rate\t"
                 "chance_rhyme_rate\tverdict\n")
        for x in rows:
            fh.write("\t".join(str(y) for y in x) + "\n")
    print(f"\n  wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
