# Gate G2: is the rhyme position really the part a scribe leaves alone?
#
# The whole case for a rhyme-word stream is that a copyist rewrites inside the
# line but cannot touch the word at the rhyme without breaking it. That is a
# claim about manuscripts, and the corpus can test it: many works survive in
# several witnesses, and the editions number the verses, so two witnesses of one
# work can be compared VERSE BY VERSE.
#
# For every verse the two witnesses share, four things are asked of the
# canonical lemmas:
#
#   the last word of the verse   -- the rhyme position, the claim;
#   the first word of the verse  -- a position with no metrical protection;
#   the words in between         -- the body of the line, token by token;
#   the masked line as a whole   -- whether the POSNoise stream's own unit
#                                   survives the copy unchanged.
#
# If the rhyme position is not markedly the most stable of these, the premise
# fails and the stream is not worth building on.
#
# Each work is compared to its longest witness rather than every pair to every
# other: one work has eighty-five witnesses here and the pairs would be
# thousands, all of them the same evidence over again.
#
#   python experiments/run_rhyme_stability.py
#
# Output: medieval/rhyme_stability.tsv and the summary on stdout

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = ROOT / "medieval" / "rhyme_stability.tsv"


def read(stream, name):
    base = ROOT / "masked" / f"mhg_witnesses_{stream}"
    lines = [l.split("\t") for l in
             (base / "bank" / f"{name}.tsv").read_text(
                 encoding="utf-8").splitlines() if l]
    vids = [v for v in (base / "verse_ids" / f"{name}.txt").read_text(
        encoding="utf-8").split("\n") if v]
    return dict(zip(vids, lines)) if len(vids) == len(lines) else {}


def kappa(obs, chance):
    """agreement above what the position's own vocabulary gives for nothing

    The first word of a Middle High German verse is very often `unde`, `do` or
    an article, so two witnesses agree there half the time by the vocabulary
    alone; the rhyme word is a content word and its chance agreement is small.
    Comparing the raw rates would therefore credit the least informative
    position with the greatest stability.
    """
    return (obs - chance) / (1 - chance) if chance < 1 else float("nan")


def self_chance(seqs):
    """the chance that two draws from this position's distribution agree"""
    from collections import Counter
    c = Counter(seqs)
    n = sum(c.values())
    return sum((v / n) ** 2 for v in c.values()) if n else float("nan")


def agree(a, b):
    """token-by-token agreement of two versions of one verse"""
    if not a or not b:
        return np.nan
    n = min(len(a), len(b))
    return sum(x == y for x, y in zip(a[:n], b[:n])) / max(len(a), len(b))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-shared", type=int, default=200)
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    with (ROOT / "masked" / "mhg_witnesses_lemma_full" /
          "MANIFEST.tsv").open(encoding="utf-8") as fh:
        man = [r for r in csv.DictReader(fh, delimiter="\t")
               if r["file"].startswith(("kcd", "hvad", "parzival", "ambraser"))
               and not r["file"].endswith("__reg")]
    works = defaultdict(list)
    for r in man:
        works[(r["file"].split("__")[0], r["work"])].append(r)
    rows = []
    for (src, work), ws in sorted(works.items()):
        if len(ws) < 2:
            continue
        ws.sort(key=lambda r: -int(r["tokens"]))
        ref = ws[0]["file"]
        R, Rm = read("lemma_full", ref), read("lemma", ref)
        if not R:
            continue
        for r in ws[1:]:
            name = r["file"]
            A, Am = read("lemma_full", name), read("lemma", name)
            shared = [v for v in A if v in R]
            if len(shared) < args.min_shared:
                continue
            ok = [v for v in shared if A[v] and R[v]]
            last = float(np.mean([A[v][-1] == R[v][-1] for v in ok]))
            first = float(np.mean([A[v][0] == R[v][0] for v in ok]))
            body = float(np.nanmean([agree(A[v], R[v]) for v in ok]))
            # a pair whose bodies do not agree at all is not aligned: the two
            # witnesses number their own lines, so one omission puts every
            # later verse against a different verse. Those pairs are counted
            # and set aside rather than averaged in.
            if body < 0.3:
                rows.append((src, work, name, ref, len(ok), round(last, 4),
                             round(first, 4), round(body, 4),
                             "", "", "not aligned"))
                continue
            ch_last = self_chance([tuple(R[v][-1:]) for v in ok])
            ch_first = self_chance([tuple(R[v][:1]) for v in ok])
            unit = [Am.get(v) == Rm.get(v) for v in ok if v in Am and v in Rm]
            rows.append((src, work, name, ref, len(ok), round(last, 4),
                         round(first, 4), round(body, 4),
                         round(kappa(last, ch_last), 4),
                         round(kappa(first, ch_first), 4), "aligned"))

    al = [x for x in rows if x[-1] == "aligned"]
    print(f"{len(rows)} witness pairs, each against the longest witness of its "
          f"work; {len(al)} of them share an edition's verse numbering\n")
    print(f"  {'source':10} {'pairs':>6} {'verses':>9} {'rhyme word':>11} "
          f"{'first word':>11} {'the body':>9}   above chance: rhyme / first")
    for s in ("kcd", "hvad", "parzival", "ambraser"):
        sel = [x for x in al if x[0] == s]
        if not sel:
            n = len([x for x in rows if x[0] == s])
            print(f"  {s:10} {n:6}       --   no shared verse numbering")
            continue
        a = np.array([[x[4], x[5], x[6], x[7], x[8], x[9]] for x in sel], float)
        print(f"  {s:10} {len(sel):6} {a[:, 0].sum():9,.0f} "
              f"{a[:, 1].mean():11.1%} {a[:, 2].mean():11.1%} "
              f"{a[:, 3].mean():9.1%}   {a[:, 4].mean():13.1%} / "
              f"{a[:, 5].mean():.1%}")
    if al:
        a = np.array([[x[5], x[6], x[7], x[8], x[9]] for x in al], float)
        print(f"\n  over the aligned pairs: the rhyme word survives the copy "
              f"{np.nanmean(a[:, 0]):.1%} of the time and the first word "
              f"{np.nanmean(a[:, 1]):.1%}, but corrected for what each "
              f"position's own vocabulary gives for nothing, the rhyme word "
              f"stands at {np.nanmean(a[:, 3]):.1%} and the first word at "
              f"{np.nanmean(a[:, 4]):.1%}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as fh:
        fh.write("source\twork\twitness\treference\tshared_verses\t"
                 "rhyme_word\tfirst_word\tbody\trhyme_above_chance\t"
                 "first_above_chance\tstatus\n")
        for x in rows:
            fh.write("\t".join(str(y) for y in x) + "\n")
    print(f"  wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
