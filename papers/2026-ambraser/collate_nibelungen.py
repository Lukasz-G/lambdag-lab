# Align ten copies of one poem by what they say, not by how they are numbered.
#
# WHY THE NUMBERS WILL NOT DO. Each transcription carries its own manuscript's
# strophe numbers, and the manuscripts do not agree on what strophe one is:
# B opens at `Ez wuohs in Burgonden` where A and C have a strophe before it, so
# A's second strophe is B's first, and from there the discrepancy grows as each
# copy adds and loses strophes of its own. Comparing equal numbers therefore
# compares unequal strophes, and it does so invisibly -- it returns a number,
# and the number is the distance between two arbitrary passages of the poem,
# which for this text is about 78 per cent of the words whichever two you take.
# Five of the ten happen to share a numbering and align at no shift at all; no
# constant shift rescues the other five, because the discrepancy is not constant.
#
# HOW THE ALIGNMENT IS FOUND. Rare wording is a better index than numbering. A
# strophe is indexed by its lemma trigrams; a strophe of one copy is matched to
# the strophe of another that shares most of them; and the matches are then cut
# down to the longest run that keeps the poem's order, so that a chance agreement
# between distant passages cannot pull the alignment out of sequence. What
# survives is a collation: which strophe of this copy is which strophe of that
# one, established from the text and from nothing else.
#
# WHAT IS MEASURED ON IT. Over the strophes a pair genuinely shares, the share of
# one copy's lemmas that the other's version of the SAME strophe does not have.
# That is the reworking, and the question it answers is whether the redactions
# the editors of this tradition distinguish are separated by more of it than two
# copies of one redaction are -- which is what a redactor with a hand of his own
# would imply, and what every positive verdict in the Ambraser study would then
# have to be read against.
#
#   python medieval/extract_nibelungen.py
#   python medieval/tag_mhdbdb.py --batch 1 \
#       --inputs medieval/hydra_in_nibelungen --outputs medieval/hydra_out_nibelungen
#   python medieval/collate_nibelungen.py
#
# Output: medieval/kc_reuse/nibelungen_collation.tsv, and the tables printed

import bisect
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SRC = HERE / "nibelungen_verses.jsonl"
TAGGED = HERE / "hydra_out_nibelungen"
OUT = ROOT / "medieval" / "kc_reuse" / "nibelungen_collation.tsv"

MIN_SHARED = 3      # trigrams a match must rest on before it is believed
K = 3


def load():
    """{manuscript: (redaction, [ [lemma, ...] per strophe, in order ])}"""
    out = {}
    for line in SRC.open(encoding="utf-8"):
        r = json.loads(line)
        f = TAGGED / f"{r['sigle']}.txt"
        if not f.exists():
            print(f"  {r['manuscript']}: not tagged yet")
            continue
        lem = [l.split("\t")[1].lower() for l in
               f.read_text(encoding="utf-8").splitlines()
               if "\t" in l and not l.startswith("@")]
        flat = [(v["n"].rsplit(".", 1)[0], w) for v in r["verses"]
                for w in v["w"]]
        if len(lem) != len(flat):
            print(f"  {r['manuscript']}: {len(lem):,} tagged against "
                  f"{len(flat):,} written; skipped")
            continue
        order, st = [], {}
        for (s, _), L in zip(flat, lem):
            if s not in st:
                st[s] = []
                order.append(s)
            st[s].append(L)
        out[r["manuscript"]] = (r["redaction"], [st[s] for s in order], order)
    return out


def trigram_index(strophes):
    idx = defaultdict(list)
    for j, w in enumerate(strophes):
        for t in {tuple(w[i:i + K]) for i in range(len(w) - K + 1)}:
            idx[t].append(j)
    return idx


def candidates(xs, idx):
    """(i, j, shared trigrams) for each strophe of x, best partner in y only"""
    out = []
    for i, w in enumerate(xs):
        c = Counter()
        for t in {tuple(w[k:k + K]) for k in range(len(w) - K + 1)}:
            for j in idx.get(t, ()):
                c[j] += 1
        if not c:
            continue
        j, s = c.most_common(1)[0]
        if s >= MIN_SHARED:
            out.append((i, j, s))
    return out


def in_order(pairs):
    """the longest run of matches that keeps the poem's order

    A copy may repeat a formula, so a strophe can match a distant one by
    accident; keeping only a rising sequence of partners throws those away,
    because a scribe does not reorder the poem even when he cuts and adds.
    """
    if not pairs:
        return []
    tails, back, idx = [], [-1] * len(pairs), []
    for p, (_, j, _) in enumerate(pairs):
        k = bisect.bisect_left(tails, j)
        if k == len(tails):
            tails.append(j)
            idx.append(p)
        else:
            tails[k] = j
            idx[k] = p
        back[p] = idx[k - 1] if k else -1
    out, p = [], idx[-1]
    while p != -1:
        out.append(pairs[p])
        p = back[p]
    return out[::-1]


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    books = load()
    sig = sorted(books, key=lambda s: (books[s][0], s))
    print(f"{len(books)} manuscripts\n")
    index = {s: trigram_index(books[s][1]) for s in sig}

    rows = []
    cover = {}
    for x in sig:
        for y in sig:
            if x == y:
                continue
            xs, ys = books[x][1], books[y][1]
            pairs = in_order(candidates(xs, index[y]))
            d = tot = 0
            for i, j, _ in pairs:
                d += sum((Counter(xs[i]) - Counter(ys[j])).values())
                tot += len(xs[i])
            cover[(x, y)] = (len(pairs), len(xs), d / tot if tot else 1.0)
            rows.append(dict(first=x, second=y,
                             first_redaction=books[x][0],
                             second_redaction=books[y][0],
                             strophes_first=len(xs), aligned=len(pairs),
                             coverage=round(len(pairs) / len(xs), 4),
                             lemmas=tot, divergent=d,
                             share=round(d / tot, 4) if tot else ""))
        print(f"  collated {x}", flush=True)

    print("\nhow much of the first copy the collation could place in the "
          "second\n")
    print("     " + "".join(f"{s:>7}" for s in sig))
    for x in sig:
        print(f"  {x:2} " + "".join(
            f"{'-':>7}" if x == y else f"{cover[(x, y)][0] / cover[(x, y)][1]:6.0%} "
            for y in sig))

    print("\nover the strophes the collation pairs: lemmas of the first the "
          "second lacks\n")
    print("     " + "".join(f"{s:>7}" for s in sig))
    same, other = [], []
    for x in sig:
        cells = []
        for y in sig:
            if x == y:
                cells.append(f"{'-':>7}")
                continue
            v = cover[(x, y)][2]
            (same if books[x][0] == books[y][0] else other).append(v)
            cells.append(f"{v:6.1%}")
        print(f"  {x:2} " + "".join(cells))

    def med(v):
        return sorted(v)[len(v) // 2] if v else float("nan")
    print(f"\n  two copies of ONE redaction   {med(same):6.1%} of lemmas differ "
          f"({len(same)} pairs)")
    print(f"  copies of DIFFERENT redactions {med(other):6.1%} of lemmas differ "
          f"({len(other)} pairs)")
    print(f"  the gap between them           {med(other) - med(same):+6.1%}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]), delimiter="\t",
                           lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    print(f"\n  wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
