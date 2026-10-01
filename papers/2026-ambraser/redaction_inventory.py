# How much text actually separates one redaction of a poem from another.
#
# THE QUESTION THIS ANSWERS, AND WHY IT COMES FIRST. A redactor reworks a poem,
# and a reworking may carry its own hand in the statistical sense; if it does,
# then two poems that were reworked together would score as one author's whether
# or not one poet wrote them, which is an alternative reading of every positive
# verdict in this study. The way to test it is to mask what the redactions share
# and score only what differs -- but that is worth building only if what differs
# is long enough to carry a question at all. The floors measured on this corpus
# are about 2,000 masked tokens for a question about a hand and nearer 5,000 for
# a question about an author, so a redaction whose own material falls far below
# them cannot be tested however the design is arranged.
#
# THE MATERIAL. The transcriptions published with Kofler's synopsis set the
# complete manuscripts under one strophe numbering, so the alignment this test
# would otherwise have to compute is published: a strophe of one manuscript and
# the same number in another are the same strophe of the poem, and what differs
# between them is the reworking rather than the arrangement.
#
# TWO WAYS A REDACTOR CAN SHOW HIMSELF, AND THEY ARE NOT EQUAL. He can add
# strophes, and he can reword the ones every copy carries. Both are counted here,
# separately, because the first turns out to be almost nothing and the second
# almost everything.
#
# SURFACE AND LEMMA, WHICH IS THE WHOLE POINT. These are diplomatic
# transcriptions, so a word differs between copies when the scribe spelt it
# differently as readily as when the redactor changed it. A count on the written
# forms therefore mixes the two, and mostly measures the first: the scribal toll
# on this corpus is 0.055 on the lemma stream against 0.151 on the scribe's
# spelling. Both streams are reported so that the difference between them is
# visible instead of assumed -- the lemma count is the one that bears on the
# question, and it is available once `tag_mhdbdb.py` has run over
# `hydra_in_nibelungen`.
#
#   python medieval/extract_nibelungen.py
#   python medieval/redaction_inventory.py
#
# Output: medieval/kc_reuse/redaction_inventory.tsv, and the tables printed

import csv
import json
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SRC = HERE / "nibelungen_verses.jsonl"
TAGGED = HERE / "hydra_out_nibelungen"
OUT = ROOT / "medieval" / "kc_reuse" / "redaction_inventory.tsv"


def strophe_of(verse_id):
    return verse_id.rsplit(".", 1)[0]


def load_surface():
    """{manuscript: (redaction, {strophe: [word, ...]})} as the scribes wrote"""
    out = {}
    for line in SRC.open(encoding="utf-8"):
        r = json.loads(line)
        st = {}
        for v in r["verses"]:
            st.setdefault(strophe_of(v["n"]), []).extend(
                w.lower() for w in v["w"])
        out[r["manuscript"]] = (r["redaction"], st)
    return out


def load_lemma():
    """the same, on the tagger's lemma, or {} until the tagging has been done

    The tagged file carries one token per line and no verse boundaries, so the
    lines are counted back onto the strophes in the order the extraction records
    them -- the same order they were written to the tagger in.
    """
    if not TAGGED.exists():
        return {}
    out = {}
    for line in SRC.open(encoding="utf-8"):
        r = json.loads(line)
        f = TAGGED / f"{r['sigle']}.txt"
        if not f.exists():
            continue
        lemmas = [l.split("\t")[1] for l in
                  f.read_text(encoding="utf-8").splitlines()
                  if "\t" in l and not l.startswith("@")]
        flat = [(strophe_of(v["n"]), w) for v in r["verses"] for w in v["w"]]
        if len(lemmas) != len(flat):
            print(f"  {r['manuscript']}: {len(lemmas):,} tagged tokens against "
                  f"{len(flat):,} written; skipped")
            continue
        st = {}
        for (s, _), lem in zip(flat, lemmas):
            st.setdefault(s, []).append(lem.lower())
        out[r["manuscript"]] = (r["redaction"], st)
    return out


def divergence(books, label):
    """over the strophes each pair shares, words of one the other lacks"""
    sig = sorted(books, key=lambda s: (books[s][0], s))
    print(f"\n{label}: over the strophes each PAIR shares, words of the first "
          f"that its\npartner's version of the same strophe does not have\n")
    print("     " + "".join(f"{s:>7}" for s in sig))
    same, other, rows = [], [], []
    for x in sig:
        cells = []
        for y in sig:
            if x == y:
                cells.append(f"{'-':>7}")
                continue
            rx, sx = books[x]
            ry, sy = books[y]
            shared = sx.keys() & sy.keys()
            d = sum(sum((Counter(sx[n]) - Counter(sy[n])).values())
                    for n in shared)
            tot = sum(len(sx[n]) for n in shared) or 1
            (same if rx == ry else other).append(d / tot)
            rows.append(dict(stream=label, first=x, second=y,
                             first_redaction=rx, second_redaction=ry,
                             shared_strophes=len(shared), words=tot,
                             divergent=d, share=round(d / tot, 4)))
            cells.append(f"{d / tot:6.1%}")
        print(f"  {x:2} " + "".join(cells))

    def med(v):
        return sorted(v)[len(v) // 2] if v else float("nan")
    print(f"\n  within one redaction  {med(same):6.1%} of words differ "
          f"({len(same)} pairs)")
    print(f"  across redactions     {med(other):6.1%} of words differ "
          f"({len(other)} pairs)")
    return rows, med(same), med(other)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    surf = load_surface()
    print(f"{len(surf)} manuscripts\n")
    print(f"{'ms':3} {'red':5} {'strophes':>9} {'words':>9} {'its own':>9} "
          f"{'words':>8}")
    carried = Counter()
    for _, st in surf.values():
        for n in st:
            carried[n] += 1
    rows = []
    for sig, (red, st) in sorted(surf.items()):
        own = [n for n in st if carried[n] == 1]
        print(f"{sig:3} {red:5} {len(st):9,} {sum(len(v) for v in st.values()):9,}"
              f" {len(own):9,} {sum(len(st[n]) for n in own):8,}")
    print("\n  `its own` counts the strophes no other manuscript here carries: "
          "the material a\n  redactor ADDED, and the upper bound on what a "
          "comparison could use if adding\n  were the only thing he did.")

    rows += divergence(surf, "the scribes' spelling")[0]
    lem = load_lemma()
    if lem:
        rows += divergence(lem, "the tagger's lemma")[0]
    else:
        print("\n  the lemma stream is not built yet: run\n"
              "    python medieval/tag_mhdbdb.py --batch 1 "
              "--inputs medieval/hydra_in_nibelungen "
              "--outputs medieval/hydra_out_nibelungen")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]), delimiter="\t",
                           lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    print(f"\n  wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
