# Repair words broken across a ruled line, using the reference corpus as lexicon.
#
# THE PROBLEM. A diplomatic transcription preserves the manuscript's ruling,
# and a scribe who reached the end of a line mid-word simply continued on the
# next one, commonly with no hyphen to say so. Concatenating the ruled lines
# therefore yields 'mache ten' for macheten and 'lieg ent' for liegent. Measured
# against the Referenzkorpus Mittelhochdeutsch's own transcription of the Vorau
# manuscript, some 1.4% of tokens are involved in such a break -- a small rate
# that falls in the worst possible place, since a verse-final fragment corrupts
# precisely the rhyme word on which the rhyme-based analysis depends.
#
# WHY A LEXICON AND NOT THE TAGGER. The tagger locates a discontinuous unit
# more often than not but resolves the whole annotation of one correctly little
# better than half the time, so it is used, where it is used at all, to find the
# break rather than to mend it. The reference corpus supplies 290,976 attested
# surface forms, which is what mending requires.
#
# THE RULE, CHOSEN BY MEASUREMENT. Four rules were tried against the gold
# alignment. Repairing a junction whenever the concatenation is attested at all
# mends the most breaks and is nonetheless the worst of them: it manufactures
# 115 false joins and agreement with the gold transcription FALLS, the scribe's
# genuine word pairs being welded into compounds he never wrote. Demanding that
# one half be unattested is too timid at the other extreme, since 'mache' and
# 'ten' are both words and 'macheten' is thereby left broken. What works is the
# middle condition retained here: the concatenation must be attested at least
# twice AND be no rarer than the rarer of the two halves. That mends 42% of the
# breaks (2,468 -> 1,424) for nineteen additional false joins, and carries
# agreement with the reference transcription from 88.5% to 90.7%.
#
# MEASURED AGAINST M121Y ONLY. ReM's M121V is an excerpt 92.3% contained in
# M121Y, so the two together are not 98,723 tokens of independent gold but
# ~85,897 with an eighth repeated; and every Vorau file must leave the lexicon,
# not merely the one named V. Excluding M121V alone leaves M121y1/M121y2 -- the
# same text -- to supply the very joined forms under test, which inflated an
# earlier version of this measurement.
#
#   python medieval/join_splits.py --measure     # repair rate against ReM gold
#
# Used by extract_kcd.py; the lexicon is built once and cached.

import json
import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
REM_DIR = Path(r"D:\Corpora\MHD")
CACHE = HERE / "raw" / "kcd" / "rem_lexicon.json"
MIN_JOINED = 2          # the concatenation must be attested more than once
# every transcription of the Vorau manuscript, for measurements against it
VORAU = ("M121V", "M121y1", "M121y2")


def fold(s):
    """Orthographic folding for comparison across transcription conventions."""
    s = unicodedata.normalize("NFD", s.lower())
    s = "".join(c for c in s if not unicodedata.combining(c))
    for a, b in (("\u017f", "s"), ("v", "u"), ("j", "i"), ("y", "i"),
                 ("\u00e6", "e"), ("\u0153", "o"), ("\u00df", "ss")):
        s = s.replace(a, b)
    return re.sub(r"[^a-z]", "", s)


def build_lexicon(exclude=()):
    """Folded surface-form counts from the reference corpus.

    Texts named in `exclude` are omitted, so that a repair measured against one
    of them is not measured against its own vocabulary.
    """
    counts = Counter()
    for p in sorted(REM_DIR.glob("*.txt")):
        if any(x in p.name for x in exclude):
            continue
        for line in open(p, encoding="utf-8", errors="replace"):
            if not line.strip():
                continue
            w = fold(line.split("\t")[0])
            if w:
                counts[w] += 1
    return counts


def load_lexicon(exclude=(), refresh=False):
    key = "|".join(sorted(exclude))
    if CACHE.exists() and not refresh:
        d = json.loads(CACHE.read_text(encoding="utf-8"))
        if d.get("exclude") == key:
            return Counter(d["counts"])
    counts = build_lexicon(exclude)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_text(json.dumps({"exclude": key, "counts": counts},
                                ensure_ascii=False), encoding="utf-8")
    return counts


def join_chunks(chunks, lex):
    """Concatenate ruled lines, mending words the ruling broke.

    Only the junction between two lines is ever considered: a break can occur
    nowhere else, and testing every adjacent pair of words would invite the
    invention of compounds throughout the text.
    """
    out, joins = [], 0
    for s in chunks:
        if not out:
            out.append(s)
            continue
        prev = out[-1]
        if prev.endswith(("-", "\u2010", "=")):      # the scribe said so
            out[-1] = prev[:-1] + s
            joins += 1
            continue
        a = prev.split()[-1] if prev.split() else ""
        b = s.split()[0] if s.split() else ""
        fa, fb = fold(a), fold(b)
        if fa and fb:
            merged = lex.get(fa + fb, 0)
            if merged >= MIN_JOINED and merged >= min(lex.get(fa, 0), lex.get(fb, 0)):
                # the two halves become one word; whatever followed on the new
                # line carries on as that line's remainder
                out[-1] = prev.rstrip() + b
                rest = s.split(None, 1)
                out.append(rest[1] if len(rest) > 1 else "")
                joins += 1
                continue
        out.append(s)
    return [x for x in out if x.strip()], joins


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--measure", action="store_true")
    args = ap.parse_args()
    lex = load_lexicon(exclude=VORAU)
    print(f"lexicon: {len(lex):,} folded surface types "
          f"(reference corpus less the text under test)")
    if args.measure:
        from measure_join import measure
        measure(lex)


if __name__ == "__main__":
    main()
