# Is an author's genre difference partly an EDITION difference?
#
# About 60% of the German harvest is diplomatic transcription that preserves the
# Fraktur typography and the historical orthography of its source; the rest is
# modernised. Which of the two a text receives is a property of the EDITION
# zeno.org digitised, not of the author who wrote it. Long s and line-break
# hyphenation were normalised at harvest time because they break the tagger, but
# historical ORTHOGRAPHY was deliberately left in place: it is genuine language
# and part of the period signal.
#
# That leaves an untested confound. If an author's verse comes from a diplomatic
# edition and his prose from a modernised one, then a cross-genre comparison is
# partly a comparison of editorial practice, and the function words POSNoise
# keeps are exactly where the difference shows -- *seyn* for *sein*, *itzt* for
# *jetzt*, *ohnerachtet*, *dieweil*. The masked stream cannot tell an author's
# habit from his editor's convention.
#
# The probe is a set of orthographic markers whose modern spellings are ordinary:
# th before a vowel (Thal, Theil), -irt/-iren for -iert/-ieren, y for i (seyn,
# Meynung), doubled vowels (Saal-type), and c for k in Latinate stems (Cultur).
# Their RATE per 1,000 tokens is an edition fingerprint, not an author's style.
#
#   python experiments/measure_transcription.py
#
# Reads the masked banks directly, so it measures exactly the stream the scorer
# saw.

import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
MASKED = HERE.parent / "masked"
GENRES = {"prose": "german_tgproseall", "verse": "german_tgverseall",
          "drama": "german_tgdramaall"}

# Historical spellings whose modern counterpart is unremarkable. Matched on whole
# tokens so that a modern word merely containing the letters cannot fire.
MARKERS = re.compile(
    r"^(th(eil|al|ur|un|at|eur)\w*|seyn|sey|seynd|ihro|itzt|jtzt|ietzt"
    r"|ohnerachtet|dieweil|allhier|alldieweil|derjenige\w*|weyl|meyn\w*"
    r"|deyn\w*|freyheit\w*|frey|beyde\w*|bey|zwey\w*|drey\w*|neu\w*e[yi]t"
    r"|\w+irt|\w+iret|\w+iren|c(ultur|onzert|orrect)\w*)$", re.I)


def rate(path):
    """Historical-spelling markers per 1,000 tokens."""
    n = hit = 0
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            for t in line.rstrip("\n").split("\t"):
                if not t or not t[0].isalpha():
                    continue
                n += 1
                if MARKERS.match(t):
                    hit += 1
            if n > 400_000:
                break
    return (1000.0 * hit / n) if n else float("nan"), n


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    per = defaultdict(dict)
    for g, ds in GENRES.items():
        for f in sorted((MASKED / ds / "bank").glob("*.tsv")):
            stem = f.stem.split("_", 1)[1] if f.stem[:3].isdigit() else f.stem
            r, n = rate(f)
            if n >= 20_000:
                per[stem][g] = r

    three = {a: v for a, v in per.items() if len(v) == 3}
    print(f"{len(three)} authors with all three genres and enough text\n")
    print(f"{'author':30s} {'prose':>7s} {'verse':>7s} {'drama':>7s} "
          f"{'spread':>7s}  markers per 1,000 tokens")
    spreads = []
    for a in sorted(three, key=lambda k: -(max(three[k].values())
                                           - min(three[k].values()))):
        v = three[a]
        sp = max(v.values()) - min(v.values())
        spreads.append(sp)
        flag = "  <- editions differ" if sp > 3 else ""
        print(f"  {a[:28]:28s} {v['prose']:7.2f} {v['verse']:7.2f} "
              f"{v['drama']:7.2f} {sp:7.2f}{flag}")

    print(f"\nwithin-author spread across genres: median {np.median(spreads):.2f}, "
          f"max {max(spreads):.2f}")

    # The comparison that matters: is the difference BETWEEN an author's genres
    # of the same order as the difference between authors? If it is, genre and
    # edition are not separable in this corpus.
    allv = [x for v in three.values() for x in v.values()]
    print(f"between-author spread (same measure): "
          f"IQR {np.percentile(allv, 25):.2f}-{np.percentile(allv, 75):.2f}, "
          f"full {min(allv):.2f}-{max(allv):.2f}")
    big = sum(1 for s in spreads if s > 3)
    print(f"\n{big}/{len(spreads)} authors differ by more than 3 per 1,000 across "
          f"their own genres,\nwhich is an EDITION difference sitting inside what "
          f"a cross-genre test reads as genre.")


if __name__ == "__main__":
    main()
