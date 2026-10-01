# Ten copies of one poem, in the form the rest of the pipeline reads.
#
# WHY THIS CORPUS. Every other question in this study compares one poem with
# another. This one compares a poem with itself: the Nibelungenlied survives in
# ten complete manuscripts, and the editors of the tradition assign them to six
# redactions, each a reworking of the poem by somebody who was not its poet. If a
# redactor leaves a hand of his own in the statistical sense, then two poems
# reworked together would score as one author's whether or not one poet wrote
# them -- which is an alternative reading of every positive verdict in this
# study, and one these ten copies can test, because here the redactional
# relations are known independently of anything measured here.
#
# THE SOURCE. The transcriptions published with Kofler's synopsis, one PDF per
# manuscript, each a diplomatic text with the strophe numbers of the synopsis's
# common numbering. Those numbers are the alignment: a strophe of one manuscript
# and the same number in another are the same strophe of the poem, so what
# differs between them is the reworking and not the arrangement.
#
# TWO THINGS THE TEXT LAYER DOES THAT HAVE TO BE UNDONE.
#
#   A superscript vowel is broken across three lines -- the letter it sits on
#   ends one line, the vowel stands alone on the next, and the rest of the word
#   opens the third. Joined naively the word becomes two, and every manuscript
#   then looks equally strange; joined properly `o`, `u`, `ch` is `ouch` again.
#
#   The caesura is set as a run of spaces inside the long line, and the
#   apparatus, page numbers and running heads are not verse at all. Only lines
#   that fall under a strophe header are kept.
#
# WHAT IS AND IS NOT DONE HERE. The text is diplomatic, so its spelling is the
# scribe's and differs from copy to copy far more than the wording does. That
# difference is the reason this extraction stops at the surface: the comparison
# these texts are for has to be made on the lemma, and the lemma comes from the
# tagger, exactly as it does for every other witness in this corpus.
#
#   python medieval/extract_nibelungen.py
#
# Output: medieval/nibelungen_verses.jsonl

import json
import re
import sys
from pathlib import Path

from pypdf import PdfReader

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SRC = HERE / "raw" / "nibelungen"
OUT = HERE / "nibelungen_verses.jsonl"

# manuscript -> (shelfmark, the redaction it stands for)
BOOKS = {
    "A": ("Muenchen, Bayerische Staatsbibliothek, Cgm 34", "*A"),
    "B": ("St. Gallen, Stiftsbibliothek, Cod. 857", "*B"),
    "C": ("Karlsruhe, Badische Landesbibliothek, Cod. Don. 63", "*C"),
    "a": ("Cologny, Bibliotheca Bodmeriana, Cod. Bodm. 117", "*C"),
    "D": ("Muenchen, Bayerische Staatsbibliothek, Cgm 31", "*D"),
    "b": ("Berlin, Staatsbibliothek, Ms. germ. fol. 855", "*D"),
    "I": ("Berlin, Staatsbibliothek, Ms. germ. fol. 474", "*I"),
    "k": ("Wien, Oesterreichische Nationalbibliothek, Cod. 15478", "*I"),
    "d": ("Wien, Oesterreichische Nationalbibliothek, Cod. Ser. nova 2663",
          "*d"),
    "n": ("Berlin, Staatsbibliothek, Ms. germ. fol. 856", "-"),
}

# a strophe opens with the manuscript's own siglum and the synopsis's number;
# the number may carry a star, where the synopsis counts a strophe the reference
# text lacks, and a letter, where a manuscript adds one of its own
HEAD = re.compile(r"^\s*([A-Za-z])\s+(\*?\d+[A-Za-z]?)\s*$")
PAGE = re.compile(r"^\s*\d+\s*$")


def mend(lines):
    """put the words back together that the text layer broke into three

    A line holding one letter and nothing else is a superscript vowel: it
    belongs at the end of the line above, and the line below continues the same
    word rather than starting a new one.
    """
    out, joining = [], False
    for ln in lines:
        s = ln.strip()
        if len(s) == 1 and s.isalpha():
            if out:
                out[-1] = out[-1].rstrip() + s
            joining = True
            continue
        if joining and out:
            out[-1] += ln.lstrip()
            joining = False
        else:
            out.append(ln)
    return out


def read_book(path, siglum):
    """[(strophe, [long line, ...]), ...] for one manuscript"""
    lines = []
    for page in PdfReader(str(path)).pages:
        lines += (page.extract_text() or "").splitlines()
    out, cur = [], None
    for ln in mend(lines):
        m = HEAD.match(ln)
        if m:
            # a header for a DIFFERENT siglum is the synopsis naming another
            # manuscript's reading, which is not this manuscript's text
            cur = m.group(2) if m.group(1) == siglum else None
            if cur:
                out.append((cur, []))
            continue
        if cur and ln.strip() and not PAGE.match(ln):
            out[-1][1].append(re.sub(r"\s+", " ", ln).strip())
    return [(n, ls) for n, ls in out if ls]


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    n_all = t_all = 0
    with OUT.open("w", encoding="utf-8") as fh:
        for sig, (shelf, red) in BOOKS.items():
            f = SRC / (f"gr-{sig}_nib.pdf" if sig.isupper()
                       else f"kl-{sig}_nib.pdf")
            if not f.exists():
                print(f"  {sig}: {f.name} is not on disk")
                continue
            book = read_book(f, sig)
            verses = []
            for st, ls in book:
                for i, line in enumerate(ls, 1):
                    w = [x for x in line.split() if any(c.isalpha() for c in x)]
                    if w:
                        # the identifier carries the strophe, which is what
                        # aligns this line with the same line of another copy
                        verses.append({"n": f"{st}.{i}", "w": w})
            ntok = sum(len(v["w"]) for v in verses)
            n_all += len(verses)
            t_all += ntok
            fh.write(json.dumps({
                # The sigla of this tradition tell an upper case letter from a
                # lower one -- A and a are different books -- and a file system
                # that folds case does not, so three of the ten would silently
                # overwrite three others. The source's own division of the
                # copies into the greater and the lesser distinguishes them
                # without relying on case, so the name carries it.
                "sigle": f"NL_{'gr' if sig.isupper() else 'kl'}{sig}",
                "manuscript": sig, "shelfmark": shelf,
                "redaction": red, "author": "Anonym",
                "title": f"Nibelungenlied ({sig})",
                "strophes": len(book), "verses": verses},
                ensure_ascii=False) + "\n")
            print(f"  {sig:2} {red:3} {len(book):5,} strophes  "
                  f"{len(verses):6,} lines  {ntok:7,} words   {shelf[:44]}")
    print(f"\n{n_all:,} lines, {t_all:,} words -> {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
