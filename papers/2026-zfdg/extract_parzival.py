# Parzival digital (Bern) -> verses and rhyme words, one record per witness.
#
# WHAT THE SOURCE IS. The Bern project's static API serves one TEI file per
# witness with the diplomatic transcription of that manuscript or fragment
# (CC BY-NC-SA 4.0; analysis only, no redistribution of the texts). Every
# verse is an <l> whose identifier carries the edition's Dreissiger-and-verse
# count, so witnesses align on the same numbering without collation. Lines
# whose identifier carries a suffix are marginalia and extra lines and are
# left out.
#
# WHAT IS TAKEN AND WHAT IS REFUSED. Abbreviations are expanded (<ex> in,
# <am> out). The scribe's corrections are read as he left them (<add> in,
# <del> out). Letters the editors supply are taken; a word with an unfilled
# gap is dropped and counted. The de-ligature glyph is resolved to its
# letters; other declared glyphs (line fillers, the finis mark) are dropped.
# Marginal notes and editorial notes never enter the text. Punctuation is
# consumed as segmentation and then discarded.
#
#   python medieval/extract_parzival.py
#
# Output: medieval/raw/parzival/parzival_verses.jsonl
#         one record per witness: work, siglum, author, title, manuscript,
#         verses, verse_ids ("Dreissiger.verse"), rhymes

import json
import re
import sys
import unicodedata
import xml.etree.ElementTree as ET
from pathlib import Path

HERE = Path(__file__).resolve().parent
RAW = HERE / "raw" / "parzival"
OUT = RAW / "parzival_verses.jsonl"
TEI = "{http://www.tei-c.org/ns/1.0}"
XMLID = "{http://www.w3.org/XML/1998/namespace}id"
PUNCT = "·.,;:!?/()[]{}«»\"'„“”‚‘’-–—~|*"
GLYPH = {"#deligatur": "de"}


def tag(e):
    return e.tag.split("}")[-1]


class Render:
    def __init__(self):
        self.out = []
        self.gap = False

    def walk(self, e):
        t = tag(e)
        if t in ("note", "del", "am", "surplus", "fw", "milestone", "anchor", "ref"):
            return
        if t == "gap":
            self.gap = True
            return
        if t == "g":
            self.out.append(GLYPH.get(e.get("ref", ""), ""))
            return
        if t == "lb":
            self.out.append("" if e.get("break") == "no" else " ")
            return
        if t == "choice":
            kids = {tag(ch): ch for ch in e}
            if "ex" in kids:
                self.walk(kids["ex"])
            elif "corr" in kids and "sic" in kids:
                self.walk(kids["sic"])
            elif "reg" in kids and "orig" in kids:
                self.walk(kids["orig"])
            else:
                for ch in e:
                    self.walk(ch)
            return
        if t == "subst":
            for ch in e:
                if tag(ch) == "add":
                    self.walk(ch)
            return
        if e.text:
            self.out.append(e.text)
        for ch in e:
            self.walk(ch)
            if ch.tail:
                self.out.append(ch.tail)


def tokens(s):
    s = unicodedata.normalize("NFC", s)
    out = []
    for t in s.split():
        t = t.strip(PUNCT)
        if any(c.isalpha() for c in t):
            out.append(t)
    return out


def read(path):
    root = ET.parse(path).getroot()
    head = root.find(TEI + "teiHeader")
    title = " / ".join(t.text or "" for t in head.iter(TEI + "title") if t.text)[:120]
    ms = head.find(".//" + TEI + "msIdentifier")
    manuscript = " ".join(x.text.strip() for x in ms.iter() if x.text and x.text.strip()) if ms is not None else "-"
    sig = path.stem
    rec = dict(work="wolfram_von_eschenbach__parzival", siglum=sig, author="Wolfram von Eschenbach",
               title=title, manuscript=manuscript[:160], verses=[], verse_ids=[], rhymes=[])
    gapped = 0
    idpat = re.compile(r"^[A-Za-z0-9]+_(\d+)\.(\d+)$")
    for l in root.find(".//" + TEI + "body").iter(TEI + "l"):
        m = idpat.match(l.get(XMLID, ""))
        if not m:
            continue
        r = Render()
        r.walk(l)
        toks = tokens("".join(r.out))
        if r.gap:
            gapped += 1          # a gap somewhere in the verse: keep the verse, count it
        if not toks:
            continue
        rec["verses"].append(toks)
        rec["verse_ids"].append("%s.%s" % (m.group(1), m.group(2)))
        rec["rhymes"].append(toks[-1])
    return rec, gapped


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    recs = []
    print("%-5s %7s %7s %6s  %s" % ("sig", "verses", "tokens", "gapped", "manuscript"))
    for f in sorted((RAW / "tei").glob("*.xml"), key=lambda p: (len(p.stem), p.stem)):
        if f.stem.startswith("syn"):
            continue
        try:
            rec, gapped = read(f)
        except ET.ParseError as e:
            print("%-5s PARSE ERROR %s" % (f.stem, e)); continue
        recs.append(rec)
        print("%-5s %7d %7d %6d  %s" % (rec["siglum"], len(rec["verses"]), sum(map(len, rec["verses"])),
                                         gapped, rec["manuscript"][:70]))
    with OUT.open("w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("%d witnesses -> %s" % (len(recs), OUT))


if __name__ == "__main__":
    main()
