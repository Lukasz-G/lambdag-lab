# Hartmann von Aue digital (Heidelberg) -> verses and rhyme words, one record
# per witness of each work.
#
# WHAT THE SOURCE IS. The editions of Iwein, Gregorius and Der arme Heinrich
# are served through the Distributed Text Services API of heiEDITIONS
# (CC BY-SA 4.0), one TEI document per witness. Each document carries two
# readings of every word side by side: the transcription of the manuscript
# (<orig>, with the scribe's abbreviation marks as <am>) and the editors'
# regularised text (<reg>, expansions as <ex>). Both are read here and
# emitted as two layers of the same verse, token for token: the diplomatic
# layer for tagging and everything that follows it, the regularised layer
# so that the effect of the editors' regularisation on the same witness can be
# measured rather than assumed.
#
# WHAT IS TAKEN AND WHAT IS REFUSED. Abbreviations are expanded in both
# layers, since an unexpanded abbreviation cannot be lemmatised; where the
# transcription encodes an abbreviation as an <orig> consisting only of a
# mark, the expansion is taken there too. The scribe's own corrections are
# read as he left them (<add> in, <del> and <surplus> out); the editors'
# corrections of slips are not applied to the diplomatic layer (<sic> in,
# <corr> out) but are applied to the regularised one. Letters the editors
# supply in damaged places are taken, since the alternative is a broken word.
# A word with an unfilled gap is dropped from both layers and counted. Line
# breaks inside a word (<lb break="no">) join. Editorial notes, headings and
# the variant apparatus never enter the text. Punctuation is consumed as
# segmentation and then discarded; the verse is the <l> element.
#
#   python medieval/extract_hvad.py
#
# Output: medieval/raw/hvad/hvad_verses.jsonl
#         one record per witness: work, siglum, author, title, manuscript,
#         verses (diplomatic), verses_reg (regularised), verse_ids, rhymes

import json
import re
import sys
import unicodedata
import xml.etree.ElementTree as ET
from pathlib import Path

HERE = Path(__file__).resolve().parent
RAW = HERE / "raw" / "hvad"
OUT = RAW / "hvad_verses.jsonl"
TEI = "{http://www.tei-c.org/ns/1.0}"
HEI = "{https://digi.ub.uni-heidelberg.de/schema/tei/heiEDITIONS}"
WORKS = {"Iwein": "hartmann_von_aue__iwein", "Gregorius": "hartmann_von_aue__gregorius",
         "ArmerHeinrich": "hartmann_von_aue__der_arme_heinrich"}
PUNCT = "·.,;:!?/()[]{}«»\"'„“”‚‘’-–—~|*"


def tag(e):
    return e.tag.split("}")[-1]


def only_marks(s):
    s = s.strip()
    return bool(s) and all(unicodedata.combining(c) or unicodedata.category(c).startswith("M")
                           for c in s)


class Render:
    """collects the diplomatic and regularised text of one verse"""

    def __init__(self):
        self.d, self.r = [], []
        self.gap = False

    def both(self, s):
        self.d.append(s); self.r.append(s)

    def walk(self, e):
        t = tag(e)
        if t in ("note", "head", "figure", "fw", "docTitle", "milestone", "anchor", "ptr"):
            return
        if t == "c":
            self.both(" ")
        elif t == "pc":
            self.both(" ")
        elif t == "lb":
            self.both("" if e.get("break") == "no" else " ")
        elif t == "gap":
            self.gap = True
        elif t == "choice":
            self.choice(e)
            return
        elif t == "subst":
            for ch in e:
                if tag(ch) == "add":
                    self.walk(ch)
            return
        elif t in ("del", "surplus", "am", "corr", "reg"):
            return          # only reachable outside their normal parents; drop
        else:
            if e.text:
                self.both(e.text)
            for ch in e:
                self.walk(ch)
        # tail text belongs to the parent and is emitted by the caller

    def children(self, e):
        for ch in e:
            self.walk(ch)
            if ch.tail:
                self.both(ch.tail)

    def choice(self, e):
        kids = {tag(ch): ch for ch in e}
        if "am" in kids or "ex" in kids:
            ex = kids.get("ex")
            if ex is not None:
                self.both(text_of(ex))
        elif "orig" in kids or "reg" in kids:
            orig, reg = kids.get("orig"), kids.get("reg")
            o = text_of(orig) if orig is not None else ""
            r = text_of(reg) if reg is not None else ""
            has_gap = orig is not None and orig.find(TEI + "gap") is not None
            sup = kids.get("supplied")
            if has_gap and sup is not None:
                self.both(text_of(sup))
            elif only_marks(o) or (o == "" and orig is not None and reg is not None and not has_gap):
                self.both(r)          # an abbreviation encoded as orig/reg
            else:
                self.d.append(o); self.r.append(r)
                if has_gap and sup is None:
                    self.gap = True
        elif "sic" in kids or "corr" in kids:
            sic, corr = kids.get("sic"), kids.get("corr")
            self.d.append(text_of(sic) if sic is not None else "")
            self.r.append(text_of(corr) if corr is not None else "")
        elif "supplied" in kids:
            self.both(text_of(kids["supplied"]))
        else:
            for ch in e:
                self.walk(ch)


def text_of(e):
    """flat text of an element with the same refusals as the walker"""
    if e is None:
        return ""
    r = Render()
    if e.text:
        r.both(e.text)
    r.children(e)
    return "".join(r.d)


def tokens(s):
    s = unicodedata.normalize("NFC", s)
    out = []
    for t in s.split():
        t = t.strip(PUNCT)
        if any(c.isalpha() for c in t):
            out.append(t)
    return out


def verse(l):
    """(diplomatic tokens, regularised tokens, dropped-word count) of one <l>"""
    dip, reg, dropped = [], [], 0
    # word by word, so that a gap drops exactly the word it sits in
    for w in l.iter(TEI + "w"):
        r = Render()
        if w.text:
            r.both(w.text)
        r.children(w)
        if r.gap:
            dropped += 1
            continue
        dt, rt = tokens("".join(r.d)), tokens("".join(r.r))
        dip += dt; reg += rt if rt else dt
    return dip, reg, dropped


def read(path, work):
    root = ET.parse(path).getroot()
    head = root.find(TEI + "teiHeader")
    title = " / ".join(t.text or "" for t in head.iter(TEI + "title") if t.text)[:120]
    ms = head.find(".//" + TEI + "msIdentifier")
    manuscript = ", ".join(x.text.strip() for x in ms.iter()
                           if tag(x) in ("settlement", "repository", "idno", "placeName", "orgName")
                           and x.text and x.text.strip() and not x.text.startswith("http")) if ms is not None else "-"
    # sigla differ by case only (A and a are different manuscripts) and the
    # filesystem does not, so lowercase sigla are stored with the suffix _lc
    siglum = path.stem[:-3] if path.stem.endswith("_lc") else path.stem
    rec = dict(work=WORKS[work], siglum=siglum, author="Hartmann von Aue", title=title,
               manuscript=manuscript[:160], verses=[], verses_reg=[], verse_ids=[], rhymes=[])
    dropped = 0
    body = root.find(".//" + TEI + "body")
    for l in body.iter(TEI + "l"):
        d, r, k = verse(l)
        dropped += k
        if not d:
            continue
        rec["verses"].append(d); rec["verses_reg"].append(r)
        rec["verse_ids"].append(l.get(HEI + "altN") or l.get("n") or "?")
        rec["rhymes"].append(d[-1])
    return rec, dropped


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    recs = []
    print("%-14s %-4s %7s %7s %6s  %s" % ("work", "sig", "verses", "tokens", "gapped", "manuscript"))
    for work in WORKS:
        d = RAW / work
        if not d.is_dir():
            continue
        for f in sorted(d.glob("*.xml")):
            try:
                rec, dropped = read(f, work)
            except ET.ParseError as e:
                print("%-14s %-4s PARSE ERROR %s" % (work, f.stem, e)); continue
            recs.append(rec)
            print("%-14s %-4s %7d %7d %6d  %s" % (work, rec["siglum"], len(rec["verses"]),
                                                  sum(map(len, rec["verses"])), dropped, rec["manuscript"][:60]))
    with OUT.open("w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("%d witnesses -> %s" % (len(recs), OUT))


if __name__ == "__main__":
    main()
