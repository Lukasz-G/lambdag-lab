# Kaiserchronik digital -> verses and rhyme words, one record per witness.
#
# WHAT THE SOURCE IS. The heiEDITIONS transcriptions are diplomatic
# <sourceDoc> documents: the unit of markup is <line>, and a <line> is a ruled
# line of the manuscript, not a verse. Early Middle High German verse was
# written continuously across the ruling, so the Vorau witness carries 13,430
# ruled lines for a text of some 17,283 verses. The verse boundary is instead
# carried by the medieval point -- the Reimpunkt the scribe set at the end of
# each verse -- which in this encoding appears either as a literal full stop or
# as <g ref="#char_pe">, the punctus elevatus, whose <glyph> declaration maps
# it to a full stop. A transcription stripped of its tags loses the second kind
# and with it a part of the verse structure, so the glyph table is resolved
# here rather than ignored.
#
# Punctuation is excluded from the analytical alphabet throughout this work,
# medieval pointing being a scribal and editorial matter rather than an
# authorial one. It is nonetheless the only carrier of verse structure in this
# material, and is accordingly consumed as segmentation and then discarded: the
# emitted verses and rhyme words contain none of it.
#
# WHAT IS REFUSED. <note> holds the editors' own prose, in German and English,
# and never enters the text. <choice> offers the scribe's abbreviation beside
# the editors' expansion; the expansion is taken, since what follows is
# lemmatisation and an unexpanded abbreviation cannot be lemmatised. A word
# broken across two ruled lines is rejoined at the hyphen.
#
#   python medieval/extract_kcd.py
#   python medieval/extract_kcd.py --abbr        # diplomatic reading instead
#
# Output: medieval/raw/kcd/kcd_verses.jsonl
#         one record per witness: siglum, recension, title, verses, rhymes

import argparse
import json
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE / "raw" / "kcd"
OUT = SRC / "kcd_verses.jsonl"
TEI = "{http://www.tei-c.org/ns/1.0}"

# A verse ends at the point. Everything here resolves to one of these before
# segmentation; nothing here survives into the emitted text.
VERSE_END = ".!?;"
# survives depointing (alphabetic), then is stripped from the emitted verse
GAPMARK = "xxlacunaxx"
DROP = {TEI + "note", TEI + "teiHeader"}


def local(tag):
    return tag.rsplit("}", 1)[-1]


def glyph_map(root):
    """xml:id of a <glyph> -> its standard character equivalent.

    The transcriptions encode special sorts (ligatures, the punctus elevatus,
    the r rotunda) as <g ref="#id"/>, with a declaration in the header giving
    the modern equivalent. Without this table the punctus elevatus vanishes and
    roughly a sixth of the verse boundaries vanish with it.
    """
    out = {}
    for g in root.iter(TEI + "glyph"):
        gid = g.get("{http://www.w3.org/XML/1998/namespace}id")
        if not gid:
            continue
        std = ""
        for m in g.findall(TEI + "mapping"):
            if m.get("ana") == "hc:CharacterEquivalentStandard" and m.text:
                std = m.text
                break
        out[gid] = std
    return out


def text_of(el, gmap, want_expan=True):
    """Readable text of one element, apparatus resolved."""
    if el.tag in DROP:
        return ""
    parts = []
    tag = local(el.tag)
    if tag == "choice":
        pick = "expan" if want_expan else "abbr"
        for ch in el:
            if local(ch.tag) == pick:
                parts.append(text_of(ch, gmap, want_expan))
        return "".join(parts) + (el.tail or "")
    if tag == "g":
        ref = (el.get("ref") or "").lstrip("#")
        # a mapped glyph yields its standard equivalent; an unmapped one its
        # own content, which for ligatures is already the resolved letters
        parts.append(gmap.get(ref) or (el.text or ""))
        return "".join(parts) + (el.tail or "")
    if tag == "gap":
        # A lacuna is not whitespace. Several witnesses are damaged down a
        # column edge -- the Karlsruhe witness loses the opening of 219 lines
        # -- and a verse whose first word the parchment no longer carries is
        # not a sample of the poet's grammar. The marker survives segmentation
        # so such verses can be identified, and is removed from the text.
        return " " + GAPMARK + " " + (el.tail or "")
    parts.append(el.text or "")
    for ch in el:
        parts.append(text_of(ch, gmap, want_expan))
    return "".join(parts) + (el.tail or "")


def witness_text(path, want_expan=True):
    root = ET.parse(path).getroot()
    gmap = glyph_map(root)
    chunks = []
    for ln in root.iter(TEI + "line"):
        s = text_of(ln, gmap, want_expan)
        # the element's own tail belongs to the parent, not to this line
        s = re.sub(r"\s+", " ", s).strip()
        if s:
            chunks.append(s)
    return chunks


def depoint(raw):
    """A verse with every mark of pointing removed."""
    v = re.sub(r"[^\w\s̀-ͯ'’]", " ", raw, flags=re.UNICODE)
    return " ".join(v.split())


def to_verses(text):
    """Split on the medieval point, then remove all pointing from the verses."""
    return [v for v in (depoint(r)
                        for r in re.split("[" + re.escape(VERSE_END) + "]", text))
            if v]


# A Middle High German rhyming couplet runs to some four to nine words; a
# segmentation that yields a mean far outside that band has found the wrong
# unit, and says so without any need to know the manuscript in advance.
PLAUSIBLE = (3.5, 10.0)


def rhyme_rate(verses, sample=4000):
    """Share of adjacent verse pairs whose final words rhyme.

    This is the decisive test of a segmentation, and it is the text's own:
    the Kaiserchronik is written in rhyming couplets, so a segmentation that
    has found the verse will show its verse-final words rhyming in pairs,
    and one that has fused or split verses will not. Rhyme is approximated by
    a shared final trigram of the vowel-bearing tail, which is coarse but
    needs no phonology of a language stage whose spelling is unnormalised;
    both phasings are tried, since a witness may open mid-couplet.
    """
    ends = [v.split()[-1].lower() for v in verses[:sample] if v.split()]
    if len(ends) < 4:
        return 0.0
    def tail(w):
        w = re.sub(r"[^a-zäöüæœ]", "", w)
        return w[-3:] if len(w) >= 3 else w
    best = 0.0
    for off in (0, 1):
        pairs = [(ends[i], ends[i + 1])
                 for i in range(off, len(ends) - 1, 2)]
        if not pairs:
            continue
        hit = sum(1 for a, b in pairs
                  if tail(a) and (tail(a) == tail(b)
                                  or tail(a)[-2:] == tail(b)[-2:]))
        best = max(best, hit / len(pairs))
    return best


def segment(chunks):
    """Verses of one witness, under whichever layout the scribe in fact used.

    Two regimes occur in this transmission and the difference is not cosmetic.
    The older witnesses, Vorau among them, write verse continuously across the
    ruling and mark each verse end with a point; there the point is the unit.
    Others -- the lead witness of the B recension is one -- write one verse to
    the ruled line and point only sporadically; there the ruling is the unit,
    and splitting on points would fuse twenty verses into one. Both
    segmentations are therefore computed and the one whose mean length falls in
    the range a couplet can occupy is returned, so that the layout is measured
    rather than assumed.
    """
    joined = ""
    for s in chunks:
        if joined.endswith(("-", "‐", "=")):
            joined = joined[:-1] + s
        elif joined:
            joined += " " + s
        else:
            joined = s
    by_point = to_verses(joined)
    by_line = [v for v in (depoint(c) for c in chunks) if v]
    cands = []
    for name, vs in (("point", by_point), ("line", by_line)):
        if not vs:
            continue
        mean = sum(len(v.split()) for v in vs) / len(vs)
        cands.append((PLAUSIBLE[0] <= mean <= PLAUSIBLE[1],
                      round(rhyme_rate(vs), 3), name, vs, mean))
    if not cands:
        return [], "none", 0.0, 0.0
    ok, rr, name, vs, mean = max(cands)
    return vs, name, mean, rr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--abbr", action="store_true",
                    help="keep the scribe's abbreviations instead of the "
                         "editors' expansions (diplomatic reading)")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    readme = (SRC / "000_README.txt").read_text(encoding="utf-8", errors="replace")
    meta = {}
    for sig, title, fn in re.findall(
            r"Sigle:\s*(\S+)\s*\nTitel:\s*([^\n]+)\s*\nDatei:\s*(\S+)", readme):
        meta[fn.strip()] = (sig.strip(), title.strip())

    n_out = 0
    with open(OUT, "w", encoding="utf-8") as fh:
        for path in sorted(SRC.glob("*.tei.xml")):
            sig, title = meta.get(path.name, ("?", ""))
            try:
                chunks = witness_text(path, want_expan=not args.abbr)
            except ET.ParseError as e:
                print(f"  {path.name}: PARSE ERROR {e}")
                continue
            verses, regime, mean, rr = segment(chunks)
            damaged = [i for i, v in enumerate(verses) if GAPMARK in v]
            verses = [" ".join(w for w in v.split() if w != GAPMARK)
                      for v in verses]
            verses = [v for v in verses if v]
            n_lines = len(chunks)
            rhymes = [v.split()[-1] for v in verses if v.split()]
            # the capital of the siglum names the recension; the lower-case
            # families are that recension's fragments
            rec = re.match(r"([A-Za-z]+)", sig)
            rec = rec.group(1).upper() if rec else "?"
            fh.write(json.dumps({
                "siglum": sig, "recension": rec, "title": title,
                "file": path.name, "n_ms_lines": n_lines,
                "layout": regime, "mean_words_per_verse": round(mean, 2),
                "n_damaged_verses": len(damaged),
                "rhyme_rate": rr,
                "n_verses": len(verses), "verses": verses,
                "rhymes": rhymes}, ensure_ascii=False) + "\n")
            n_out += 1
            print(f"  {sig:6s} {len(verses):>6,} verses  {n_lines:>6,} ruled "
                  f"lines  [{regime}, {mean:.1f} w/v, rhyme {rr:.2f}]  {title[:26]}")
    print(f"\n{n_out} witnesses -> {OUT.name}")


if __name__ == "__main__":
    main()
