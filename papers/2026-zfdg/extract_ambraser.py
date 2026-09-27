# Ambraser Heldenbuch -> verses and rhyme words, one record per work.
#
# WHAT THE SOURCE IS. Klarer's transcription (Zenodo 10.5281/zenodo.6025611,
# CC BY 4.0) is ALLOGRAPHIC: it records which of Hans Ried's letter shapes
# stands on the page, so a final n is an eng, a final s a closed reversed open
# e or a sigma, an r after a rounded letter an r rotunda, and so on. The
# transcription guidelines list every such shape with the grapheme it
# realises, and that list is applied here in reverse: allographs collapse to
# the grapheme, superscripts that carry meaning (the e- and o-superscripts)
# are kept as combining marks in the form the reference corpus uses, and the
# breve, which the guidelines describe as a purely graphic device for telling
# u from n, is dropped. Long s is kept, as the reference corpus keeps it.
#
# The codex writes verse continuously; verse ends are pointed and the
# transcription tags each point with the verse number of the standard
# edition (<endOfVerse value="N">), strophic works additionally with
# <endOfStrophe>. Words broken across a ruled line are tagged on both halves
# (<hyphenation>), so here, unlike in the Kaiserchronik witnesses, the join
# is exact and needs no lexicon. Abbreviations come with the editors'
# expansion (<choice>), which is taken. The scribe's insertion marks carry
# the inserted text as an attribute, which is emitted in place. Struck-through
# letters and letters the transcription tags as slips are dropped, and the
# count of each is reported, since both are decisions about the scribe's text.
#
# Punctuation is consumed as segmentation and then discarded, as in every
# medieval setting of this work.
#
#   python medieval/extract_ambraser.py
#
# Output: medieval/raw/ambraser/ambraser_verses.jsonl
#         one record per work: no, key, author, title, incipit, siglum "d",
#         verses, verse_ids (edition numbering; strophe.verse for strophic
#         works), rhymes

import json
import re
import sys
import unicodedata
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from join_splits import MIN_JOINED, fold, load_lexicon  # noqa: E402

SRC = (HERE / "raw" / "ambraser"
       / "ambraser-heldenbuch-transcription-and-scientific-dataset"
       / "Ambraser_Heldenbuch_Transkription_und_wissenschaftliches_Datenset.xml")
OUT = HERE / "raw" / "ambraser" / "ambraser_verses.jsonl"

# The twenty-five texts in codex order. Authorship as the handbooks give it;
# "anon." where none is transmitted. Key = author__work, the label under
# which the work travels through the rest of the pipeline.
WORKS = [
    ("stricker__frauenehre", "Der Stricker", "Frauenehre"),
    ("anon__moriz_von_craun", "anon.", "Moriz von Craûn"),
    ("hartmann_von_aue__iwein", "Hartmann von Aue", "Iwein"),
    ("hartmann_von_aue__klage", "Hartmann von Aue", "Die Klage (Büchlein)"),
    ("anon__zweites_buechlein", "anon.", "Das zweite Büchlein"),
    ("anon__der_mantel", "anon.", "Der Mantel"),
    ("hartmann_von_aue__erec", "Hartmann von Aue", "Erec"),
    ("anon__dietrichs_flucht", "anon.", "Dietrichs Flucht"),
    ("anon__rabenschlacht", "anon.", "Rabenschlacht"),
    ("anon__nibelungenlied", "anon.", "Nibelungenlied"),
    ("anon__nibelungenklage", "anon.", "Die Klage (Nibelungenklage)"),
    ("anon__kudrun", "anon.", "Kudrun"),
    ("anon__biterolf_und_dietleib", "anon.", "Biterolf und Dietleib"),
    ("anon__ortnit", "anon.", "Ortnit"),
    ("anon__wolfdietrich_a", "anon.", "Wolfdietrich A"),
    ("anon__von_dem_uebelen_wibe", "anon.", "Von dem übelen wîbe"),
    ("herrand_von_wildonie__die_treue_gattin", "Herrand von Wildonie", "Die treue Gattin"),
    ("herrand_von_wildonie__der_verkehrte_wirt", "Herrand von Wildonie", "Der verkehrte Wirt"),
    ("herrand_von_wildonie__der_nackte_kaiser", "Herrand von Wildonie", "Der nackte Kaiser"),
    ("herrand_von_wildonie__die_katze", "Herrand von Wildonie", "Die Katze"),
    ("ulrich_von_liechtenstein__frauenbuch", "Ulrich von Liechtenstein", "Frauenbuch"),
    ("wernher_der_gartenaere__helmbrecht", "Wernher der Gartenaere", "Helmbrecht"),
    ("stricker__pfaffe_amis", "Der Stricker", "Pfaffe Amis"),
    ("wolfram_von_eschenbach__titurel", "Wolfram von Eschenbach", "Titurel"),
    ("priester_wernher__maria", "Priester Wernher", "Maria"),
]

# Allograph -> grapheme, from the guidelines' Zeichenübersicht (§2.1, §2.3).
ALLOGRAPH = {
    "ſ": "ſ",   # long s: kept, the reference corpus keeps it
    "ɞ": "s",        # closed reversed open e = final s
    "σ": "s",        # sigma = final s
    "": "ſ",   # enlarged long s
    "ŋ": "n",        # eng = final n
    "ɱ": "m",        # m with hook = final m
    "ꝛ": "r",        # r rotunda
    "ʒ": "z",        # ezh = z
    "ꜩ": "tz",       # tz ligature
    "∂": "d",        # partial differential = d variant
    "ɧ": "h",        # heng with hook = h variant
    "ỽ": "v",        # middle-Welsh v = v variant
    "ⱳ": "w",        # w with hook = w variant
    "ᷓ": "ͤ",   # flattened open a above = the e-superscript
    "ͦ": "ͦ",   # o-superscript: kept
    "̈": "̈",   # diaeresis: kept
    "̆": "",         # breve: graphic only (u vs n), dropped
    "ŭ": "u",        # the same breve, precomposed in the source twice
    "̄": "",         # macron outside an expanded abbreviation: dropped
    "ˀ": "",         # glottal-stop hook, likewise
    "̃": "",         # tilde, likewise
    "̇": "",
    "⸗": "",         # double oblique hyphen (line-break mark)
    "·": " ", "": " ", "~": " ", ":": " ", "∧": " ",
    ".": " ", ",": " ", ";": " ", "/": " ", "|": " ",
}
_MAP = {ord(k): v for k, v in ALLOGRAPH.items()}

TAG = re.compile(r"<(/?)([A-Za-z]+)([^>]*?)(/?)>", re.S)


def grapheme(s):
    return unicodedata.normalize("NFC", s.translate(_MAP))


def tokens(s, lex=None, joins=None):
    """tokens of a verse; at each unmarked ruled-line junction (\\x01) the two
    neighbouring tokens are joined under the lexicon rule of join_splits.py,
    the rule chosen by measurement against the reference corpus's own
    transcription of a Kaiserchronik witness: the concatenation must be
    attested at least MIN_JOINED times and be no rarer than the rarer half"""
    raw = grapheme(s).split()
    out = []
    i = 0
    while i < len(raw):
        t = raw[i]
        if t == "\x01":
            if lex is not None and out and i + 1 < len(raw) and raw[i + 1] != "\x01":
                a, b = out[-1], raw[i + 1]
                fa, fb = fold(a), fold(b)
                if fa and fb:
                    merged = lex.get(fa + fb, 0)
                    if merged >= MIN_JOINED and merged >= min(lex.get(fa, 0), lex.get(fb, 0)):
                        out[-1] = a + b
                        if joins is not None:
                            joins[0] += 1
                        i += 2
                        continue
            i += 1
            continue
        if any(c.isalpha() for c in t):
            out.append(t)
        i += 1
    return out


def extract(xml, lex=None):
    body = xml[xml.find("<text"):]
    works = []
    cur = None
    buf = []          # text of the verse under construction, with a glue marker
    skip = 0          # depth of abbr/note/incipit/hi/mistake/folio content to drop
    strophe = 0
    counts = {"mistake": 0, "struck": 0, "insertion": 0, "hyphenation": 0, "lexjoin": 0}
    joins = [0]

    def flush(vid):
        text = "".join(buf)
        buf.clear()
        toks = tokens(text.replace("\x00", ""), lex, joins)
        counts["lexjoin"] = joins[0]
        if not toks:
            return
        cur["verses"].append(toks)
        cur["verse_ids"].append(vid)
        cur["rhymes"].append(toks[-1])

    pos = 0
    in_incipit = False    # inside the scribe's title line of the current work
    for m in TAG.finditer(body):
        text = body[pos:m.start()]
        if cur is not None:
            if skip == 0:
                buf.append(text)
            elif in_incipit:
                cur["incipit"] += text
        pos = m.end()
        closing, name, attrs, selfclose = m.groups()
        if name == "startText":
            if not closing:
                if cur is not None:
                    works.append(cur)
                key, author, title = WORKS[len(works)]
                cur = dict(no=len(works) + 1, key=key, author=author, title=title,
                           incipit="", siglum="d", verses=[], verse_ids=[], rhymes=[])
                strophe = 0
                buf.clear()
                # a work's title line, where the scribe wrote one, wraps this marker
                in_incipit = skip > 0
            continue
        # dropped regions are tracked whether or not a work has started, so the
        # table of contents and the codex's own incipit cannot unbalance the count
        if name in ("abbr", "note", "hi", "mistake", "folio", "incipit"):
            if cur is not None and name == "mistake" and not closing:
                counts["mistake"] += 1
            if cur is not None and name == "hi" and not closing:
                counts["struck"] += 1
            if selfclose:
                continue
            skip = max(0, skip - 1) if closing else skip + 1
            if name == "incipit" and closing:
                in_incipit = False
            continue
        if cur is None or skip:
            continue
        if name == "hyphenation":
            # first half ends in the hyphen mark, second half follows on the next
            # ruled line: glue them by suppressing the space the <lb> would add
            if not closing:
                counts["hyphenation"] += 1
            elif buf and buf[-1].rstrip().endswith("⸗"):
                buf.append("\x00")
            continue
        if name == "insertion":
            # the scribe's caret; the inserted text is the attribute, the caret
            # itself is punctuation and falls out with the rest
            ins = re.search(r"insert='([^']*)'", attrs)
            if not closing and ins:
                buf.append(" " + ins.group(1) + " ")
                counts["insertion"] += 1
            continue
        if name == "lb":
            if buf and buf[-1] == "\x00":
                buf.pop()
            else:
                buf.append(" \x01 ")     # an unmarked ruled-line junction
            continue
        if name == "endOfVerse" and not closing:
            v = re.search(r"value='(\d+)'", attrs)
            vid = v.group(1) if v else "?"
            if strophe:
                vid = "%d.%s" % (strophe, vid)
            flush(vid)
            continue
        if name == "endOfStrophe" and not closing:
            strophe += 1
            continue
        if name == "endOfVerse" and closing:
            # the point itself sits between the tags and was already dropped by flush
            continue
    if cur is not None:
        works.append(cur)
    for w in works:
        w["incipit"] = " ".join(grapheme(w["incipit"]).split()) or "-"
    # strophic works: the first strophe's verses were flushed before any
    # endOfStrophe was seen, so their ids lack a strophe prefix; number them 1
    for w in works:
        if any("." in v for v in w["verse_ids"]):
            w["verse_ids"] = [v if "." in v else "1." + v for v in w["verse_ids"]]
    return works, counts


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    xml = SRC.read_text(encoding="utf-8")
    # the reference corpus ends two centuries before this codex was written,
    # so nothing in it is the text under test and nothing is excluded
    lex = load_lexicon()
    works, counts = extract(xml, lex)
    with OUT.open("w", encoding="utf-8") as f:
        for w in works:
            f.write(json.dumps(w, ensure_ascii=False) + "\n")
    print("%d works -> %s" % (len(works), OUT))
    print("dropped: %d slips, %d struck letters; %d insertions emitted; %d marked line-break joins; "
          "%d unmarked junctions joined by the lexicon rule"
          % (counts["mistake"], counts["struck"], counts["insertion"], counts["hyphenation"] // 2,
             counts["lexjoin"]))
    print("%3s %-42s %-24s %7s %7s  %s" % ("no", "key", "author", "verses", "tokens", "incipit"))
    for w in works:
        print("%3d %-42s %-24s %7d %7d  %s" % (w["no"], w["key"], w["author"][:24], len(w["verses"]),
                                              sum(len(v) for v in w["verses"]), w["incipit"][:40]))
    left = {}
    for w in works:
        for v in w["verses"]:
            for t in v:
                for c in t:
                    if ord(c) > 127:
                        left[c] = left.get(c, 0) + 1
    print("\nnon-ASCII characters left in the output:")
    for c, n in sorted(left.items(), key=lambda x: -x[1]):
        print("  U+%04X %-3r %8d  %s" % (ord(c), c, n, unicodedata.name(c, "?")))


if __name__ == "__main__":
    main()
