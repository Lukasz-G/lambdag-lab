# Dates and display names for the witnesses, shared by the reuse figures.
#
# One parser, because two would drift. Three date conventions occur in the
# corpus's manifest and all three are approximate by nature, so every year here
# is good for ORDERING witnesses and for nothing else:
#
#   1504-1516 (Hans Ried)    an explicit year or span
#   2. Viertel 13. Jh.       a named part of a century
#   12,2-13,1                the reference corpus's own century,half notation
#
# Display names come from the manifest's work field, which carries the edition's
# own label; nothing is written from memory.

import re

# one colour and one marker per named text, used identically in every figure
STYLE = {
    "rem__M013O": ("#0b5fa5", "o", "Annolied"),
    "rem__M205P": ("#b8860b", "s", "Rolandslied P"),
    "rem__M205A": ("#8c2d2d", "^", "Rolandslied A"),
}

INK = "#1a1a1a"
GREY = "#7d7d7d"

_PART = {"1. Viertel": 12, "2. Viertel": 37, "3. Viertel": 62,
         "4. Viertel": 87, "1. Hälfte": 25, "2. Hälfte": 75,
         "1. Drittel": 17, "2. Drittel": 50, "3. Drittel": 83,
         "Anfang": 10, "Mitte": 50, "Ende": 90}

_REM = re.compile(r"^(\d{1,2})(?:,([12]))?(?:-(\d{1,2}),([12]))?$")


def _half(c, h):
    return (int(c) - 1) * 100 + (25 if h == "1" else 75 if h == "2" else 50)


def year_of(date):
    """an approximate year, for ordering only; None where the field says nothing"""
    d = (date or "").strip()
    if not d or d == "-":
        return None
    yrs = [int(y) for y in re.findall(r"\b(1[0-5]\d{2})\b", d)]
    if yrs:
        return (min(yrs) + max(yrs)) // 2
    m = _REM.match(d)
    if m:
        a = _half(m.group(1), m.group(2))
        return (a + _half(m.group(3), m.group(4))) // 2 if m.group(3) else a
    c = re.search(r"(1[0-5])\.\s*Jh", d)
    if not c:
        return None
    base = (int(c.group(1)) - 1) * 100
    for key, off in _PART.items():
        if key in d:
            return base + off
    return base + 50


_ORD = {"1": "1st", "2": "2nd", "3": "3rd", "4": "4th"}


def date_label(date):
    """the manifest's own dating, in English, for a figure a reader reads

    The two conventions are translated term for term and nothing is rounded: a
    century half stays a century half. Anything the translation does not cover
    is passed through as the manifest wrote it.
    """
    d = (date or "").strip()
    if not d or d == "-":
        return "no date recorded"
    m = _REM.match(d)
    if m:
        def part(c, h):
            return f"{_ORD[h]} half {c}th c." if h else f"{c}th c."
        a = part(m.group(1), m.group(2))
        return (f"{a[:-3]}–{part(m.group(3), m.group(4))}" if m.group(3)
                else a)
    out = d
    for de, en in (("Viertel", "quarter"), ("Hälfte", "half"),
                   ("Drittel", "third")):
        out = re.sub(r"(\d)\.\s*" + de,
                     lambda mm: f"{_ORD[mm.group(1)]} {en}", out)
    for de, en in (("Anfang", "early"), ("Mitte", "mid"), ("Ende", "late"),
                   ("um", "c."), ("vor", "before"), ("nach", "after")):
        out = re.sub(r"\b" + de + r"\b", en, out)
    # a century whose "Jh." is carried by a later member of a compound, as in
    # "Ende 13. / Anfang 14. Jh."
    out = re.sub(r"(\d{1,2})\.\s*(?=[/,])", r"\1th ", out)
    return re.sub(r"(\d{1,2})\.\s*Jh\.?", r"\1th c.", out)


_LOWER = {"von", "vom", "dem", "den", "der", "die", "das", "des", "und", "zu",
          "im", "in", "ein", "eine", "einem", "uebelen", "a", "b"}


def display_name(name, row=None):
    """a title a reader recognises, from the edition's own label"""
    if name.startswith("rem__") and row:
        w = row["work"].replace("'", "").replace("’", "")
        # the reference corpus writes a witness's siglum and its editor into one
        # parenthesis, as in "Annolied (O: Opitz)"; the siglum alone identifies it
        w = re.sub(r"\(([A-Za-z0-9]+):[^)]*\)", r"(\1)", w)
        # and a gloss on the siglum, as in "Alexanderlied (V [Vorauer
        # Alexander])", which a figure has no room for
        w = re.sub(r"\s*\[[^\]]*\]", "", w)
        pre, _, rest = w.partition(": ")
        return rest if rest and "(" not in pre else w
    parts = name.split("__")
    if len(parts) >= 4:
        words = parts[2].split("_")
        title = " ".join(w if i and w in _LOWER else w.capitalize()
                         for i, w in enumerate(words))
        return f"{title} ({parts[3]})"
    return name
