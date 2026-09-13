# Bank the Wikisource letter records: strip the EDITOR'S text, keep the
# author's, and log the dose of both.
#
# A letter record arrives with three layers. The editorial apparatus -- the
# Conard numbering-and-addressee headline ("9. AU MEME.", "12. A ERNEST
# CHEVALIER") -- is the editor's prose and must not enter an authorship bank.
# The dateline ("Rouen, ce 11 septembre 1833.") and the salutation ("Cher
# Ernest,") are the AUTHOR'S OWN writing conventions and stay: stripping them
# would delete authorial habit, which is the object of study. What the
# banking step owes downstream is not silence but a LOGGED DOSE: how many
# header lines were removed and how many letters open with a salutation
# formula, so that any later worry about formula inflation has a number to
# start from.
#
#   python data_prep/bank_ws_letters.py
#
# Output: data_prep/raw/wikisource/french_wsletters_preprocessed.jsonl
#         (one record per author, standard schema, ready for masking)

import json
import re
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE / "raw" / "wikisource" / "fr_ws_letters.jsonl"
OUT = HERE / "raw" / "wikisource" / "french_wsletters_preprocessed.jsonl"

# the Conard headline: a number, then an all-caps addressee line
HEAD = re.compile(r"^\s*\d{1,4}\s*\.?\s*(AU |A |À |AUX |A LA |À LA |AU MÊME"
                  r"|A SA |À SA |A SON |À SON )[A-ZÉÈÀÂÇÔÛ' ,.-]{0,60}\.?\s*$")
CAPS = re.compile(r"^[0-9 .]*[A-ZÉÈÀÂÇÔÛ' ,.-]{6,70}\.?$")
SALUT = re.compile(r"^\s*(cher|chère|mon |ma |vieux|bien-aimé|pauvre|carissimo)",
                   re.I)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    per_author = {}
    dose = Counter()
    for line in open(SRC, encoding="utf-8"):
        d = json.loads(line)
        lines = d["text"].splitlines()
        body, stripped = [], 0
        for i, l in enumerate(lines):
            # editorial headline: only in the first few lines, numbered or
            # fully capitalised address
            if i < 3 and (HEAD.match(l) or CAPS.match(l.strip())):
                stripped += 1
                continue
            body.append(l)
        txt = "\n".join(body).strip()
        if len(txt.split()) < 15:
            dose["dropped_short"] += 1
            continue
        dose["header_lines_stripped"] += stripped
        if any(SALUT.match(l) for l in body[:4] if l.strip()):
            dose["opens_with_salutation"] += 1
        dose["letters"] += 1
        a = d["author_id"]
        per_author.setdefault(a, {"author_id": a,
                                  "author_name": d["author_name"],
                                  "id_source": "wikisource-fr:letters",
                                  "parts": []})
        per_author[a]["parts"].append(txt)
    with open(OUT, "w", encoding="utf-8") as fh:
        for a in sorted(per_author):
            r = per_author[a]
            text = "\n\n".join(r.pop("parts"))
            r["text"] = text
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
            print(f"  {a}: {len(text.split()):,} words")
    print("dose:", dict(dose))
    print(f"-> {OUT.name}")


if __name__ == "__main__":
    main()
