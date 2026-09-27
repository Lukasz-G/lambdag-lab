# The normalised-edition corpus as verses: surface, lemma and part of speech.
#
# WHAT THE SOURCE GIVES. One TEI per work, `<l n="...">` for the verse and `<w>`
# for the word, with the part of speech on the element and the lemma held as a
# pointer into the lexicon (`lemmaRef="lexicon.xml#lemma_1331"`). The lemma forms
# therefore have to be resolved through that file; a word whose pointer is
# missing keeps its own form, and how often that happens is counted rather than
# passed over.
#
# WHAT IS EMITTED. One record per work: the author, the register's sigle and
# title, and the verses, each a list of (surface, lemma, pos). Verse numbering is
# the edition's own `n`, which is what the printed text counts and what a
# citation in the paper would name. Lines with no word in them are dropped.
#
# The part of speech is MHDBDB's, not the HiTS scheme the witness banks carry and
# not Universal Dependencies, so it is passed through unmapped: the rhyme stream
# needs no tags, and anything that does need them owes a map first.
#
#   python medieval/extract_mhdbdb.py
#   python medieval/extract_mhdbdb.py --only "Rudolf"
#
# Output: medieval/mhdbdb_verses.jsonl (one work per line)

import argparse
import json
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RAW = HERE / "raw" / "mhdbdb"
OUT = HERE / "mhdbdb_verses.jsonl"
T = "{http://www.tei-c.org/ns/1.0}"


def lexicon():
    """lemma id -> the lemma's form, from the authority file"""
    out = {}
    for ev, e in ET.iterparse(RAW / "lexicon.xml", events=("end",)):
        if e.tag != T + "entry":
            continue
        i = e.get("{http://www.w3.org/XML/1998/namespace}id")
        f = e.find(f"{T}form/{T}orth")
        if f is None:
            f = e.find(f"{T}form")
        if i and f is not None and (f.text or "").strip():
            out[i] = f.text.strip().lower()
        e.clear()
    return out


def registry():
    """sigle -> (author, title)"""
    out = {}
    for b in ET.parse(RAW / "works.xml").getroot().iter(T + "bibl"):
        s = b.find(f"{T}idno[@type='sigle']")
        a = b.find(f"{T}author")
        t = b.find(f"{T}title")
        if s is not None and s.text:
            out[s.text.strip()] = ((a.text or "").strip() if a is not None
                                   else "",
                                   (t.text or "").strip() if t is not None
                                   else "")
    return out


def verses_of(path, lex, miss):
    """the work's verses as lists of (surface, lemma, pos)"""
    out = []
    for ev, e in ET.iterparse(path, events=("end",)):
        if e.tag != T + "l":
            continue
        v = []
        for w in e.iter(T + "w"):
            s = "".join(w.itertext()).strip()
            if not s:
                continue
            ref = (w.get("lemmaRef") or "").split("#")[-1]
            lem = lex.get(ref)
            if lem is None:
                miss[bool(ref)] += 1
                lem = s.lower()
            v.append((s, lem, (w.get("pos") or "").split()[0]
                      if w.get("pos") else ""))
        if v:
            out.append(dict(n=e.get("n") or str(len(out) + 1), w=v))
        e.clear()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="", help="substring of the author's name")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    reg = registry()
    lex = lexicon()
    print(f"lexicon: {len(lex):,} lemma forms; register: {len(reg):,} works")
    miss = Counter()
    files = sorted((RAW / "tei").glob("*.tei.xml"))
    n_out = 0
    with OUT.open("w", encoding="utf-8") as fh:
        for p in files:
            sig = p.name.replace(".tei.xml", "")
            author, title = reg.get(sig, ("", ""))
            if args.only and args.only.lower() not in author.lower():
                continue
            vs = verses_of(p, lex, miss)
            if not vs:
                print(f"  {sig:8} {author:30} no verses")
                continue
            fh.write(json.dumps(dict(sigle=sig, author=author, title=title,
                                     verses=vs), ensure_ascii=False) + "\n")
            n_out += 1
            if n_out % 25 == 0:
                print(f"  {n_out:4}/{len(files)} works", flush=True)
    tok = miss[True] + miss[False]
    print(f"\n{n_out} works -> {OUT.relative_to(ROOT)}")
    print(f"unresolved lemmas: {miss[True]:,} with a pointer the lexicon does "
          f"not hold, {miss[False]:,} with no pointer at all"
          + (f" ({tok:,} words keep their own form)" if tok else ""))


if __name__ == "__main__":
    main()
