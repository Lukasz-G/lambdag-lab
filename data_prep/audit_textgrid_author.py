# Provenance audit for one TextGrid author, before his texts enter a bank.
#
# THE PROBLEM. The Digitale Bibliothek reproduces public-domain editions without
# recording whether the text is the author's own composition. Four things arrive
# under one name and none of them is flagged:
#
#   1. UNAUTHORISED / THIRD-PARTY EDITED editions. Karl May's Fischer-Ausgabe
#      (1902-03) was published against his will and its text was edited by Paul
#      Staberow; it appears simply as "Karl May".
#   2. DISPUTED works. May's Muenchmeyer colportage novels -- he claimed the
#      objectionable passages were interpolated, the manuscripts are lost, and it
#      cannot be settled either way.
#   3. TRANSLATIONS AND ADAPTATIONS. "Der Waldlaeufer" is May's reworking of
#      Gabriel Ferry via Fuellner's German translation, so its sentence structure
#      descends from a translator. For a method that measures grammatical habit
#      this is worse than a topic confound: the nuisance variable is another
#      writer's syntax.
#   4. NON-RUNNING TEXT. Riddles, proverb collections, open letters, tables.
#
# For authorship verification a contaminated identity label is not a metadata
# nuisance -- it is a wrong label on the dependent variable. This script surfaces
# the evidence; the judgements go into textgrid_exclusions.json with reasons, so
# they are auditable and defensible in the paper.
#
# WHAT IT REPORTS, and why each signal matters:
#   collections     the Digitale Bibliothek groups an author's works by SOURCE
#                   edition, so a rogue edition is usually one whole collection.
#                   This is the strongest signal available -- for May every
#                   disputed title sat in a single collection.
#   stub share      objects whose payload is an ORE/RDF manifest rather than TEI.
#                   They yield no text, so a collection that is entirely stubs is
#                   INERT: it looks alarming in a title list and contributes
#                   nothing. Distinguishing inert from ingested prevents both
#                   false alarms and false comfort.
#   repeated titles the same work reaching the bank twice becomes known/questioned
#                   leakage. Reported separately from near-duplicate CONTENT,
#                   because volume parts legitimately share a title.
#   markers         title patterns for adaptation, translation, editorship and
#                   posthumous completion, in German and Latin abbreviations.
#   outliers        a single object far larger than the rest is usually a whole
#                   collected volume that duplicates the individual works.
#
#   python data_prep/audit_textgrid_author.py "May, Karl"
#   python data_prep/audit_textgrid_author.py --slug tieck_ludwig --titles
#   python data_prep/audit_textgrid_author.py --risky      # who to audit next
#
# Output: a report on stdout, plus data_prep/cache/textgrid/audit_<slug>.json.

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from xml.etree import ElementTree as ET

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from fetch_textgrid import (CACHE, TEIC, classify, dupkeys, list_authors,  # noqa: E402
                            load_exclusions, slug, text_of)

# Title markers for texts that are not the author's own composition. German
# first, then the abbreviations German title pages actually use.
MARKERS = {
    "adaptation": r"\bfrei nach\b|\bnach dem (englischen|franz|span|ital|russ)|"
                  r"\bbearbeit|\bbearb\.|\bumgearbeitet|\bnacherz[aä]hl",
    "translation": r"\b[uü]bersetz|\b[uü]bertragen|\b[uü]bers\.|\baus dem "
                   r"(englischen|franz|lat|griech|russ)",
    "editorship": r"\bherausgegeben|\bhrsg\.|\bredigiert|\bgesammelt von|"
                  r"\bmitgeteilt von",
    "collaboration": r"\bund anderen\b|\bu\. a\.\b|\bgemeinsam mit|\bin "
                     r"verbindung mit",
    "posthumous": r"\bnachla[sß]|\bvollendet von|\berg[aä]nzt von|"
                  r"\bfortgesetzt von|\baus dem nachlass",
    "attributed": r"\bzugeschrieben|\bangeblich|\bpseudo-|\buntergeschoben",
    "not_running_text": r"r[aä]thsel|r[aä]tsel|spr[uü]chw[oö]rter|sprichw[oö]rter|"
                        r"inhaltsverzeichnis|register|verzeichnis|"
                        r"chronologie|bibliographie",
}
COMPILED = {k: re.compile(v, re.I) for k, v in MARKERS.items()}


def load_objects(author_slug):
    f = CACHE / f"objs_{author_slug}.json"
    if not f.exists():
        return None
    return json.loads(f.read_text(encoding="utf-8"))


def measure(o):
    """(has_text, words, genre) for a listed object, from the TEI cache."""
    f = TEIC / f"{o['uri'].replace(':', '_')}.xml"
    if not f.exists():
        return (None, 0, "")
    try:
        root = ET.fromstring(f.read_text(encoding="utf-8", errors="replace"))
    except ET.ParseError:
        return (False, 0, "")
    t = text_of(root)
    w = len(t.split())
    if w < 20:
        return (False, 0, "")
    return (True, w, classify(root)), t


def audit(author_slug, show_titles=False):
    objs = load_objects(author_slug)
    if objs is None:
        print(f"no cached listing for {author_slug!r}; run fetch_textgrid.py "
              f"--authors \"<Name>\" first")
        return None

    excl = load_exclusions().get(author_slug)
    colls = defaultdict(lambda: {"n": 0, "text": 0, "stub": 0, "words": 0,
                                 "titles": []})
    titles = defaultdict(int)
    sizes, seen, dup_content = [], set(), []
    flagged = defaultdict(list)

    for o in objs:
        coll = o["path"][0] if o.get("path") else "(none)"
        c = colls[coll]
        c["n"] += 1
        titles[o["title"]] += 1
        for kind, pat in COMPILED.items():
            if pat.search(o["title"]):
                flagged[kind].append(o["title"])
        got = measure(o)
        if not got or got[0] is None:
            continue
        (has, w, genre), text = got if isinstance(got, tuple) and len(got) == 2 \
            else (got, "")
        if not has:
            c["stub"] += 1
            continue
        c["text"] += 1
        c["words"] += w
        c["titles"].append((o["title"], w, genre))
        sizes.append((w, o["title"]))
        ks = dupkeys(text)
        if ks & seen:
            dup_content.append((o["title"], w))
        else:
            seen |= ks

    total_w = sum(c["words"] for c in colls.values())
    print(f"\n=== {author_slug} ===")
    print(f"{len(objs)} objects listed, {total_w:,} extractable words\n")

    print(f"{'collection':26s} {'objs':>5s} {'text':>5s} {'stubs':>6s} "
          f"{'words':>11s}  status")
    for name, c in sorted(colls.items(), key=lambda kv: -kv[1]["words"]):
        if c["words"] == 0 and c["stub"]:
            status = "INERT (manifests only -- contributes nothing)"
        elif c["words"] == 0:
            status = "empty"
        elif c["stub"] > c["text"]:
            status = "mostly stubs -- coverage thinner than it looks"
        else:
            status = ""
        print(f"{name[:26]:26s} {c['n']:5d} {c['text']:5d} {c['stub']:6d} "
              f"{c['words']:11,d}  {status}")

    print("\n-- provenance markers in titles --")
    if flagged:
        for kind, ts in sorted(flagged.items()):
            print(f"  {kind:18s} {len(ts):3d}  e.g. {'; '.join(ts[:3])[:90]}")
    else:
        print("  none")

    rep = [(t, n) for t, n in titles.items() if n > 1]
    print(f"\n-- repeated titles: {len(rep)} "
          f"({sum(n - 1 for _, n in rep)} redundant listings) --")
    for t, n in sorted(rep, key=lambda x: -x[1])[:8]:
        print(f"  x{n}  {t[:76]}")
    print(f"-- near-duplicate CONTENT actually ingested: {len(dup_content)} --")
    for t, w in dup_content[:8]:
        print(f"  {w:>8,}  {t[:70]}")

    if sizes:
        sizes.sort(reverse=True)
        med = sizes[len(sizes) // 2][0]
        # A large object is only suspicious if a SMALLER object's title is
        # contained in it -- that is the collected-volume-plus-its-own-parts
        # pattern, which double-counts. Size alone says nothing: an author of
        # both novels and short prose has a wide distribution by nature.
        small = [(w, t) for w, t in sizes if w <= 3 * med]
        nested = []
        for w, t in sizes:
            if w <= 3 * med:
                continue
            tl = t.lower()
            inner = [st for sw, st in small
                     if len(st) > 8 and st.lower() in tl and st.lower() != tl]
            if inner:
                nested.append((w, t, inner[:3]))
        print(f"\n-- containment check (median object {med:,} words, "
              f"largest {sizes[0][0]:,}) --")
        for w, t, inner in nested[:6]:
            print(f"  {w:>9,}  {t[:60]}  contains: {'; '.join(inner)[:50]}")
        if not nested:
            print("  no collected volume appears to repeat its own parts")

    if excl:
        hit = [o["title"] for o in objs
               if excl.search(" | ".join(o.get("path", []) + [o["title"]]))]
        print(f"\n-- current exclusions match {len(hit)} listed objects --")
    else:
        print("\n-- no exclusions recorded for this author yet --")

    out = {"slug": author_slug, "objects": len(objs), "words": total_w,
           "collections": {k: {kk: vv for kk, vv in v.items() if kk != "titles"}
                           for k, v in colls.items()},
           "markers": {k: v for k, v in flagged.items()},
           "repeated_titles": dict(rep),
           "duplicate_content": dup_content}
    (CACHE / f"audit_{author_slug}.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    return out


def risky(index_file):
    """Who to audit next: authors most likely to carry a messy record.

    Ranked on what we can see without fetching -- object count (a large,
    long-published oeuvre attracts rival and posthumous editions), the number of
    distinct collections (each collection is a different source edition), and
    whether the harvest already tripped a marker.
    """
    idx = json.loads(Path(index_file).read_text(encoding="utf-8"))
    rows = []
    for a, v in idx.items():
        objs = load_objects(a) or []
        colls = {o["path"][0] for o in objs if o.get("path")}
        marks = sum(1 for o in objs
                    for p in COMPILED.values() if p.search(o["title"]))
        words = sum(g["words"] for g in v.get("genres", {}).values())
        rows.append((len(colls), marks, v.get("aggregation_stubs", 0), words,
                     v["author_name"], a))
    rows.sort(key=lambda r: (-r[0], -r[1], -r[3]))
    print(f"{'author':34s} {'colls':>5s} {'marks':>5s} {'stubs':>6s} {'words':>11s}")
    for c, m, s, w, name, a in rows[:30]:
        print(f"{name[:34]:34s} {c:5d} {m:5d} {s:6d} {w:11,d}")
    print("\nHigh collection count = several source editions = the case where an "
          "unauthorised or edited edition hides among the authentic ones.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("author", nargs="?", default="",
                    help='TextGrid author name, e.g. "May, Karl"')
    ap.add_argument("--slug", default="", help="author slug instead of the name")
    ap.add_argument("--titles", action="store_true",
                    help="print every title with its word count")
    ap.add_argument("--risky", action="store_true",
                    help="rank harvested authors by audit priority")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    if args.risky:
        f = HERE / "raw" / "textgrid" / "german_textgrid_index.json"
        if not f.exists():
            print("no harvest index yet; run fetch_textgrid.py first")
            return
        risky(f)
        return

    s = args.slug or slug(args.author)
    if not s or s == "x":
        ap.error("give an author name or --slug, or use --risky")
    audit(s, show_titles=args.titles)


if __name__ == "__main__":
    main()
