# Is the prose bank pure prose? And what utilitarian text does the cache hold?
#
# WHY. Genre in this harvest is classified from TEI MARKUP (fetch_textgrid.py:
# classify() -- speeches, verse lines, else prose), because the catalogue has no
# usable genre field. That was the right call for separating drama and verse,
# but it has a blind spot: a letter, a diary entry and an essay are paragraphs,
# exactly like a novel, so anything epistolary or utilitarian the Digitale
# Bibliothek carries for an author has been filed into the PROSE bank silently.
# Before letters and notebooks are added as domains of their own, two questions
# must be answered with numbers: (1) how much of the existing prose banks is
# not belletristic prose at all -- a purity statement owed to every result
# already computed on them; (2) what utilitarian material the cache already
# holds per author -- the inventory that gates the expansion.
#
# HOW. Every cached TEI object carries the Digitale Bibliothek's own shelf mark
# in an n="..." attribute: /Literatur/<letter>/<Author>/<Collection>/<title...>.
# The COLLECTION segment is the library's native kind label (Briefe, Gedichte,
# Romane, Tagebuecher, Schriften, ...), assigned by its editors -- far better
# evidence than any keyword guess of ours. Each object is also re-classified
# from markup exactly as the harvest did, so the audit reports the two signals
# side by side and disagreements (a verse epistle; a prose section inside
# Gedichte) become visible instead of silent.
#
#   python data_prep/audit_domain_purity.py
#   python data_prep/audit_domain_purity.py --top 30      # widen the report
#
# Output: data_prep/periods/../domain_inventory.tsv  (author x collection)
#         printed purity summary per bank author, worst offenders first.

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
TEI_DIR = HERE / "cache" / "textgrid" / "tei"
RAW = HERE / "raw" / "textgrid"
OUT = HERE / "domain_inventory.tsv"

N_ATTR = re.compile(r'\bn="(/Literatur/[^"]+)"')
TEXT_RE = re.compile(r"<text[ >].*?</text>", re.S)
TAG_RE = re.compile(r"<[^>]+>")
SP_RE = re.compile(r"<sp[ >]")
L_RE = re.compile(r"<l[ >]")
P_RE = re.compile(r"<p[ >]")

# The Digitale Bibliothek's collection labels, rolled up into working domains.
# Anything not listed is reported under its own name rather than forced into a
# bucket -- the point of the audit is to SEE the taxonomy, not to flatten it.
DOMAIN = {
    "briefe": "letters", "briefwechsel": "letters",
    "tagebuecher": "diary", "tagebücher": "diary", "tagebuch": "diary",
    "autobiographisches": "autobiography", "erinnerungen": "autobiography",
    "memoiren": "autobiography", "selbstzeugnisse": "autobiography",
    "reiseberichte": "travel", "reisebilder": "travel",
    "schriften": "essay", "aesthetische schriften": "essay",
    "ästhetische schriften": "essay", "theoretische schriften": "essay",
    "abhandlungen": "essay", "aufsaetze": "essay", "aufsätze": "essay",
    "kritiken": "essay", "essays": "essay", "aphorismen": "essay",
    "romane": "belletristic", "erzaehlungen": "belletristic",
    "erzählungen": "belletristic", "novellen": "belletristic",
    "maerchen": "belletristic", "märchen": "belletristic",
    "prosa": "belletristic", "epen": "belletristic",
    "gedichte": "verse-side", "lyrik": "verse-side",
    "dramen": "drama-side", "stuecke": "drama-side", "stücke": "drama-side",
    "libretti": "drama-side",
    # labels discovered by the first pass over the cache, 2026-09-10
    "roman": "belletristic", "novelle": "belletristic",
    "erzählung": "belletristic", "erzaehlung": "belletristic",
    "einzelne erzählungen": "belletristic", "jugenderzählungen": "belletristic",
    "erzählprosa": "belletristic", "erzählungen und märchen": "belletristic",
    "märchen-sammlung": "belletristic", "autobiographischer roman": "belletristic",
    # May's Reiseerzählungen are adventure FICTION, not reportage
    "reiseerzählungen": "belletristic",
    "reisebeschreibungen": "travel",
    # Heine: literary travel prose interleaved with letters; kept as travel,
    # flagged for per-work audit before any bank is built from it
    "reisebilder und reisebriefe": "travel",
    "theoretische schrift": "essay", "traktate": "essay",
    "essays i: über deutschland": "essay", "essays ii: über frankreich": "essay",
    "essays iii: aufsätze und streitschriften": "essay",
    "essays, reden, vorträge": "essay",
    "theologiekritische und philosophische schriften": "essay",
    "aufsätze, reden, offene briefe und sonstiges": "essay",
    "komödien": "drama-side", "tragödien": "drama-side",
    "historien": "drama-side", "drama": "drama-side",
    "verserzählungen": "verse-side", "versepos": "verse-side", "epos": "verse-side",
    # translations of scripture -- not the author's idiolect; must be excluded
    # from any authorship bank, and checked for presence in the current one
    "luther-bibel 1912": "scripture-translation",
    "luther-bibel 1545": "scripture-translation",
}


def classify_markup(xml):
    """The harvest's own rule, applied to the raw string."""
    if len(SP_RE.findall(xml)) >= 5:
        return "drama"
    lines, paras = len(L_RE.findall(xml)), len(P_RE.findall(xml))
    if lines >= 8 and lines > paras:
        return "verse"
    return "prose"


def words_of(xml):
    m = TEXT_RE.search(xml)
    if not m:
        return 0
    return len(TAG_RE.sub(" ", m.group(0)).split())


def slugify(name):
    sys.path.insert(0, str(HERE))
    from fetch_textgrid import slug
    return slug(name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=18)
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    idx = json.loads((RAW / "german_textgrid_index.json").read_text(encoding="utf-8"))
    bank_words = {a: {g: v["words"] for g, v in rec["genres"].items()}
                  for a, rec in idx.items()}

    # ---- pass over the cache ------------------------------------------------
    inv = defaultdict(lambda: Counter())          # (author, collection) -> words
    cnt = defaultdict(lambda: Counter())          # (author, collection) -> objects
    disagree = Counter()                          # (domain, markup) crosses
    nofn = 0
    scan_f = HERE / "domain_scan.jsonl"
    rows_scan = []
    if scan_f.exists():
        for line in open(scan_f, encoding="utf-8"):
            rows_scan.append(json.loads(line))
        print(f"  reusing {len(rows_scan)} scanned objects from {scan_f.name}")
    else:
        files = sorted(TEI_DIR.glob("*.xml"))
        for i, f in enumerate(files):
            xml = f.read_text(encoding="utf-8", errors="replace")
            m = N_ATTR.search(xml)
            if not m:
                nofn += 1
                continue
            seg = [s for s in m.group(1).split("/") if s]
            if len(seg) < 4:
                continue
            rows_scan.append({"uri": f.stem, "author": seg[2], "coll": seg[3],
                              "title": seg[-1][:120],
                              "markup": classify_markup(xml),
                              "words": words_of(xml)})
            if (i + 1) % 3000 == 0:
                print(f"  ...{i + 1}/{len(files)} objects", flush=True)
        with open(scan_f, "w", encoding="utf-8") as fh:
            for r in rows_scan:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    for r in rows_scan:
        a = slugify(r["author"])
        inv[a][r["coll"]] += r["words"]
        cnt[a][r["coll"]] += 1
        disagree[(DOMAIN.get(r["coll"].lower(), r["coll"].lower()),
                  r["markup"])] += r["words"]

    # ---- write the inventory ------------------------------------------------
    with open(OUT, "w", encoding="utf-8") as fh:
        fh.write("author\tcollection\tdomain\tobjects\twords\n")
        for a in sorted(inv):
            for coll, w in inv[a].most_common():
                fh.write(f"{a}\t{coll}\t{DOMAIN.get(coll.lower(), '?')}\t"
                         f"{cnt[a][coll]}\t{w}\n")
    print(f"\n{sum(len(v) for v in inv.values())} (author, collection) rows -> "
          f"{OUT.name};  {nofn} objects without a DigiBib path (ELTeC etc.)")

    # ---- question 1: purity of the existing prose banks ---------------------
    print("\nPROSE-BANK PURITY (cache words by domain, PROSE-classified markup "
          "only,\nfor authors present in the banks; upper bound -- caps and "
          "dedup mean not\nevery cached word entered the bank)")
    print(f"  {'author':30s} {'bank prose':>10s} {'belletr.':>9s} "
          f"{'letters':>8s} {'diary':>7s} {'essay':>7s} {'autob.':>7s} "
          f"{'travel':>7s} {'other':>7s}")
    tot = Counter()
    rows = []
    for a in sorted(inv):
        if a not in bank_words or "prose" not in bank_words[a]:
            continue
        by = Counter()
        for coll, w in inv[a].items():
            d = DOMAIN.get(coll.lower(), "other")
            if d in ("verse-side", "drama-side"):
                continue
            by[d] += w
        util = sum(v for k, v in by.items() if k not in ("belletristic",))
        rows.append((util, a, by))
        tot += by
    for util, a, by in sorted(rows, reverse=True)[:args.top]:
        print(f"  {a[:28]:30s} {bank_words[a]['prose']:10d} "
              f"{by['belletristic']:9d} {by['letters']:8d} {by['diary']:7d} "
              f"{by['essay']:7d} {by['autobiography']:7d} {by['travel']:7d} "
              f"{by['other']:7d}")
    allw = sum(tot.values())
    if allw:
        print(f"\n  cache totals across bank authors: "
              + ", ".join(f"{k} {100*v/allw:.1f}%" for k, v in tot.most_common()))

    # ---- question 2: the expansion inventory --------------------------------
    print("\nEXPANSION INVENTORY (authors with >= 20k cached words in a "
          "utilitarian domain)")
    for dom in ("letters", "diary", "essay", "autobiography", "travel"):
        ok = []
        for a in inv:
            w = sum(v for c, v in inv[a].items()
                    if DOMAIN.get(c.lower()) == dom)
            if w >= 20_000:
                ok.append((w, a, a in bank_words))
        ok.sort(reverse=True)
        inb = sum(1 for _, _, b in ok if b)
        print(f"  {dom:14s} {len(ok):3d} authors >=20k ({inb} already in banks): "
              + ", ".join(f"{a}({w//1000}k)" for w, a, _ in ok[:8])
              + (" ..." if len(ok) > 8 else ""))

    # ---- the kind-audit trap, measured --------------------------------------
    print("\nCOLLECTION x MARKUP disagreements worth reading (words):")
    for (d, mk), w in disagree.most_common():
        odd = (d == "verse-side" and mk != "verse") or \
              (d == "drama-side" and mk != "drama") or \
              (d in ("letters", "diary", "essay") and mk != "prose")
        if odd and w > 5000:
            print(f"  {d:14s} classified as {mk:6s} {w:9d} words")


if __name__ == "__main__":
    main()
