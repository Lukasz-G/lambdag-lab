# Which manuscript stands behind each printed text, and how many there are.
#
# WHY THIS IS A SCRIPT AND NOT A PARAGRAPH. Every relational question in the
# Ambraser study is asked twice, once of the scribe's copy and once of the poem
# in a normalised edition, and what the second reading is worth depends entirely
# on what the editor had in front of him. Where the edition rests on a witness of
# its own, the second reading crosses the hand, the orthography AND the
# manuscript; where the poem survives in the codex alone, the editor can only
# have normalised the codex, and the second reading crosses the layer and
# nothing else. The two cases carry different weight and must be told apart in
# every table, so the fact is read out of the sources rather than remembered.
#
# TWO SOURCES, BOTH MACHINE-READABLE.
#
#   The edition. The corpus's TEI records the printed source of every text in
#   `sourceDesc`: the version, the editors, the series and, where the edition
#   states it, the manuscript it prints. That is where `Wolfdietrich A.
#   [handschrift A]` comes from.
#
#   The transmission. The same header carries the work's identifier in the
#   Handschriftencensus, whose work page lists every surviving witness. A poem
#   with one witness that is the codex cannot have an independent edition; the
#   count settles it without recourse to a handbook.
#
# The fetched pages are cached, so a rerun costs nothing and the table can be
# regenerated without a network.
#
#   python medieval/edition_provenance.py
#
# Output: medieval/kc_reuse/edition_provenance.tsv, and the same table printed

import csv
import re
import sys
import time
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
TEI = HERE / "raw" / "mhdbdb" / "tei"
CACHE = HERE / "raw" / "hsc"
OUT = ROOT / "medieval" / "kc_reuse" / "edition_provenance.tsv"
NS = "{http://www.tei-c.org/ns/1.0}"
UA = {"User-Agent": "authorship-verification-research/1.0"}
HSC = "https://handschriftencensus.de/werke/"

# the poems the study asks about, in the order the chapters take them
SIGLES = ["NBB", "KLA", "KU", "BRF", "ORT", "WDA", "RAB", "DFL", "CR", "WEN"]

# the four relational questions, as (known text, candidate, how it is named)
QUESTIONS = [
    ("NBB", "KLA", "the Klage against the Nibelungenlied"),
    ("DFL", "RAB", "the Rabenschlacht against Dietrichs Flucht"),
    ("WDA", "ORT", "Ortnit against Wolfdietrich"),
    ("KU", "NBB", "the Nibelungenlied against Kudrun"),
]

# the codex, as the Handschriftencensus names it, and its own description, which
# lists the contents with the siglum each poem carries in that poem's own
# tradition. This is what settles the question the work pages cannot: a work may
# survive in a dozen books while the VERSION an edition prints survives in one.
AMBRASER = "Ser. nova 2663"
AMBRASER_MS = "3766"

# a line of the codex's contents: a link to the poem, then -- outside the link,
# where an alternative title inside it cannot be mistaken for one -- the siglum
# this book carries in that poem's tradition, then the leaves it occupies
CONTENTS = re.compile(
    r'href="/werke/(\d+)">.*?</a>\s*(?:\(([A-Za-z]{1,3})\))?\s*\[Bl', re.S)


def txt(e, path):
    n = e.find(path)
    return " ".join((n.text or "").split()) if n is not None and n.text else ""


def fetch(ident, works=True):
    """a Handschriftencensus page, cached on disk

    `works` chooses between a work's page, which lists the copies that survive
    of it, and a manuscript's, which lists what that book contains.
    """
    CACHE.mkdir(parents=True, exist_ok=True)
    p = CACHE / (f"werke_{ident}.html" if works else f"hs_{ident}.html")
    url = f"{HSC}{ident}" if works else f"{HSC.rsplit('/werke/', 1)[0]}/{ident}"
    if p.exists() and p.stat().st_size > 0:
        return p.read_text(encoding="utf-8", errors="replace")
    for k in range(3):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=60) as fh:
                h = fh.read().decode("utf-8", errors="replace")
            p.write_text(h, encoding="utf-8")
            time.sleep(1)
            return h
        except (urllib.error.URLError, TimeoutError) as e:
            if k == 2:
                print(f"  could not fetch {url}: {e}")
                return ""
            time.sleep(2 * (k + 1))
    return ""


# one witness, as the work page lists it: a symbol naming the kind of book,
# then a link into the manuscript's own description carrying the shelfmark
WITNESS = re.compile(
    r'<li><a href="/abkverz"[^>]*title="([^"]*)"[^>]*>.*?'
    r'<a href="/\d+">([^<]+)</a>(.*?)</li>', re.S)


def witnesses(html):
    """(complete codices, fragments, lost, is the Ambraser among them)

    A poem's transmission decides what an edition of it can rest on, so the
    witnesses are counted by kind. A copy the page marks as lost or burnt is
    listed but cannot have been read by anyone since, and a fragment cannot
    carry a whole poem; neither can make an edition independent of the codex,
    so both are counted separately from the complete books.
    """
    if not html:
        return 0, 0, 0, False
    full = frag = gone = 0
    seen_ambraser = False
    for kind, shelf, tail in WITNESS.findall(html):
        lost = "verschollen" in tail or "verbrannt" in tail
        if lost:
            gone += 1
        elif "fragment" in kind.lower():
            frag += 1
        else:
            full += 1
        if AMBRASER in shelf:
            seen_ambraser = True
    return full, frag, gone, seen_ambraser


def decide(sig, codices, in_amb, codex_siglum, version, edition):
    """can the printed text rest on a book other than this codex?

    Three sourced facts answer it, and they are kept beside the answer so that a
    reader can see the reasoning rather than take the verdict on trust.

      the edition's own words. Where the title names the manuscript it prints,
      that settles the matter outright, in either direction: Amelung and
      Jaenicke print `Wolfdietrich A. [handschrift A]`, and A is this codex.
      Where it names several, as Batts's parallel print of A, B and C does, the
      edition is independent whatever else is true.

      how many copies survive. A poem with one complete copy, and that copy in
      this book, cannot have an edition that is anything but this book
      normalised.

      what the codex is called in that poem's tradition. A work may survive in a
      dozen books while the version an edition prints survives in one, and the
      siglum is what connects the edition's version to this copy.
    """
    named = re.findall(r"Handschriften?\s+([A-Z](?:,?\s*[A-Zu]+\.?\s*[A-Z])*)",
                       edition)
    if named:
        return dict(independent="yes",
                    grounds=f"the edition prints manuscripts {named[0]}")
    m = re.search(r"\[handschrift\s+([A-Z])\]", version, re.I)
    if m:
        same = codex_siglum and m.group(1) == codex_siglum
        return dict(independent="no" if same else "yes",
                    grounds=(f"the edition prints manuscript {m.group(1)}, "
                             f"which is this codex" if same else
                             f"the edition prints manuscript {m.group(1)}"))
    if in_amb and codices <= 1:
        return dict(independent="no",
                    grounds="the poem survives in this codex alone")
    if codices > 1:
        return dict(independent="yes",
                    grounds=f"{codices} complete copies survive")
    return dict(independent="unknown", grounds="")


def codex_sigla():
    """the siglum the codex carries for each poem it contains

    The census prints the contents as `'Wolfdietrich' A`, where the letter is
    this book's siglum in that poem's tradition. Amelung and Jaenicke's edition
    says it prints `Wolfdietrich A. [handschrift A]`, so for that poem the
    version named in the edition and the copy in this codex are the same thing,
    and the printed text can only be this codex normalised -- although the work
    as a whole survives in eleven books.
    """
    html = fetch(AMBRASER_MS, works=False)
    out = {}
    for work_id, siglum in CONTENTS.findall(html):
        out.setdefault(work_id, siglum or "")
    return out


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    sigla = codex_sigla()
    out = []
    for sig in SIGLES:
        f = TEI / f"{sig}.tei.xml"
        if not f.exists():
            print(f"  {sig}: no TEI on disk")
            continue
        r = ET.parse(f).getroot()
        sd = r.find(f".//{NS}sourceDesc")
        wid = txt(sd, f'.//{NS}idno[@type="handschriftencensus"]')
        eds = [" ".join(x for x in (txt(e, f"{NS}forename"),
                                    txt(e, f"{NS}surname")) if x)
               for e in sd.iter(f"{NS}editor")]
        full, frag, gone, in_amb = (witnesses(fetch(wid)) if wid
                                    else (0, 0, 0, False))
        out.append(dict(
            siglum=sig,
            work=txt(sd, f".//{NS}msName"),
            # the edition's own description of what it prints; for a poem
            # transmitted in several versions this names the version, and
            # sometimes the very manuscript
            version=txt(sd, f'.//{NS}title[@level="a"]'),
            edition=txt(sd, f'.//{NS}title[@level="m"]'),
            series=txt(sd, f'.//{NS}title[@level="s"]'),
            editors="; ".join(eds),
            year=txt(sd, f".//{NS}date"),
            hsc_work=wid,
            codices=full,
            fragments=frag,
            lost=gone,
            ambraser_among_them="yes" if in_amb else "no",
            codex_siglum=sigla.get(wid, ""),
            **decide(sig, full, in_amb, sigla.get(wid, ""),
                     txt(sd, f'.//{NS}title[@level="a"]'),
                     txt(sd, f'.//{NS}title[@level="m"]')),
        ))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    cols = list(out[0]) if out else []
    with OUT.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, delimiter="\t",
                           lineterminator="\n")
        w.writeheader()
        w.writerows(out)

    print(f"{'sig':5} {'work':23} {'cod':>4} {'frg':>4} {'in codex as':>12} "
          f"{'independent':>12}   grounds")
    for r in out:
        print(f"{r['siglum']:5} {r['work'][:23]:23} {r['codices']:4} "
              f"{r['fragments']:4} {r['codex_siglum'] or '-':>12} "
              f"{r['independent']:>12}   {r['grounds']}")
    print(f"\n  wrote {OUT.relative_to(ROOT)}")
    print("  `independent` is whether the printed text can rest on a book other "
          "than the codex.\n  Where it cannot, the second reading of that "
          "question crosses the layer alone.")

    # what each relational question's second reading is therefore worth. Both
    # sides have to be independent for the second reading to be a second
    # WITNESS; where one is not, the arm still changes the hand and the
    # orthography, which is worth reporting and is not the same claim.
    by = {r["siglum"]: r for r in out}
    print(f"\n{'question':40} {'crosses'}")
    for known, cand, label in QUESTIONS:
        ok = [by[x]["independent"] == "yes" for x in (known, cand) if x in by]
        crosses = ("the layer, the hand and the manuscript" if all(ok) else
                   "the layer and the hand alone" if not any(ok) else
                   "the layer throughout, the manuscript on one side")
        print(f"  {label[:38]:38} {crosses}")


if __name__ == "__main__":
    main()
