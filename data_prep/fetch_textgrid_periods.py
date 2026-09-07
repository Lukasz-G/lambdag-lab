# Period labels for the TextGrid harvest, in the schema fetch_periods.py emits.
#
# WHY A SEPARATE SCRIPT. fetch_periods.py covers ELTeC, DraCor and PoeTree, each
# of which ships a metadata table listing a year per work. TextGrid does not:
# a leaf text object's own metadata carries a title and a format and nothing
# else -- no dateOfPublication, no editionTitle, no agent (verified against
# textgrid:4bp60.0, "Durchs Wilde Kurdistan"). The bibliographic record lives on
# the PARENT EDITION, so dating a TextGrid work means walking up the aggregation
# path and reading <source><bibliographicCitation> there:
#
#   <dateOfPublication date="1862"/>
#   <editionTitle>Birlinger, Anton: Sitten und Gebraeuche ... 1862.</editionTitle>
#   <placeOfPublication><value>Breisgau</value></placeOfPublication>
#
# AND THE DATE IS USUALLY NOT A PUBLICATION DATE. Measured on Karl May: 185 of
# 197 works carry dateOfPublication 2016-06, which is when TextGrid built the
# digital edition. Taking that field at face value would date a nineteenth-
# century oeuvre to the twenty-first. So the year is accepted ONLY when it falls
# inside the author's lifetime, and discarded as a digitisation date otherwise --
# a test the record validates itself against.
#
# WHAT CARRIES THE PERIOD INSTEAD is the GND authority identifier, which the same
# citation does give reliably:
#
#   <author id="http://d-nb.info/gnd/118818651">May, Karl</author>
#
# resolved through lobid.org to birth and death dates (118818651 -> May, Karl,
# 1842-1912). Author lifespan is a weaker label than a first-edition year but it
# is the honest one here, and fetch_periods.py already carries the same
# lifespan_from/floruit fallback for corpora whose per-work dates are missing.
#
#   python data_prep/fetch_textgrid_periods.py
#   python data_prep/fetch_textgrid_periods.py --authors may_karl,tieck_ludwig
#
# Output: data_prep/periods/textgrid_work_periods.tsv    folder/namekey/work/year
#         data_prep/periods/textgrid_author_periods.tsv  same columns as
#                                                        author_periods.tsv
# Both use the column names of fetch_periods.py's outputs so the two can simply
# be concatenated.

import argparse
import json
import re
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path
from urllib.parse import quote
from xml.etree import ElementTree as ET

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from _net import get  # noqa: E402
from fetch_textgrid import (CACHE, RAW, SEARCH, drop_eltec, namekey,  # noqa: E402
                            parse_result, slug, NS)

OUT = HERE / "periods"; OUT.mkdir(parents=True, exist_ok=True)
INFO = CACHE / "info"; INFO.mkdir(parents=True, exist_ok=True)
BASE = "https://textgridlab.org/1.0/tgsearch-public"

GENRE_FOLDER = {"prose": "german_tgprose", "verse": "german_tgverse",
                "drama": "german_tgdrama"}


# ---- listings, upgraded in place to carry parent URIs -----------------------

def objects_with_parents(author, cap=200, page=100):
    """Cached listing for one author, re-fetched once if it predates the
    path_uris field (the early harvest kept path TITLES only, and a title is not
    addressable -- the parent edition has to be fetched by URI)."""
    cf = CACHE / f"objs_{slug(author)}.json"
    if cf.exists():
        objs = json.loads(cf.read_text(encoding="utf-8"))
        if objs and all("path_uris" in o for o in objs):
            return objs
    out, start = [], None
    while len(out) < cap:
        url = (f"{SEARCH}?q=*&limit={page}&path=true"
               f"&filter={quote('edition.agent.value:' + author)}"
               f"&filter={quote('format:text/xml')}")
        if start:
            url += f"&start={quote(start)}"
        xml = get(url, as_json=False, timeout=120)
        if not xml:
            break
        try:
            root = ET.fromstring(xml)
        except ET.ParseError:
            break
        got = 0
        for res in root.findall("tgs:result", NS):
            rec = parse_result(res)
            if rec:
                out.append(rec); got += 1
        nxt = root.get("next")
        if not got or not nxt or nxt == start:
            break
        start = nxt
        time.sleep(0.2)
    if out:
        cf.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    return out


# ---- the bibliographic record on the parent edition -------------------------

YEAR = re.compile(r"(1[0-9]{3}|20[0-2][0-9])")


def edition_record(uri):
    """{year, edition_title, place} for a TextGrid edition/collection URI."""
    cf = INFO / f"{uri.replace(':', '_')}.xml"
    if cf.exists():
        xml = cf.read_text(encoding="utf-8", errors="replace")
    else:
        xml = get(f"{BASE}/info/{uri}", as_json=False, timeout=60)
        if not xml:
            return None
        cf.write_text(xml, encoding="utf-8")
    try:
        root = ET.fromstring(xml)
    except ET.ParseError:
        return None
    d = root.find(".//{*}dateOfPublication")
    year, field = None, ""
    if d is not None:
        raw = d.get("date") or (d.text or "")
        m = YEAR.search(raw or "")
        if m:
            year, field = int(m.group(1)), "dateOfPublication"
    et = root.findtext(".//{*}editionTitle") or ""
    if year is None and et:
        # many records give the year only inside the citation string
        m = YEAR.findall(et)
        if m:
            year, field = int(m[-1]), "editionTitle"
    place = root.findtext(".//{*}placeOfPublication/{*}value") or ""
    gnd = ""
    a = root.find(".//{*}bibliographicCitation/{*}author")
    if a is not None:
        m = re.search(r"gnd/([0-9X\-]+)", a.get("id") or "")
        if m:
            gnd = m.group(1)
    return {"year": year, "year_field": field, "gnd": gnd,
            "edition_title": " ".join(et.split())[:220], "place": place.strip()}


# ---- author lifespan from the GND authority record --------------------------

def gnd_lifespan(gnd):
    """(birth, death) as years, via lobid.org's GND API. Cached."""
    if not gnd:
        return (None, None)
    cf = INFO / f"gnd_{gnd}.json"
    if cf.exists():
        d = json.loads(cf.read_text(encoding="utf-8"))
    else:
        d = get(f"https://lobid.org/gnd/{gnd}.json", as_json=True, timeout=60)
        if not d:
            return (None, None)
        cf.write_text(json.dumps(d, ensure_ascii=False), encoding="utf-8")
        time.sleep(0.1)

    def yr(key):
        v = d.get(key) or []
        if isinstance(v, str):
            v = [v]
        for s in v:
            m = YEAR.search(str(s))
            if m:
                return int(m.group(1))
        return None

    return (yr("dateOfBirth"), yr("dateOfDeath"))


def date_objects(objs):
    """Attach a year to each leaf by walking up its aggregation path."""
    cacheh, rows = {}, []
    for o in objs:
        rec = None
        for uri in reversed(o.get("path_uris") or []):   # nearest parent first
            if uri not in cacheh:
                cacheh[uri] = edition_record(uri)
                time.sleep(0.05)
            r = cacheh[uri]
            if r and r["year"]:
                rec = r
                break
        rows.append((o, rec))
    return rows


# ---- output ----------------------------------------------------------------

def band(y):
    """Coarse period band, matching the granularity ELTeC's time-slots use."""
    return f"{(y // 25) * 25}-{(y // 25) * 25 + 24}" if y else ""


def write_tsv(path, rows, cols):
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\t".join(cols) + "\n")
        for r in rows:
            fh.write("\t".join(str(r.get(c, "")) for c in cols) + "\n")
    print(f"  {len(rows):5d} rows -> {path.name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--authors", default="",
                    help="comma-separated author slugs (default: the whole "
                         "harvest index)")
    ap.add_argument("--cap", type=int, default=200)
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    idx_file = RAW / "german_textgrid_index.json"
    if not idx_file.exists():
        print("no harvest index; run fetch_textgrid.py first")
        return
    index = json.loads(idx_file.read_text(encoding="utf-8"))
    todo = [a.strip() for a in args.authors.split(",") if a.strip()] or sorted(index)

    work_rows, author_rows = [], []
    for i, a in enumerate(todo, 1):
        if a not in index:
            print(f"{a}: not in the harvest index, skipped", flush=True)
            continue
        meta = index[a]
        name = meta["author_name"]
        objs = drop_eltec(objects_with_parents(name, cap=args.cap))
        dated = date_objects(objs)
        nk = "_".join(sorted(namekey(name)))

        gnds = [r["gnd"] for _, r in dated if r and r.get("gnd")]
        gnd = max(set(gnds), key=gnds.count) if gnds else ""
        birth, death = gnd_lifespan(gnd)

        # A year is a composition date only if the author was alive for it;
        # anything later is TextGrid's digitisation date wearing the same tag.
        years, ndig = [], 0
        for o, r in dated:
            if not r or not r["year"]:
                continue
            y = r["year"]
            live = (birth is None or y >= birth) and (death is None or y <= death + 5)
            if not live:
                ndig += 1
                continue
            years.append(y)
            work_rows.append({"folder": "german_textgrid", "namekey": nk,
                              "work": o["title"][:120], "year": y,
                              "year_field": r["year_field"],
                              "edition_title": r["edition_title"],
                              "place": r["place"]})
        # floruit: measured work years when any survived the lifetime test,
        # otherwise mid-career from the lifespan -- birth+35 is the convention
        # fetch_periods.py uses, flagged so the two are never confused.
        if years:
            flor, src = int(statistics.median(years)), "works"
        elif birth:
            flor = min(birth + 35, death) if death else birth + 35
            src = "lifespan"
        else:
            flor, src = None, ""

        for g in meta.get("genres", {}):
            author_rows.append({
                "folder": GENRE_FOLDER.get(g, f"german_tg{g}"),
                "language": "german", "corpus": f"textgrid-{g}",
                "author_raw": name, "slug": a, "namekey": nk,
                "n_works": len(objs), "n_dated": len(years),
                "year_min": min(years) if years else "",
                "year_max": max(years) if years else "",
                "year_med": int(statistics.median(years)) if years else "",
                "birth": birth or "", "death": death or "",
                "band": band(flor), "lifespan_from": src, "floruit": flor or "",
                "gnd": gnd, "digitisation_dates_discarded": ndig})
        print(f"[{i}/{len(todo)}] {name:32s} {len(objs):4d} works "
              f"{len(years):4d} dated  "
              + (f"{birth}-{death}" if birth else "no lifespan")
              + f"  floruit {flor or '?'} ({src or 'none'})"
              + (f"  -{ndig} digitisation dates" if ndig else ""), flush=True)

    print()
    write_tsv(OUT / "textgrid_work_periods.tsv", work_rows,
              ["folder", "namekey", "work", "year", "year_field",
               "edition_title", "place"])
    write_tsv(OUT / "textgrid_author_periods.tsv", author_rows,
              ["folder", "language", "corpus", "author_raw", "slug", "namekey",
               "n_works", "n_dated", "year_min", "year_max", "year_med",
               "birth", "death", "band", "lifespan_from", "floruit", "gnd",
               "digitisation_dates_discarded"])
    tot = sum(r["n_works"] for r in author_rows)
    dat = sum(r["n_dated"] for r in author_rows)
    print(f"\ncoverage: {dat}/{tot} works dated "
          f"({100 * dat / tot:.0f}%)" if tot else "\nno works")
    nlife = sum(1 for r in author_rows if r["lifespan_from"] == "lifespan")
    print(f"{nlife}/{len(author_rows)} author rows fall back to lifespan "
          f"(birth+35) because every work year failed the lifetime test.")
    print("NB: a surviving work year is still an EDITION year, so it is an "
          "upper bound on composition -- not ELTeC's first-edition field.")


if __name__ == "__main__":
    main()
