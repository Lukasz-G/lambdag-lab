# Real publication dates for the harvest, via Wikidata, joined on GND.
#
# WHY, and what it fixes. fetch_textgrid_periods.py can only offer the year of
# the EDITION TextGrid digitised, which for most authors collapses to a single
# modern date -- 185 of Karl May's 197 works read 2016 -- so the surviving years
# are useless and the fallback is a floruit guessed as birth + 35. That is
# enough to band an author but not to date his individual works, and dating
# works is what a period-controlled experiment needs: an author's verse and his
# prose are rarely contemporaneous, so an uncontrolled cross-genre comparison is
# partly a cross-decade one.
#
# Wikidata carries first-publication dates (P577) for literary works and links
# them to authors by P50. The join key is the GND identifier (P227), which the
# TextGrid edition records already give us, so no text is re-harvested and no
# name matching is needed -- name matching being exactly where identity errors
# enter.
#
# MEASURED WORTH: a period gap of two centuries between impostor and case is
# worth about -0.08 in per-token score, against roughly -0.05 for the whole
# cross-genre author effect. Left uncontrolled, period heterogeneity in the
# impostor pool inflates cross-genre results by around a third.
#
#   python data_prep/fetch_wikidata_periods.py
#   python data_prep/fetch_wikidata_periods.py --authors tieck_ludwig
#
# Output: data_prep/periods/wikidata_author_periods.tsv  (one row per author:
#         n_works, first/median/last publication year, source)
#         data_prep/periods/wikidata_work_periods.tsv    (one row per work)

import argparse
import json
import sys
import time
import urllib.parse
from pathlib import Path

import csv
import statistics

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from _net import get  # noqa: E402
from fetch_textgrid import CACHE, RAW, slug  # noqa: E402
from fetch_textgrid_periods import (CORPUS_LATEST, edition_record,  # noqa: E402
                                    objects_with_parents)

OUT = HERE / "periods"; OUT.mkdir(parents=True, exist_ok=True)
WD = CACHE / "wikidata"; WD.mkdir(parents=True, exist_ok=True)
ENDPOINT = "https://query.wikidata.org/sparql"

QUERY = """
SELECT ?work ?workLabel ?date WHERE {
  ?author wdt:P227 "%s" .
  ?work wdt:P50 ?author ; wdt:P577 ?date .
  SERVICE wikibase:label { bd:serviceParam wikibase:language "de,en". }
}
"""


def works_for(gnd):
    """[(title, year)] for one GND, cached. Empty list is a valid answer."""
    cf = WD / f"{gnd}.json"
    if cf.exists():
        return json.loads(cf.read_text(encoding="utf-8"))
    url = f"{ENDPOINT}?format=json&query=" + urllib.parse.quote(QUERY % gnd)
    d = get(url, as_json=True, timeout=120)
    if d is None:
        return None                      # distinguish failure from "no works"
    out = []
    for r in d.get("results", {}).get("bindings", []):
        y = r.get("date", {}).get("value", "")[:4]
        if y.isdigit():
            out.append([r.get("workLabel", {}).get("value", "")[:120], int(y)])
    cf.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    time.sleep(0.4)                      # the public endpoint is shared
    return out


def lifespans():
    """{slug: (birth, death)} from the GND lifespans already resolved."""
    out = {}
    f = OUT / "textgrid_author_periods.tsv"
    if not f.exists():
        return out
    for r in csv.DictReader(open(f, encoding="utf-8"), delimiter="	"):
        b = int(r["birth"]) if r.get("birth") else None
        d = int(r["death"]) if r.get("death") else None
        if b or d:
            out[r["slug"]] = (b, d)
    return out


def in_lifetime(years, span):
    """Keep only years the author could have published in.

    P577 is a publication date, not a FIRST-publication date, so a work carries
    a row for every edition Wikidata knows -- Lessing (d. 1781) came out with a
    median of 1888 and a span reaching 2016, Hans Sachs (d. 1576) with 1881.
    Unfiltered, these are worse than the lifespan estimate they would replace.
    The author's own dates are the only check available, and they are decisive.
    """
    b, d = span if span else (None, None)
    lo = b if b else 1400
    hi = (d + 5) if d else CORPUS_LATEST
    return [y for y in years if lo <= y <= hi]


def gnd_for(author_name):
    for o in objects_with_parents(author_name, cap=60)[:12]:
        for uri in (o.get("path_uris") or []):
            rec = edition_record(uri)
            if rec and rec.get("gnd"):
                return rec["gnd"]
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--authors", default="")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    idx = json.loads((RAW / "german_textgrid_index.json").read_text(encoding="utf-8"))
    todo = [a.strip() for a in args.authors.split(",") if a.strip()] or sorted(idx)

    SPANS = lifespans()
    arows, wrows, nofind = [], [], []
    for i, a in enumerate(todo, 1):
        if a not in idx:
            continue
        name = idx[a]["author_name"]
        gnd = gnd_for(name)
        works = works_for(gnd) if gnd else None
        if not works:
            nofind.append(a)
            print(f"[{i}/{len(todo)}] {name[:32]:32s} gnd={gnd or '-':>12s}  "
                  f"no dated works", flush=True)
            continue
        raw = sorted(w[1] for w in works)
        yrs = in_lifetime(raw, SPANS.get(a))
        ndrop = len(raw) - len(yrs)
        if not yrs:
            nofind.append(a)
            print(f"[{i}/{len(todo)}] {name[:32]:32s} {len(raw):4d} works, "
                  f"none within the author's lifetime", flush=True)
            continue
        # A publication year outside the author's plausible span is a data error
        # in either source; drop rather than let it move the median.
        arows.append({"slug": a, "author_raw": name, "gnd": gnd,
                      "n_works": len(yrs), "year_first": yrs[0],
                      "year_median": int(statistics.median(yrs)),
                      "year_last": yrs[-1], "n_dropped": ndrop,
                      "source": "wikidata-P577-lifetime"})
        keep = set(yrs)
        for t, y in works:
            if y in keep:
                wrows.append({"slug": a, "work": t, "year": y})
        print(f"[{i}/{len(todo)}] {name[:32]:32s} {len(yrs):4d} works  "
              f"{yrs[0]}-{yrs[-1]}  median {int(statistics.median(yrs))}"
              + (f"  (-{ndrop} outside lifetime)" if ndrop else ""),
              flush=True)

    def write(path, rows, cols):
        with open(path, "w", encoding="utf-8") as fh:
            fh.write("\t".join(cols) + "\n")
            for r in rows:
                fh.write("\t".join(str(r.get(c, "")) for c in cols) + "\n")
        print(f"  {len(rows):5d} rows -> {path.name}")

    print()
    write(OUT / "wikidata_author_periods.tsv", arows,
          ["slug", "author_raw", "gnd", "n_works", "year_first", "year_median",
           "year_last", "n_dropped", "source"])
    write(OUT / "wikidata_work_periods.tsv", wrows, ["slug", "work", "year"])
    print(f"\n{len(arows)}/{len(todo)} authors dated from real publication years; "
          f"{len(nofind)} without")


if __name__ == "__main__":
    main()
