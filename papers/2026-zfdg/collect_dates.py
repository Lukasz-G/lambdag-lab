# Manuscript dates for every witness in the inventory, so that each witness
# can be placed in the period of the reference corpus that covers it (Middle
# High German to 1350, Early New High German after).
#
# The Heidelberg editions (Hartmann von Aue digital, Kaiserchronik digital)
# carry no date in their TEI; their witness pages are rendered from the
# library's search index, one query per witness, and that index is asked here
# directly. The title field names the witness ("Iwein (A)", "Kaiserchronik B
# (VB)"); for the Kaiserchronik the siglum is taken from the TEI file's own
# editorial siglum instead, keyed by the digitisation identifier, since the
# index titles of composite manuscripts name no siglum. The Bern Parzival
# metadata carries the date inside the descriptive prose of each witness.
# The Ambraser Heldenbuch was written 1504-1516.
#
#   python medieval/collect_dates.py
#
# Output: medieval/raw/witness_dates.json
#         {"<source>__<work>__<siglum>": {"date": ..., "year": ..., "manuscript": ...}}

import html
import json
import re
import sys
import urllib.parse
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "raw" / "witness_dates.json"
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
INDEX = "https://digi.ub.uni-heidelberg.de/Suchindex/cgi-bin/graphq.cgi"
PAGES = {
    ("hvad", "hartmann_von_aue__iwein", "Iwein"): "https://digi.ub.uni-heidelberg.de/de/iwd/edition/handschriften.html",
    ("hvad", "hartmann_von_aue__gregorius", "Gregorius"): "https://digi.ub.uni-heidelberg.de/de/grd/edition/handschriften.html",
    ("hvad", "hartmann_von_aue__der_arme_heinrich", "Der arme Heinrich"): "https://digi.ub.uni-heidelberg.de/de/ahd/edition/handschriften.html",
    ("kcd", "anon__kaiserchronik", "Kaiserchronik"): "https://digi.ub.uni-heidelberg.de/de/kcd/kaiserchronik/handschriften.html",
}
SUB = str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")


def fetch(url):
    return urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=90).read().decode("utf-8")


def plain(h):
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", h)))


def year_of(date):
    """a point estimate in years for period assignment"""
    m = re.search(r"1[2-6]\d{2}", date)
    if m:
        return int(m.group(0))
    m = re.search(r"(1[2-6])\.(?:\s*(?:/|oder)\s*(?:Anfang |Anf\. )?(1[2-6])\.)? ?Jh", date)
    if not m:
        return None
    c = (int(m.group(1)) - 1) * 100
    if m.group(2):
        return c + 100
    q = date.lower()
    if "1. viertel" in q: return c + 12
    if "2. viertel" in q: return c + 37
    if "3. viertel" in q: return c + 62
    if "4. viertel" in q: return c + 87
    if "1. drittel" in q: return c + 17
    if "2. drittel" in q: return c + 50
    if "3. drittel" in q: return c + 83
    if "1. hälfte" in q or "anfang" in q or "anf." in q: return c + 25
    if "2. hälfte" in q or "ende" in q: return c + 75
    return c + 50


def index_rows(page_url):
    """(digitisation id, title, date, manuscript) for every witness form on a page"""
    h = fetch(page_url)
    rows = []
    for i in sorted(set(re.findall(r"<input name=\"q\" value='id:([^' ]+)'", h))):
        u = INDEX + "?" + urllib.parse.urlencode({"ui_lang": "de_DE", "q": "id:" + i, "include_protected": "true"})
        try:
            j = json.loads(fetch(u))
        except Exception as e:
            print("   index error", i, e, file=sys.stderr)
            continue
        for d in j.get("_hl_meta", {}).get("response", {}).get("docs", []):
            if d.get("class_s") != "meta":
                continue
            rows.append((i.split("/")[-1].replace("\\", ""), " ".join(d.get("meta_title_txt", [])),
                         " ".join(d.get("meta_date_txt", [])),
                         ", ".join(" ".join(d.get(k, [])) for k in ("meta_physicallocation_txt", "meta_shelflocator_txt"))))
    return rows


def kcd_sigla():
    """digitisation id (file stem) -> editorial siglum, from the TEI files"""
    out = {}
    for f in (HERE / "raw" / "kcd").glob("*.tei.xml"):
        m = re.search(r'<altIdentifier ana="hc:EditorialSiglum"><idno>([^<]+)</idno>', f.read_text(encoding="utf-8")[:30000])
        if m:
            out[f.stem.replace(".tei", "")] = m.group(1).strip()
    return out


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    dates = {}
    sigla = kcd_sigla()
    for (src, work, label), url in PAGES.items():
        n = 0
        for did, title, date, ms in index_rows(url):
            if src == "kcd":
                sig = sigla.get(did)
                if sig is None:          # three files carry no editorial siglum in their header
                    m = re.search(r"Kaiserchronik(?: [ABC](?: \+ [ABC])?)? \(([A-Za-z0-9]+)\)", title)
                    sig = m.group(1) if m else None
                if sig:
                    sig = sig.replace("alpha", "α").replace("beta", "β").replace("gamma", "γ")
            else:
                m = re.search(re.escape(label) + r" \((?:Fragment |Handschrift )?([A-Za-z0-9₀-₉]+)\)", title)
                sig = m.group(1).translate(SUB) if m else None
                if sig is None and did == "cpg341":      # Der arme Heinrich Ba, a miscellany
                    sig = "Ba"
            if not sig or not date:
                if src == "kcd" and "Kaiserchronik" in title:
                    print("   undated or unnamed:", did, "|", title[:60], "|", date, file=sys.stderr)
                continue
            key = "%s__%s__%s" % (src, work, sig)
            if key not in dates:
                dates[key] = {"date": date, "year": year_of(date), "manuscript": ms}
                n += 1
        print("%-12s %-30s %3d witnesses dated" % (src, label, n))
    j = json.load((HERE / "raw" / "parzival" / "json" / "metadata-nomenclature.json").open(encoding="utf-8"))
    DATE = re.compile(r"((?:Anfang|Mitte|Ende|[123]\. ?(?:Drittel|Hälfte|Viertel)|um|nach|vor|ca\.)?\s*(?:des )?"
                      r"1[2-6]\.(?:/1[2-6]\.)? ?Jh\.?|um 1[2-6]\d{2}|1[2-6]\d{2}(?:/\d{2,4})?)")
    n = 0
    for sec in ("codices", "fragments"):
        for w in j[sec]:
            d = DATE.search(plain(w.get("info", "")))
            if d:
                loc = ", ".join(p["loc"] + " " + p["id"] for p in w.get("part", []))
                dates["parzival__wolfram_von_eschenbach__parzival__%s" % w["handle"]] = {
                    "date": d.group(0).strip(), "year": year_of(d.group(0)), "manuscript": loc}
                n += 1
    print("%-12s %-30s %3d witnesses dated" % ("parzival", "Parzival", n))
    dates["ambraser__*"] = {"date": "1504-1516 (Hans Ried)", "year": 1510,
                            "manuscript": "Wien, Österreichische Nationalbibliothek, Cod. ser. nova 2663"}
    OUT.write_text(json.dumps(dates, ensure_ascii=False, indent=1), encoding="utf-8")
    print("%d entries -> %s" % (len(dates), OUT))


if __name__ == "__main__":
    main()
