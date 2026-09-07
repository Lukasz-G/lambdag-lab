# TextGrid Repository "Digitale Bibliothek" -> per-genre JSONL, in the schema the
# other fetchers emit ({author_id, author_name, id_source, text}).
#
# WHY THIS SOURCE. The cross-genre experiments need authors attested in more than
# one genre, and our three existing corpora each cover one genre only, so a
# cross-genre author has to be recovered by matching names ACROSS corpora -- which
# is exactly where the identity risk sits. The Digitale Bibliothek carries prose,
# verse and drama for the same ~600 German-language authors inside ONE catalogue
# with one author authority, so a multi-genre author is established without any
# cross-source name matching at all.
#
# ACCESS (probed 2026-09-07, all unauthenticated):
#   author list   GET /1.0/tgsearch-public/search?q=*&limit=0
#                     &facet=edition.agent.value&facetlimit=N
#   objects       GET /1.0/tgsearch-public/search?filter=edition.agent.value:<name>
#                     &filter=format:text/xml&limit=..&start=<next>&path=true
#   TEI payload   GET /1.0/tgcrud-public/rest/<textgrid-uri>/data
# Paging past 10,000 hits must use the `next` attribute of the previous response,
# not a numeric offset. The /1.0/aggregator/teicorpus/ route also exists but times
# out on large aggregations, so leaf `text/xml` objects are fetched directly.
#
# GENRE is NOT a reliable metadata field here, so it is read off the TEI markup
# itself, which is the more robust signal anyway: <sp>/<speaker> => drama,
# a high <l>/<lg> density => verse, otherwise prose. See classify().
#
# LICENCE: the Digitale Bibliothek is a TextGrid modification of the editura.de
# (zeno.org) data stock, published under CC BY 3.0 DE; the underlying texts are
# public domain. Attribution belongs in the paper's data-availability statement.
#
# ELTeC editions are ALSO hosted in TextGrid and are dropped here, because those
# novels already reach us through fetch_eltec.py and would otherwise be counted
# twice (see drop_eltec()).
#
#   python data_prep/fetch_textgrid.py --list-authors        # enumerate + cache
#   python data_prep/fetch_textgrid.py --min-objects 20      # harvest (default)
#   python data_prep/fetch_textgrid.py --authors "Tieck, Ludwig"
#
# Output: data_prep/raw/textgrid/german_tg{prose,verse,drama}_preprocessed.jsonl
#         plus german_textgrid_index.json (per author/genre object and word counts)

import argparse
import json
import re
import sys
import time
import unicodedata
from collections import defaultdict
from pathlib import Path
from urllib.parse import quote
from xml.etree import ElementTree as ET

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from _net import get  # noqa: E402

RAW = HERE / "raw" / "textgrid"; RAW.mkdir(parents=True, exist_ok=True)
CACHE = HERE / "cache" / "textgrid"; CACHE.mkdir(parents=True, exist_ok=True)
TEIC = CACHE / "tei"; TEIC.mkdir(parents=True, exist_ok=True)

BASE = "https://textgridlab.org/1.0"
SEARCH = f"{BASE}/tgsearch-public/search"
CRUD = f"{BASE}/tgcrud-public/rest"
TEI = "{http://www.tei-c.org/ns/1.0}"
NS = {"tgs": "http://www.textgrid.info/namespaces/middleware/tgsearch"}

GENRES = ("prose", "verse", "drama")


# ---- author enumeration ----------------------------------------------------

def list_authors(limit=4000, refresh=False):
    """All edition.agent.value facet values, as {name: object count}."""
    cf = CACHE / "authors.json"
    if cf.exists() and not refresh:
        return json.loads(cf.read_text(encoding="utf-8"))
    url = (f"{SEARCH}?q=*&limit=0&facet=edition.agent.value"
           f"&facetlimit={limit}")
    xml = get(url, as_json=False, timeout=180)
    if not xml:
        return {}
    out = {}
    for m in re.finditer(r'<tgs:facet count="(\d+)">(.*?)</tgs:facet>', xml, re.S):
        out[unescape(m.group(2))] = int(m.group(1))
    cf.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    return out


def unescape(s):
    for a, b in (("&amp;", "&"), ("&lt;", "<"), ("&gt;", ">"), ("&quot;", '"'),
                 ("&apos;", "'")):
        s = s.replace(a, b)
    return s.strip()


def namekey(s):
    """Order-insensitive token set, so 'Tieck, Ludwig' == 'ludwig_tieck'.
    PoeTree lists the surname first and our bank stems the given name first, so
    matching has to ignore order; the slug already folds diacritics and ß."""
    return frozenset(t for t in slug(s).split("_") if len(t) > 1)


def want_list(dirs):
    """Bank stems of the genres we already hold -> TextGrid author names.

    The cross-genre pool is limited by IDENTITY, not by tokens: German has 127
    drama and 114 poetry bank authors but only 28 novelists, so the useful
    harvest is the authors we already hold in one genre and lack in another.
    """
    want = {}
    for d in dirs:
        p = Path(d)
        if not p.is_dir():
            print(f"  (no such bank dir: {d})", flush=True)
            continue
        for f in sorted(p.glob("*.tsv")):
            stem = re.sub(r"^\d+_", "", f.stem)
            k = namekey(stem)
            if len(k) >= 2:
                want.setdefault(k, stem)
    return want


# ---- object listing --------------------------------------------------------

def objects_for(author, cap=600, page=100):
    """Leaf text/xml objects of one author: [{uri, title, date, path}]."""
    cf = CACHE / f"objs_{slug(author)}.json"
    if cf.exists():
        return json.loads(cf.read_text(encoding="utf-8"))
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
    cf.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    return out


def parse_result(res):
    uri = res.findtext(".//{*}textgridUri")
    if not uri:
        return None
    title = res.findtext(".//{*}title") or ""
    date = ""
    d = res.find(".//{*}dateOfPublication")
    if d is not None:
        date = d.get("date") or (d.text or "").strip()
    path = [e.text for e in res.findall(".//tgs:pathGroup//tgs:title", NS)
            if e.text]
    # Parent URIs as well as titles: a leaf text object carries NO publication
    # date of its own (verified), the date lives on the parent edition, so
    # fetch_textgrid_periods.py has to walk up this path to date the work.
    puris = [e.text for e in res.findall(".//tgs:pathGroup//tgs:textgridUri", NS)
             if e.text]
    return {"uri": uri.strip(), "title": title.strip(), "date": date,
            "path": path, "path_uris": [p.strip() for p in puris]}


# ---- author provenance, from the GND authority record -----------------------
#
# The want-list is matched on NAME, and our German drama and poetry banks contain
# foreign authors (staged in German production, anthologised in German
# translation). The Digitale Bibliothek then supplies them either in the original
# language or, worse, in GERMAN TRANSLATION -- which no language test can catch,
# because the text really is German. Its grammatical habit is the translator's,
# and several of our own authors WERE the translators (Tieck did Shakespeare), so
# a translated foreign author puts one hand in the bank under two identities.
# The check therefore has to be on the AUTHOR, not the text: the GND authority
# record gives a language code and a geographic area, and lobid.org resolves it.
GERMAN_AREAS = {
    "deutschland", "österreich", "schweiz", "liechtenstein", "luxemburg",
    "preußen", "bayern", "sachsen", "württemberg", "baden", "böhmen",
    "deutsches reich", "heiliges römisches reich", "österreich-ungarn",
    "elsass", "schlesien", "ostpreußen", "siebenbürgen", "baltikum",
}
EDITOR_IN_NAME = re.compile(r"\((hg|hrsg|ed|red)\.?\)", re.I)


def _gnd_of(objs, edition_lookup, limit=8):
    for o in objs[:limit]:
        for uri in (o.get("path_uris") or []):
            rec = edition_lookup(uri)
            if rec and rec.get("gnd"):
                return rec["gnd"]
    return None


def gnd_record(gnd):
    """The lobid.org GND record, cached."""
    if not gnd:
        return None
    cf = CACHE / "info" / f"gnd_{gnd}.json"
    cf.parent.mkdir(parents=True, exist_ok=True)
    if cf.exists():
        return json.loads(cf.read_text(encoding="utf-8"))
    d = get(f"https://lobid.org/gnd/{gnd}.json", as_json=True, timeout=60)
    if not d:
        return None
    cf.write_text(json.dumps(d, ensure_ascii=False), encoding="utf-8")
    time.sleep(0.1)
    return d


def gnd_by_name(name):
    """GND records matching this author name exactly, when the edition records
    carry no id (common for the oldest entries -- which are exactly the
    pre-modern foreign authors the gate must catch).

    Returns ALL exact-name matches rather than the first. Names are ambiguous:
    "Keats, John" resolves to two people, an American and the British poet, and
    taking whichever came first would drop the right author on the wrong
    evidence. A verdict is only drawn when every candidate agrees.
    """
    cf = CACHE / "info" / f"gndsearch_{slug(name)}.json"
    cf.parent.mkdir(parents=True, exist_ok=True)
    if cf.exists():
        d = json.loads(cf.read_text(encoding="utf-8"))
    else:
        q = quote(f'preferredName:"{name}" and type:Person')
        d = get(f"https://lobid.org/gnd/search?q={q}&size=6&format=json",
                as_json=True, timeout=60)
        if d is None:
            return []
        cf.write_text(json.dumps(d, ensure_ascii=False), encoding="utf-8")
        time.sleep(0.1)
    want = " ".join(slug(name).split("_"))
    return [h for h in (d.get("member") or [])
            if " ".join(slug(h.get("preferredName", "")).split("_")) == want]


def _is_german(d):
    """None when the record says nothing either way."""
    langs = [str(x.get("label", "")).rsplit("/", 1)[-1].lower()
             for x in (d.get("languageCode") or [])]
    areas = [str(x.get("label", "")).lower()
             for x in (d.get("geographicAreaCode") or [])]
    if langs:
        return any(l.startswith("ger") or l.startswith("deu") for l in langs)
    if areas:
        return any(a in GERMAN_AREAS for a in areas)
    return None


def author_provenance(name, gnd):
    """('keep'|'drop', reason). Conservative: an author is dropped only on
    positive evidence of foreign provenance, never on absence of a record."""
    if EDITOR_IN_NAME.search(name):
        return ("drop", "name marks an editor/compiler, not an author")
    d = gnd_record(gnd)
    if d is not None:
        verdict = _is_german(d)
        if verdict is False:
            return ("drop", f"GND {gnd}: not a German-language author")
        if verdict is True:
            return ("keep", "")

    # No id on the edition, or the record is silent: fall back to exact-name
    # matches, and decide only if they ALL agree.
    hits = gnd_by_name(name)
    verdicts = [v for v in (_is_german(h) for h in hits) if v is not None]
    if verdicts and not any(verdicts):
        return ("drop", f"all {len(verdicts)} GND records for this name are "
                        f"non-German")
    return ("keep", "no decisive GND evidence" if not verdicts else "")


def drop_eltec(objs):
    """ELTeC editions reach us via fetch_eltec.py; do not count them twice."""
    return [o for o in objs
            if not any("eltec" in (t or "").lower()
                       for t in o["path"] + [o["title"]])]


# ---- TEI payload -----------------------------------------------------------

def fetch_tei(uri):
    dest = TEIC / f"{uri.replace(':', '_')}.xml"
    if dest.exists():
        return dest.read_text(encoding="utf-8", errors="replace")
    xml = get(f"{CRUD}/{uri}/data", as_json=False, timeout=90)
    if not xml:
        return None
    dest.write_text(xml, encoding="utf-8")
    return xml


# The TextGrid repository is NOT German-only, and the want-list can pull a
# foreign author in: "Shakespeare, William" matches our German drama bank (which
# holds him for German productions) and the Digitale Bibliothek serves his plays
# in ENGLISH -- 1.6M words of English entered german_tgdrama before this check
# existed. The parent records' <language> element is not reliably reachable from
# a leaf object, so the text is tested directly, which is self-validating and
# needs no metadata. Function words are the right probe here for the same reason
# they carry the authorship signal: they are frequent, closed-class, and survive
# any subject matter.
DE_FW = {"der", "die", "das", "und", "nicht", "sich", "ist", "zu", "den", "ein",
         "von", "mit", "dem", "es", "auf", "eine", "als", "auch", "wie", "aber"}
EN_FW = {"the", "and", "of", "to", "is", "that", "it", "in", "for", "with",
         "was", "his", "he", "as", "but", "not", "this", "have", "from", "they"}


def looks_german(text, sample=4000):
    """True when German function words clearly outweigh English ones."""
    toks = [w.strip(".,;:!?()[]«»\"'-–—").lower()
            for w in text.split()[:sample]]
    toks = [t for t in toks if t]
    if len(toks) < 50:
        return True                      # too short to judge; keep it
    de = sum(t in DE_FW for t in toks)
    en = sum(t in EN_FW for t in toks)
    return de >= max(en * 1.5, 0.02 * len(toks))


def classify(root):
    """drama / verse / prose, read off the markup rather than the metadata."""
    sp = len(root.findall(f".//{TEI}sp"))
    lines = len(root.findall(f".//{TEI}l"))
    paras = len(root.findall(f".//{TEI}p"))
    if sp >= 5:
        return "drama"
    if lines >= 8 and lines > paras:
        return "verse"
    return "prose"


# Roughly 60% of the harvested words come from DIPLOMATIC transcriptions that
# keep the Fraktur typography -- 176k long s, plus r rotunda. spaCy's German
# model is trained on modern orthography, so "Geſellſchaft" is out of vocabulary
# and mis-tagged, and POSNoise depends entirely on the tag. Worse, whether an
# author's text is diplomatic or modernised is a property of the EDITION, not of
# him, so leaving it in adds a transcription-convention confound that cuts across
# the corpus exactly as a genre or period effect would.
# Only typography is normalised here. Historical ORTHOGRAPHY ("Abtheilung",
# "seyn") is left alone: it is genuine language, and flattening it would erase
# part of the period signal the period axis is meant to measure.
HIST_CHARS = {"ſ": "s", "ꝛ": "r", "": "s"}
# German suspends hyphens across a conjunction ("Kunst- und Literaturgeschichte"),
# which must NOT be joined; a line-break hyphen must.
SUSPENDED = r"(?!und\b|oder\b|bzw\b|sowie\b|wie\b|als\b|noch\b|aber\b)"
HYPHEN_LB = re.compile(r"(\w)[-‐‑]\s+" + SUSPENDED + r"([a-zäöüß])")


def normalise_historical(text):
    """Fold Fraktur typography and repair line-break hyphenation."""
    for a, b in HIST_CHARS.items():
        if a in text:
            text = text.replace(a, b)
    return HYPHEN_LB.sub(r"\1\2", text)


def text_of(root):
    body = root.find(f".//{TEI}body")
    if body is None:
        return ""
    for bad in body.iter():
        if bad.tag in (f"{TEI}note", f"{TEI}figure", f"{TEI}speaker"):
            bad.clear()
    txt = " ".join(t for t in body.itertext() if t)
    txt = unicodedata.normalize("NFC", txt)
    txt = normalise_historical(txt)
    return re.sub(r"[ \t ]+", " ", txt).strip()


def slug(s):
    s = unicodedata.normalize("NFKD", s.lower())
    s = "".join(c for c in s if not unicodedata.combining(c))
    for a, b in (("ß", "ss"), ("ä", "ae"), ("ö", "oe"), ("ü", "ue")):
        s = s.replace(a, b)
    return re.sub(r"[^a-z0-9]+", "_", s).strip("_")[:60] or "x"


def dupkeys(text):
    """Near-duplicate keys. The Digitale Bibliothek carries the same work in
    several collections (Tieck: 30 same-title objects in a 200-object slice) and
    the copies differ in orthography and in edition-specific front matter, so a
    single hash of the opening misses most of them. Fold case and diacritics,
    keep letters only, then key on THREE windows at different depths and treat a
    match on any one as a duplicate -- front matter shifts the early window but
    not the later ones. Duplicates matter here beyond wasted tokens: the same
    text reaching both the known and the questioned side is leakage."""
    t = unicodedata.normalize("NFKD", text.lower())
    t = "".join(c for c in t if c.isalpha())
    if len(t) < 600:
        return {t}
    keys = {t[400:2400]}
    for off in (5000, 20000):
        if len(t) >= off + 2000:
            keys.add(t[off:off + 2000])
    return keys


# ---- harvest ---------------------------------------------------------------

def load_exclusions():
    """Per-author title exclusions, compiled. See textgrid_exclusions.json for
    why each author is listed: the Digitale Bibliothek reproduces third-party
    edited editions, unauthorised editions and an author's adaptations of other
    writers under his own name, which for authorship work is a contaminated
    identity label rather than a metadata nuisance."""
    f = HERE / "textgrid_exclusions.json"
    if not f.exists():
        return {}
    spec = json.loads(f.read_text(encoding="utf-8"))
    out = {}
    for author, v in spec.items():
        if author.startswith("_") or not isinstance(v, dict):
            continue
        pats = v.get("titles") or []
        if pats:
            out[author] = re.compile("|".join(pats), re.I)
    return out


def harvest(authors, cap, min_objects, min_words, drop_title=None,
            exclusions=None, provenance=None):
    per = {g: {} for g in GENRES}          # genre -> author -> [texts]
    index, rejected = {}, []
    exclusions = exclusions or {}
    for i, (name, total) in enumerate(authors, 1):
        objs = drop_eltec(objects_for(name, cap=cap))
        if len(objs) < min_objects:
            continue
        if provenance:
            gnd = _gnd_of(objs, provenance)
            verdict, why = author_provenance(name, gnd)
            if verdict == "drop":
                print(f"[{i}/{len(authors)}] {name:38s} SKIPPED -- {why}",
                      flush=True)
                rejected.append({"author": name, "gnd": gnd, "reason": why})
                continue
        buckets = defaultdict(list)
        dates, seen, ndup, nexcl, nstub, nforeign = [], set(), 0, 0, 0, 0
        audit = exclusions.get(slug(name))
        for o in objs:
            if drop_title and drop_title.search(o["title"]):
                nexcl += 1
                continue
            # Audited exclusions match the collection path as well as the title:
            # a work held as an aggregation exposes its text through leaf objects
            # titled by chapter, which no work-title regex would catch, but every
            # leaf carries the parent collection in its path.
            if audit and audit.search(" | ".join(o["path"] + [o["title"]])):
                nexcl += 1
                continue
            xml = fetch_tei(o["uri"])
            if not xml:
                continue
            try:
                root = ET.fromstring(xml)
            except ET.ParseError:
                continue
            t = text_of(root)
            if len(t.split()) < 20:
                # An aggregation: tgcrud returns its ORE/RDF manifest, not TEI,
                # so the work's text lives in leaf objects listed separately.
                # Counted because a high stub share means the cap is being spent
                # on manifests and the author's coverage is thinner than it looks.
                nstub += 1
                continue
            if not looks_german(t):      # foreign-language text under a
                nforeign += 1            # want-listed author; see looks_german
                continue
            ks = dupkeys(t)
            if ks & seen:            # same work under another collection/edition
                ndup += 1
                continue
            seen |= ks
            buckets[classify(root)].append(t)
            if o["date"]:
                dates.append(o["date"])
        kept = {}
        for g, texts in buckets.items():
            nw = sum(len(t.split()) for t in texts)
            if nw >= min_words:
                per[g][slug(name)] = texts
                kept[g] = {"objects": len(texts), "words": nw}
        if kept:
            index[slug(name)] = {"author_name": name, "repo_objects": total,
                                 "genres": kept, "duplicates_dropped": ndup,
                                 "titles_excluded": nexcl, "aggregation_stubs": nstub, "foreign_language": nforeign,
                                 "dates": sorted(set(dates))[:40]}
            print(f"[{i}/{len(authors)}] {name:38s} "
                  + "  ".join(f"{g}={kept[g]['words']:,}w" for g in GENRES
                              if g in kept)
                  + (f"  (-{ndup} dup)" if ndup else "")
                  + (f"  (-{nexcl} excl)" if nexcl else "")
                  + (f"  ({nstub} stubs)" if nstub else "")
                  + (f"  (-{nforeign} FOREIGN)" if nforeign else ""), flush=True)
    return per, index, rejected


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list-authors", action="store_true",
                    help="enumerate and cache the author facet, then exit")
    ap.add_argument("--authors", default="",
                    help="semicolon-separated exact TextGrid author names")
    ap.add_argument("--want-from",
                    default="masked/german_dracor/bank,masked/german_poetree/bank",
                    help="comma-separated bank dirs whose authors we want to "
                         "fill out in the other genres (default: the German "
                         "drama and poetry banks)")
    ap.add_argument("--top-authors", type=int, default=250,
                    help="cap on authors harvested, richest first (default 250)")
    ap.add_argument("--cap", type=int, default=600,
                    help="max objects listed per author (default 600)")
    ap.add_argument("--min-objects", type=int, default=5)
    ap.add_argument("--min-words", type=int, default=8000,
                    help="minimum words for a genre to be kept (default 8000)")
    ap.add_argument("--no-provenance", action="store_true",
                    help="skip the GND author-provenance gate (see "
                         "author_provenance): foreign authors reached through "
                         "German translation are otherwise dropped")
    ap.add_argument("--drop-title",
                    default=r"shakespeare|quijote|quixote|übersetz|uebersetz|"
                            r"nach dem englischen|nach dem franz",
                    help="regex on titles to exclude; guards against works the "
                         "author TRANSLATED being ingested as his own writing")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    known = list_authors()
    print(f"{len(known)} authors in the TextGrid author facet", flush=True)
    if args.list_authors:
        for n, c in sorted(known.items(), key=lambda kv: -kv[1])[:40]:
            print(f"  {c:7,d}  {n}")
        return

    if args.authors:
        wanted = [(n.strip(), known.get(n.strip(), 0))
                  for n in args.authors.split(";") if n.strip()]
    elif args.want_from:
        want = want_list(args.want_from.split(","))
        bykey = {}
        for n, c in known.items():
            bykey.setdefault(namekey(n), []).append((n, c))
        wanted, missing = [], []
        for k, stem in sorted(want.items(), key=lambda kv: kv[1]):
            hit = bykey.get(k)
            if hit:
                wanted.append(max(hit, key=lambda nc: nc[1]))
            else:
                missing.append(stem)
        print(f"want-list {len(want)} bank authors -> {len(wanted)} matched in "
              f"TextGrid, {len(missing)} absent", flush=True)
        (CACHE / "want_unmatched.json").write_text(
            json.dumps(sorted(missing), ensure_ascii=False, indent=1),
            encoding="utf-8")
        wanted = sorted(wanted, key=lambda nc: -nc[1])[:args.top_authors]
    else:
        wanted = sorted(known.items(), key=lambda kv: -kv[1])[:args.top_authors]
        wanted = [(n, c) for n, c in wanted if n.lower() != "anonymous"]

    drop = re.compile(args.drop_title, re.I) if args.drop_title else None
    excl = load_exclusions()
    if excl:
        print(f"title exclusions loaded for {len(excl)} audited author(s): "
              f"{', '.join(sorted(excl))}", flush=True)
    prov = None
    if not args.no_provenance:
        from fetch_textgrid_periods import edition_record as prov
    per, index, rejected = harvest(wanted, args.cap, args.min_objects,
                                   args.min_words, drop_title=drop,
                                   exclusions=excl, provenance=prov)

    for g in GENRES:
        if not per[g]:
            continue
        out = RAW / f"german_tg{g}_preprocessed.jsonl"
        nw = 0
        with open(out, "w", encoding="utf-8") as fh:
            for a in sorted(per[g]):
                text = "\n".join(per[g][a])
                nw += len(text.split())
                fh.write(json.dumps(
                    {"author_id": a, "author_name": index[a]["author_name"],
                     "id_source": "textgrid-digitale-bibliothek",
                     "text": text}, ensure_ascii=False) + "\n")
        print(f"{g:6s} {len(per[g]):4d} authors  {nw:>11,} words -> {out.name}")

    if rejected:
        (RAW / "german_textgrid_rejected.json").write_text(
            json.dumps(rejected, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"\n{len(rejected)} authors rejected on provenance:")
        for r in rejected:
            print(f"  {r['author'][:34]:34s} {r['reason']}")
    (RAW / "german_textgrid_index.json").write_text(
        json.dumps(index, ensure_ascii=False, indent=1), encoding="utf-8")
    multi = [a for a, v in index.items() if len(v["genres"]) >= 2]
    three = [a for a, v in index.items() if len(v["genres"]) == 3]
    print(f"\n{len(index)} authors kept; {len(multi)} in >=2 genres, "
          f"{len(three)} in all three")


if __name__ == "__main__":
    main()
