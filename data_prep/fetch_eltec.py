# ELTeC (COST Action CA16204 "Distant Reading") -> one JSONL per language, in the
# schema the prepr_data_novels notebook consumes ({author_id, author_name, id_source, text}).
#
# Each language repo is downloaded once as a zip from GitHub; level-1 TEI files are
# parsed with the standard library only. Extension repos are MERGED into their base
# language (ELTeC-fra + fra-ext1/2/3 -> french), because the brief asks for one corpus
# per language and the extensions are simply further novels of the same collection.
#
#   python data_prep/fetch_eltec.py                  # every language
#   python data_prep/fetch_eltec.py --langs fra,hun  # selected base codes
#
# Output: data_prep/raw/eltec/{language}_preprocessed.jsonl  (one line per author)

import argparse, json, re, sys, unicodedata, zipfile
from collections import defaultdict
from pathlib import Path
from xml.etree import ElementTree as ET

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from _net import download, get

RAW = HERE / "raw" / "eltec"; RAW.mkdir(parents=True, exist_ok=True)
ZIPS = HERE / "cache" / "eltec"; ZIPS.mkdir(parents=True, exist_ok=True)
TEI = "{http://www.tei-c.org/ns/1.0}"

# ELTeC three-letter code -> (folder name, ISO-639-1 used in output filenames)
LANGS = {
    "cze": ("czech", "cs"),      "deu": ("german", "de"),      "eng": ("english", "en"),
    "eus": ("basque", "eu"),     "fra": ("french", "fr"),      "geo": ("georgian", "ka"),
    "gle": ("irish", "ga"),      "gre": ("greek", "el"),       "gsw": ("swissgerman", "gsw"),
    "hrv": ("croatian", "hr"),   "hun": ("hungarian", "hu"),   "ita": ("italian", "it"),
    "lav": ("latvian", "lv"),    "lit": ("lithuanian", "lt"),  "nld": ("dutch", "nl"),
    "nor": ("norwegian", "no"),  "pol": ("polish", "pl"),      "por": ("portuguese", "pt"),
    "rom": ("romanian", "ro"),   "rus": ("russian", "ru"),     "slv": ("slovenian", "sl"),
    "spa": ("spanish", "es"),    "srp": ("serbian", "sr"),     "swe": ("swedish", "sv"),
    "ukr": ("ukrainian", "uk"),
}


def repos_for(lang):
    """Base repo plus any extension repos of the same language."""
    names = [f"ELTeC-{lang}"]
    for suf in ("-ext", "-ext1", "-ext2", "-ext3"):
        names.append(f"ELTeC-{lang}{suf}")
    return names


def norm_author(name):
    """'Christen, Ada' -> 'ada christen'; strip dates and bracketed notes."""
    name = re.sub(r"\(.*?\)|\[.*?\]", " ", name)
    name = re.sub(r"\d{3,4}\s*-\s*\d{0,4}", " ", name)
    name = name.replace(".", " ").strip(" ,;")
    if "," in name:
        last, _, first = name.partition(",")
        name = f"{first.strip()} {last.strip()}"
    name = " ".join(name.lower().split())
    return unicodedata.normalize("NFC", name)


def text_of(root):
    """Plain running text of the TEI <body>, paragraph by paragraph."""
    body = root.find(f".//{TEI}text/{TEI}body")
    if body is None:
        return ""
    parts = []
    for p in body.iter():
        if p.tag in (f"{TEI}p", f"{TEI}l"):
            s = " ".join(t.strip() for t in p.itertext() if t and t.strip())
            if s:
                parts.append(s)
    return "\n".join(parts)


def author_of(root):
    for path in (f".//{TEI}titleStmt/{TEI}author", f".//{TEI}sourceDesc//{TEI}author"):
        el = root.find(path)
        if el is not None:
            ref = el.get("ref") or ""
            name = " ".join(t.strip() for t in el.itertext() if t and t.strip())
            if name:
                return norm_author(name), name.strip(), ("ref" if ref else "name")
    return None, None, None


def harvest(lang, folder):
    """Download every repo of `lang`, parse level-1 TEI, aggregate text per author."""
    by_author, names, srcs, nfiles = defaultdict(list), {}, {}, 0
    for repo in repos_for(lang):
        zp = ZIPS / f"{repo}.zip"
        if not zp.exists():
            ok = None
            for branch in ("master", "main"):
                ok = download(f"https://codeload.github.com/COST-ELTeC/{repo}/zip/refs/heads/{branch}", zp)
                if ok:
                    break
            if not ok:
                continue                                   # extension repo does not exist
        try:
            zf = zipfile.ZipFile(zp)
        except zipfile.BadZipFile:
            zp.unlink(missing_ok=True); continue
        members = [m for m in zf.namelist()
                   if m.lower().endswith(".xml") and "/level1/" in m.lower().replace("\\", "/")]
        if not members:                                    # some repos keep TEI at the root
            members = [m for m in zf.namelist() if m.lower().endswith(".xml")
                       and "/schema" not in m.lower() and "/doc" not in m.lower()]
        for m in members:
            try:
                root = ET.fromstring(zf.read(m))
            except ET.ParseError:
                continue
            key, disp, src = author_of(root)
            if not key or key in ("anonymous", "anonym", "unknown"):
                continue
            txt = text_of(root)
            if len(txt.split()) < 100:
                continue
            by_author[key].append(txt); names[key] = disp; srcs[key] = src; nfiles += 1
    return by_author, names, srcs, nfiles


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--langs", default="", help="comma-separated ELTeC codes (default: all)")
    args = ap.parse_args()
    todo = [l.strip() for l in args.langs.split(",") if l.strip()] or list(LANGS)

    summary = []
    for lang in todo:
        if lang not in LANGS:
            print(f"unknown ELTeC code {lang!r}, skipped", flush=True); continue
        folder, iso = LANGS[lang]
        by_author, names, srcs, nfiles = harvest(lang, folder)
        if not by_author:
            print(f"{lang:4s} -> no usable texts", flush=True); continue
        out = RAW / f"{folder}_preprocessed.jsonl"
        nwords = 0
        with open(out, "w", encoding="utf-8") as fh:
            for a in sorted(by_author):
                text = "\n".join(by_author[a])
                nwords += len(text.split())
                fh.write(json.dumps({"author_id": a, "author_name": names[a],
                                     "id_source": srcs[a], "text": text}, ensure_ascii=False) + "\n")
        print(f"{lang:4s} {folder:12s} {nfiles:4d} novels  {len(by_author):4d} authors  "
              f"{nwords:>10,} words -> {out.name}", flush=True)
        summary.append((folder, iso, len(by_author), nwords))

    print(f"\n{len(summary)} languages written to {RAW}")


if __name__ == "__main__":
    main()
