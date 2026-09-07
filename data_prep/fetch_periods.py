# Period labels for every author in the three corpora, so pairs can be matched
# for date instead of drawn at random.
#
# WHY: a cross-genre or cross-lingual same-author pair is also, almost always, a
# same-PERIOD pair, and function-word and syntactic-rhythm distributions drift
# with period. POSNoise and catsrank both preserve exactly the closed-class
# material that carries that drift, so neither removes the confound. Without
# date-matched impostors an above-chance result cannot be attributed to
# authorship rather than era.
#
# SOURCES (two of three need no network -- the harvest caches hold the originals)
#   novels   ELTeC     data_prep/cache/eltec/ELTeC-<repo>.zip -> <repo>_metadata.tsv
#                      (author-birth, author-death, reference-year, first-edition,
#                      and ELTeC's own time-slot band)
#   poetree  PoeTree   data_prep/cache/poetree/<code>.zip -> per-poem JSON
#                      (year_created, source.year_published, author.born/died)
#   dracor   DraCor    API /corpora/<c> for plays+authors and /corpora/<c>/metadata
#                      for yearNormalized|Written|Printed|Premiered, cached to
#                      data_prep/cache/periods/
#
# THE JOIN IS THE DELICATE PART. A bank file is named
# `{index:03d}_{slug(author)}.tsv` by mask_corpora.py, where slug() lowercases and
# maps every non-[a-z0-9] run to "_" WITHOUT transliterating -- so Moellhausen
# becomes `m_llhausen`. The TM drivers then key on namekey(): strip the leading
# index, split on "_", sort the parts. This script reuses each fetcher's own
# author normaliser and replicates slug/namekey exactly, then REPORTS the match
# rate against the real bank filenames. A period table that does not join is
# worse than none, so the coverage report is not optional output.
#
#   python data_prep/fetch_periods.py                  # everything in data/
#   python data_prep/fetch_periods.py --langs german,czech
#   python data_prep/fetch_periods.py --genres novels,poetree   # skip the network
#
# Output: data_prep/periods/work_periods.tsv    one row per work
#         data_prep/periods/author_periods.tsv  one row per (folder, author)
#         data_prep/periods/coverage.json       match rate against masked/*/bank

import argparse
import json
import re
import statistics
import sys
import time
import unicodedata
import zipfile
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

from _net import get  # noqa: E402  (also injects the OS trust store)
from fetch_eltec import LANGS as ELTEC_LANGS, norm_author, repos_for  # noqa: E402

DATA = ROOT / "data"
MASKED = ROOT / "masked"
CACHE = HERE / "cache"
OUT = HERE / "periods"
API = "https://dracor.org/api/v1"


def slug(s):
    """Byte-identical to mask_corpora.slug -- the bank filenames depend on it."""
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")[:60] or "x"


def namekey(stem):
    """Byte-identical to the namekey() in the Julia drivers."""
    s = re.sub(r"^\d+_", "", stem)
    return "_".join(sorted(p for p in s.split("_") if p))


def author_key_of(stem):
    return namekey(stem)


def _int(v):
    if v is None:
        return None
    m = re.search(r"(1[0-9]{3}|20[0-9]{2})", str(v))
    return int(m.group(1)) if m else None


# ---------------------------------------------------------------- ELTeC (offline)
def eltec_tei_rows(z):
    """Fallback for works absent from the metadata TSV.

    The TSV lists only ELTeC's canonical 100-novel selection, but the harvest
    read every TEI file in the repo, so bank authors can exist with no TSV row
    (Conrad Ferdinand Meyer in ELTeC-deu is one, and he is a cross-genre author).
    Pull the author from <persName> and the year from the print-source bibl or a
    <date> attribute.
    """
    rows = []
    for n in z.namelist():
        if not n.endswith(".xml"):
            continue
        try:
            s = z.read(n).decode("utf-8", errors="replace")
        except Exception:
            continue
        head = s[:20000]
        m = re.search(r"<author>.*?<persName[^>]*>(.*?)</persName>", head, re.S)
        if not m:
            m = re.search(r"<author[^>]*>([^<]{3,80})</author>", head, re.S)
        if not m:
            continue
        raw = re.sub(r"<[^>]+>", " ", m.group(1))
        raw = " ".join(raw.split())
        if not raw:
            continue
        yr = None
        b = re.search(r'<bibl type="print_source">(.*?)</bibl>', head, re.S)
        if b:
            ys = re.findall(r"\b(1[5-9][0-9]{2}|20[0-2][0-9])\b", b.group(1))
            if ys:
                yr = int(ys[-1])
        if yr is None:
            d = re.search(r"<date[^>]*\bwhen=\"(\d{4})", head)
            if d:
                yr = int(d.group(1))
        rows.append({"author_raw": norm_author(raw), "work": Path(n).stem,
                     "year": yr, "year_field": "tei:print_source", "birth": None,
                     "death": None, "band": ""})
    return rows


def eltec_works(iso):
    """iso ('de') -> rows. ELTeC ships a metadata TSV per repo; extension repos
    (fra-ext1 etc.) were merged into the base language at harvest, so merge here.
    TSV rows win; TEI headers fill in works the TSV does not list."""
    repo_code = next((k for k, (_, i) in ELTEC_LANGS.items() if i == iso), None)
    if repo_code is None:
        return []
    rows = []
    for name in repos_for(repo_code):
        zp = CACHE / "eltec" / f"{name}.zip"
        if not zp.exists():
            continue
        with zipfile.ZipFile(zp) as z:
            seen_works = set()
            tsvs = [n for n in z.namelist() if n.endswith("_metadata.tsv")]
            for t in tsvs:
                lines = z.read(t).decode("utf-8", errors="replace").splitlines()
                if len(lines) < 2:
                    continue
                cols = lines[0].split("\t")
                ix = {c: i for i, c in enumerate(cols)}
                for line in lines[1:]:
                    f = line.split("\t")
                    if len(f) < len(cols):
                        continue
                    raw = f[ix.get("author-name", 0)]
                    if not raw or raw.upper() == "NA":
                        continue
                    year = (_int(f[ix["reference-year"]]) if "reference-year" in ix else None) \
                        or (_int(f[ix["first-edition"]]) if "first-edition" in ix else None)
                    rows.append({
                        "author_raw": norm_author(raw),
                        "work": f[ix.get("xmlid", 1)],
                        "year": year,
                        "year_field": "reference-year",
                        "birth": _int(f[ix["author-birth"]]) if "author-birth" in ix else None,
                        "death": _int(f[ix["author-death"]]) if "author-death" in ix else None,
                        "band": f[ix["time-slot"]] if "time-slot" in ix else "",
                    })
                    seen_works.add(f[ix.get("xmlid", 1)])
            for r in eltec_tei_rows(z):
                if r["work"] not in seen_works:
                    rows.append(r)
    return rows


# -------------------------------------------------------------- PoeTree (offline)
def poetree_works(code, max_poems=0):
    zp = CACHE / "poetree" / f"{code}.zip"
    if not zp.exists():
        return []
    rows = []
    with zipfile.ZipFile(zp) as z:
        members = [n for n in z.namelist() if n.endswith(".json")]
        if max_poems:
            members = members[:max_poems]
        for i, m in enumerate(members):
            try:
                p = json.loads(z.read(m))
            except Exception:
                continue
            a = p.get("author") or {}
            if isinstance(a, str):
                name, born, died = a, None, None
            else:
                name = a.get("name") or ""
                born, died = _int(a.get("born")), _int(a.get("died"))
            if not name:
                continue
            src = p.get("source") or {}
            year = _int(p.get("year_created")) or _int(src.get("year_published"))
            rows.append({
                "author_raw": " ".join(str(name).lower().split()),
                "work": p.get("id", ""),
                "year": year,
                "year_field": "year_created|source.year_published",
                "birth": born, "death": died, "band": "",
            })
            if i and i % 20000 == 0:
                print(f"    poetree {code}: {i}/{len(members)} poems", flush=True)
    return rows


# ---------------------------------------------------------------- DraCor (network)
def dracor_author_of(p):
    """Mirrors fetch_dracor.author_of: 'first last', lowercased."""
    au = (p.get("authors") or [{}])[0]
    name = au.get("name") or ""
    if not name:
        first, last = au.get("firstName") or "", au.get("lastName") or ""
        name = f"{first.strip()} {last.strip()}"
    return " ".join(name.lower().split())


def dracor_works(corpus):
    cdir = CACHE / "periods"
    cdir.mkdir(parents=True, exist_ok=True)
    cp, cm = cdir / f"{corpus}_plays.json", cdir / f"{corpus}_metadata.json"
    if cp.exists():
        plays = json.loads(cp.read_text(encoding="utf-8"))
    else:
        meta = get(f"{API}/corpora/{corpus}")
        if not meta:
            print(f"    dracor {corpus}: corpus endpoint unavailable", flush=True)
            return []
        plays = meta.get("plays", [])
        cp.write_text(json.dumps(plays, ensure_ascii=False), encoding="utf-8")
    if cm.exists():
        md = json.loads(cm.read_text(encoding="utf-8"))
    else:
        md = get(f"{API}/corpora/{corpus}/metadata") or []
        cm.write_text(json.dumps(md, ensure_ascii=False), encoding="utf-8")
    years = {}
    for r in md:
        y = (_int(r.get("yearNormalized")) or _int(r.get("yearWritten"))
             or _int(r.get("yearPrinted")) or _int(r.get("yearPremiered")))
        if r.get("name"):
            years[r["name"]] = y
    rows = []
    for p in plays:
        a = dracor_author_of(p)
        if not a or a == "anonymous":
            continue
        rows.append({"author_raw": a, "work": p.get("name", ""),
                     "year": years.get(p.get("name")), "year_field": "yearNormalized",
                     "birth": None, "death": None, "band": ""})
    return rows


# ------------------------------------------------------------------------ driver
def targets(langs, genres):
    """masked/{language}_{corpus} <- data/{language}/av_reference_{corpus}_{code}"""
    out = []
    if not DATA.is_dir():
        return out
    for d in sorted(DATA.iterdir()):
        if not d.is_dir() or (langs and d.name not in langs):
            continue
        for f in sorted(d.glob("av_reference_*.jsonl")):
            parts = f.stem.split("_", 3)
            if len(parts) < 4:
                continue
            corpus, code = parts[2], parts[3]
            if genres and corpus not in genres:
                continue
            out.append((d.name, corpus, code))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--langs", default="")
    ap.add_argument("--genres", default="novels,dracor,poetree")
    ap.add_argument("--max-poems", type=int, default=0,
                    help="cap poems scanned per PoeTree language (smoke tests)")
    args = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    langs = {x.strip() for x in args.langs.split(",") if x.strip()}
    genres = {x.strip() for x in args.genres.split(",") if x.strip()}
    OUT.mkdir(parents=True, exist_ok=True)

    tg = targets(langs, genres)
    print(f"targets: {len(tg)}")
    work_rows, author_rows, coverage = [], [], {}

    for language, corpus, code in tg:
        t0 = time.time()
        if corpus == "novels":
            rows = eltec_works(code)
        elif corpus == "poetree":
            rows = poetree_works(code, args.max_poems)
        elif corpus == "dracor":
            rows = dracor_works(code)
        else:
            rows = []
        folder = f"{language}_{corpus}"
        if not rows:
            print(f"  {folder:26s} no metadata ({time.time()-t0:.0f}s)", flush=True)
            continue

        by = defaultdict(list)
        for r in rows:
            by[r["author_raw"]].append(r)
        for a, rs in by.items():
            st = slug(a)
            nk = namekey(st)
            ys = sorted(x["year"] for x in rs if x["year"])
            births = [x["birth"] for x in rs if x["birth"]]
            deaths = [x["death"] for x in rs if x["death"]]
            bands = [x["band"] for x in rs if x["band"]]
            author_rows.append({
                "folder": folder, "language": language, "corpus": corpus,
                "author_raw": a, "slug": st, "namekey": nk,
                "n_works": len(rs), "n_dated": len(ys),
                "year_min": ys[0] if ys else "", "year_max": ys[-1] if ys else "",
                "year_med": int(statistics.median(ys)) if ys else "",
                "birth": births[0] if births else "", "death": deaths[0] if deaths else "",
                "band": bands[0] if bands else "",
            })
            for x in rs:
                work_rows.append({"folder": folder, "namekey": nk, "work": x["work"],
                                  "year": x["year"] or "", "year_field": x["year_field"]})

        # coverage against the bank the experiments actually read
        bank = MASKED / folder / "bank"
        have = {namekey(f.stem) for f in bank.glob("*.tsv")} if bank.is_dir() else set()
        known = {r["namekey"] for r in author_rows if r["folder"] == folder}
        dated = {r["namekey"] for r in author_rows
                 if r["folder"] == folder and r["year_med"] != ""}
        hit = len(have & known); hitd = len(have & dated)
        coverage[folder] = {"bank_authors": len(have), "metadata_authors": len(known),
                            "matched": hit, "matched_with_year": hitd,
                            "match_rate": round(hit / len(have), 3) if have else None,
                            "dated_rate": round(hitd / len(have), 3) if have else None}
        print(f"  {folder:26s} {len(rs) and len(rows):5d} works  "
              f"{len(known):4d} authors  bank={len(have):4d}  matched={hit:4d}  "
              f"dated={hitd:4d}  ({time.time()-t0:.0f}s)", flush=True)

    # ---- cross-corpus backfill -------------------------------------------------
    # Birth and death are properties of the AUTHOR, not of a work, so an author
    # who is undated in one corpus can borrow them from another corpus that has
    # them (PoeTree carries author.born/died; ELTeC carries author-birth/death;
    # DraCor carries neither). This matters: Conrad Ferdinand Meyer is absent
    # from the ELTeC-deu metadata yet is one of the cross-genre authors, so
    # without the backfill he would be unusable for date-matched pairing.
    lifespan = {}
    for r in author_rows:
        if r["birth"] != "" or r["death"] != "":
            lifespan.setdefault(r["namekey"], (r["birth"], r["death"], r["folder"]))
    filled = 0
    for r in author_rows:
        if r["birth"] == "" and r["death"] == "" and r["namekey"] in lifespan:
            b, d, src = lifespan[r["namekey"]]
            r["birth"], r["death"] = b, d
            r["lifespan_from"] = src
            filled += 1
        else:
            r.setdefault("lifespan_from", "")
    # a floruit estimate is better than nothing for date matching: mid-career is
    # conventionally taken around age 40
    for r in author_rows:
        r["floruit"] = r["year_med"] if r["year_med"] != "" else (
            int(r["birth"]) + 40 if r["birth"] != "" else "")
    print(f"cross-corpus backfill: {filled} authors given a lifespan from another corpus")

    def write_tsv(path, rows):
        if not rows:
            return
        cols = list(rows[0].keys())
        with open(path, "w", encoding="utf-8", newline="\n") as fh:
            fh.write("\t".join(cols) + "\n")
            for r in rows:
                fh.write("\t".join(str(r[c]) for c in cols) + "\n")
        print(f"wrote {path} ({len(rows)} rows)")

    write_tsv(OUT / "author_periods.tsv", author_rows)
    write_tsv(OUT / "work_periods.tsv", work_rows)
    (OUT / "coverage.json").write_text(json.dumps(coverage, indent=2), encoding="utf-8")
    print(f"wrote {OUT / 'coverage.json'}")

    bad = [k for k, v in coverage.items() if v["dated_rate"] is not None and v["dated_rate"] < 0.8]
    if bad:
        print("\nLOW DATE COVERAGE (<80% of bank authors carry a year):")
        for k in bad:
            print(f"  {k}: {coverage[k]['dated_rate']:.0%} "
                  f"({coverage[k]['matched_with_year']}/{coverage[k]['bank_authors']})")


if __name__ == "__main__":
    main()
