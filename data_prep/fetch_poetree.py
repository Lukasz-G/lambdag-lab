# PoeTree (Zenodo record 17414036, v1.0.0 -- 11 languages) -> one JSONL of
# {corpus, author, text}, matching the schema the prepr_data_poetree notebook consumes.
#
# Each poem is a JSON file whose `body` is a list of verse LINES; every line carries the
# surface `text` plus `words`, a full Universal Dependencies analysis (form, lemma, upos,
# feats, head, deprel). We keep the running text here, and optionally dump the UD layer
# (--with-ud), which matters because it lets Czech and Hungarian -- the two PoeTree
# languages with no spaCy model -- be content-masked from the corpus's own annotation
# via POSNoiseMasker.mask_tagged(), with no tagger of ours in the loop at all.
#
#   python data_prep/fetch_poetree.py                 # all 11 languages
#   python data_prep/fetch_poetree.py --langs cs,hu --with-ud
#
# Output: data_prep/raw/authors_poetree.jsonl  (+ raw/poetree_ud/{lang}.jsonl with --with-ud)

import argparse, json, sys, zipfile
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from _net import get, download

RECORD = "17414036"                      # v1.0.0: cs de en es fr hu it no pt ru sl
RAW = HERE / "raw"; RAW.mkdir(parents=True, exist_ok=True)
ZIPS = HERE / "cache" / "poetree"; ZIPS.mkdir(parents=True, exist_ok=True)

NAMES = {"cs": "czech", "de": "german", "en": "english", "es": "spanish", "fr": "french",
         "hu": "hungarian", "it": "italian", "no": "norwegian", "pt": "portuguese",
         "ru": "russian", "sl": "slovenian"}


def author_key(poem):
    """Canonical author key. `author` is normally a dict, but a few poems carry a
    list of collaborators -- we take the first, so joint works stay attributable."""
    a = poem.get("author") or {}
    if isinstance(a, list):
        a = next((x for x in a if isinstance(x, dict)), {})
    if not isinstance(a, dict):
        return None
    name = (a.get("name") or "").strip()
    if not name:
        return None
    return " ".join(name.lower().split())


def poem_text(poem):
    body = poem.get("body") or []
    lines = []
    for ln in body:
        t = (ln or {}).get("text")
        if t and t.strip():
            lines.append(t.strip())
    return "\n".join(lines)


def poem_ud(poem):
    """Flat [[(form, upos, lemma), ...], ...] -- one list per verse line."""
    out = []
    for ln in poem.get("body") or []:
        toks = [(w.get("form", ""), w.get("upos", "X"), w.get("lemma", "_"))
                for w in (ln.get("words") or []) if w.get("form")]
        if toks:
            out.append(toks)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--langs", default="", help="comma-separated ISO codes (default: all 11)")
    ap.add_argument("--with-ud", action="store_true", help="also dump the corpus's UD annotation")
    args = ap.parse_args()

    rec = get(f"https://zenodo.org/api/records/{RECORD}")
    if not rec:
        sys.exit("could not reach Zenodo")
    files = {f["key"][:-4]: f["links"]["self"] for f in rec["files"] if f["key"].endswith(".zip")}
    todo = [l.strip() for l in args.langs.split(",") if l.strip()] or sorted(files)
    print(f"PoeTree v{rec['metadata'].get('version')} | languages available: {sorted(files)}", flush=True)

    ud_dir = RAW / "poetree_ud"
    if args.with_ud:
        ud_dir.mkdir(parents=True, exist_ok=True)

    out = RAW / "authors_poetree.jsonl"
    mode = "a" if out.exists() and args.langs else "w"
    with open(out, mode, encoding="utf-8") as fh:
        for lang in todo:
            if lang not in files:
                print(f"  {lang}: not in the record, skipped", flush=True); continue
            zp = ZIPS / f"{lang}.zip"
            if not download(files[lang], zp):
                print(f"  {lang}: download failed", flush=True); continue
            try:
                zf = zipfile.ZipFile(zp)
            except zipfile.BadZipFile:
                print(f"  {lang}: corrupt archive, delete {zp} and retry", flush=True); continue

            by_author, npoems = defaultdict(list), 0
            udfh = open(ud_dir / f"{lang}.jsonl", "w", encoding="utf-8") if args.with_ud else None
            for m in zf.namelist():
                if not m.endswith(".json"):
                    continue
                try:
                    poem = json.loads(zf.read(m))
                except (json.JSONDecodeError, KeyError):
                    continue
                if poem.get("duplicate"):                      # PoeTree flags its own duplicates
                    continue
                a = author_key(poem)
                txt = poem_text(poem)
                if not a or not txt:
                    continue
                by_author[a].append(txt); npoems += 1
                if udfh:
                    udfh.write(json.dumps({"author": a, "id": poem.get("id"),
                                           "lines": poem_ud(poem)}, ensure_ascii=False) + "\n")
            if udfh:
                udfh.close()

            corpus = NAMES.get(lang, lang)
            nwords = 0
            for a in sorted(by_author):
                text = "\n".join(by_author[a])
                nwords += len(text.split())
                fh.write(json.dumps({"corpus": corpus, "author": a, "text": text},
                                    ensure_ascii=False) + "\n")
            print(f"  {lang} {corpus:11s} {npoems:6d} poems  {len(by_author):5d} authors  "
                  f"{nwords:>12,} words", flush=True)

    print(f"wrote -> {out}")


if __name__ == "__main__":
    main()
