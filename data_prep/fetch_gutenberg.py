# Project Gutenberg -> per-genre JSONL for English and French, in the schema
# the other fetchers emit ({author_id, author_name, id_source, text}).
#
# WHY THIS SOURCE. One catalogue, one author authority, all three genres --
# the property that made TextGrid the German source. The offline catalogue
# (pg_catalog.csv.gz, ~79k text rows) supplies author names WITH LIFE DATES,
# language, subjects and shelves, so the multi-genre inventory is computed
# without a single API call.
#
# THE TRAP, met on the first sweep: translations are catalogued under the
# TRANSLATED author -- Balzac and Dumas rank among the most prolific "English"
# authors. Two independent gates answer it. (1) Every selected book's metadata
# is fetched from the gutendex API, whose `translators` field is explicit: a
# non-empty list excludes the book. (2) Every downloaded text must pass a
# function-word language test in the HARVEST language; a Kipling item in
# French dies here regardless of metadata. Neither gate alone sufficed for
# German, and no gate replaces reading the text where the audit flags it.
#
# GENRE. The catalogue subject/shelf strings nominate a kind (poetry / drama /
# fiction); the text's own shape must then agree: a verse candidate must be
# short-lined, a drama candidate must carry speaker turns. Disagreements are
# not resolved silently -- the book is dropped and logged, because the German
# audits showed the mislabelled residue is precisely where artefacts breed
# (verse epics shelved as poetry, closet drama shelved as verse).
#
#   python data_prep/fetch_gutenberg.py --language en
#   python data_prep/fetch_gutenberg.py --language fr --min-books 2
#   python data_prep/fetch_gutenberg.py --language en --authors "Kipling, Rudyard"
#
# Output: data_prep/raw/gutenberg/{en,fr}_pg{prose,verse,drama}_preprocessed.jsonl
#         + {lang}_pg_index.json (per author/kind books, words, exclusions)

import argparse
import csv
import gzip
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from _net import get  # noqa: E402
from fetch_textgrid import dupkeys, slug  # noqa: E402


def load_exclusions():
    """Per-author title patterns, compiled; see gutenberg_exclusions.json."""
    f = HERE / "gutenberg_exclusions.json"
    if not f.exists():
        return {}
    spec = json.loads(f.read_text(encoding="utf-8"))
    out = {}
    for a, v in spec.items():
        if a.startswith("_") or not isinstance(v, dict):
            continue
        pats = v.get("titles") or []
        if pats:
            out[a] = re.compile("|".join(pats), re.I)
    return out

RAW = HERE / "raw" / "gutenberg"
CACHE = HERE / "cache" / "gutenberg"
CATALOG = CACHE / "pg_catalog.csv.gz"

# Function-word gates. A text qualifies for a language when its rate of that
# language's closed-class words is a multiple of the competing language's.
FW = {
    "en": set("the of and to in that it is was he for as with his on be at by "
              "had not are but from or have an they which one you were".split()),
    "fr": set("le la les de des du et un une que qui dans pour pas sur est "
              "il elle au aux ce cette ne se son sa ses mais avec tout".split()),
    "de": set("der die das und nicht ist ich sie ein eine zu den dem mit von "
              "auf für als auch es an werden aus er hat dass war wie im".split()),
}

KIND_PAT = {
    "verse": re.compile(r"\bpoetry\b|\bpoems?\b|po[eé]sie", re.I),
    "drama": re.compile(r"\bdrama\b|\bplays?\b|th[eé][aâ]tre|tragedies|comedies",
                        re.I),
    "prose": re.compile(r"\bfiction\b|\bnovels?\b|short stories|\bromans?\b"
                        r"|\btales?\b|nouvelles", re.I),
}
AUTH = re.compile(r"^(.*?),\s*(\d{3,4})\??-(\d{3,4})?")
# Utilitarian kinds, matched on SUBJECTS -- "Letters on England" is essays and
# "Letters of Two Brides" a novel, so titles never decide. The letters kind
# additionally refuses two title shapes learned the hard way: volumes of
# letters addressed TO the author, and posthumous "Life and letters"
# compilations, which are a biographer's prose wrapped round the letters.
UTIL_PAT = {
    "letters": re.compile(r"correspondence", re.I),
    "diary": re.compile(r"diaries", re.I),
    "memoir": re.compile(r"autobiograph|memoirs", re.I),
    "essay": re.compile(r"essays|essais", re.I),
}
LETTER_TRAP = re.compile(r"letters to|life and letters", re.I)
PG_START = re.compile(r"\*\*\*\s*START OF (THE|THIS) PROJECT GUTENBERG.*?\*\*\*",
                      re.I | re.S)
PG_END = re.compile(r"\*\*\*\s*END OF (THE|THIS) PROJECT GUTENBERG", re.I)
# Two dramatic conventions: the English inline turn ("HAMLET. Words...") and
# the French centred speaker alone on his line ("CYRANO." / "LE BRET"), the
# dialogue following beneath. The first sweep recognised only the former and
# rejected French plays wholesale (Rostand lost five) -- the label said drama,
# the shape test could not see it.
SPEAKER = re.compile(r"^\s{0,8}([A-Z][A-Za-z .'-]{1,28}|[A-Z .'-]{2,28})[.:]\s")
SPEAKER_ALONE = re.compile(
    r"^\s{0,30}[A-ZÉÈÀÂÇÔÛ][A-ZÉÈÀÂÇÔÛa-zéèàâçôû .'-]{1,26}\s*[.:,]?\s*$")
WORD_CAP = 600_000        # per author-kind; a bank never needs more


def catalog_rows(lang):
    with gzip.open(CATALOG, "rt", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["Type"] == "Text" and row["Language"] == lang:
                yield row


def kinds_of(row):
    s = (row["Subjects"] or "") + " ; " + (row["Bookshelves"] or "")
    ks = {k for k, p in KIND_PAT.items() if p.search(s)}
    if "drama" in ks:
        ks.discard("prose")
    return ks


def gutendex(book_id):
    cf = CACHE / f"gd_{book_id}.json"
    if cf.exists():
        return json.loads(cf.read_text(encoding="utf-8"))
    d = get(f"https://gutendex.com/books/{book_id}", as_json=True, timeout=40)
    if d is None:
        return None
    cf.write_text(json.dumps(d, ensure_ascii=False), encoding="utf-8")
    time.sleep(0.25)
    return d


def text_of(book_id):
    cf = CACHE / f"pg_{book_id}.txt"
    if cf.exists():
        return cf.read_text(encoding="utf-8", errors="replace")
    for u in (f"https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt",
              f"https://www.gutenberg.org/files/{book_id}/{book_id}-0.txt",
              f"https://www.gutenberg.org/files/{book_id}/{book_id}.txt"):
        t = get(u, as_json=False, timeout=90)
        if t and len(t) > 2000:
            cf.write_text(t, encoding="utf-8")
            time.sleep(0.35)
            return t
    return None


def strip_pg(t):
    m = PG_START.search(t)
    if m:
        t = t[m.end():]
    m = PG_END.search(t)
    if m:
        t = t[:m.start()]
    return t.strip()


def lang_ok(t, lang):
    toks = re.findall(r"[a-zA-Zàâçéèêëîïôùûüœæ'']+", t[:40_000].lower())
    if len(toks) < 300:
        return False
    own = sum(1 for w in toks if w in FW[lang])
    rival = max(sum(1 for w in toks if w in FW[l]) for l in FW if l != lang)
    return own > 2 * rival and own > 0.12 * len(toks)


def shape_of(t):
    """verse / drama / prose from the text's own lines, as the German harvest
    classified from markup: the label nominates, the shape confirms."""
    lines = [l for l in t.splitlines() if l.strip()]
    if len(lines) < 80:
        return "prose"
    body = lines[40:5040]
    n = len(body)
    short = sum(1 for l in body if 1 <= len(l.split()) <= 9)
    def alone(l):
        m = SPEAKER_ALONE.match(l)
        if not m: return False
        t = l.strip().rstrip(".:,")
        # Roman-numeral and numeric section markers litter verse collections
        # and are capital-heavy short lines; they are headings, not speakers
        if re.fullmatch(r"[IVXLCDM]+|\d+", t): return False
        w = l.split()
        return len(w) <= 4 and sum(c.isupper() for c in l) >= 0.4 * sum(c.isalpha() for c in l)
    speak = sum(1 for l in body if SPEAKER.match(l) or alone(l))
    if speak > 0.04 * n:
        return "drama"
    if short > 0.55 * n:
        return "verse"
    return "prose"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--language", required=True, choices=("en", "fr"))
    ap.add_argument("--min-books", type=int, default=2,
                    help="per kind, for an author to qualify in that kind")
    ap.add_argument("--born", default="1750-1900",
                    help="author birth-year window")
    ap.add_argument("--authors", default="", help="restrict to these names; SEMICOLON-separated, since catalogue names contain commas")
    ap.add_argument("--min-words", type=int, default=8000)
    ap.add_argument("--util", action="store_true",
                    help="harvest the utilitarian kinds (letters/diary/memoir/"
                         "essay) INSTEAD of the literary genres, restricted to "
                         "authors already kept by the literary harvest -- the "
                         "join with the literary banks is the point")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    RAW.mkdir(parents=True, exist_ok=True)
    b_lo, b_hi = (int(x) for x in args.born.split("-"))

    kind_pat = UTIL_PAT if args.util else KIND_PAT
    pool = None
    if args.util:
        idx = json.loads((RAW / f"{args.language}_pg_index.json")
                         .read_text(encoding="utf-8"))
        pool = {v["author_name"] for v in idx.values() if v.get("kinds")}
        print(f"utilitarian mode: restricted to {len(pool)} literary-pool "
              f"authors", flush=True)
    # ---- inventory from the offline catalogue -------------------------------
    excl = load_exclusions()
    n_excl = 0
    per = defaultdict(lambda: defaultdict(list))     # author -> kind -> [ids]
    for row in catalog_rows(args.language):
        a = (row["Authors"] or "").split(";")[0].strip()
        m = AUTH.match(a)
        if not m:
            continue
        name = m.group(1).strip()
        birth = int(m.group(2))
        if not (b_lo <= birth <= b_hi):
            continue
        if pool is not None and name not in pool:
            continue
        pat = excl.get(slug(name))
        if pat and pat.search(row["Title"] or ""):
            n_excl += 1
            continue
        s_ = (row["Subjects"] or "") + " ; " + (row["Bookshelves"] or "")
        ks = {k for k, p2 in kind_pat.items() if p2.search(s_)}
        if not args.util and "drama" in ks:
            ks.discard("prose")
        if args.util and "letters" in ks and LETTER_TRAP.search(row["Title"] or ""):
            n_excl += 1
            continue
        for k in ks:
            per[name][k].append(int(row["Text#"]))
    want = {x.strip() for x in args.authors.split(";") if x.strip()}
    sel = {}
    for name, ks in per.items():
        if want and name not in want:
            continue
        good = {k: ids for k, ids in ks.items() if len(ids) >= args.min_books}
        if args.util:
            if good:
                sel[name] = good
        # verse is the bottleneck genre everywhere; an author with no verse
        # and no drama duplicates what ELTeC already supplies
        elif len(good) >= 2 and ("verse" in good or "drama" in good):
            sel[name] = good
    print(f"{args.language}: {len(sel)} candidate authors "
          f"({n_excl} books excluded by title rules)", flush=True)

    # ---- harvest -------------------------------------------------------------
    out = {k: {} for k in (("letters", "diary", "memoir", "essay")
                           if args.util else ("prose", "verse", "drama"))}
    index = {}
    for ni, name in enumerate(sorted(sel), 1):
        aslug = slug(name)
        rec = {"author_name": name, "kinds": {}, "excluded": defaultdict(int)}
        for kind, ids in sorted(sel[name].items()):
            seen, parts, words = set(), [], 0
            for bid in sorted(ids):
                if words >= WORD_CAP:
                    break
                gd = gutendex(bid)
                if not gd:
                    rec["excluded"]["no-metadata"] += 1
                    continue
                if gd.get("translators"):
                    rec["excluded"]["translation"] += 1
                    continue
                t = text_of(bid)
                if not t:
                    rec["excluded"]["no-text"] += 1
                    continue
                t = strip_pg(t)
                if not lang_ok(t, args.language):
                    rec["excluded"]["wrong-language"] += 1
                    continue
                want_shape = "prose" if args.util else kind
                if shape_of(t) != want_shape:
                    rec["excluded"][f"shape!={want_shape}"] += 1
                    continue
                keys = dupkeys(t)
                if any(k in seen for k in keys):
                    rec["excluded"]["duplicate"] += 1
                    continue
                seen.update(keys)
                parts.append(re.sub(r"[ \t]+", " ", t))
                words += len(t.split())
            if words >= args.min_words:
                out[kind][aslug] = {"author_id": aslug, "author_name": name,
                                    "id_source": f"gutenberg:{args.language}",
                                    "text": "\n\n".join(parts)}
                rec["kinds"][kind] = {"books": len(parts), "words": words}
        index[aslug] = rec
        got = ", ".join(f"{k}:{v['words'] // 1000}k" for k, v in
                        rec["kinds"].items())
        print(f"  [{ni}/{len(sel)}] {name[:34]:36s} {got or '-'}"
              + ("  excl: " + dict(rec["excluded"]).__repr__()
                 if rec["excluded"] else ""), flush=True)

    for kind, d in out.items():
        keep = {a: v for a, v in d.items()}
        p = RAW / f"{args.language}_pg{kind}_preprocessed.jsonl"
        with open(p, "w", encoding="utf-8") as fh:
            for a in sorted(keep):
                fh.write(json.dumps(keep[a], ensure_ascii=False) + "\n")
        print(f"{kind}: {len(keep)} authors -> {p.name}")
    (RAW / (f"{args.language}_pg_util_index.json" if args.util
            else f"{args.language}_pg_index.json")).write_text(
        json.dumps(index, ensure_ascii=False, indent=1, default=dict),
        encoding="utf-8")
    multi = [a for a in index
             if sum(1 for _ in index[a]["kinds"]) >= 2]
    print(f"\n{len(multi)} authors kept in >=2 kinds")


if __name__ == "__main__":
    main()
