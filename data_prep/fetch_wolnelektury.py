# Wolne Lektury -> per-genre JSONL, in the schema the other fetchers emit
# ({author_id, author_name, id_source, text}).
#
# WHY THIS SOURCE. Polish is the thinnest cross-genre corpus we hold: 14 drama
# and 17 novel authors, ONE author in both, and no poetry at all. Wolne Lektury
# carries all three kinds for the same authors in one catalogue, so a multi-genre
# author is established without matching names across corpora.
#
# WHAT THE METADATA GIVES US, and it is more than the German source did. Each of
# the four hazards that had to be inferred there is an explicit field here:
#
#   kinds        Epika / Liryka / Dramat -- the genre, stated, not inferred from
#                markup as it had to be for TEI
#   epochs       Romantyzm, Pozytywizm, Modernizm... -- a period label, where the
#                German source's publication date was the digitisation date
#   translators  a NON-EMPTY list means the text is a translation and the prose
#                is the translator's, not the named author's. Direct evidence,
#                where the German pipeline had to infer foreign provenance from
#                an authority record. Wolne Lektury carries world literature in
#                Polish (Verne, Dickens, Homer), so this filter is load-bearing.
#   language     "pol"; anything else is dropped outright
#
# The one thing the metadata does NOT handle is the plain-text wrapper: every
# .txt opens with the author, title and ISBN and closes with a licence notice,
# the source edition, the publisher, and -- the part that matters -- the names of
# the editors who prepared the text. Left in, that is another hand's prose
# entering the bank under the author's name, which is the same class of error as
# a translation. strip_boilerplate() removes it.
#
# LICENCE: Wolne Lektury publishes public-domain texts with editorial matter
# under free licences (CC BY-SA / free-art), per book. Attribution belongs in the
# paper's data-availability statement.
#
#   python data_prep/fetch_wolnelektury.py --list          # catalogue summary
#   python data_prep/fetch_wolnelektury.py --min-words 5000
#   python data_prep/fetch_wolnelektury.py --authors "Adam Mickiewicz"
#
# Output: data_prep/raw/wolnelektury/polish_wl{prose,verse,drama}_preprocessed.jsonl
#         plus polish_wolnelektury_index.json (per author/genre counts, epochs)

import argparse
import json
import re
import sys
import time
import unicodedata
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from _net import get  # noqa: E402

RAW = HERE / "raw" / "wolnelektury"; RAW.mkdir(parents=True, exist_ok=True)
CACHE = HERE / "cache" / "wolnelektury"; CACHE.mkdir(parents=True, exist_ok=True)
TXTC = CACHE / "txt"; TXTC.mkdir(parents=True, exist_ok=True)

API = "https://wolnelektury.pl/api"

# Wolne Lektury's "kind" -> our genre vocabulary
KINDS = {"epika": "prose", "liryka": "verse", "dramat": "drama"}
GENRES = ("prose", "verse", "drama")


# ---- catalogue -------------------------------------------------------------

def catalogue(refresh=False):
    """The whole book list: one flat JSON array, ~7.6k records."""
    cf = CACHE / "books.json"
    if cf.exists() and not refresh:
        return json.loads(cf.read_text(encoding="utf-8"))
    d = get(f"{API}/books/?format=json", as_json=True, timeout=180)
    if not d:
        return []
    cf.write_text(json.dumps(d, ensure_ascii=False), encoding="utf-8")
    return d


def book_detail(slug):
    """Per-book record, which is where translators/language/txt live."""
    cf = CACHE / "detail" / f"{slug}.json"
    cf.parent.mkdir(parents=True, exist_ok=True)
    if cf.exists():
        return json.loads(cf.read_text(encoding="utf-8"))
    d = get(f"{API}/books/{slug}/?format=json", as_json=True, timeout=60)
    if not d:
        return None
    cf.write_text(json.dumps(d, ensure_ascii=False), encoding="utf-8")
    time.sleep(0.1)
    return d


# ---- the plain-text wrapper ------------------------------------------------
#
# Head: author, title, optional subtitle, ISBN, then the work.
# Foot: an "-----" rule, then the licence notice, the source edition, the
# publisher, the project blurb and the editorial credits. The editors' names are
# the reason this must go: their prose is not the author's.
FOOT_MARKERS = (
    "wszystkie zasoby wolnych lektur",
    "ten utwór nie jest objęty majątkowym prawem autorskim",
    "tekst opracowany na podstawie",
    "wydawca:",
    "publikacja zrealizowana w ramach projektu wolne lektury",
    "opracowanie redakcyjne i przypisy",
    "wesprzyj wolne lektury",
    "fundacja nowoczesna polska",
    "utwór opracowany został w ramach projektu",
)
ISBN = re.compile(r"^\s*ISBN[- ]?[\d\-–]+\s*$", re.I | re.M)
RULE = re.compile(r"^\s*-{4,}\s*$", re.M)


def strip_boilerplate(text, author=None, title=None):
    """Remove the catalogue wrapper, leaving the work.

    The foot is found by the earliest marker rather than by the rule alone,
    because not every book carries the rule and some carry several. The head is
    trimmed only as far as the ISBN line, which every record has and which sits
    after the title block -- cutting a fixed number of lines instead would eat
    dedications and epigraphs, which ARE the author's.
    """
    low = text.lower()
    cut = len(text)
    m = RULE.search(text)
    if m:
        cut = min(cut, m.start())
    for marker in FOOT_MARKERS:
        i = low.find(marker)
        if i != -1:
            cut = min(cut, i)
    body = text[:cut]

    m = ISBN.search(body)
    if m:
        body = body[m.end():]
    else:                                   # no ISBN: drop the title block only
        for line in (title, author):
            if line:
                i = body.find(line)
                if 0 <= i < 400:
                    body = body[i + len(line):]
    return body.strip()


def fetch_text(detail):
    url = detail.get("txt")
    if not url:
        return None
    slug = detail.get("slug") or url.rsplit("/", 1)[-1].rsplit(".", 1)[0]
    dest = TXTC / f"{slug}.txt"
    if dest.exists():
        return dest.read_text(encoding="utf-8", errors="replace")
    t = get(url, as_json=False, timeout=90)
    if not t:
        return None
    dest.write_text(t, encoding="utf-8")
    time.sleep(0.1)
    return t


# ---- provenance ------------------------------------------------------------

def usable(detail):
    """(ok, reason). Translations and non-Polish text are rejected outright."""
    if (detail.get("language") or "").lower() not in ("pol", "pl", ""):
        return (False, f"language {detail.get('language')!r}")
    tr = detail.get("translators") or []
    if tr:
        names = ", ".join(t.get("name", "?") for t in tr)[:60]
        return (False, f"translated ({names})")
    return (True, "")


def slug_name(s):
    s = unicodedata.normalize("NFKD", s.lower())
    s = "".join(c for c in s if not unicodedata.combining(c))
    for a, b in (("ł", "l"), ("ż", "z"), ("ź", "z"), ("ś", "s"), ("ć", "c"),
                 ("ń", "n"), ("ó", "o"), ("ą", "a"), ("ę", "e")):
        s = s.replace(a, b)
    return re.sub(r"[^a-z0-9]+", "_", s).strip("_")[:60] or "x"


# ---- harvest ---------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true",
                    help="summarise the catalogue and exit")
    ap.add_argument("--authors", default="",
                    help="semicolon-separated author names (default: every "
                         "author present in at least two kinds)")
    ap.add_argument("--min-words", type=int, default=5000,
                    help="minimum words for a genre to be kept")
    ap.add_argument("--min-kinds", type=int, default=2)
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    books = catalogue()
    print(f"{len(books)} books in the catalogue", flush=True)

    bykind = defaultdict(set)
    for b in books:
        for k in (b.get("kind") or "").split(","):
            k = KINDS.get(k.strip().lower())
            if k:
                bykind[b.get("author", "?")].add(k)
    if args.list:
        n2 = sum(1 for v in bykind.values() if len(v) >= 2)
        n3 = sum(1 for v in bykind.values() if len(v) >= 3)
        print(f"{len(bykind)} authors; {n2} in >=2 kinds, {n3} in all three")
        print("NB these are WORK counts and a lyric 'work' is one poem, so a "
              "multi-kind author may still be far too thin on his second kind; "
              "the token counts below are what decide.")
        return

    wanted = ([a.strip() for a in args.authors.split(";") if a.strip()]
              or sorted(a for a, v in bykind.items() if len(v) >= args.min_kinds))
    print(f"harvesting {len(wanted)} authors", flush=True)

    byauthor = defaultdict(lambda: defaultdict(list))
    epochs = defaultdict(set)
    rejected = defaultdict(int)
    for i, author in enumerate(wanted, 1):
        slugs = [b for b in books if b.get("author") == author]
        for b in slugs:
            det = book_detail(b.get("slug", ""))
            if not det:
                continue
            ok, why = usable(det)
            if not ok:
                rejected[why.split(" (")[0]] += 1
                continue
            kinds = [KINDS.get((k.get("slug") or "").lower())
                     for k in (det.get("kinds") or [])]
            kinds = [k for k in kinds if k]
            if len(kinds) != 1:                 # ambiguous kind: skip, not guess
                rejected["ambiguous kind"] += 1
                continue
            raw = fetch_text(det)
            if not raw:
                continue
            body = strip_boilerplate(raw, author=author, title=det.get("title"))
            if len(body.split()) < 50:
                continue
            byauthor[author][kinds[0]].append(body)
            for e in (det.get("epochs") or []):
                epochs[author].add(e.get("name", ""))
        got = {g: sum(len(t.split()) for t in ts)
               for g, ts in byauthor[author].items()}
        got = {g: n for g, n in got.items() if n >= args.min_words}
        if got:
            print(f"[{i}/{len(wanted)}] {author[:34]:34s} "
                  + "  ".join(f"{g}={n:,}w" for g, n in sorted(got.items())),
                  flush=True)

    index, per = {}, {g: {} for g in GENRES}
    for author, kinds in byauthor.items():
        kept = {}
        for g, ts in kinds.items():
            nw = sum(len(t.split()) for t in ts)
            if nw >= args.min_words:
                per[g][slug_name(author)] = ts
                kept[g] = {"works": len(ts), "words": nw}
        if kept:
            index[slug_name(author)] = {"author_name": author, "genres": kept,
                                        "epochs": sorted(epochs[author])}

    for g in GENRES:
        if not per[g]:
            continue
        out = RAW / f"polish_wl{g}_preprocessed.jsonl"
        nw = 0
        with open(out, "w", encoding="utf-8") as fh:
            for a in sorted(per[g]):
                text = "\n".join(per[g][a])
                nw += len(text.split())
                fh.write(json.dumps(
                    {"author_id": a, "author_name": index[a]["author_name"],
                     "id_source": "wolnelektury", "text": text},
                    ensure_ascii=False) + "\n")
        print(f"{g:6s} {len(per[g]):4d} authors  {nw:>11,} words -> {out.name}")

    (RAW / "polish_wolnelektury_index.json").write_text(
        json.dumps(index, ensure_ascii=False, indent=1), encoding="utf-8")
    multi = [a for a, v in index.items() if len(v["genres"]) >= 2]
    three = [a for a, v in index.items() if len(v["genres"]) == 3]
    print(f"\n{len(index)} authors kept; {len(multi)} in >=2 genres, "
          f"{len(three)} in all three")
    if rejected:
        print("rejected books: "
              + ", ".join(f"{k} {v}" for k, v in sorted(rejected.items())))


if __name__ == "__main__":
    main()
