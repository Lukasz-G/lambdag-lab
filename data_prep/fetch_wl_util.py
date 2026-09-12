# Wolne Lektury utilitarian kinds -> per-kind JSONL, for the literary pool.
#
# The literary harvest (fetch_wolnelektury.py) took Epika/Liryka/Dramat by the
# catalogue's kind field; this pass takes the utilitarian GENRES the same
# catalogue states outright -- List/Listy (letters), Dziennik (diary),
# Pamiętnik (memoir), Esej/Rozprawa/Reportaż (essays) -- restricted to authors
# the literary harvest already kept, because the join with the literary banks
# is the point of the register axis. All gates are the literary run's own:
# book_detail() supplies translators and language, strip_boilerplate() the
# wrapper, and the same word floor applies.
#
#   python data_prep/fetch_wl_util.py
#
# Output: data_prep/raw/wolnelektury/polish_wl{letters,diary,memoir,essay}_preprocessed.jsonl

import json
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from fetch_wolnelektury import (book_detail, catalogue, fetch_text,  # noqa: E402
                                strip_boilerplate, usable)
from fetch_textgrid import dupkeys, slug  # noqa: E402

RAW = HERE / "raw" / "wolnelektury"
GENRE_MAP = {"list": "letters", "listy": "letters",
             "dziennik": "diary",
             "pamiętnik": "memoir", "pamietnik": "memoir",
             "esej": "essay", "rozprawa": "essay", "reportaż": "essay",
             "reportaz": "essay"}
MIN_WORDS = 5000


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    pool = set()
    for g in ("prose", "verse", "drama"):
        p = RAW / f"polish_wl{g}_preprocessed.jsonl"
        for line in open(p, encoding="utf-8"):
            pool.add(json.loads(line)["author_name"])
    print(f"literary pool: {len(pool)} authors")

    books = catalogue()
    picks = defaultdict(lambda: defaultdict(list))
    for b in books:
        kind = GENRE_MAP.get((b.get("genre") or "").lower())
        a = b.get("author") or ""
        if kind and a in pool:
            picks[a][kind].append(b["slug"])
    print(f"{len(picks)} pool authors hold utilitarian works")

    out = defaultdict(dict)
    for a in sorted(picks):
        for kind, slugs in sorted(picks[a].items()):
            seen, parts, words = set(), [], 0
            rej = defaultdict(int)
            for s in sorted(slugs):
                d = book_detail(s)
                if not d:
                    rej["no-detail"] += 1
                    continue
                ok, why = usable(d)
                if not ok:
                    rej[why.split()[0]] += 1
                    continue
                t = fetch_text(d)
                if not t:
                    rej["no-text"] += 1
                    continue
                t = strip_boilerplate(t, author=a, title=d.get("title"))
                keys = dupkeys(t)
                if any(k in seen for k in keys):
                    rej["duplicate"] += 1
                    continue
                seen.update(keys)
                parts.append(t)
                words += len(t.split())
            if words >= MIN_WORDS:
                aslug = slug(a)
                out[kind][aslug] = {"author_id": aslug, "author_name": a,
                                    "id_source": f"wolnelektury:{kind}",
                                    "text": "\n\n".join(parts)}
            if parts or rej:
                print(f"  {a[:28]:30s} {kind:8s} {words:8,d}w"
                      + (f"  rej {dict(rej)}" if rej else ""), flush=True)

    for kind in ("letters", "diary", "memoir", "essay"):
        p = RAW / f"polish_wl{kind}_preprocessed.jsonl"
        with open(p, "w", encoding="utf-8") as fh:
            for a in sorted(out[kind]):
                fh.write(json.dumps(out[kind][a], ensure_ascii=False) + "\n")
        print(f"{kind}: {len(out[kind])} authors -> {p.name}")


if __name__ == "__main__":
    main()
