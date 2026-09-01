# DraCor -> one JSONL of {language, corpus:{author: text}}, matching the schema the
# prepr_data_dracor notebook already consumes (dracor_authors_corpus_all_lang.jsonl).
#
# Selection policy (defaults follow the project brief):
#   * keep corpora with >= MIN_PLAYS plays  (Polish 50 / Ukrainian 47 are the cut-off)
#   * drop single-author or few-author collections -- Calderon, Ibsen, Shakespeare and
#     the ancient corpora are period/author showcases, useless for authorship
#     verification.  This is enforced BOTH by an explicit deny-list and by a
#     data-driven MIN_AUTHORS check, so a corpus never sneaks in on play count alone.
#   * text = spoken text only (stage directions and speaker labels excluded).
#
#   python data_prep/fetch_dracor.py                 # all eligible corpora
#   python data_prep/fetch_dracor.py --corpora pol,u # just these
#
# Output: data_prep/raw/dracor_authors_corpus_all_lang.jsonl

import argparse, json, os, sys, time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _net import get as _get, SESSION

API = "https://dracor.org/api/v1"
HERE = Path(__file__).resolve().parent
RAW = HERE / "raw"; RAW.mkdir(parents=True, exist_ok=True)
CACHE = HERE / "cache" / "dracor"; CACHE.mkdir(parents=True, exist_ok=True)

# author-showcase / few-author collections: no use for AV
DENY = {"cal", "ibs", "shake", "gersh", "greek", "rom"}
MIN_PLAYS = 40
MIN_AUTHORS = 10

def get(url, as_json=True):
    return _get(url, as_json=as_json, session=SESSION)


def play_text(corpus, play):
    """Spoken text of one play, cached on disk so re-runs are free."""
    fn = CACHE / corpus / f"{play}.txt"
    if fn.exists():
        return fn.read_text(encoding="utf-8")
    txt = get(f"{API}/corpora/{corpus}/plays/{play}/spoken-text", as_json=False) or ""
    fn.parent.mkdir(parents=True, exist_ok=True)
    fn.write_text(txt, encoding="utf-8")
    return txt


def author_of(play):
    """Canonical author key: lower-cased full name, as the notebook expects."""
    auths = play.get("authors") or []
    if not auths:
        return None
    a = auths[0]
    name = a.get("fullname") or a.get("name") or a.get("shortname")
    if not name:
        return None
    if "," in name:                                   # "Fredro, Aleksander" -> "aleksander fredro"
        last, _, first = name.partition(",")
        name = f"{first.strip()} {last.strip()}"
    return " ".join(name.lower().split())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpora", default="", help="comma-separated corpus names (default: auto-select)")
    ap.add_argument("--min-plays", type=int, default=MIN_PLAYS)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    corpora = get(f"{API}/corpora?include=metrics") or []
    sizes = {c["name"]: c.get("metrics", {}).get("plays", 0) for c in corpora}
    if args.corpora:
        chosen = [c.strip() for c in args.corpora.split(",") if c.strip()]
    else:
        chosen = sorted((n for n, p in sizes.items() if p >= args.min_plays and n not in DENY),
                        key=lambda n: -sizes[n])
    print(f"corpora considered: {len(sizes)} | selected by size/deny-list: {chosen}", flush=True)

    out = RAW / "dracor_authors_corpus_all_lang.jsonl"
    written = 0
    with open(out, "w", encoding="utf-8") as fh:
        for corpus in chosen:
            meta = get(f"{API}/corpora/{corpus}")
            if not meta:
                print(f"  {corpus}: metadata unavailable, skipped", flush=True); continue
            plays = meta.get("plays", [])
            by_author = {}
            for p in plays:
                a = author_of(p)
                if a and a != "anonymous":
                    by_author.setdefault(a, []).append(p["name"])
            if len(by_author) < MIN_AUTHORS:
                print(f"  {corpus}: only {len(by_author)} authors (<{MIN_AUTHORS}) -- "
                      f"author-showcase corpus, skipped", flush=True)
                continue

            t0 = time.time()
            jobs = [(a, pl) for a, pls in by_author.items() for pl in pls]
            texts = {}
            with ThreadPoolExecutor(max_workers=args.workers) as ex:
                for (a, pl), txt in zip(jobs, ex.map(lambda j: play_text(corpus, j[1]), jobs)):
                    if txt and txt.strip():
                        texts.setdefault(a, []).append(txt.strip())
            corpus_map = {a: "\n".join(t) for a, t in texts.items() if t}
            fh.write(json.dumps({"language": corpus, "corpus": corpus_map}, ensure_ascii=False) + "\n")
            nwords = sum(len(t.split()) for t in corpus_map.values())
            written += 1
            print(f"  {corpus}: {len(plays)} plays, {len(corpus_map)} authors, "
                  f"{nwords:,} words ({time.time()-t0:.0f}s)", flush=True)

    print(f"wrote {written} corpora -> {out}")


if __name__ == "__main__":
    main()
