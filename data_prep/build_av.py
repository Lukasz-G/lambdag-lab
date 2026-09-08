# Turn the harvested per-author corpora into LambdaG-ready AV datasets, in exactly the
# format of the existing german/ folder: av_test_{corpus}_{lang}.jsonl (pairs) and
# av_reference_{corpus}_{lang}.jsonl (disjoint reference authors).
#
# The pairing logic is a faithful port of the cell shared by the three prepr_data_*
# notebooks -- same seed, same asymmetric design (questioned D_U of QUERY_WORDS vs
# enrollment D_A of KNOWN_WORDS, drawn from DISJOINT spans of a same-author's text so
# no case can leak), same 50/50 balance, same author-disjoint reference split.
#
#   python data_prep/build_av.py                     # every corpus/language available
#   python data_prep/build_av.py --genres novels     # one genre only
#
# Output: data/{language}/av_{test,reference}_{corpus}_{lang}.jsonl

import argparse, json, random, sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RAW = HERE / "raw"
OUT = ROOT / "data"

SEED = 42
# Defaults reproduce the notebooks' asymmetric design, and the novels/dracor/
# poetree datasets are built with them, so they must not change: those are the
# published baselines.
#
# --symmetric overrides both to one length. Cross-genre work needs it: there the
# open question is how much evidence each SIDE requires, and an asymmetric pair
# confounds that with the fixed 1000/5000 split. See --symmetric in main().
QUERY_WORDS = 1000        # length of each questioned document D_U
KNOWN_WORDS = 5000        # length of each author's known / enrollment sample D_A
REF_AUTHOR_FRACTION = 0.5
N_PAIRS = 2000

# DraCor corpus name -> (output folder, iso code kept in the filename)
DRACOR = {"fre": ("french", "fre"), "ger": ("german", "ger"), "eng": ("english", "eng"),
          "dutch": ("dutch", "dutch"), "rus": ("russian", "rus"), "ita": ("italian", "ita"),
          "ro": ("romanian", "ro"), "hun": ("hungarian", "hun"), "swe": ("swedish", "swe"),
          "pol": ("polish", "pol"), "u": ("ukrainian", "u"), "am": ("american", "am")}
# ELTeC folder name -> iso
ELTEC_ISO = {"czech": "cs", "german": "de", "english": "en", "basque": "eu", "french": "fr",
             "georgian": "ka", "irish": "ga", "greek": "el", "swissgerman": "gsw",
             "croatian": "hr", "hungarian": "hu", "italian": "it", "latvian": "lv",
             "lithuanian": "lt", "dutch": "nl", "norwegian": "no", "polish": "pl",
             "portuguese": "pt", "romanian": "ro", "russian": "ru", "slovenian": "sl",
             "spanish": "es", "serbian": "sr", "swedish": "sv", "ukrainian": "uk"}
POETREE_ISO = {"czech": "cs", "german": "de", "english": "en", "spanish": "es", "french": "fr",
               "hungarian": "hu", "italian": "it", "norwegian": "no", "portuguese": "pt",
               "russian": "ru", "slovenian": "sl"}


def build(texts_by_author, corpus, lang, folder, quiet=False):
    """Faithful port of the notebooks' dataset-construction cell."""
    random.seed(SEED)
    need = KNOWN_WORDS + QUERY_WORDS
    authors = [a for a in texts_by_author if len(texts_by_author[a].split()) >= need]
    random.shuffle(authors)
    if len(authors) < 4:
        if not quiet:
            print(f"  {corpus}/{lang}: only {len(authors)} authors with >= {need} words -- skipped")
        return None
    n_ref = max(1, round(len(authors) * REF_AUTHOR_FRACTION))
    ref_authors = set(authors[:n_ref])
    test_authors = [a for a in authors if a not in ref_authors]
    assert ref_authors.isdisjoint(test_authors)

    known_sample, query_docs = {}, {}
    for a in test_authors:
        w = texts_by_author[a].split()
        known_sample[a] = " ".join(w[:KNOWN_WORDS])
        rest = w[KNOWN_WORDS:]
        docs = [" ".join(rest[i:i + QUERY_WORDS]) for i in range(0, len(rest), QUERY_WORDS)]
        query_docs[a] = [d for d in docs if len(d.split()) == QUERY_WORDS]
    elig = [a for a in test_authors if query_docs[a]]

    half = N_PAIRS // 2
    seen, same_pairs, diff_pairs = set(), [], []
    attempts, max_attempts = 0, half * 500
    while len(same_pairs) < half and elig and attempts < max_attempts:
        attempts += 1
        a = random.choice(elig); qi = random.randrange(len(query_docs[a]))
        key = ("S", a, qi)
        if key in seen:
            continue
        seen.add(key)
        same_pairs.append({"pair": [query_docs[a][qi], known_sample[a]], "label": 1, "authors": [a, a]})
    attempts = 0
    while len(diff_pairs) < half and len(elig) >= 2 and attempts < max_attempts:
        attempts += 1
        a, b = random.sample(elig, 2); qi = random.randrange(len(query_docs[a]))
        key = ("D", a, qi, b)
        if key in seen:
            continue
        seen.add(key)
        diff_pairs.append({"pair": [query_docs[a][qi], known_sample[b]], "label": 0, "authors": [a, b]})

    n = min(len(same_pairs), len(diff_pairs))
    if n == 0:
        if not quiet:
            print(f"  {corpus}/{lang}: no usable pairs -- skipped")
        return None
    test_set = same_pairs[:n] + diff_pairs[:n]
    random.shuffle(test_set)
    for k, item in enumerate(test_set):
        item["id"] = k
    reference = [{"author": a, "text": texts_by_author[a]} for a in sorted(ref_authors)]

    d = OUT / folder; d.mkdir(parents=True, exist_ok=True)
    tp, rp = d / f"av_test_{corpus}_{lang}.jsonl", d / f"av_reference_{corpus}_{lang}.jsonl"
    with open(tp, "w", encoding="utf-8") as fh:
        for it in test_set:
            fh.write(json.dumps(it, ensure_ascii=False) + "\n")
    with open(rp, "w", encoding="utf-8") as fh:
        for it in reference:
            fh.write(json.dumps(it, ensure_ascii=False) + "\n")
    if not quiet:
        print(f"  {folder:12s} {corpus:8s} {len(authors):4d} usable authors -> "
              f"{len(ref_authors):3d} ref / {len(elig):3d} test-eligible | {2*n:5d} pairs")
    return {"folder": folder, "corpus": corpus, "lang": lang, "authors": len(authors),
            "ref": len(ref_authors), "test": len(elig), "pairs": 2 * n}


def build_full_bank(texts_by_author, corpus, lang, folder, quiet=False):
    """A reference file holding EVERY author, plus a token test file.

    Why this is needed. build() splits authors 50/50 into reference and test, and
    mask_corpora builds masked/{folder}_{corpus}/bank/ from the reference half
    alone -- the test half survives only as 5,000-word pair slices. That is right
    for pair evaluation, where a reference model must not have seen the test
    author.

    It is wrong for cross-genre work, which matches an author's identity ACROSS
    genres and therefore needs him in the bank on BOTH sides. The split is drawn
    independently per genre, so an author must fall on the reference side twice:
    expected retention is 0.25, and measured, the German harvest's 67 cross-genre
    authors came through as 15, and its 17 three-genre authors as 2.

    So the bank for cross-genre work is built separately, under its own corpus
    tag, and the evaluation datasets are left exactly as they are. No leakage is
    introduced: the cross-genre drivers exclude both case authors from the donor
    pool per case, rather than relying on a corpus-level split.
    """
    tag = f"{corpus}all"
    reference = [{"author": a, "text": t} for a, t in sorted(texts_by_author.items())]
    if len(reference) < 4:
        return None
    d = OUT / folder
    d.mkdir(parents=True, exist_ok=True)
    with open(d / f"av_reference_{tag}_{lang}.jsonl", "w", encoding="utf-8") as fh:
        for it in reference:
            fh.write(json.dumps(it, ensure_ascii=False) + "\n")
    # mask_corpora keys a dataset on the presence of its test file, so write a
    # minimal one; only bank/ is wanted from this tag.
    names = [r["author"] for r in reference[:4]]
    stub = []
    for i, a in enumerate(names):
        w = texts_by_author[a].split()
        stub.append({"id": i, "label": 1, "authors": [a, a],
                     "pair": [" ".join(w[:200]), " ".join(w[200:400])]})
    with open(d / f"av_test_{tag}_{lang}.jsonl", "w", encoding="utf-8") as fh:
        for it in stub:
            fh.write(json.dumps(it, ensure_ascii=False) + "\n")
    if not quiet:
        print(f"  {folder:12s} {tag:10s} {len(reference):4d} authors -> full bank")
    return {"folder": folder, "corpus": tag, "lang": lang,
            "authors": len(reference), "ref": len(reference), "test": 0,
            "pairs": 0}


def load_eltec():
    for p in sorted((RAW / "eltec").glob("*_preprocessed.jsonl")):
        folder = p.name.replace("_preprocessed.jsonl", "")
        texts = {}
        for line in open(p, encoding="utf-8"):
            o = json.loads(line); texts[o["author_id"]] = o["text"]
        yield texts, "novels", ELTEC_ISO.get(folder, folder), folder


def load_dracor():
    p = RAW / "dracor_authors_corpus_all_lang.jsonl"
    if not p.exists():
        return
    for line in open(p, encoding="utf-8"):
        o = json.loads(line)
        name = o["language"]
        folder, iso = DRACOR.get(name, (name, name))
        yield o["corpus"], "dracor", iso, folder


def load_poetree():
    p = RAW / "authors_poetree.jsonl"
    if not p.exists():
        return
    by_corpus = defaultdict(dict)
    for line in open(p, encoding="utf-8"):
        o = json.loads(line)
        by_corpus[o["corpus"]][o["author"]] = o["text"]
    for corpus, texts in sorted(by_corpus.items()):
        yield texts, "poetree", POETREE_ISO.get(corpus, corpus), corpus


def load_textgrid():
    """The multi-genre German harvest (fetch_textgrid.py).

    Unlike the other three sources this one supplies ALL THREE genres for the
    same authors from a single catalogue, so its genre lives in the file name
    rather than in the source: german_tgprose / _tgverse / _tgdrama. The corpus
    tag is kept distinct from novels/dracor/poetree so that the ELTeC, DraCor
    and PoeTree baselines stay untouched and comparable.
    """
    for p in sorted((RAW / "textgrid").glob("*_preprocessed.jsonl")):
        stem = p.name.replace("_preprocessed.jsonl", "")   # german_tgprose
        folder, _, corpus = stem.partition("_")
        if not corpus:
            continue
        texts = {}
        for line in open(p, encoding="utf-8"):
            o = json.loads(line)
            texts[o["author_id"]] = o["text"]
        yield texts, corpus, ELTEC_ISO.get(folder, folder), folder


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--genres", default="novels,dracor,poetree")
    ap.add_argument("--full-bank", action="store_true",
                    help="ALSO write a {corpus}all dataset whose reference file "
                         "holds every author, so cross-genre identity matching "
                         "is not thinned by the 50/50 reference split")
    ap.add_argument("--symmetric", type=int, default=0, metavar="N",
                    help="build pairs with N words on BOTH sides instead of the "
                         "default 1000 questioned / 5000 known. Required for "
                         "cross-genre work, where how much evidence each side "
                         "needs is the question under test. Raises the per-author "
                         "minimum to 2N words, so fewer authors qualify.")
    args = ap.parse_args()
    genres = {g.strip() for g in args.genres.split(",")}
    if args.symmetric:
        global QUERY_WORDS, KNOWN_WORDS
        QUERY_WORDS = KNOWN_WORDS = args.symmetric
        print(f"symmetric pairs: {args.symmetric} words on both sides "
              f"(per-author minimum {2 * args.symmetric:,} words)")

    rows = []
    loaders = [("novels", load_eltec), ("dracor", load_dracor),
               ("poetree", load_poetree), ("textgrid", load_textgrid)]
    for genre, loader in loaders:
        if genre not in genres:
            continue
        print(f"\n=== {genre} ===")
        for texts, corpus, iso, folder in loader():
            r = build(texts, corpus, iso, folder)
            if r:
                rows.append(r)
            if args.full_bank:
                r = build_full_bank(texts, corpus, iso, folder)
                if r:
                    rows.append(r)

    print(f"\n{len(rows)} datasets written under {OUT}")
    cov = defaultdict(set)
    for r in rows:
        cov[r["folder"]].add(r["corpus"])
    print("\nlanguage coverage (genres per language):")
    for lang in sorted(cov):
        print(f"  {lang:12s} {', '.join(sorted(cov[lang]))}")
    with open(OUT / "coverage.json", "w", encoding="utf-8") as fh:
        json.dump(rows, fh, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
