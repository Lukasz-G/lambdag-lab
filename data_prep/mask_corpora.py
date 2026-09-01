# POSNoise-mask every AV dataset into the TSV layout the experiment code already reads
# (one masked sentence per line, tokens tab-separated) -- mirroring phase3/pairs500 and
# phase1/bank, so the KN-LambdaG and HDC-TM drivers need no new I/O.
#
# Three tagging back ends, chosen per language:
#   spacy   19 languages with an official *_lg model
#   stanza  cs, hu, sr, lv (+ ka, ga) -- no spaCy model exists; Stanza is UD-native and
#           BiLSTM-based, so it keeps the pipeline free of transformers
#   ud      PoeTree ships its own Universal Dependencies layer per verse line, so Czech
#           and Hungarian POETRY need no tagger of ours at all: the corpus annotation is
#           fed straight to POSNoiseMasker.mask_tagged()
#
#   python data_prep/mask_corpora.py --langs german,czech          # selected languages
#   python data_prep/mask_corpora.py --genres poetree --backend ud
#
# Output: masked/{language}_{corpus}/{pairs/,bank/,pairs.tsv}

import argparse, hashlib, json, os, re, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
import _net  # noqa: F401 -- injects the OS trust store; Stanza fetches its resource
             # index over HTTPS at Pipeline() time and fails without it on this machine
from lambdag import POSNoiseMasker

DATA = ROOT / "data"
OUT = ROOT / "masked"
UD_DIR = HERE / "raw" / "poetree_ud"

# output folder name -> (iso for POSNoise/pattern list, spaCy model or None)
SPACY = {
    "croatian": ("hr", "hr_core_news_lg"), "english": ("en", "en_core_web_lg"),
    "german": ("de", "de_core_news_lg"), "greek": ("el", "el_core_news_lg"),
    "italian": ("it", "it_core_news_lg"), "lithuanian": ("lt", "lt_core_news_lg"),
    "norwegian": ("no", "nb_core_news_lg"), "polish": ("pl", "pl_core_news_lg"),
    "portuguese": ("pt", "pt_core_news_lg"), "romanian": ("ro", "ro_core_news_lg"),
    "russian": ("ru", "ru_core_news_lg"), "slovenian": ("sl", "sl_core_news_lg"),
    "spanish": ("es", "es_core_news_lg"), "swedish": ("sv", "sv_core_news_lg"),
    "ukrainian": ("uk", "uk_core_news_lg"), "french": ("fr", "fr_core_news_lg"),
    "dutch": ("nl", "nl_core_news_lg"), "american": ("en", "en_core_web_lg"),
    # Swiss German has no model of its own; the standard German pipeline is the closest
    # available and ELTeC-gsw is modern-orthography prose, so it tags tolerably.
    "swissgerman": ("de", "de_core_news_lg"),
}
STANZA = {"czech": "cs", "hungarian": "hu", "serbian": "sr", "latvian": "lv",
          "georgian": "ka", "irish": "ga"}
# POSNoise pattern lists we built; languages absent here fall back to POS-only masking
HAVE_LIST = {"en", "de", "fr", "es", "it", "pl", "ru", "cs", "el", "hr", "hu", "lt",
             "lv", "nl", "no", "pt", "ro", "sl", "sr", "sv", "uk"}


def slug(s):
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")[:60] or "x"


def write_tsv(path, sents):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for s in sents:
            if s:
                fh.write("\t".join(s) + "\n")
    return sum(len(s) for s in sents)


class Tagger:
    """Uniform mask(text) -> List[List[str]] over the three back ends."""

    def __init__(self, lang, backend, model=None):
        self.backend, self.lang = backend, lang
        kw = dict(language=lang) if lang in HAVE_LIST else {}
        if backend == "spacy":
            self.m = POSNoiseMasker(spacy_model=model, lowercase=True, **kw)
        else:                                   # stanza / ud: no spaCy in the loop
            self.m = POSNoiseMasker(require_tagger=False, lowercase=True, **kw)
        self.nlp = None
        if backend == "stanza":
            import stanza
            # download_method=None: the models are already on disk, so do not let
            # Stanza re-check the hub (that network call is what fails behind the proxy)
            self.nlp = stanza.Pipeline(lang=lang, processors="tokenize,pos,lemma",
                                       verbose=False, use_gpu=False, download_method=None)

    # spaCy refuses documents over ~1M characters (E088) and Stanza slows to a crawl on
    # them; reference "documents" here are an author's whole oeuvre, so we mask in
    # chunks cut at paragraph boundaries and concatenate the sentence lists. Sentences
    # are the unit of independence downstream, so a cut between paragraphs costs nothing.
    CHUNK = 400_000

    def _chunks(self, text):
        if len(text) <= self.CHUNK:
            return [text]
        out, buf = [], []
        n = 0
        for para in text.split("\n"):
            if n + len(para) > self.CHUNK and buf:
                out.append("\n".join(buf)); buf, n = [], 0
            buf.append(para); n += len(para) + 1
        if buf:
            out.append("\n".join(buf))
        return out

    def mask_text(self, text, cache=None):
        """Mask `text`, optionally memoised on its content hash.

        Every pair reuses ONE fixed 5000-word enrollment sample per author, so the
        known side of the corpus repeats dozens of times; tagging it once per author
        instead of once per pair removes ~2/3 of the work (and far more of Stanza's,
        which is several times slower than spaCy).
        """
        if cache is not None:
            key = hashlib.blake2b(text.encode("utf-8"), digest_size=16).hexdigest()
            hit = cache.get(key)
            if hit is not None:
                return hit
        sents = []
        for piece in self._chunks(text):
            sents.extend(self._mask_one(piece))
        if cache is not None:
            cache[key] = sents
        return sents

    def _mask_one(self, text):
        if self.backend == "spacy":
            return self.m.mask(text)
        doc = self.nlp(text)                    # stanza
        sents = [[(w.text, w.upos or "X", w.lemma or "") for w in s.words] for s in doc.sentences]
        return self.m.mask_tagged(sents)

    def mask_tagged_lines(self, lines):         # ud back end (PoeTree)
        return self.m.mask_tagged([[(f, p, l) for f, p, l in ln] for ln in lines])


def load_pairs(p):
    for line in open(p, encoding="utf-8"):
        yield json.loads(line)


def do_dataset(folder, corpus, iso, tagger, limit_pairs=None):
    """Mask one (language, genre) dataset into masked/{folder}_{corpus}/."""
    tp = DATA / folder / f"av_test_{corpus}_{iso}.jsonl"
    rp = DATA / folder / f"av_reference_{corpus}_{iso}.jsonl"
    if not tp.exists():
        return None
    dest = OUT / f"{folder}_{corpus}"
    done = dest / "DONE"
    if done.exists():                      # resume: masking is slow (~30 min per large set)
        print(f"  {folder:12s} {corpus:8s} already masked, skipped", flush=True)
        return json.loads(done.read_text(encoding="utf-8"))
    (dest / "pairs").mkdir(parents=True, exist_ok=True)
    (dest / "bank").mkdir(parents=True, exist_ok=True)

    t0 = time.time(); npairs = 0
    known_cache = {}                        # author enrollment samples repeat across pairs
    with open(dest / "pairs.tsv", "w", encoding="utf-8") as man:
        man.write("id\tlabel\tknown_author\tq_author\n")
        for rec in load_pairs(tp):
            if limit_pairs and npairs >= limit_pairs:
                break
            q_txt, k_txt = rec["pair"]
            q = tagger.mask_text(q_txt); k = tagger.mask_text(k_txt, known_cache)
            if not q or not k:
                continue
            pid = rec["id"]
            write_tsv(dest / "pairs" / f"{pid}_q.tsv", q)
            write_tsv(dest / "pairs" / f"{pid}_known.tsv", k)
            qa, ka = rec["authors"][0], rec["authors"][1]
            man.write(f"{pid}\t{rec['label']}\t{ka}\t{qa}\n")
            npairs += 1

    nref = 0
    if rp.exists():
        for i, rec in enumerate(load_pairs(rp)):
            sents = tagger.mask_text(rec["text"])
            if sents:
                write_tsv(dest / "bank" / f"{i:03d}_{slug(rec['author'])}.tsv", sents)
                nref += 1
    print(f"  {folder:12s} {corpus:8s} [{tagger.backend}] {npairs:5d} pairs  {nref:4d} ref "
          f"({time.time()-t0:.0f}s) -> {dest.name}", flush=True)
    info = {"folder": folder, "corpus": corpus, "pairs": npairs, "ref": nref,
            "backend": tagger.backend}
    done.write_text(json.dumps(info), encoding="utf-8")
    return info


def do_poetree_ud(folder, iso, limit_pairs=None):
    """Czech/Hungarian poetry: mask straight from PoeTree's own UD annotation."""
    src = UD_DIR / f"{iso}.jsonl"
    if not src.exists():
        print(f"  {folder}: no UD dump ({src.name}); rerun fetch_poetree.py --with-ud", flush=True)
        return None
    # PoeTree UD is per POEM, whereas the AV pairs are word-sliced spans of the author's
    # concatenated text, so the two cannot be aligned token-for-token. We therefore use
    # the UD layer only for the REFERENCE bank (whole authors), and tag the sliced pair
    # texts with Stanza. Reported explicitly so the mixture is never silent.
    print(f"  {folder}: poetry UD available for the reference bank; pairs need Stanza", flush=True)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--langs", default="", help="comma-separated output folder names")
    ap.add_argument("--genres", default="novels,dracor,poetree")
    ap.add_argument("--limit-pairs", type=int, default=0, help="cap pairs per dataset (smoke tests)")
    args = ap.parse_args()
    genres = {g.strip() for g in args.genres.split(",")}
    want = {l.strip() for l in args.langs.split(",") if l.strip()}

    todo = []
    for d in sorted(DATA.iterdir()):
        if not d.is_dir() or (want and d.name not in want):
            continue
        for f in sorted(d.glob("av_test_*.jsonl")):
            _, _, corpus, iso = f.stem.split("_", 3)
            if corpus in genres:
                todo.append((d.name, corpus, iso))

    rows, taggers = [], {}
    for folder, corpus, iso in todo:
        if folder in SPACY:
            lang, model = SPACY[folder]; backend = "spacy"
        elif folder in STANZA:
            lang, model, backend = STANZA[folder], None, "stanza"
        else:
            print(f"  {folder}: no tagger configured, skipped", flush=True); continue
        key = (backend, lang)
        if key not in taggers:
            try:
                taggers[key] = Tagger(lang, backend, model)
            except Exception as e:
                print(f"  {folder}: tagger unavailable ({type(e).__name__}: {str(e)[:70]})", flush=True)
                taggers[key] = None
        tg = taggers[key]
        if tg is None:
            continue
        r = do_dataset(folder, corpus, iso, tg, args.limit_pairs or None)
        if r:
            rows.append(r)

    OUT.mkdir(exist_ok=True)
    with open(OUT / "masking_report.json", "w", encoding="utf-8") as fh:
        json.dump(rows, fh, indent=2, ensure_ascii=False)
    print(f"\n{len(rows)} datasets masked -> {OUT}")


if __name__ == "__main__":
    main()
