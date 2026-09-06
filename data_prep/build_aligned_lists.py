# Category-annotated companion files for the POSNoise pattern lists
# (cross-lingual alignment layer).
#
# The shipped pattern lists are flat: one entry per line, no part of speech, no
# functional category. For cross-lingual work we need every entry annotated with
# (a) its dominant UD POS and (b) a functional class drawn from a small alphabet
# shared across languages, so that a kept token in any language can be collapsed
# to a language-independent symbol (AUX, LVERB, ADV, DET, ADP, PRON, ...).
#
# Method: REGENERATION FROM TREEBANK EVIDENCE, never annotation from memory.
# Every entry of the shipped list is looked up in the same UD treebanks the list
# was built from (surface and lemma occurrences pooled); its dominant UPOS over
# all occurrences decides the class. Entries with no treebank evidence are
# written with class UNK and flagged in the report -- downstream users fall back
# to the generic function-word symbol for them.
#
# Output, per language Xx:
#   posnoise_lists/aligned/POSNoise_Aligned_<Xx>_v1.0.tsv
#     columns: entry <TAB> upos <TAB> class <TAB> evidence_count
#   posnoise_lists/aligned/aligned_report.json   (coverage + class histograms)
#
# The medieval stages (gmh/gml) are excluded here: their lists carry HiTS-derived
# tags via a separate pipeline (build_hist_posnoise_lists.py) and would need the
# hits_to_ud map, not UD treebanks.
#
#   python data_prep/build_aligned_lists.py --langs en,de,fr,pl,cs,hu
#   python data_prep/build_aligned_lists.py --langs all

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from build_posnoise import TREEBANKS, load_treebank  # noqa: E402

ROOT = HERE.parent
LISTS = ROOT / "posnoise_lists"
OUT = LISTS / "aligned"

# Languages whose lists were not built by build_posnoise.py but for which a UD
# treebank supplies the same kind of evidence (en/de upstream lists included:
# annotating them from EWT/GSD keeps the whole layer on one evidence standard).
EXTRA_TREEBANKS = {
    "en": ["UD_English-EWT"],
    "de": ["UD_German-GSD"],
    "fr": ["UD_French-GSD"],
    "es": ["UD_Spanish-AnCora"],
    "it": ["UD_Italian-ISDT"],
    "pl": ["UD_Polish-PDB"],
    "ru": ["UD_Russian-SynTagRus"],
}

ALL_TREEBANKS = {**TREEBANKS, **EXTRA_TREEBANKS}

# UPOS -> shared functional class. AUX and clause-taking verbs are what the
# lists exist to rescue; the closed classes are already POS-transparent but the
# class label makes the entry usable as a cross-language symbol; anything
# content-like that slipped into a list maps to OTH and is reported.
UPOS_CLASS = {
    "AUX": "AUX", "VERB": "LVERB", "ADV": "ADV",
    "DET": "DET", "ADP": "ADP", "PRON": "PRON",
    "CCONJ": "CCONJ", "SCONJ": "SCONJ", "PART": "PART",
    "INTJ": "INTJ", "NUM": "NUM",
}
OTHER_CLASS = "OTH"   # NOUN/ADJ/PROPN/X/SYM/PUNCT-dominant entries
UNSEEN_CLASS = "UNK"  # no treebank evidence

# SECOND EVIDENCE CHANNEL: syntactic function, consulted only when the dominant
# UPOS is content-like and the entry would otherwise be discarded as OTH.
#
# The tag alone is the wrong instrument in languages that do not mark adverbs
# morphologically. German is the clear case: *absolut*, *aktuell*, *absichtlich*
# are adverbs, but German adverbs are formally identical to predicative
# adjectives, so UD tags them ADJ and 237 of the 312 German OTH entries were
# adjective-dominant. Czech, which marks adverbs with -e/-o, has one classless
# entry in the whole list. Reading the dependency relation instead of the tag
# asks what the word DOES rather than what it looks like, and it stays entirely
# treebank-derived -- no entry is classified from memory.
DEPREL_CLASS = {
    "advmod": "ADV", "det": "DET", "case": "ADP", "mark": "SCONJ",
    "cc": "CCONJ", "aux": "AUX", "aux:pass": "AUX", "cop": "AUX",
    "expl": "PRON", "nummod": "NUM",
}


def orth_variants(e):
    """Spelling variants to try when a direct lookup fails.

    Purely mechanical -- no lexical knowledge. The German list carries Swiss
    orthography (*abschliessend*) where the treebank has the eszett form
    (*abschliessend* -> *abschließend*), and the reverse occurs too.
    """
    seen = [e]
    for v in (e.replace("ß", "ss"), e.replace("ss", "ß")):
        if v not in seen:
            seen.append(v)
    return seen


def mine_corpus(code, needed, max_tokens=2_000_000):
    """UPOS + deprel evidence for entries the TREEBANK never shows, mined from
    the target corpora themselves.

    Why this is needed at all: the UD treebanks are contemporary prose, while
    the corpora are literary texts of roughly 1840-1920. The German residue
    after the syntactic-function fix is dominated by archaic and literary
    function words -- *alldieweil*, *allenthalben*, *allerorten*, *allerlei* --
    which a modern newspaper treebank was never going to contain. The evidence
    has to come from the period, so it comes from the corpus.

    The precedent is the medieval pipeline, where the gmh/gml v0.2 lists were
    likewise augmented with corpus-derived forms rather than hand-written ones.
    Provenance is recorded per entry in the `source` column, so a corpus-derived
    class is never mistaken for treebank evidence.

    Only tokens whose surface or lemma is in `needed` are counted, which keeps
    this to one pass and a small table.
    """
    sys.path.insert(0, str(HERE))
    import mask_corpora as mc

    # several folders can share an iso (german/swissgerman -> de); take the
    # plainest name, which is the language's own corpus
    cands = [f for f, (iso, model) in mc.SPACY.items() if iso == code and model]
    if not cands:
        return {}, 0
    folder = sorted(cands, key=len)[0]
    model = mc.SPACY[folder][1]

    src = ROOT / "data" / folder
    if not src.is_dir():
        return {}, 0
    try:
        import spacy
        nlp = spacy.load(model, exclude=["ner"])
    except Exception as e:                                  # noqa: BLE001
        print(f"  {code}: corpus augmentation unavailable ({type(e).__name__})",
              flush=True)
        return {}, 0

    # SAMPLE ACROSS THE WHOLE CORPUS, not off the front of it. Filling the
    # budget by reading records in order spends it all on the first file and its
    # first few authors -- for German, entirely on drama, with novels and poetry
    # never opened. The residue we are trying to reach is archaic literary
    # vocabulary, so breadth of author and genre matters more than depth in any
    # one of them. A per-record quota, taken from the middle of each text to
    # skip front matter, spreads the same budget over every author.
    files = sorted(src.glob("av_reference_*.jsonl"))
    nrec = 0
    for f in files:
        with f.open(encoding="utf-8") as fh:
            nrec += sum(1 for line in fh if line.strip())
    if nrec == 0:
        return {}, 0
    quota = max(500, max_tokens // nrec)

    texts = []
    ntok = 0
    for f in files:
        for line in f.open(encoding="utf-8"):
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            paras = [p.strip() for p in (rec.get("text") or "").split("\n") if p.strip()]
            if not paras:
                continue
            start = len(paras) // 4          # skip title pages and front matter
            got = 0
            for para in paras[start:]:
                texts.append(para[:100_000])
                n = para.count(" ") + 1
                got += n
                ntok += n
                if got >= quota:
                    break
    if not texts:
        return {}, 0

    # TWO BOUNDS ON THE TAGGING, because the entries we are chasing are a
    # Zipfian tail: waiting for every one of them to be decided is a condition
    # that never becomes true, and without a bound the pass simply tags
    # everything collected.
    #   - a hard token budget, so the cost is predictable whatever was collected
    #   - a saturation exit: stop when a long stretch of text has resolved no
    #     entry that was not already resolved
    stats = {}
    seen = 0
    stale = 0
    # Generous: rare entries are separated by long stretches of text, so a
    # short stale window quits while evidence is still arriving (measured:
    # at 40k it stopped early and lost ground against a smaller front-loaded
    # sample). The token budget is the real bound; this only saves time when
    # a language genuinely runs dry.
    STALE_LIMIT = 250_000         # tokens without a newly-resolved entry
    for doc in nlp.pipe(texts, batch_size=64):
        before = len(stats)
        for tok in doc:
            seen += 1
            for s in {tok.text.lower(), tok.lemma_.lower()}:
                if s in needed:
                    rec = stats.get(s)
                    if rec is None:
                        rec = stats[s] = (Counter(), Counter())
                    rec[0][tok.pos_] += 1
                    rec[1][tok.dep_] += 1
        stale = 0 if len(stats) > before else stale + len(doc)
        if seen >= max_tokens or stale >= STALE_LIMIT:
            break
    return stats, seen


def find_list(code):
    cands = sorted(LISTS.glob(f"POSNoise_PatternList_{code.title()}_v*.txt"))
    return cands[-1] if cands else None


def load_entries(path):
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            out.append(line)
    return out


def mine_upos(repos):
    """string (surface or lemma, lowercased) -> (UPOS Counter, deprel Counter)."""
    stats = {}
    n_sent = 0
    for repo in repos:
        for sent in load_treebank(repo):
            n_sent += 1
            for tok in sent:
                for s in {tok["form"], tok["lemma"]}:
                    rec = stats.get(s)
                    if rec is None:
                        rec = stats[s] = (Counter(), Counter())
                    rec[0][tok["upos"]] += 1
                    rec[1][tok["deprel"]] += 1
    return stats, n_sent


def _lookup(e, stats):
    """Entry -> (upos Counter, deprel Counter), trying spelling variants."""
    for v in orth_variants(e):
        rec = stats.get(v)
        if rec:
            return rec
    return None


def annotate(entry, stats):
    """(upos, class, evidence) for one list entry."""
    e = entry.lower()
    if " " in e:
        # fixed multiword unit: class is its own symbol; UPOS of the first word
        # is recorded for reference only
        rec = _lookup(e.split()[0], stats)
        upos = max(rec[0], key=rec[0].get) if rec else "-"
        return upos, "MWE", sum(rec[0].values()) if rec else 0
    rec = _lookup(e, stats)
    if not rec:
        return "-", UNSEEN_CLASS, 0
    cu, cd = rec
    upos = max(cu, key=cu.get)
    cls = UPOS_CLASS.get(upos)
    if cls is None:
        # content-dominant tag: ask what the word DOES before discarding it
        deprel = max(cd, key=cd.get).split(":")[0] if cd else ""
        cls = DEPREL_CLASS.get(deprel, OTHER_CLASS)
    return upos, cls, sum(cu.values())


def build_lang(code, augment=False):
    lp = find_list(code)
    if lp is None:
        return None, f"no pattern list for {code}"
    repos = ALL_TREEBANKS.get(code)
    if repos is None:
        return None, f"no treebank mapping for {code}"
    entries = load_entries(lp)
    stats, n_sent = mine_upos(repos)
    if n_sent == 0:
        return None, f"no treebank sentences loaded for {code}"
    rows = [[e, *annotate(e, stats), "ud"] for e in entries]
    for r in rows:
        if r[2] == UNSEEN_CLASS:
            r[4] = "-"

    n_corpus_tok = 0
    rescued = 0
    if augment:
        # Only entries the treebank could not place are re-examined, so corpus
        # evidence never overrides treebank evidence.
        needed = {r[0].lower() for r in rows if r[2] == UNSEEN_CLASS and " " not in r[0]}
        if needed:
            cstats, n_corpus_tok = mine_corpus(code, needed)
            for r in rows:
                if r[2] != UNSEEN_CLASS:
                    continue
                rec = cstats.get(r[0].lower())
                if not rec:
                    continue
                cu, cd = rec
                upos = max(cu, key=cu.get)
                cls = UPOS_CLASS.get(upos)
                if cls is None:
                    dep = max(cd, key=cd.get).split(":")[0] if cd else ""
                    cls = DEPREL_CLASS.get(dep, OTHER_CLASS)
                r[1], r[2], r[3], r[4] = upos, cls, sum(cu.values()), "corpus"
                rescued += 1

    OUT.mkdir(exist_ok=True)
    ver = "v1.2" if augment else "v1.1"
    op = OUT / f"POSNoise_Aligned_{code.title()}_{ver}.tsv"
    with open(op, "w", encoding="utf-8", newline="\n") as f:
        f.write(f"# aligned companion of {lp.name}; evidence: "
                f"{'+'.join(repos)} ({n_sent} sentences)"
                + (f" + target corpus ({n_corpus_tok} tokens tagged)" if augment else "")
                + "\n")
        f.write("# entry\tupos\tclass\tevidence_count\tsource\n")
        for e, upos, cls, n, srcname in rows:
            f.write(f"{e}\t{upos}\t{cls}\t{n}\t{srcname}\n")

    hist = Counter(r[2] for r in rows)
    seen = sum(1 for r in rows if r[2] != UNSEEN_CLASS)
    rep = {"list": lp.name, "treebanks": repos, "sentences": n_sent,
           "entries": len(rows), "with_evidence": seen,
           "coverage": round(seen / len(rows), 4),
           "corpus_tokens_tagged": n_corpus_tok, "corpus_rescued": rescued,
           "by_class": dict(hist.most_common())}
    return rep, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--langs", default="all")
    ap.add_argument("--augment", action="store_true",
                    help="mine the target corpora for entries the treebank "
                         "never shows (writes v1.2)")
    args = ap.parse_args()
    codes = (sorted(ALL_TREEBANKS) if args.langs == "all"
             else [c.strip().lower() for c in args.langs.split(",")])

    rp = OUT / "aligned_report.json"
    report = json.loads(rp.read_text(encoding="utf-8")) if rp.exists() else {}
    for code in codes:
        rep, err = build_lang(code, args.augment)
        if err:
            print(f"{code}: SKIP ({err})", flush=True)
            continue
        report[code] = rep
        print(f"{code}: {rep['entries']} entries, coverage {rep['coverage']:.1%}, "
              f"classes {rep['by_class']}", flush=True)
    OUT.mkdir(exist_ok=True)
    rp.write_text(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True),
                  encoding="utf-8")


if __name__ == "__main__":
    main()
