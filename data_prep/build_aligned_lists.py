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
    """string (surface or lemma, lowercased) -> Counter of UPOS occurrences."""
    stats = {}
    n_sent = 0
    for repo in repos:
        for sent in load_treebank(repo):
            n_sent += 1
            for tok in sent:
                for s in {tok["form"], tok["lemma"]}:
                    stats.setdefault(s, Counter())[tok["upos"]] += 1
    return stats, n_sent


def annotate(entry, stats):
    """(upos, class, evidence) for one list entry."""
    e = entry.lower()
    if " " in e:
        # fixed multiword unit: class is its own symbol; UPOS of the first word
        # is recorded for reference only
        head = stats.get(e.split()[0])
        upos = max(head, key=head.get) if head else "-"
        return upos, "MWE", sum(head.values()) if head else 0
    c = stats.get(e)
    if not c:
        return "-", UNSEEN_CLASS, 0
    upos = max(c, key=c.get)
    return upos, UPOS_CLASS.get(upos, OTHER_CLASS), sum(c.values())


def build_lang(code):
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
    rows = [(e, *annotate(e, stats)) for e in entries]

    OUT.mkdir(exist_ok=True)
    op = OUT / f"POSNoise_Aligned_{code.title()}_v1.0.tsv"
    with open(op, "w", encoding="utf-8", newline="\n") as f:
        f.write(f"# aligned companion of {lp.name}; evidence: "
                f"{'+'.join(repos)} ({n_sent} sentences)\n")
        f.write("# entry\tupos\tclass\tevidence_count\n")
        for e, upos, cls, n in rows:
            f.write(f"{e}\t{upos}\t{cls}\t{n}\n")

    hist = Counter(cls for _, _, cls, _ in rows)
    seen = sum(1 for _, _, cls, _ in rows if cls != UNSEEN_CLASS)
    rep = {"list": lp.name, "treebanks": repos, "sentences": n_sent,
           "entries": len(rows), "with_evidence": seen,
           "coverage": round(seen / len(rows), 4),
           "by_class": dict(hist.most_common())}
    return rep, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--langs", default="all")
    args = ap.parse_args()
    codes = (sorted(ALL_TREEBANKS) if args.langs == "all"
             else [c.strip().lower() for c in args.langs.split(",")])

    rp = OUT / "aligned_report.json"
    report = json.loads(rp.read_text(encoding="utf-8")) if rp.exists() else {}
    for code in codes:
        rep, err = build_lang(code)
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
