# Build POSNoise safe-pattern lists for new languages from UD treebanks, following the
# method documented in POSNOISE_LISTS.md (nothing here is written from memory: every
# entry is attested in a treebank, and every entry is validated against it).
#
# What a list must rescue -- POSNoise masks {ADJ, ADV, AUX, NOUN, NUM, PROPN, SYM, VERB, X}
# and keeps the closed classes automatically, so the list exists for:
#   1. AUXILIARIES      complete lemma inventory of AUX             (measured)
#   2. LIGHT/MODAL VERBS VERB lemmas that mostly take a clausal complement -- "can, must,
#                        begin, seem": functional, not topical      (measured, thresholded)
#   3. FUNCTION ADVERBS  the highest-frequency ADV lemmas (negation, degree, time, place);
#                        manner adverbs are left to be masked       (measured, thresholded)
#   4. MULTIWORD UNITS   `fixed` chains whose head carries a functional deprel
#                        ("sin embargo", "potому что", "ainsi que")  (measured)
#   5. CLOSED CLASSES    frequent DET/ADP/PRON/CCONJ/SCONJ/PART -- redundant with the POS
#                        rule, but the shipped en/de lists carry them and they cover
#                        tagger slips                                (measured)
# Every candidate is then VALIDATED: anything whose dominant tag in the treebank is
# NOUN/PROPN is dropped and reported (this is the check that caught FR `point` and
# PL `czasem` when the shipped lists were built).
#
#   python data_prep/build_posnoise.py --langs cs,hu,lt      # selected
#   python data_prep/build_posnoise.py                       # every language we need
#
# Output: posnoise_lists/POSNoise_PatternList_{Xx}_v1.0.txt  + a provenance report.

import argparse, io, json, re, sys, zipfile
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from _net import download

LISTS = ROOT / "posnoise_lists"
CACHE = HERE / "cache" / "ud"; CACHE.mkdir(parents=True, exist_ok=True)

# language -> UD treebank repos (pooled; small languages need every token they can get)
TREEBANKS = {
    "cs": ["UD_Czech-PDT"],
    "el": ["UD_Greek-GDT"],
    "hr": ["UD_Croatian-SET"],
    "hu": ["UD_Hungarian-Szeged", "UD_Hungarian-POUD"],
    "lt": ["UD_Lithuanian-ALKSNIS", "UD_Lithuanian-HSE"],
    "lv": ["UD_Latvian-LVTB"],
    "nl": ["UD_Dutch-Alpino", "UD_Dutch-LassySmall"],
    "no": ["UD_Norwegian-Bokmaal"],
    "pt": ["UD_Portuguese-Bosque"],
    "ro": ["UD_Romanian-RRT"],
    "sl": ["UD_Slovenian-SSJ"],
    "sr": ["UD_Serbian-SET"],
    "sv": ["UD_Swedish-Talbanken"],
    "uk": ["UD_Ukrainian-IU"],
}
CLOSED = {"DET", "ADP", "PRON", "CCONJ", "SCONJ", "PART"}
MASKED = {"ADJ", "ADV", "AUX", "NOUN", "NUM", "PROPN", "SYM", "VERB", "X"}
FUNC_DEPREL = {"case", "mark", "cc", "advmod", "cop", "aux", "det", "fixed", "expl"}


def load_treebank(repo):
    """Yield sentences as lists of token dicts, from the repo's CoNLL-U files."""
    zp = CACHE / f"{repo}.zip"
    if not zp.exists():
        ok = None
        for branch in ("master", "main", "dev"):
            ok = download(f"https://codeload.github.com/UniversalDependencies/{repo}/zip/refs/heads/{branch}", zp)
            if ok:
                break
        if not ok:
            return
    try:
        zf = zipfile.ZipFile(zp)
    except zipfile.BadZipFile:
        zp.unlink(missing_ok=True); return
    for m in zf.namelist():
        if not m.endswith(".conllu"):
            continue
        sent = []
        for line in io.TextIOWrapper(zf.open(m), encoding="utf-8", errors="replace"):
            line = line.rstrip("\n")
            if not line:
                if sent:
                    yield sent
                sent = []
                continue
            if line.startswith("#"):
                continue
            f = line.split("\t")
            if len(f) != 10 or "-" in f[0] or "." in f[0]:
                continue
            sent.append({"id": int(f[0]), "form": f[1].lower(), "lemma": (f[2] or f[1]).lower(),
                         "upos": f[3], "head": int(f[6]) if f[6].isdigit() else 0, "deprel": f[7]})
        if sent:
            yield sent


def mine(lang, repos):
    pos_of = defaultdict(Counter)          # lemma -> POS counts   (for validation)
    form_pos = defaultdict(Counter)
    aux, adv, closed = Counter(), Counter(), Counter()
    verb_tot, verb_clausal = Counter(), Counter()
    mwe = Counter()
    ntok = 0

    for repos_i in repos:
        for sent in load_treebank(repos_i):
            idx = {t["id"]: t for t in sent}
            for t in sent:
                ntok += 1
                lem, up = t["lemma"], t["upos"]
                pos_of[lem][up] += 1
                form_pos[t["form"]][up] += 1
                if up == "AUX":
                    aux[lem] += 1
                elif up == "ADV":
                    adv[lem] += 1
                elif up in CLOSED:
                    closed[lem] += 1
                elif up == "VERB":
                    verb_tot[lem] += 1
                    if any(c["head"] == t["id"] and c["deprel"] in ("xcomp", "ccomp")
                           for c in sent):
                        verb_clausal[lem] += 1
                # multiword functional units: a `fixed` chain hanging off a functional head
                if t["deprel"] == "fixed":
                    h = idx.get(t["head"])
                    if h is not None and h["deprel"] in FUNC_DEPREL and h["id"] < t["id"]:
                        parts = [h["form"]] + [x["form"] for x in sent
                                               if x["deprel"] == "fixed" and x["head"] == h["id"]]
                        unit = " ".join(parts)
                        if 1 < len(parts) <= 4 and all(re.search(r"\w", p) for p in parts):
                            mwe[unit] += 1
    return dict(pos_of=pos_of, form_pos=form_pos, aux=aux, adv=adv, closed=closed,
                verb_tot=verb_tot, verb_clausal=verb_clausal, mwe=mwe, ntok=ntok)


def function_adverbs(adv, top_adv):
    """Split ADV lemmas into function adverbs (keep) and manner adverbs (mask).

    POSNoise keeps degree/time/place/negation/focus adverbs and masks manner adverbs.
    Rather than hand-writing a suffix list per language (which the project forbids --
    entries must be measured, not recalled), we exploit a typological regularity that
    is measurable in any treebank: manner adverbs are DERIVED and productive, so their
    ending is shared by very many distinct ADV types (-nie/-o in Polish, -ment in
    French, -mente in Spanish, -ly in English, -ai in Lithuanian), whereas function
    adverbs form a small, high-frequency, morphologically idiosyncratic core.

    So: an ADV is treated as manner when its final 2- or 3-character string is shared
    by a large number of other ADV types -- unless it is itself among the most frequent
    adverbs, where the closed-class core lives regardless of how it happens to end.
    """
    types = list(adv)
    if not types:
        return []
    suf1, suf2, suf3 = Counter(), Counter(), Counter()
    for t in types:
        if len(t) >= 2:
            suf1[t[-1:]] += 1
        if len(t) >= 3:
            suf2[t[-2:]] += 1
        if len(t) >= 4:
            suf3[t[-3:]] += 1
    ntypes = len(types)
    prod_cut = max(8, int(0.04 * ntypes))       # "productive" = >=4% of all ADV types
    # a single-character ending (Polish -o, Russian -о) needs a stricter bar, since one
    # letter is shared by chance far more often than a two- or three-letter string
    prod_cut1 = max(20, int(0.15 * ntypes))
    core = {l for l, _ in adv.most_common(max(25, top_adv // 4))}   # always-keep core

    kept = []
    for lem, c in adv.most_common(top_adv):
        if lem in core:
            kept.append(lem); continue
        if max(suf2.get(lem[-2:], 0), suf3.get(lem[-3:], 0)) >= prod_cut:
            continue                             # derived/manner -> let POSNoise mask it
        if suf1.get(lem[-1:], 0) >= prod_cut1:
            continue
        kept.append(lem)
    return kept


def build(lang, stats, report):
    n = stats["ntok"]
    scale = max(1.0, n / 200_000)                     # thresholds relative to treebank size
    min_mwe = 3 if n >= 150_000 else 2                # small treebanks need a lower bar
    top_adv = int(120 * min(2.0, max(0.6, scale)))
    min_closed = max(2, int(3 * scale))
    min_verb = max(5, int(10 * scale))

    cand = {}
    for lem, c in stats["aux"].items():               # 1. every attested auxiliary
        cand[lem] = "aux"
    for lem, c in stats["verb_tot"].items():          # 2. light / modal verbs
        if c >= min_verb and stats["verb_clausal"][lem] / c >= 0.30:
            cand.setdefault(lem, "light-verb")
    for lem in function_adverbs(stats["adv"], top_adv):   # 3. function (not manner) adverbs
        cand.setdefault(lem, "adverb")
    for lem, c in stats["closed"].items():            # 5. closed classes (belt and braces)
        if c >= min_closed:
            cand.setdefault(lem, "closed")
    for unit, c in stats["mwe"].items():              # 4. multiword functional units
        if c >= min_mwe:
            cand.setdefault(unit, "multiword")

    # ---- validation: drop anything the treebank says is really a noun --------
    kept, dropped = {}, []
    for entry, cat in cand.items():
        if " " in entry:
            kept[entry] = cat; continue
        pc = stats["pos_of"].get(entry) or stats["form_pos"].get(entry)
        if not pc:
            dropped.append((entry, cat, "unattested")); continue
        dom, dom_n = pc.most_common(1)[0]
        if dom in ("NOUN", "PROPN") and dom_n / sum(pc.values()) > 0.5:
            dropped.append((entry, cat, f"dominantly {dom} ({dom_n}/{sum(pc.values())})")); continue
        kept[entry] = cat
    report[lang] = {"tokens": n, "kept": len(kept), "dropped": len(dropped),
                    "by_category": Counter(kept.values()),
                    "dropped_examples": dropped[:15],
                    "thresholds": {"top_adv": top_adv, "min_mwe": min_mwe,
                                   "min_closed": min_closed, "min_verb": min_verb}}
    return sorted(kept)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--langs", default="", help="comma-separated ISO codes (default: all)")
    args = ap.parse_args()
    todo = [l.strip() for l in args.langs.split(",") if l.strip()] or sorted(TREEBANKS)

    report = {}
    for lang in todo:
        if lang not in TREEBANKS:
            print(f"no treebank configured for {lang!r}", flush=True); continue
        stats = mine(lang, TREEBANKS[lang])
        if not stats["ntok"]:
            print(f"{lang}: treebank download failed", flush=True); continue
        entries = build(lang, stats, report)
        out = LISTS / f"POSNoise_PatternList_{lang.capitalize()}_v1.0.txt"
        out.write_text("\n".join(entries) + "\n", encoding="utf-8")
        r = report[lang]
        print(f"{lang}  {r['tokens']:>9,} tok  {len(entries):5d} entries "
              f"({dict(r['by_category'])})  {r['dropped']} dropped -> {out.name}", flush=True)

    with open(HERE / "posnoise_build_report.json", "w", encoding="utf-8") as fh:
        json.dump({k: {kk: (dict(vv) if isinstance(vv, Counter) else vv) for kk, vv in v.items()}
                   for k, v in report.items()}, fh, indent=2, ensure_ascii=False)
    print(f"\nprovenance report -> {HERE / 'posnoise_build_report.json'}")


if __name__ == "__main__":
    main()
