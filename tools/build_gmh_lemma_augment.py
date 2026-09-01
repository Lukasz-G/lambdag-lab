# -*- coding: utf-8 -*-
"""
Corpus-derived lemma augmentation of the Middle High German (gmh) POSNoise list.

The shipped list ``POSNoise_PatternList_Gmh_v0.1.txt`` is in normalised orthography
and matches surface forms; but ReM's own *lemmas* differ from a few list entries
(``wërden`` vs ``werden``), so a scribal spelling variant of a function word whose
particular surface is not listed gets masked. Emitting lemmas (``emit="lemma"``)
makes this worse unless the list also carries the lemma forms.

This tool closes that gap **from the corpus, not from memory** (the standing rule
for gmh/gml lists). Over a ReM corpus of ``surface<TAB>lemma<TAB>POS<TAB>morph``
files it adds a lemma L to the list iff:

  * some token with lemma L has a surface that is ALREADY whitelisted (so L is a
    confirmed function-word form), and
  * that token's HiTS->UD POS is a function class:
      - ADV or AUX                              (function adverbs, auxiliaries), or
      - VERB whose lemma L also occurs as AUX   (light verbs: wërden, sîn, haben, ...)

Content classes (NOUN, PROPN, ADJ, NUM, SYM, X) are excluded, so a function surface
that is homographic with an inflected content word (``man`` vs noun ``mann``) never
drags a content lemma into the list.

Writes ``posnoise_lists/POSNoise_PatternList_Gmh_v0.2.txt`` = v0.1 + the new lemmas,
which ``_find_pattern_list`` then auto-selects as the highest version.

Usage:  python build_gmh_lemma_augment.py [CORPUS_DIR]
        (CORPUS_DIR default: D:\\Corpora\\MHD)
"""
import sys
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lambdag import load_hits_to_ud, _find_pattern_list  # noqa: E402

CORPUS_DIR = Path(sys.argv[1] if len(sys.argv) > 1 else r"D:\Corpora\MHD")
CONTENT = {"NOUN", "PROPN", "ADJ", "NUM", "SYM", "X"}
DROP_LEMMA = {"[!]"}

# Group B: curator-approved function adverbs (manner / degree / negation) that the
# corpus leaves masked. Exact ReM lemma strings (corpus-attested, not from memory);
# each is validated below and only added if it actually occurs as an ADV lemma.
CURATED_ADV = {
    "gërne", "lange", "niène", "vërre", "schône", "balde", "vruo",
    "lèider", "übel(e)", "wær-lîche", "êwig-lîche",
}

HITS = load_hits_to_ud()

def ud(pos):
    head = pos.split("+", 1)[0]
    return HITS.get(head, HITS.get(head.split(".")[0], ""))

def clean_lemma(l):
    l = l.lower()
    if not l or " " in l or "+" in l or "/" in l or l in DROP_LEMMA:
        return False
    if l.count("(") != l.count(")"):      # reject malformed ReM lemmas, e.g. "biz)"
        return False
    return True

# ---- base list (always the shipped v0.1, so the build is idempotent) -------
base_path = _find_pattern_list("gmh").with_name("POSNoise_PatternList_Gmh_v0.1.txt")
base_lines = base_path.read_text(encoding="utf-8").splitlines()
base_single = {ln.strip().lower() for ln in base_lines if ln.strip() and " " not in ln.strip()}
base_all = {ln.strip().lower() for ln in base_lines if ln.strip()}
print(f"base list: {base_path.name}  ({len(base_lines)} entries, {len(base_single)} single-token)")

# ---- scan the corpus ------------------------------------------------------
files = sorted(p for p in CORPUS_DIR.glob("*") if p.is_file())
print(f"scanning {len(files)} files in {CORPUS_DIR} ...")

lemma_uds = defaultdict(set)          # lemma -> set of UD tags it is ever seen with
cand = defaultdict(set)               # lemma -> UD tags where surface was whitelisted
fusion_cand = defaultdict(set)        # aux+clitic fusion lemma -> set of head lemmas
pav_lemmas = set()                    # Group A: pronominal-adverb lemmas (PAV* tags)
n_tok = 0
for f in files:
    for line in f.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        c = line.split("\t")
        if len(c) < 3:
            continue
        surf, lem, pos = c[0], c[1], c[2]
        if lem in DROP_LEMMA or "FM" in pos.split("+"):
            continue
        n_tok += 1
        u = ud(pos)
        head_pos = pos.split("+", 1)[0]
        # Group A: pronominal adverbs (PAVD/PAVAP/PAVG/PAVREL/PAVW), a closed function
        # class (darumbe=dâr/+umbe, dazu=dâr/+zuo, damit, danach, ...). Collected raw
        # here; lemmas that embed a content VERB (separated particle+verb constructions,
        # ane/dâr++ge-dènken, ane/.+dâr+sëhen) are filtered out after the scan.
        if head_pos.startswith("PAV") and " " not in lem and lem not in DROP_LEMMA:
            pav_lemmas.add(lem.lower())
        # aux + enclitic-pronoun univerbations (biſtu = sîn+dû): a '+'-lemma (no '/'
        # separated notation) whose head POS is AUX. Recorded for a second pass.
        if "+" in lem and "/" not in lem and u == "AUX":
            fusion_cand[lem.lower()].add(lem.split("+", 1)[0].lower())
        if not clean_lemma(lem):
            continue
        L = lem.lower()
        lemma_uds[L].add(u)
        if surf.lower() in base_single:
            cand[L].add(u)

print(f"scanned {n_tok:,} tokens, {len(lemma_uds):,} distinct clean lemmas")

# ---- decide additions -----------------------------------------------------
aux_lemmas = {L for L, uds in lemma_uds.items() if "AUX" in uds}
additions = set()
for L, uds in cand.items():
    if L in base_all:
        continue
    if uds & {"ADV", "AUX"}:
        additions.add(L)
    elif "VERB" in uds and L in aux_lemmas:      # light / auxiliary verb
        additions.add(L)
    # tokens seen only under content POS with a whitelisted (homographic) surface: skip

# safety: never add a lemma that is a content word EVERYWHERE it appears
additions = {L for L in additions if lemma_uds[L] - CONTENT}

# A lemma "embeds a content verb" if any of its notation-split components is a known
# VERB lemma -- used to reject separated particle+verb / aux+verb periphrases.
import re
verb_lemmas = {L for L, uds in lemma_uds.items() if "VERB" in uds}
def _embeds_verb(lem):
    return any(p in verb_lemmas for p in re.split(r"[+/.>]+", lem) if p)

# aux + enclitic-pronoun univerbations: head lemma is a whitelisted auxiliary AND the
# fusion embeds no content verb (rules out wërden+sëhen, sol(e)n+ge-sëhen periphrases).
aug_so_far = base_all | additions
fusions = {L for L, heads in fusion_cand.items()
           if L not in base_all and any(h in aug_so_far for h in heads)
           and not _embeds_verb(L)}
additions |= fusions
print(f"  of which {len(fusions)} are aux+clitic univerbation lemmas (biſtu = sîn+dû, ...)")

# Group A: pronominal adverbs (POS-gated closed function class), minus '++' multi-
# component constructions and any that embed a content verb.
def _messy(lem):
    return "++" in lem or _embeds_verb(lem)
pav_new = {L for L in pav_lemmas if L not in base_all and not _messy(L)}
dropped_verb = sum(1 for L in pav_lemmas if L not in base_all and _messy(L))
additions |= pav_new
print(f"  + {len(pav_new)} pronominal-adverb lemmas (darumbe, dazu, hier, warumbe, ...) "
      f"[{dropped_verb} verb-embedding constructions dropped]")

# Group B: curator-approved function adverbs, validated as ADV lemmas in the corpus
curated_ok = {L for L in CURATED_ADV if "ADV" in lemma_uds.get(L, set()) and L not in base_all}
missing = {L for L in CURATED_ADV if "ADV" not in lemma_uds.get(L, set())}
additions |= curated_ok
print(f"  + {len(curated_ok)} curated function adverbs (gërne, lange, niène, ...)")
if missing:
    print(f"  ! curated adverbs NOT found as ADV lemmas in the corpus (check spelling): {sorted(missing)}")

# ---- coverage before / after (ADV & AUX token occurrences) ----------------
def rescuable(listset):
    cov = tot = 0
    for f in files:
        for line in f.read_text(encoding="utf-8", errors="replace").splitlines():
            c = line.split("\t")
            if len(c) < 3:
                continue
            surf, lem, pos = c[0], c[1], c[2]
            if lem in DROP_LEMMA or "FM" in pos.split("+"):
                continue
            if ud(pos) in ("ADV", "AUX"):
                tot += 1
                if surf.lower() in listset or lem.lower() in listset:
                    cov += 1
    return cov, tot

before = rescuable(base_all)
after = rescuable(base_all | additions)
print(f"\nADV/AUX occurrences rescuable (surface or lemma listed):")
print(f"   before: {before[0]:,}/{before[1]:,} = {before[0]/before[1]:.1%}")
print(f"   after : {after[0]:,}/{after[1]:,} = {after[0]/after[1]:.1%}")

# ---- write v0.2 -----------------------------------------------------------
out = base_path.with_name("POSNoise_PatternList_Gmh_v0.2.txt")
out.write_text("\n".join(base_lines + sorted(additions)) + "\n", encoding="utf-8")
print(f"\n+{len(additions)} lemma forms -> {out}")
print("sample additions:", ", ".join(sorted(additions)[:40]))
