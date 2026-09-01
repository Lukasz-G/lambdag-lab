# -*- coding: utf-8 -*-
"""
Corpus-derived lemma augmentation of the Middle Low German (gml) POSNoise list.

This is the Middle **Low** German twin of ``build_gmh_lemma_augment.py``: it applies
the *same* corpus-derived construction that produced ``Gmh_v0.2`` to the Middle Low
German list, over the **ReN** corpus (Referenzkorpus Mittelniederdeutsch/Nieder-
rheinisch) instead of ReM. The shipped ``POSNoise_PatternList_Gml_v0.1.txt`` is a
*surface* list (~2981 attested spellings). When the masker emits lemmas
(``emit="lemma"``) or meets a scribal spelling whose particular surface is unlisted,
a function word gets masked because ReN's *lemma* forms (``hebben``, ``schȫlen``)
differ from the listed surfaces. This tool adds those lemma forms **from the corpus,
not from memory** (the standing rule for gmh/gml lists).

Input is the tab extraction produced by ``rem_extract.ipynb``'s ReN cell, i.e.
``surface<TAB>lemma<TAB>POS<TAB>morph`` files (POS = ReN's HiTS variant, mapped to
UD by ``hits_to_ud.json``). A lemma L is added iff:

  * some token with lemma L has a surface that is ALREADY whitelisted (so L is a
    confirmed function-word form), and
  * that token's HiTS->UD POS is a function class:
      - ADV or AUX                              (function adverbs, auxiliaries), or
      - VERB whose lemma L also occurs as AUX   (light verbs: hebben, wēsen, wērden,
                                                 schȫlen, willen, mȫgen, ...)

plus two closed POS-gated groups, exactly as in the gmh build:

  * aux + enclitic-pronoun univerbations  (hestu = hebben+dû, scaltu = schȫlen+dû,
    Bistu = wēsen+dû): a '+'-lemma whose head POS is AUX/modal and which embeds no
    content verb (rules out the aux+participle periphrasis hefftvpsporen =
    hebben+upspȫren), and
  * pronominal adverbs, the closed PAV* class (dâr+ümme = "darum", dâr+nâ, dâr+in,
    hîr, wôr, ...), minus '++' multi-component and verb-embedding constructions.

Content classes (NOUN, PROPN, ADJ, NUM, SYM, X) are excluded, so a function surface
homographic with an inflected content word never drags a content lemma into the list.
Wholly foreign (Latin) tokens carry ``FM`` and are skipped; Germanic function words
that are homographs of Latin (``in``, ``an``, ``ut``) are tagged as their Germanic
class here (PAV*/APPR/AVD) and so are kept -- see POSNOISE_LISTS.md.

Unlike the gmh build, ``CURATED_ADV`` is **empty by default**: the eleven curated
adverbs added to Gmh_v0.2 were hand-approved Middle *High* German forms, and inventing
Middle Low German adverbs from memory would violate the corpus-derived rule (and gml
is the list we trust least; see POSNOISE_LISTS.md). The mechanism is kept -- populate
it with corpus-attested ReN lemma strings and rerun to extend.

Writes ``posnoise_lists/POSNoise_PatternList_Gml_v0.2.txt`` = v0.1 + the new lemmas,
which ``_find_pattern_list`` then auto-selects as the highest version.

Usage:  python build_gml_lemma_augment.py [CORPUS_DIR]
        (CORPUS_DIR default: D:\\Corpora\\ReN-v1.1_tab)
"""
import sys
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lambdag import load_hits_to_ud, _find_pattern_list  # noqa: E402

CORPUS_DIR = Path(sys.argv[1] if len(sys.argv) > 1 else r"D:\Corpora\ReN-v1.1_tab")
CONTENT = {"NOUN", "PROPN", "ADJ", "NUM", "SYM", "X"}
DROP_LEMMA = {"[!]", "--"}

# Curator-approved function adverbs -- intentionally EMPTY for Middle Low German
# (see module docstring). To extend, add exact ReN lemma strings; each is validated
# below and only added if it actually occurs as an ADV lemma in the corpus.
CURATED_ADV = set()

HITS = load_hits_to_ud()

def ud(pos):
    head = pos.split("+", 1)[0]
    return HITS.get(head, HITS.get(head.split(".")[0], ""))

def clean_lemma(l):
    l = l.lower()
    if not l or " " in l or "+" in l or "/" in l or l in DROP_LEMMA:
        return False
    if l.count("(") != l.count(")"):      # reject malformed ReN lemmas, e.g. "biz)"
        return False
    return True

# ---- base list (always the shipped v0.1, so the build is idempotent) -------
base_path = _find_pattern_list("gml").with_name("POSNoise_PatternList_Gml_v0.1.txt")
base_lines = base_path.read_text(encoding="utf-8").splitlines()
base_single = {ln.strip().lower() for ln in base_lines if ln.strip() and " " not in ln.strip()}
base_all = {ln.strip().lower() for ln in base_lines if ln.strip()}
print(f"base list: {base_path.name}  ({len(base_lines)} entries, {len(base_single)} single-token)")

# ---- scan the corpus ------------------------------------------------------
files = sorted(p for p in CORPUS_DIR.glob("*.txt") if p.is_file())
print(f"scanning {len(files)} files in {CORPUS_DIR} ...")

lemma_uds = defaultdict(set)          # lemma -> set of UD tags it is ever seen with
cand = defaultdict(set)               # lemma -> UD tags where surface was whitelisted
fusion_cand = defaultdict(set)        # aux+clitic fusion lemma -> set of head lemmas
fusion_periphrasis = set()            # fusion lemmas with a content-VERB tail POS
pav_lemmas = set()                    # pronominal-adverb lemmas (PAV* tags)
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
        # pronominal adverbs (PAVAP/PAVKO/PAVD/PAVW/...), a closed function class
        # (dâr+ümme=darum, dâr+nâ=danach, dâr+in, hîr, wôr, ...). Collected raw here;
        # lemmas that embed a content VERB are filtered out after the scan.
        if head_pos.startswith("PAV") and " " not in lem and lem not in DROP_LEMMA:
            pav_lemmas.add(lem.lower())
        # aux + enclitic-pronoun univerbations (hestu = hebben+dû): a '+'-lemma (no
        # '/'-separated notation) whose head POS is AUX/modal. Recorded for a 2nd pass.
        # If any *tail* POS component is itself a content verb (VAFIN+VVPP = perfect
        # periphrasis "hebben+upspȫren"), flag it: this catches periphrases whose
        # participle never occurs standalone, which the lemma-membership test misses.
        if "+" in lem and "/" not in lem and u == "AUX":
            L = lem.lower()
            fusion_cand[L].add(lem.split("+", 1)[0].lower())
            if any(ud(p) == "VERB" for p in pos.split("+")[1:]):
                fusion_periphrasis.add(L)
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

def _tail_embeds_verb(lem):
    """True if a NON-head component of a fusion lemma is a content verb.

    The head (component before the first '+') is a confirmed AUX by construction of
    ``fusion_cand`` (recorded only when ``ud(pos) == "AUX"``). Only a *following*
    component being a content verb makes this an aux+participle **periphrasis**
    (``hebben+upspoeren`` = "has tracked down") rather than the aux+enclitic
    **contraction** we want (``hestu`` = ``hebben+du``).

    This is where the gml build MUST diverge from ``build_gmh_lemma_augment.py``:
    ReN tags its auxiliary/modal lemmas (hebben, wesen, schoelen, kuennen, willen,
    moegen, ...) as ``VERB`` too -- their main-verb readings -- so all 149 fusion
    heads land in ``verb_lemmas``. Checking the *whole* lemma (as the gmh build
    safely can, because ReM does not double-tag these heads) would wrongly drop
    every contraction (measured: 149 -> 0). Checking only the tail keeps the 126
    real contractions and still drops the periphrases. ``++`` multi-component
    fusions are excluded as messy, mirroring the PAV group.
    """
    parts = [p for p in re.split(r"[+/.>]+", lem) if p]
    return "++" in lem or any(p in verb_lemmas for p in parts[1:])

# aux + enclitic-pronoun univerbations: head lemma is a whitelisted auxiliary AND no
# *following* component is a content verb (rules out aux+participle periphrases).
aug_so_far = base_all | additions
fusions = {L for L, heads in fusion_cand.items()
           if L not in base_all and any(h in aug_so_far for h in heads)
           and L not in fusion_periphrasis
           and not _tail_embeds_verb(L)}
additions |= fusions
print(f"  of which {len(fusions)} are aux+clitic univerbation lemmas (hestu = hebben+dû, ...)")

# pronominal adverbs (POS-gated closed function class), minus '++' multi-component
# and any that embed a content verb.
def _messy(lem):
    return "++" in lem or _embeds_verb(lem)
pav_new = {L for L in pav_lemmas if L not in base_all and not _messy(L)}
dropped_verb = sum(1 for L in pav_lemmas if L not in base_all and _messy(L))
additions |= pav_new
print(f"  + {len(pav_new)} pronominal-adverb lemmas (dâr+ümme, dâr+nâ, hîr, wôr, ...) "
      f"[{dropped_verb} verb-embedding constructions dropped]")

# curator-approved function adverbs, validated as ADV lemmas in the corpus
curated_ok = {L for L in CURATED_ADV if "ADV" in lemma_uds.get(L, set()) and L not in base_all}
missing = {L for L in CURATED_ADV if "ADV" not in lemma_uds.get(L, set())}
additions |= curated_ok
print(f"  + {len(curated_ok)} curated function adverbs")
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
out = base_path.with_name("POSNoise_PatternList_Gml_v0.2.txt")
out.write_text("\n".join(base_lines + sorted(additions)) + "\n", encoding="utf-8")
print(f"\n+{len(additions)} lemma forms -> {out}")
print("sample additions:", ", ".join(sorted(additions)[:40]))
