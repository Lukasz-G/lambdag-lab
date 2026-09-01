# -*- coding: utf-8 -*-
"""
Build POSNoise safe-pattern lists for Middle High German (gmh) and
Middle Low German (gml).

Empirical base -- neither list is written from memory:
  gmh : ReM (Referenzkorpus Mittelhochdeutsch 1050-1350) v1.0, via the
        ReM-derived token->HiTS inventory shipped in cltk/gmh_models_cltk
        (61,312 token types).
  gml : ReN (Referenzkorpus Mittelniederdeutsch/Niederrheinisch 1200-1650) v0.6,
        via the ReN-trained NLTK backoff tagger in cltk/gml_models_cltk
        (24,431 token types recovered from its UnigramTagger model).

Both corpora use HiTS (Historisches Tagset, Ruhr-Uni Bochum), not UD, so the
POS names are mapped below. ReM and ReN use slightly different HiTS variants
(ReM has DDART, ReN has DDARTA), hence both spellings appear in the map.

Why this works where a hand-written list could not: historical German has no
standardised orthography. ReM attests *19 spellings of the negation particle*
(ne, niht, nieht, niet, niuht, niwet, niwiht, niuwet, niewet, nieuht, niuweht,
niut, niwit, ...). ReN attests ~15 of the preposition an (an, ane, aen, ahn,
ahne, am, ame, amm, amme, ...) and enclitic fusions such as `hestu` (hest+du)
or `datck` (dat+ick). Those are exactly the forms a modern-German intuition
would miss, and exactly the ones that carry authorial/scribal signal.
"""
import pickle, re, json, sys
from collections import Counter, defaultdict

# --------------------------------------------------------------------------- #
# HiTS (Historisches Tagset) -> Universal Dependencies POS
# --------------------------------------------------------------------------- #
HITS_TO_UD = {
    # --- determiners / articles (ReM: DDART, ReN: DDARTA)
    "DDART": "DET", "DDARTA": "DET", "DIART": "DET", "DIARTA": "DET",
    "DDA": "DET", "DDS": "DET", "DDN": "DET", "DDD": "DET",
    "DIA": "DET", "DIS": "DET", "DIN": "DET", "DID": "DET", "DIDA": "DET",
    "DGA": "DET", "DGS": "DET", "DGN": "DET",
    "DPOSA": "DET", "DPOSS": "DET", "DPOSN": "DET", "DPOSD": "DET",
    "DRELS": "PRON", "DRELA": "DET", "DPIS": "PRON", "DPIA": "DET",
    "DDART*": "DET", "DINEG": "DET",
    # --- pronouns
    "PPER": "PRON", "PRF": "PRON", "PI": "PRON", "PW": "PRON", "PG": "PRON",
    "PWS": "PRON", "PWA": "DET", "PRELS": "PRON", "PRELAT": "DET",
    "PAVAP": "ADV", "PAVD": "ADV", "PAVG": "ADV", "PAVW": "ADV", "PAVREL": "ADV",
    "PPOSS": "PRON", "PPOSAT": "DET", "PNEG": "PRON",
    # --- verbs: full -> VERB, auxiliary and modal -> AUX (as in UD)
    "VVFIN": "VERB", "VVINF": "VERB", "VVPP": "VERB", "VVIMP": "VERB",
    "VVPS": "VERB", "VVPP*": "VERB",
    "VAFIN": "AUX", "VAINF": "AUX", "VAPP": "AUX", "VAIMP": "AUX", "VAPS": "AUX",
    "VMFIN": "AUX", "VMINF": "AUX", "VMPP": "AUX", "VMIMP": "AUX",
    # --- adpositions
    "APPR": "ADP", "APPRART": "ADP", "APPO": "ADP", "APZR": "ADP", "APPR*": "ADP",
    # --- conjunctions
    "KON": "CCONJ", "KOUS": "SCONJ", "KOKOM": "SCONJ", "KO*": "CCONJ",
    "KOUS*": "SCONJ", "AVKO": "ADV", "AVD-KO*": "ADV", "KOA": "CCONJ",
    # --- adverbs
    "AVD": "ADV", "AVG": "ADV", "AVW": "ADV", "AVREL": "ADV", "AVNEG": "ADV",
    # --- adjectives
    "ADJA": "ADJ", "ADJD": "ADJ", "ADJN": "ADJ", "ADJS": "ADJ", "ADJV": "ADJ",
    # --- nouns
    "NA": "NOUN", "NE": "PROPN",
    # --- particles
    "PTKNEG": "PART", "PTKVZ": "PART", "PTKA": "PART", "PTKREL": "PART",
    "PTKANT": "PART", "PTKZU": "PART", "PTK": "PART",
    # --- numerals, interjections, foreign, other
    "CARDA": "NUM", "CARDN": "NUM", "CARDD": "NUM", "CARDS": "NUM", "ORDA": "ADJ",
    "OA": "ADJ", "ORD": "ADJ",
    "ITJ": "INTJ", "FM": "X", "XY": "X", "$_": "PUNCT", "$.": "PUNCT",
}

# HiTS tags whose members are function words wholesale -> take every attested
# spelling variant straight from the corpus.
CLOSED_TAGS = {
    "DDART", "DDARTA", "DIART", "DIARTA", "DDA", "DDS", "DDN", "DDD",
    "DIA", "DIS", "DIN", "DID", "DIDA", "DGA", "DGS", "DGN",
    "DPOSA", "DPOSS", "DPOSN", "DPOSD", "DRELS", "DRELA", "DPIS", "DPIA",
    "PPER", "PRF", "PI", "PW", "PG", "PWS", "PWA", "PRELS", "PRELAT",
    "PAVAP", "PAVD", "PAVG", "PAVW", "PAVREL", "PPOSS", "PPOSAT",
    "APPR", "APPRART", "APPO", "APZR",
    "KON", "KOUS", "KOKOM", "KO*", "AVKO", "AVD-KO*",
    "PTKNEG", "PTKVZ", "PTKA", "PTKREL", "PTKANT", "PTKZU",
    "VAFIN", "VAINF", "VAPP", "VAIMP",          # sin / haben / werden
    "VMFIN", "VMINF", "VMPP", "VMIMP",          # kunnen / mugen / suln / wellen / muezen
}

# Function adverbs (Table 6: degree / frequency / place / time / focusing /
# conjunctive / pronominal). AVD in ReM has 2809 types and in ReN 1029, most of
# them MANNER adverbs (mhd. -lîche(n), mnd. -liken) which must stay masked.
# These are curated from the standard grammars and then validated against the
# corpus inventory below -- anything not attested as an adverb is reported.
ADV_GMH = [
    # degree
    "vil", "harte", "gar", "sêre", "genuoc", "vaste", "wol", "baz", "meist",
    "michel", "lützel", "wênic", "mê", "mêr", "mêre", "alze", "ze", "sô", "als",
    "alsô", "alsus", "aleine", "halp", "nâch",
    # frequency / time
    "nû", "dô", "danne", "denne", "dannoch", "ie", "iemer", "niemer", "nie",
    "dicke", "ofte", "selten", "schiere", "sâ", "sân", "zehant", "iezuo", "iezunt",
    "ê", "êr", "êrst", "êrste", "sît", "sider", "hiute", "morgen", "gestern",
    "immer", "wîlen", "etewenne", "underwîlen", "alrêst", "noch", "vore", "vor",
    # place
    "hie", "dâ", "dar", "dan", "dannen", "hin", "her", "hinne", "ûz", "ûf",
    "abe", "an", "în", "obe", "under", "umbe", "wâ", "war", "wannen", "anderswâ",
    "allenthalben", "iergen", "niergen", "nindert", "ûzen", "innen",
    # focusing / conjunctive / modal particles
    "ouch", "aber", "doch", "iedoch", "wan", "niuwan", "eine", "aleine", "sunder",
    "besunder", "vürbaz", "dâvon", "dârumbe", "dârnâch", "dâbî", "dermite",
    "alsam", "sam", "reht", "rehte", "eht", "halt", "joch", "zwâre", "entriuwen",
    "vil lîhte", "lîhte", "vielleicht", "wænlîch",
    # interrogative / pronominal
    "wie", "warumbe", "wâvon", "wanne", "wenne", "wannen", "wiech",
    # negation-adjacent
    "niht", "ne", "en",
]
ADV_GML = [
    # degree
    "vele", "sere", "gantz", "gar", "genoch", "vaste", "wol", "bet", "meist",
    "lutken", "weinich", "mer", "mere", "alto", "so", "also", "alsus", "alse",
    # frequency / time
    "nu", "do", "denne", "dan", "dennoch", "je", "jummer", "nummer", "nie",
    "vaken", "dicke", "selden", "schire", "tohant", "ersten", "erst", "eer",
    "sint", "sider", "hude", "morgen", "gisteren", "noch", "vore", "vor", "na",
    "alrede", "allrede", "wanner", "wanneer",
    # place
    "hir", "dar", "daer", "dan", "dannen", "hen", "her", "ut", "up", "af",
    "an", "in", "boven", "under", "umme", "war", "wor", "wo", "anderswor",
    "allenthaluen", "nergen", "buten", "binnen", "aldar", "alhyr",
    # focusing / conjunctive / modal particles
    "ock", "ok", "auer", "aver", "doch", "jodoch", "wente", "sunder", "besunder",
    "vortmer", "darvan", "darumme", "darna", "darbi", "darmede", "sulves",
    "recht", "rechte", "jo", "twar", "villichte", "lichte", "moglik",
    # interrogative / pronominal
    "wo", "worumme", "wovan", "wanner", "wente",
    # negation-adjacent
    "nicht", "ne", "en",
]

# Light / delexical verbs (Table 6). These are tagged VVFIN etc. -- i.e. VERB --
# so they would be masked to "Oe" unless whitelisted. That is precisely the point
# of the category. Given as lemmas AND common finite stems.
LIGHT_GMH = ["tuon", "tuot", "tuo", "tet", "tete", "tâten", "getân",
             "hân", "hâst", "hât", "hæte", "haben", "habe", "hete",
             "wesen", "sîn", "bin", "bist", "ist", "sint", "was", "wâren", "wære",
             "werden", "wirde", "wirt", "wart", "wurden", "worden",
             "geben", "gap", "gâben", "gegeben", "nemen", "nam", "nâmen",
             "lâzen", "lât", "liez", "liezen", "gân", "gên", "gât", "gienc",
             "komen", "kam", "kom", "quam", "komen", "heizen", "hiez",
             "sprechen", "sprach", "sagen", "seit", "sehen", "sach"]
LIGHT_GML = ["don", "doen", "deit", "dede", "deden", "gedan",
             "hebben", "hebbe", "hest", "hefft", "hadde", "hadden", "gehat",
             "wesen", "sin", "bin", "bist", "is", "sint", "was", "weren",
             "werden", "werde", "wert", "wart", "worden",
             "geven", "gaff", "geven", "nemen", "nam", "nemen",
             "laten", "let", "leth", "gan", "geit", "ginck",
             "komen", "quam", "kumpt", "heten", "het", "hetet",
             "spreken", "sprack", "seggen", "secht", "sen", "sach"]

# --------------------------------------------------------------------------- #
NOISE = re.compile(r"""^$|[\[\]|(){}<>*]|^--$|^[\W\d_]+$|^.$|,,""", re.VERBOSE)

# Known transcription / segmentation errors observed in the ReM/ReN exports:
#   'die wîledaz' -> missing space (should be 'die wîle daz', already present)
#   'den worten'  -> dative NP fragment mis-harvested as functional
#   'mit den daz' -> mis-split variant of the valid 'mit deme daz'
_TRANSCRIPTION_ARTIFACTS = {"die wîledaz", "den worten", "mit den daz", "den daz"}
# Forms that are unambiguously Latin *in these corpora* (quotations, glosses).
# Deliberately EXCLUDES Germanic function words that are mere homographs of Latin:
#   ut  = MLG "out" (cf. Du. uit, Ger. aus, Eng. out)
#   in  = MLG/MHG "in" (Germanic cognate of Eng. in)
#   an  = MLG/MHG "on/at" (Germanic)
# These must stay whitelisted; only their spelling collides with Latin.
LATIN = {"cum", "aut", "et", "adhuc", "abintus", "acutissime", "item", "jtem",
         "ite", "vel", "sed", "quod", "qui", "ad", "non", "sic"}


def load_gmh():
    d = pickle.load(open("gmh_models_cltk-master/taggers/pos/tokens_pos.pickle", "rb"))
    inv = defaultdict(set)
    for tok, tags in d.items():
        for t in tags:
            inv[tok].add(t)
    return inv


def load_gml():
    d = pickle.load(open("gml_token_pos.pickle", "rb"))
    inv = defaultdict(set)
    for tok, tag in d.items():
        inv[tok.lower()].add(tag)
    return inv


def clean(tok):
    t = tok.strip().lower()
    if NOISE.search(t) or len(t) > 30:
        return None
    if t in _TRANSCRIPTION_ARTIFACTS:
        return None
    if not re.match(r"^[a-zäöüáàâéèêíìîóòôúùûæœëïÿšžçþðāēīōūăĕĭŏŭãõñýŵŷẅḧᵉ'’\- ]+$", t):
        return None
    return t


def build(lang, inv, adv_cur, light_cur):
    entries, report = {}, {"adv_unattested": [], "light_unattested": []}

    # (1) closed classes: every attested spelling variant, straight from the corpus
    closed = set()
    for tok, tags in inv.items():
        if tags & CLOSED_TAGS:
            c = clean(tok)
            if c and c not in LATIN:
                closed.add(c)
    entries["closed_class_corpus"] = sorted(closed)

    # (2) auxiliaries + modals broken out (they are what the whitelist must rescue,
    #     since HiTS VA*/VM* -> UD AUX, and AUX is masked to "Oe")
    for name, tagset in [("auxiliary_verbs", {"VAFIN", "VAINF", "VAPP", "VAIMP"}),
                         ("modal_verbs", {"VMFIN", "VMINF", "VMPP", "VMIMP"}),
                         ("negation", {"PTKNEG"})]:
        s = set()
        for tok, tags in inv.items():
            if tags & tagset:
                c = clean(tok)
                if c and c not in LATIN:
                    s.add(c)
        entries[name] = sorted(s)

    # (3) curated function adverbs, validated against the corpus inventory.
    #     The ReN tagger stores a single argmax tag per token, so a word can be a
    #     genuine function adverb yet be recorded as ADJA ('vele') or APPR ('up',
    #     'ut'). Requiring an adverb tag would delete those. The check that matters
    #     is the one used for the modern lists: attested at all, and not purely a
    #     noun. That still catches what it must -- e.g. modern-German intrusions
    #     like 'immer' (mhd. iemer) or 'vielleicht' (mhd. vil lihte).
    adv_ok = set()
    NOUNY = {"NA", "NE"}
    for w in adv_cur:
        c = clean(w)
        if c is None:
            continue
        if " " in c:
            adv_ok.add(c); continue
        if c not in inv:
            report["adv_unattested"].append(w); continue
        if inv[c] <= NOUNY:
            report["adv_unattested"].append(w + " [noun-only]"); continue
        adv_ok.add(c)
    entries["adverbs_function"] = sorted(adv_ok)

    # (4) curated light/delexical verbs, validated as verbs in the corpus
    VTAGS = {"VVFIN", "VVINF", "VVPP", "VVIMP", "VVPS",
             "VAFIN", "VAINF", "VAPP", "VAIMP", "VMFIN", "VMINF"}
    light_ok = set()
    for w in light_cur:
        c = clean(w)
        if c is None:
            continue
        if c in inv and (inv[c] & VTAGS):
            light_ok.add(c)
        else:
            report["light_unattested"].append(w)
    entries["light_verbs"] = sorted(light_ok)
    return entries, report


OUT = {}
for lang, loader, adv, light in [("gmh", load_gmh, ADV_GMH, LIGHT_GMH),
                                 ("gml", load_gml, ADV_GML, LIGHT_GML)]:
    inv = loader()
    e, rep = build(lang, inv, adv, light)
    flat = sorted({w for v in e.values() for w in v})
    OUT[lang] = flat
    print(f"\n===== {lang.upper()}: {len(flat)} entries (from {len(inv)} corpus token types) =====")
    for k, v in e.items():
        print(f"    {k:<26} {len(v):>5}")
    if rep["adv_unattested"]:
        print(f"    !! curated adverbs NOT attested as adverbs in the corpus "
              f"({len(rep['adv_unattested'])}) -> dropped:")
        print("       " + ", ".join(repr(w) for w in rep["adv_unattested"][:24]))
    if rep["light_unattested"]:
        print(f"    !! curated light verbs NOT attested as verbs ({len(rep['light_unattested'])}) -> dropped:")
        print("       " + ", ".join(repr(w) for w in rep["light_unattested"][:24]))

from pathlib import Path as _Path
_LISTS_DIR = _Path(__file__).resolve().parent / "posnoise_lists"
_LISTS_DIR.mkdir(exist_ok=True)
# hist_lists.json is an intermediate; hits_to_ud.json is read at runtime by
# lambdag.load_hits_to_ud(), so it must land in posnoise_lists/.
json.dump(OUT, open("hist_lists.json", "w"), ensure_ascii=False)
json.dump(HITS_TO_UD, open(_LISTS_DIR / "hits_to_ud.json", "w"), ensure_ascii=False, indent=0)
print(f"\nwrote hist_lists.json (cwd) + hits_to_ud.json ({_LISTS_DIR})")
