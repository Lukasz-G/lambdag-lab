# Loader and helpers for the category-annotated POSNoise companion files
# (posnoise_lists/aligned/, built by data_prep/build_aligned_lists.py).
#
# The aligned files give every pattern-list entry a dominant UD POS and a
# functional class from a small alphabet shared across languages. Four places
# in the pipeline consume this layer:
#
#   1. Cross-lingual encoding ("cats" mode in xling_pilot.py): a kept function
#      word is collapsed to its functional class instead of a bare W or a
#      frequency-rank bucket, giving a finer language-independent alphabet.
#   2. Shared symbol atoms in the hyperdimensional encoder: tokens of the same
#      class across languages can share (or correlate) their atoms, so profiles
#      from different languages inhabit a common region of the code space.
#   3. Readable literals in per-author profile classifiers: a clause literal
#      over "AUX_ADV_PRON" reads as a schematic construction; the class map
#      supplies the naming.
#   4. Category back-off for out-of-vocabulary function words in monolingual
#      profiles: an unseen function word backs off to its class symbol instead
#      of a generic unknown.
#
# Classes: AUX LVERB ADV DET ADP PRON CCONJ SCONJ PART INTJ NUM MWE OTH UNK.
# UNK (no treebank evidence) and OTH (content-dominant stray) are treated as
# "no class" by the helpers: callers get their fallback symbol instead.

from pathlib import Path

HERE = Path(__file__).resolve().parent
ALIGNED = HERE.parent / "posnoise_lists" / "aligned"

CLASSES = ["AUX", "LVERB", "ADV", "DET", "ADP", "PRON", "CCONJ", "SCONJ",
           "PART", "INTJ", "NUM", "MWE"]
_NOCLASS = {"UNK", "OTH"}

# corpus-directory language names -> ISO codes used in the aligned filenames
LANG_CODE = {"german": "de", "english": "en", "french": "fr", "polish": "pl",
             "czech": "cs", "hungarian": "hu", "spanish": "es", "italian": "it",
             "russian": "ru", "croatian": "hr", "dutch": "nl", "greek": "el",
             "latvian": "lv", "lithuanian": "lt", "norwegian": "no",
             "portuguese": "pt", "romanian": "ro", "serbian": "sr",
             "slovenian": "sl", "swedish": "sv", "ukrainian": "uk"}


def load_aligned(code):
    """entry -> (upos, class) for one language; {} if the file is missing."""
    cands = sorted(ALIGNED.glob(f"POSNoise_Aligned_{code.title()}_v*.tsv"))
    if not cands:
        return {}
    table = {}
    for line in cands[-1].read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        f = line.split("\t")
        if len(f) >= 3:
            table[f[0]] = (f[1], f[2])
    return table


def class_of(token, table, fallback="W"):
    """Functional class of a kept token, or `fallback` when the aligned file
    has no usable class for it (unlisted, UNK, or OTH)."""
    got = table.get(token.lower())
    if got is None or got[1] in _NOCLASS:
        return fallback
    return got[1]


def backoff_symbol(token, table, vocab):
    """Monolingual category back-off: the token itself while in-vocabulary,
    its functional class when out-of-vocabulary but listed, else `fallback`
    semantics via class_of."""
    t = token.lower()
    if t in vocab:
        return t
    return class_of(t, table)
