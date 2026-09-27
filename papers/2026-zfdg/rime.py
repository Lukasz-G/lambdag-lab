# The rime of a Middle High German word: what has to survive for a rhyme to.
#
# A rhyme binds the sound from the last STRESSED vowel to the end of the word,
# which for a masculine ending is the final syllable (`sprach : gesach` -> ach)
# and for a feminine ending is the last two (`minne : sinne` -> inne). Taking
# the final vowel alone, as a first version did, makes every word ending in -e
# rhyme with every other, which is useless as an alphabet even though it is
# harmless as a diagnostic.
#
# The spelling has to go first. A scribe writes the same sound a dozen ways, so
# the forms are folded before the rime is cut: the letters are lowered, the
# length and quality marks dropped, i/y merged, w written as v, doubled letters
# collapsed and everything that is not a letter removed. w is NOT merged into u,
# although the two spell one sound: it would turn `gewan` into `geuan`, whose
# last vowel group is then `eua`, and the rime would reach back over a
# consonant that is there. What is left is not a
# phonological transcription and is not meant to be; it is the part of the
# written form that survives copying, and how much of it survives is measured
# rather than assumed (about three quarters of rhyme positions across three
# traditions).
#
#   from rime import fold, rime
#   rime("Minne") -> "inne";  rime("sprach") -> "ach";  rime("gewan") -> "an"

import re

FOLD = str.maketrans({"y": "i", "w": "v", "j": "i", "ſ": "s",
                      "ë": "e", "ê": "e", "é": "e", "è": "e", "â": "a",
                      "á": "a", "à": "a", "î": "i", "í": "i", "ì": "i",
                      "ô": "o", "ó": "o", "ò": "o", "û": "u", "ú": "u",
                      "ù": "u", "æ": "e", "œ": "o", "ā": "a", "ē": "e",
                      "ī": "i", "ō": "o", "ū": "u", "ÿ": "i", "ä": "e",
                      "ö": "o", "ü": "u", "ç": "c", "ß": "s"})

VOWELS = set("aeiou")
# the codas that an unstressed final -e may carry; with any of them the rhyme
# reaches back to the syllable before
WEAK_CODA = re.compile(r"^(n|r|l|m|s|t|nt|st|rn|ns|ts|nts|rt)?$")


def fold(w):
    """the written form with the spelling taken out"""
    out, prev = [], ""
    for c in w.lower().translate(FOLD):
        if not c.isalpha():
            continue
        if c == prev:                 # doubling is spelling, not sound
            continue
        out.append(c)
        prev = c
    return "".join(out)


def groups(f):
    """the start index of every vowel group in a folded form"""
    out, inside = [], False
    for i, c in enumerate(f):
        if c in VOWELS:
            if not inside:
                out.append(i)
            inside = True
        else:
            inside = False
    return out


def rime(w, folded=False):
    """from the last stressed vowel to the end"""
    f = w if folded else fold(w)
    g = groups(f)
    if not g:
        return f
    last = g[-1]
    # a final -e with no coda or a weak one is unstressed, so the rhyme carries
    # back to the syllable before it
    if len(g) > 1 and f[last] == "e" and WEAK_CODA.match(f[last + 1:]):
        return f[g[-2]:]
    return f[last:]
