# Masking semantics on the spaCy-free path (mask_tagged): placeholder map,
# kept function classes, punctuation-verbatim, window tiling.
from lambdag import DEFAULT_ABBREV_POS_TAGS, POSNoiseMasker

SENT = [("The", "DET", "the"), ("cat", "NOUN", "cat"), ("sat", "VERB", "sit"),
        ("on", "ADP", "on"), ("a", "DET", "a"), ("mat", "NOUN", "mat"),
        (",", "PUNCT", ","), ("slowly", "ADV", "slowly"), (".", "PUNCT", ".")]


def test_placeholders_and_kept_classes():
    m = POSNoiseMasker.pretagged("en")
    [out] = m.mask_tagged([SENT])
    noun, verb = DEFAULT_ABBREV_POS_TAGS["NOUN"], DEFAULT_ABBREV_POS_TAGS["VERB"]
    adv = DEFAULT_ABBREV_POS_TAGS["ADV"]
    assert out[1] == noun and out[5] == noun     # content nouns masked
    assert out[2] in (verb, "sat")               # masked unless whitelisted
    assert out[3].lower() == "on" and out[4].lower() == "a"   # ADP/DET kept
    assert out[6] == "," and out[8] == "."       # punctuation verbatim
    assert out[7] in (adv, "slowly")             # ADV masked unless whitelisted


def test_window_tiling_ignores_sentence_boundaries():
    m = POSNoiseMasker.pretagged("en", segment="window", window=4)
    out = m.mask_tagged([SENT, SENT])
    flat = [t for w in out for t in w]
    assert len(flat) == 2 * len(SENT)
    assert all(len(w) == 4 for w in out[:-1])
    assert len(out[-1]) in range(1, 5)


def test_pattern_list_discovered():
    m = POSNoiseMasker.pretagged("en")
    assert m.pattern_list_path is not None and m.pattern_list_path.exists()
