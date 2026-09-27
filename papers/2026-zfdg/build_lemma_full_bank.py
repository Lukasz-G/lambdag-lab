# The same witnesses with NO masking at all: every token as its canonical lemma.
#
# WHY. POSNoise keeps the function words and replaces every content word with a
# placeholder, which is what authorship verification wants and the opposite of
# what textual reuse wants: a borrowed passage is recognised by its NOUNS and
# VERBS, and the masked stream throws exactly those away. The reuse detector has
# been running on the masked stream and finding the documented partners in spite
# of the handicap; this bank lets it run on the lemmata themselves, which is the
# usual basis for reuse work on medieval texts, where orthography varies from
# scribe to scribe and the lemma is the only stable unit.
#
# HOW. Nothing is re-derived. The bank is built by the very same routine that
# builds the masked banks (medieval/build_masked_banks.py) -- the same tagged
# files, the same verse boundaries, the same dropped tokens, the same lemma
# canonicalisation to the reference corpus's convention -- with the masker
# replaced by one that emits each kept token's lemma. Because POSNoise replaces
# rather than deletes, the masked banks and this one are token for token and
# unit for unit the same text, so a span found here names the same lines of the
# masked bank and no coordinate has to be mapped.
#
#   python medieval/build_lemma_full_bank.py [--only NAME_SUBSTRING]
#
# Output: masked/mhg_witnesses_lemma_full/ (bank, verse_ids, MANIFEST.tsv);
# the manifest's masked_tokens column holds the unmasked token count.

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import build_masked_banks as B  # noqa: E402

OUT = ROOT / "masked" / "mhg_witnesses_lemma_full"


class PlainLemma:
    """the masker's interface, with no masking behind it"""

    pattern_list_path = Path("none (lemmata in full)")

    def mask_tagged(self, units):
        return [[t[2].lower() for t in u] for u in units]


class NoMasker:
    @staticmethod
    def pretagged(*args, **kwargs):
        return PlainLemma()


def main():
    # the builder writes one directory per entry of its OUT table and names the
    # token-count column after the key "lemma", so the single dataset is
    # registered under that key and pointed at the new directory
    B.OUT = {"lemma": OUT}
    B.POSNoiseMasker = NoMasker
    B.main()


if __name__ == "__main__":
    main()
