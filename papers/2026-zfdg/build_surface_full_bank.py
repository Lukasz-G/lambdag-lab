# The witnesses with no masking, as the scribes wrote them.
#
# The lemma banks are normalised to the reference corpus's convention, which is
# what every authorship experiment wants and exactly what a rhyme test cannot
# use: only surfaces rhyme. The canonical lemmas of a rhyming pair need not
# share an ending at all, since the rhyme is carried by the inflected forms.
# This bank therefore emits each kept token as the scribe's own form, so that
# the verse-final words can be asked whether they rhyme.
#
# It is built by the same routine as the masked banks with the masker replaced,
# so units, dropped tokens and verse numbering are identical to every other
# bank and nothing has to be mapped between them.
#
#   python medieval/build_surface_full_bank.py [--only NAME_SUBSTRING]
#
# Output: masked/mhg_witnesses_surface_full/

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import build_masked_banks as B  # noqa: E402

OUT = ROOT / "masked" / "mhg_witnesses_surface_full"


class PlainSurface:
    """the masker's interface, emitting the scribe's own form"""

    pattern_list_path = Path("none (surfaces in full)")

    def mask_tagged(self, units):
        return [[t[0].lower() for t in u] for u in units]


class NoMasker:
    @staticmethod
    def pretagged(*args, **kwargs):
        return PlainSurface()


def main():
    B.OUT = {"lemma": OUT}
    B.POSNoiseMasker = NoMasker
    B.main()


if __name__ == "__main__":
    main()
