# Every shipped pattern list must be discoverable and parse; gmh/gml also need
# the HiTS->UD tag map. Guards against a broken lists directory in the package.
import pytest

from lambdag import SUPPORTED_LANGUAGES, POSNoiseMasker


@pytest.mark.parametrize("lang", SUPPORTED_LANGUAGES)
def test_list_loads(lang):
    m = POSNoiseMasker.pretagged(lang)
    assert m.pattern_list_path is not None and m.pattern_list_path.exists()


def test_version_glob_takes_highest():
    m = POSNoiseMasker.pretagged("gmh")
    assert "v0.2" in m.pattern_list_path.name
