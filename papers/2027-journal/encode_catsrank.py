# Pre-encode masked banks to the universal class-conditioned rank alphabet.
# Emits masked_catsrank/{ds}/bank/*.tsv (one sentence per line, tab-separated
# symbols), so downstream scorers -- Python or Julia, local or remote -- read
# plain symbol streams with no encoding dependencies.
#
#   python experiments/encode_catsrank.py [--datasets ds1,ds2,...]
#   python experiments/encode_catsrank.py --shared-group ds1,ds2,...
#
# --shared-group pools the listed same-language datasets into ONE within-class
# rank map and writes each as {ds}_shared -- required when streams from the
# listed banks must be directly comparable (e.g. cross-genre within one
# language); per-dataset maps would let the same symbol denote different words
# in different genres.

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from aligned_utils import LANG_CODE, load_aligned  # noqa: E402
from xling_pilot import class_rank_map, encode  # noqa: E402

MASKED = HERE.parent / "masked"
OUT = HERE.parent / "masked_catsrank"

DATASETS = ["german_novels", "english_novels", "french_novels",
            "polish_novels", "czech_novels", "hungarian_novels"]


def read_bank(ds):
    bank = {}
    for f in sorted((MASKED / ds / "bank").glob("*.tsv")):
        bank[f.stem] = [line.split("\t") for line in
                        f.read_text(encoding="utf-8").splitlines() if line]
    return bank


def write_bank(ds, bank, table, crmap, suffix=""):
    od = OUT / (ds + suffix) / "bank"
    od.mkdir(parents=True, exist_ok=True)
    for a, sents in bank.items():
        with open(od / f"{a}.tsv", "w", encoding="utf-8", newline="\n") as fh:
            for s in sents:
                fh.write("\t".join(encode(s, "catsrank", {}, table, crmap))
                         + "\n")
    print(f"{ds}{suffix}: {len(bank)} authors, {len(set(crmap.values()))} "
          f"class-rank symbols", flush=True)


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if "--shared-group" in sys.argv:
        dss = sys.argv[sys.argv.index("--shared-group") + 1].split(",")
        langs = {ds.split("_")[0] for ds in dss}
        if len(langs) != 1:
            raise SystemExit(f"--shared-group spans languages: {langs}")
        table = load_aligned(LANG_CODE[langs.pop()])
        banks = {ds: read_bank(ds) for ds in dss}
        flat = {f"{ds}|{a}": [t for s in sents for t in s]
                for ds, bank in banks.items() for a, sents in bank.items()}
        crmap = class_rank_map(flat, table)
        for ds, bank in banks.items():
            write_bank(ds, bank, table, crmap, suffix="_shared")
        return
    dss = (sys.argv[sys.argv.index("--datasets") + 1].split(",")
           if "--datasets" in sys.argv else DATASETS)
    for ds in dss:
        lang = ds.split("_")[0]
        table = load_aligned(LANG_CODE[lang])
        if not table:
            raise SystemExit(f"no aligned companion file for {lang}")
        bank = read_bank(ds)
        flat = {a: [t for s in sents for t in s] for a, sents in bank.items()}
        crmap = class_rank_map(flat, table)
        write_bank(ds, bank, table, crmap)


if __name__ == "__main__":
    main()
