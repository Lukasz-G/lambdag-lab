# The normalised editions on the witnesses' own annotation basis.
#
# The bank builder reads a directory of tagged texts, their one-verse-per-line
# originals and a manifest, and writes the POSNoise-masked lemma and surface
# streams. Everything it does applies here unchanged -- the mapping of the model's
# lemmas to the reference convention, the dropping of foreign material and of
# tokens carrying no analysis, the masker itself -- so the builder is reused and
# only the four paths are redirected. Nothing about the masking differs between
# the witnesses and the editions, which is the whole point: a calibration class
# drawn from these texts is then on the same annotation basis as the case.
#
# The date the manifest carries is the poem's, from build_mhdbdb_banks, because
# the experiments select their reference population and their cohort by period and
# an edition carries no date of its own.
#
#   python medieval/tag_mhdbdb.py && python medieval/build_mhdbdb_masked_banks.py
#
# Output: masked/mhdbdb_{lemma,surface}/

import csv
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import build_masked_banks as B  # noqa: E402
from build_mhdbdb_banks import AUTHOR_DATE, WORK_DATE, slug  # noqa: E402

IN = HERE / "hydra_in_mhdbdb"
MAN = IN / "MANIFEST.tsv"
OUT = {"lemma": ROOT / "masked" / "mhdbdb_lemma",
       "surface": ROOT / "masked" / "mhdbdb_surface"}


def manifest_with_dates():
    """the tagger's manifest, with the poem's date and the layer added"""
    rows = list(csv.DictReader(MAN.open(encoding="utf-8"), delimiter="\t"))
    out = IN / "MANIFEST_dated.tsv"
    cols = ["file", "source", "author", "work", "siglum", "layer", "date"]
    with out.open("w", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, delimiter="\t",
                           lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow(dict(file=r["file"], source="mhdbdb", author=r["author"],
                            work=f"{r['work']} ({r['siglum']})",
                            siglum=r["siglum"], layer="normalised edition",
                            date=str(WORK_DATE.get(r["siglum"],
                                             AUTHOR_DATE.get(r["author"], "-")))))
    return out


def verse_ids():
    """the edition's own verse numbers, keyed by the bank's file name

    The builder reads a witness's verse numbers out of the extraction records and
    expects a witness's three-part name; an edition has no siglum and no
    manuscript, so its numbers come from the extractor's records instead.
    """
    import json
    out = {}
    for line in (HERE / "mhdbdb_verses.jsonl").open(encoding="utf-8"):
        r = json.loads(line)
        out[f"{slug(r['author'])}__{r['sigle']}"] = [str(v["n"])
                                                    for v in r["verses"]]
    return out


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ids = verse_ids()
    B.verse_ids_of = lambda name: ids.get(name)
    B.TAGGED = HERE / "hydra_out_mhdbdb"
    B.ORIG = IN
    B.MAN = manifest_with_dates()
    B.REM_GOLD = Path("none (the reference corpus belongs to the witness bank)")
    B.OUT = OUT
    B.main()


if __name__ == "__main__":
    main()
