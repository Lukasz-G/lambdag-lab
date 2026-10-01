# The normalised editions in the form the tagger takes.
#
# The witness banks carry the tagger's own annotation; the editions carry the
# source database's, on a different tagset and with a fifth of the words untagged.
# A calibration class drawn from the editions is therefore on a different
# annotation basis from the case it calibrates, which is the one thing a
# calibration class must not be. Tagging the editions with the SAME model that
# annotated the witnesses removes that difference at its root -- and it is the
# same model, not merely the same tagger: `medieval/build_masked_banks.py` reads
# the vocabulary of `runs/s_crf`, so that is the checkpoint to use here.
#
# The form is the one make_hydra_inputs.py writes for the witnesses: a header
# line naming the text and its size, then one verse per line, words separated by
# single spaces. The editions need no punctuation stripping -- the extractor
# keeps only the source's word elements, and its punctuation elements were never
# collected -- but the assumption is checked rather than trusted.
#
#   python medieval/make_hydra_inputs_mhdbdb.py
#   # then, in the tagger's directory:
#   python -m hydra.cli tag --model runs/s_crf/model_only.pt \
#       --input <medieval/hydra_in_mhdbdb> --output <medieval/hydra_out_mhdbdb>
#
# Output: medieval/hydra_in_mhdbdb/<author>__<sigle>.txt + MANIFEST.tsv

import csv
import json
import sys
import unicodedata
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE / "mhdbdb_verses.jsonl"
OUT = HERE / "hydra_in_mhdbdb"
COLS = ["file", "source", "author", "work", "siglum", "verses", "tokens"]


def slug(s):
    s = unicodedata.normalize("NFKD", s.lower())
    s = "".join(c for c in s if not unicodedata.combining(c))
    return "".join(c if c.isalnum() else "_" for c in s).strip("_")


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    OUT.mkdir(parents=True, exist_ok=True)
    rows, odd = [], 0
    with SRC.open(encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            name = f"{slug(r['author'])}__{r['sigle']}"
            verses = []
            for v in r["verses"]:
                w = [x[0].strip() for x in v["w"] if x[0].strip()]
                # a word that is not a word would reach the tagger as one, so
                # anything without a letter is dropped and the count reported
                keep = [x for x in w if any(c.isalpha() for c in x)]
                odd += len(w) - len(keep)
                if keep:
                    verses.append(" ".join(keep))
            if not verses:
                continue
            n = sum(len(v.split()) for v in verses)
            (OUT / f"{name}.txt").write_text(
                f"@ {name} ({len(verses)} verses)\n" + "\n".join(verses)
                + "\n", encoding="utf-8")
            rows.append(dict(file=name, source="mhdbdb", author=r["author"],
                             work=r["title"], siglum=r["sigle"],
                             verses=len(verses), tokens=n))
    with (OUT / "MANIFEST.tsv").open("w", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=COLS, delimiter="\t",
                           lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"{len(rows)} texts, {sum(r['verses'] for r in rows):,} verses, "
          f"{sum(r['tokens'] for r in rows):,} tokens -> "
          f"{OUT.relative_to(HERE.parent)}")
    print(f"{odd} tokens without a letter dropped")
    print(f"{len({r['author'] for r in rows})} named authors")


if __name__ == "__main__":
    main()
