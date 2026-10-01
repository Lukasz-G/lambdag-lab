# The ten copies of the Nibelungenlied in the form the tagger takes.
#
# These are diplomatic transcriptions, so their spelling is each scribe's own and
# differs between copies far more than the wording does -- a count of differing
# WORDS across the ten puts twenty-five to eighty thousand between every pair,
# and most of that is `Seyfrid` against `Sivrit`. A comparison made on those
# forms would measure orthography and answer the wrong question, which is why
# these texts are tagged: the lemma is what carries the reworking, and the
# scribal toll on this corpus is 0.055 on the lemma stream against 0.151 on the
# scribe's spelling.
#
# The model is the one every other text in this corpus was annotated with, for
# the same reason as everywhere else: a class annotated by a different model is
# on a different basis from the case it is compared with.
#
#   python medieval/make_hydra_inputs_nibelungen.py
#   python medieval/tag_nibelungen.py [--batch 1]
#
# Output: medieval/hydra_in_nibelungen/NL<siglum>.txt + MANIFEST.tsv

import csv
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SRC = HERE / "nibelungen_verses.jsonl"
OUT = HERE / "hydra_in_nibelungen"
COLS = ["file", "source", "author", "work", "siglum", "manuscript",
        "redaction", "shelfmark", "strophes", "verses", "tokens"]


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    OUT.mkdir(parents=True, exist_ok=True)
    rows, odd = [], 0
    with SRC.open(encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            name = r["sigle"]
            verses = []
            for v in r["verses"]:
                keep = [w for w in v["w"] if any(c.isalpha() for c in w)]
                odd += len(v["w"]) - len(keep)
                if keep:
                    verses.append(" ".join(keep))
            n = sum(len(v.split()) for v in verses)
            (OUT / f"{name}.txt").write_text(
                f"@ {name} ({len(verses)} verses)\n" + "\n".join(verses) + "\n",
                encoding="utf-8")
            rows.append(dict(file=name, source="nibelungen", author=r["author"],
                             work=r["title"], siglum=name,
                             manuscript=r["manuscript"],
                             redaction=r["redaction"], shelfmark=r["shelfmark"],
                             strophes=r["strophes"], verses=len(verses),
                             tokens=n))
    with (OUT / "MANIFEST.tsv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLS, delimiter="\t",
                           lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} manuscripts, {sum(r['verses'] for r in rows):,} lines, "
          f"{sum(r['tokens'] for r in rows):,} words -> "
          f"{OUT.relative_to(ROOT)}")
    print(f"{odd} tokens without a letter dropped")
    for r in rows:
        print(f"  {r['file']:5} {r['redaction']:4} {r['tokens']:7,}")


if __name__ == "__main__":
    main()
