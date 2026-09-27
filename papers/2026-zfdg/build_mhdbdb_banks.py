# The normalised-edition corpus in the banks' own layout, one verse per line.
#
# The runners read a bank as a directory of tab-separated files, one line per
# unit, plus a manifest and the units' identifiers; the witness banks keep a
# manuscript per file. Here a file is a WORK, because that is the unit this
# corpus has, and the identifier is the edition's verse number. Everything else
# is the same, so the rhyme bank, the frequency-profile runner and the cell
# runners read it without a special case.
#
# Two streams are written, as for the witnesses: `lemma_full` is the canonical
# form of every word and `surface_full` the text's own, both in full and unmasked.
# The lemma has been resolved against the corpus's lexicon by the extractor.
#
# THESE BANKS MAY NOT BE MIXED WITH THE WITNESS BANKS in one comparison: these
# are editors' normalised texts and those are diplomatic transcriptions, and the
# difference between the two is a difference of hand, not of author. They answer
# the author question on their own.
#
#   python medieval/extract_mhdbdb.py && python medieval/build_mhdbdb_banks.py
#
# Output: masked/mhdbdb_{lemma,surface}_full/{bank,verse_ids,MANIFEST.tsv}

import csv
import json
import sys
import unicodedata
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SRC = HERE / "mhdbdb_verses.jsonl"
COLS = ["file", "source", "author", "work", "siglum", "layer", "date", "units",
        "tokens", "masked_tokens", "annotation", "genre"]

# The source dates the EDITION, not the poem, so the poem's date is supplied
# here: one approximate year in the middle of each author's working life, as the
# handbooks give it. It is in the bank because it belongs to the corpus rather
# than to any one experiment, and because the language stage moves over these
# three centuries -- an experiment that separates a fourteenth-century author
# from a twelfth-century one may be separating their German and not their hands.
AUTHOR_DATE = {
    "Lamprecht der Pfaffe": 1150, "Der Wilde Mann": 1170,
    "Heinrich von Veldeke": 1180, "Hartmann von Aue": 1195,
    "Walther von der Vogelweide": 1200, "Wolfram von Eschenbach": 1210,
    "Gottfried von Straßburg": 1210, "Konrad von Heimesfurt": 1225,
    "Heinrich von dem Türlin": 1230, "Der Stricker": 1235,
    "Rudolf von Ems": 1235, "Lamprecht von Regensburg": 1240,
    "Ulrich von Türheim": 1240, "Ulrich von Liechtenstein": 1255,
    "Herrand von Wildonie": 1260, "Ulrich von Winterstetten": 1260,
    "Der Pleier": 1270, "Konrad von Würzburg": 1270, "Anonym": 1290,
    "Ulrich von Etzenbach (Eschenbach)": 1290,
    "Mönch von Heilsbronn": 1345, "Der Mönch von Salzburg": 1390,
    "Heinrich Kaufringer": 1400, "Hans Rosenplüt": 1450,
    "Sebastian Brant": 1494,
}


def slug(s):
    s = unicodedata.normalize("NFKD", s.lower())
    s = "".join(c for c in s if not unicodedata.combining(c))
    return "".join(c if c.isalnum() else "_" for c in s).strip("_")


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    outs = {k: ROOT / "masked" / f"mhdbdb_{k}_full" for k in
            ("lemma", "surface")}
    for o in outs.values():
        (o / "bank").mkdir(parents=True, exist_ok=True)
        (o / "verse_ids").mkdir(parents=True, exist_ok=True)
    rows, per_author = [], Counter()
    with SRC.open(encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            name = f"mhdbdb__{slug(r['author'])}__{r['sigle']}"
            vs = r["verses"]
            for k, i in (("surface", 0), ("lemma", 1)):
                (outs[k] / "bank" / f"{name}.tsv").write_text(
                    "\n".join("\t".join(w[i] for w in v["w"]) for v in vs)
                    + "\n", encoding="utf-8")
            (outs["lemma"] / "verse_ids" / f"{name}.txt").write_text(
                "\n".join(str(v["n"]) for v in vs) + "\n", encoding="utf-8")
            (outs["surface"] / "verse_ids" / f"{name}.txt").write_text(
                "\n".join(str(v["n"]) for v in vs) + "\n", encoding="utf-8")
            ntok = sum(len(v["w"]) for v in vs)
            rows.append(dict(
                file=name, source="mhdbdb", author=r["author"],
                # the sigle is carried in the work's name because two works may
                # share a title and the runners tell works apart by this field
                work=f"{r['title']} ({r['sigle']})", siglum=r["sigle"],
                layer="normalised edition",
                date=str(AUTHOR_DATE.get(r["author"], "-")), units=len(vs),
                tokens=ntok, masked_tokens=ntok, annotation="mhdbdb",
                genre="-"))
            per_author[r["author"]] += 1
    for o in outs.values():
        with (o / "MANIFEST.tsv").open("w", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=COLS, delimiter="\t",
                               lineterminator="\n")
            w.writeheader()
            for r in rows:
                w.writerow(r)
    print(f"{len(rows)} works, {sum(r['tokens'] for r in rows):,} tokens, "
          f"{sum(r['units'] for r in rows):,} verses, "
          f"{len(per_author)} named authors")
    for a, n in per_author.most_common(12):
        print(f"  {a:34} {n:3} works")
    for k, o in outs.items():
        print(f"  wrote {o.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
