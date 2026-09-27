# The rhyme-word stream: one token per verse, the word that stands at the rhyme.
#
# WHY THIS POSITION. A scribe copying Middle High German verse rewrites freely
# inside the line -- his own spelling, his own dialect forms, sometimes his own
# words -- but he cannot touch the word at the end of the verse without breaking
# the rhyme. The rhyme position is therefore the one place in the line where the
# poet's choice survives the copy, which is why the philological tradition reads
# rhyme for authorship and why Kestemont's authorship work takes the rhyme words
# as its features.
#
# WHAT THE STREAM IS. The last token of every verse, emitted as its canonical
# lemma, in the order the verses stand; no couplets are formed and no rhyme is
# extracted. The units are the verses, as in every other bank, so the runners
# re-chunk it into units of a hundred exactly as they do the masked stream, and
# a case drawn under one stream is the same case under the other.
#
# WHAT IT IS NOT. Lemmatising the rhyme word removes the rhyme itself: the
# forms that rhyme are the inflected ones, and their dictionary forms need share
# no ending at all. What is measured here is therefore THE VOCABULARY THAT
# STANDS AT THE RHYME POSITION, which is the part the scribe cannot move, and
# the paper should call it that.
#
# ONLY THE EDITION-BASED WITNESSES. The reference corpus's texts are cut into
# twenty-five-token units and carry no verse boundaries -- its XML marks the
# lines of the manuscript, which do not coincide with the metrical verses -- so
# their "last token of a unit" would be an arbitrary word. They are left out,
# and the number left out is reported rather than passed over.
#
#   python medieval/build_rhyme_bank.py [--stream lemma_full] [--only SUBSTR]
#
# Output: masked/mhg_witnesses_rhyme/ (bank, verse_ids, MANIFEST.tsv)

import argparse
import csv
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from rime import rime as rime_of  # noqa: E402

VERSE_SOURCES = ("kcd", "hvad", "parzival", "ambraser")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stream", default="lemma_full",
                    help="the bank the rhyme words are taken from; "
                         "lemma_full for the canonical form, surface_full for "
                         "the scribe's")
    ap.add_argument("--emit", choices=["lemma", "rime"], default="lemma",
                    help="the rhyme word's canonical lemma, or the rime of the "
                         "scribe's own form. The measured difference between "
                         "them is that a copyist keeps the sound and not the "
                         "word: at the rhyme the rime survives a copy far "
                         "better than the form does, while the lemma is no "
                         "more stable there than anywhere else in the verse.")
    ap.add_argument("--bank", default="mhg_witnesses",
                    help="mhg_witnesses for the manuscripts, mhdbdb for the "
                         "normalised editions. The editions carry no manuscript "
                         "line at all: their verse is the printed one, so the "
                         "restriction to texts with verse boundaries does not "
                         "apply to them and every work is kept.")
    ap.add_argument("--out", default="")
    ap.add_argument("--only", default="")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    stream = "surface_full" if args.emit == "rime" else args.stream
    src = ROOT / "masked" / f"{args.bank}_{stream}"
    out = ROOT / "masked" / (args.out or
                             (f"{args.bank}_rime" if args.emit == "rime"
                              else f"{args.bank}_rhyme"
                              if stream == "lemma_full"
                              else f"{args.bank}_rhyme_{stream}"))
    (out / "bank").mkdir(parents=True, exist_ok=True)
    (out / "verse_ids").mkdir(parents=True, exist_ok=True)
    with (src / "MANIFEST.tsv").open(encoding="utf-8") as fh:
        man = list(csv.DictReader(fh, delimiter="\t"))

    rows, skipped = [], 0
    for r in man:
        name = r["file"]
        if args.only and args.only not in name:
            continue
        if args.bank == "mhg_witnesses" and not name.startswith(VERSE_SOURCES):
            skipped += 1
            continue
        lines = [l.split("\t") for l in
                 (src / "bank" / f"{name}.tsv").read_text(
                     encoding="utf-8").splitlines() if l]
        vids = [v for v in (src / "verse_ids" / f"{name}.txt").read_text(
            encoding="utf-8").split("\n") if v]
        if len(vids) != len(lines):
            vids = [str(i + 1) for i in range(len(lines))]
        rhymes = [u[-1] for u in lines if u and u[-1]]
        if args.emit == "rime":
            rhymes = [rime_of(w) or w for w in rhymes]
        keep = [v for u, v in zip(lines, vids) if u and u[-1]]
        (out / "bank" / f"{name}.tsv").write_text(
            "\n".join(rhymes) + "\n", encoding="utf-8")
        (out / "verse_ids" / f"{name}.txt").write_text(
            "\n".join(keep) + "\n", encoding="utf-8")
        row = dict(r)
        row["units"] = len(rhymes)
        row["masked_tokens"] = len(rhymes)
        rows.append(row)
        print(f"{name:56} {len(lines):7,} verses -> {len(rhymes):7,} "
              f"rhyme words")

    with (out / "MANIFEST.tsv").open("w", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=man[0].keys(), delimiter="\t",
                           lineterminator="\n")
        w.writeheader()
        for row in rows:
            w.writerow(row)
    print(f"\n{len(rows)} witnesses -> {out.relative_to(ROOT)}; "
          f"{skipped} left out for having no verse boundaries")
    print(f"total rhyme words: {sum(r['masked_tokens'] for r in rows):,}")


if __name__ == "__main__":
    main()
