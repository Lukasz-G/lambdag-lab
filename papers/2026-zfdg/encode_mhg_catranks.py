# Re-encode the Middle High German witness banks to the shared class-conditioned
# rank alphabet, keeping everything else about the bank intact.
#
# The witness scorer reads masked/mhg_witnesses_{stream}/ and needs the verse
# numbering to cut disjoint known and questioned passages, so the encoding is
# token for token and line for line: MANIFEST.tsv and verse_ids/ are copied
# unchanged, and only the symbols in bank/*.tsv are replaced. An encoded bank is
# therefore a drop-in for the scorer, and a case drawn under one encoding is the
# same case under the other.
#
# WHERE THE RANK MAP IS FITTED, which decides what the result may be used for:
#
#   --fit-on bank   (default) the map is estimated from the very bank it encodes.
#                   The cases and the donors are inside it, so the alphabet has
#                   seen the test material. This is a LEAKY setting and gives the
#                   alphabet its best case; it is what a feasibility gate wants,
#                   because an alphabet that loses discrimination even under
#                   favourable fitting will not gain it under honest fitting.
#   --fit-on FILE   a text file of bank names, one per line, disjoint from the
#                   cases and the donors. This is the held-out protocol, and it is
#                   what any REPORTED number must use.
#
# The suffix records which was done, so a leaky bank cannot be mistaken for a
# clean one downstream.
#
#   python experiments/encode_mhg_catranks.py --stream lemma
#   python experiments/encode_mhg_catranks.py --stream lemma --fit-on heldout.txt
#
# Output: masked/mhg_witnesses_{stream}_{suffix}/

import argparse
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from aligned_utils import LANG_CODE, class_of, load_aligned  # noqa: E402
from xling_pilot import CR_EDGES, PLACEHOLDERS  # noqa: E402

MASKED = ROOT / "masked"


def read_lines(path):
    return [line.split("\t") for line in
            path.read_text(encoding="utf-8").splitlines() if line]


def rank_map(files, table):
    """token -> class-conditioned rank-bucket symbol, e.g. DET1, ADV3, W10.

    Identical in construction to xling_pilot.class_rank_map, over files rather
    than an in-memory bank so that the fitting set can be named explicitly.
    """
    by_cls = defaultdict(Counter)
    for f in files:
        for sent in read_lines(f):
            for t in sent:
                if t not in PLACEHOLDERS and any(ch.isalpha() for ch in t):
                    by_cls[class_of(t, table)][t] += 1
    out = {}
    for cls, c in by_cls.items():
        for i, (tok, _n) in enumerate(c.most_common(), start=1):
            bucket = next((str(e) for e in CR_EDGES if i <= e), "X")
            out[tok] = f"{cls}{bucket}"
    return out


def encode_token(t, crmap):
    if t in PLACEHOLDERS:
        return t
    if any(ch.isalpha() for ch in t):
        return crmap.get(t, "WX")
    return t                      # punctuation is absent from these banks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stream", choices=["lemma", "surface"], default="lemma")
    ap.add_argument("--fit-on", default="bank",
                    help="'bank' (leaky, for a gate), a directory of masked "
                         "held-out texts, or a file of bank names")
    ap.add_argument("--suffix", default=None)
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    src = MASKED / f"mhg_witnesses_{args.stream}"
    if not (src / "bank").exists():
        sys.exit(f"no bank at {src}")

    table = load_aligned(LANG_CODE["mhg"])
    if not table:
        sys.exit("no aligned companion file for gmh; run "
                 "data_prep/build_aligned_gmh.py first")

    files = sorted((src / "bank").glob("*.tsv"))
    if args.fit_on == "bank":
        fit_files = files
        suffix = args.suffix or "catranks_leaky"
        note = ("FITTED ON THE BANK IT ENCODES -- the cases and the donors are "
                "inside the fitting set.\n  Valid as a feasibility gate, which "
                "this favours; not valid for a reported number.")
    elif Path(args.fit_on).is_dir():
        # a corpus of masked texts standing OUTSIDE this bank, which is what the
        # held-out protocol wants: the map never sees a case or a donor
        root = Path(args.fit_on)
        fit_files = sorted((root / "bank").glob("*.tsv")) or \
            sorted(root.glob("*.tsv"))
        if not fit_files:
            sys.exit(f"no masked texts under {root}")
        overlap = {f.stem for f in fit_files} & {f.stem for f in files}
        if overlap:
            sys.exit(f"the fitting corpus shares {len(overlap)} texts with the "
                     f"bank it would encode, e.g. {sorted(overlap)[:3]}")
        suffix = args.suffix or "catranks"
        note = (f"fitted on {len(fit_files)} held-out texts in {root.name}, "
                f"disjoint from this bank")
    else:
        names = {l.strip() for l in Path(args.fit_on).read_text(
            encoding="utf-8").splitlines() if l.strip()}
        fit_files = [f for f in files if f.stem in names]
        missing = names - {f.stem for f in fit_files}
        if missing:
            sys.exit(f"{len(missing)} named fitting texts are not in the bank, "
                     f"e.g. {sorted(missing)[:3]}")
        if not fit_files:
            sys.exit("the fitting set is empty")
        suffix = args.suffix or "catranks"
        note = (f"fitted on {len(fit_files)} held-out texts named in "
                f"{Path(args.fit_on).name}")

    print(f"encoding mhg_witnesses_{args.stream} to the shared alphabet")
    print(f"  companion file: {len(table):,} entries")
    print(f"  rank map: {note}")

    crmap = rank_map(fit_files, table)
    print(f"  {len(crmap):,} tokens mapped, "
          f"{len(set(crmap.values()))} distinct symbols")

    dst = MASKED / f"mhg_witnesses_{args.stream}_{suffix}"
    (dst / "bank").mkdir(parents=True, exist_ok=True)
    total = unknown = 0
    for f in files:
        out = []
        for sent in read_lines(f):
            row = []
            for t in sent:
                s = encode_token(t, crmap)
                if any(ch.isalpha() for ch in t) and t not in PLACEHOLDERS:
                    total += 1
                    unknown += s == "WX"
                row.append(s)
            out.append("\t".join(row))
        (dst / "bank" / f.name).write_text("\n".join(out) + "\n",
                                           encoding="utf-8")

    for item in ("MANIFEST.tsv", "lemma_canon.tsv"):
        if (src / item).exists():
            shutil.copy2(src / item, dst / item)
    if (src / "verse_ids").exists():
        if (dst / "verse_ids").exists():
            shutil.rmtree(dst / "verse_ids")
        shutil.copytree(src / "verse_ids", dst / "verse_ids")

    (dst / "ENCODING.txt").write_text(
        f"CatRanks alphabet over the gmh companion file\n"
        f"source bank: mhg_witnesses_{args.stream}\n"
        f"rank map: {note}\n"
        f"symbols: {len(set(crmap.values()))}\n"
        f"tokens falling outside the map: {unknown}/{total} "
        f"({unknown / max(total, 1):.2%}) encoded as WX\n", encoding="utf-8")

    print(f"  wrote {dst.relative_to(ROOT)} "
          f"({len(files)} texts, manifest and verse numbering copied)")
    print(f"  outside the rank map: {unknown:,}/{total:,} "
          f"({unknown / max(total, 1):.2%}) as WX")


if __name__ == "__main__":
    main()
