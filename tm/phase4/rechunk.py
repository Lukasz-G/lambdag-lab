# Re-segment masked TSVs into fixed token windows (pseudo-sentences of N tokens), the
# segmentation-as-variable protocol: the same windowed units feed BOTH KN LambdaG (sentence
# unit = window: padding + reference sampling) and the HDC-TM sketcher. No spaCy needed —
# masked TSVs are the ordered token stream, so windowing is a pure re-chunk.
#
#   python phase4/rechunk.py <src_dir_or_file> <dst_dir> <window>
#   e.g. python phase4/rechunk.py phase3/pairs500 phase4/pairs500_w20 20

import sys
from pathlib import Path

def rechunk_file(src: Path, dst: Path, win: int):
    toks = []
    with open(src, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if line:
                toks.extend(line.split("\t"))
    with open(dst, "w", encoding="utf-8") as f:
        for i in range(0, len(toks), win):
            chunk = toks[i:i + win]
            if chunk:
                f.write("\t".join(chunk) + "\n")

def main():
    src, dst, win = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
    dst.mkdir(parents=True, exist_ok=True)
    files = [src] if src.is_file() else sorted(src.glob("*.tsv"))
    for f in files:
        rechunk_file(f, dst / f.name, win)
    print(f"rechunked {len(files)} files -> {dst} (window={win})")

if __name__ == "__main__":
    main()
