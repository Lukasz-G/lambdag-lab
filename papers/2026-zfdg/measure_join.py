# How much of the ruled-line damage does the lexicon join actually mend?
#
# The Referenzkorpus Mittelhochdeutsch transcribes 12,826 tokens of the very
# manuscript the Kaiserchronik edition's A1 witness contains, independently and
# by other editorial conventions. Aligning the two gives a gold answer to a
# question that otherwise has only an estimate behind it: how often the ruling
# breaks a word, and how much of that a lexicon can repair without inventing
# compounds the scribe never wrote.
#
# The measurement counts BOUNDARY-ONLY differences -- stretches where the two
# transcriptions carry the same letters divided into different tokens -- before
# and after the join, and reports what else moved, since a repair that mends
# ten breaks whilst manufacturing twenty is no repair.
#
#   python medieval/measure_join.py

import difflib
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from join_splits import REM_DIR, fold, join_chunks, load_lexicon  # noqa: E402
import extract_kcd as X  # noqa: E402

GOLD = REM_DIR / "M121V-G1.txt"


def gold_tokens():
    return [l.split("\t")[0]
            for l in open(GOLD, encoding="utf-8", errors="replace") if l.strip()]


def stream_from(chunks):
    """Verse tokens of one witness, given its ruled-line chunks."""
    verses, _, _, _ = X.segment(chunks)
    return [w for v in verses for w in v.split()]


def score(gold, mine, label):
    g = [fold(t) for t in gold]
    m = [fold(t) for t in mine]
    # anchor on the gold opening so both streams start at the same place
    start = 0
    needle = [t for t in g[:40] if t][:8]
    for i in range(len(m) - 8):
        if m[i:i + 8] == needle:
            start = i
            break
    # The reference corpus samples its excerpt from THROUGHOUT the manuscript
    # rather than taking one continuous stretch, so the comparison must be
    # allowed the whole remainder of the witness; capping it to the excerpt's
    # own length finds only the first passage and understates agreement
    # threefold (28% against 90%).
    seg = m[start:]
    sm = difflib.SequenceMatcher(a=g, b=seg, autojunk=False)
    blocks = [b for b in sm.get_matching_blocks() if b.size >= 5]
    cov = sum(b.size for b in blocks)
    boundary = other = 0
    for t, i1, i2, j1, j2 in sm.get_opcodes():
        if t == "equal":
            continue
        if t == "replace" and "".join(g[i1:i2]) == "".join(seg[j1:j2]):
            boundary += 1
        else:
            other += 1
    print(f"{label:26s} coverage {cov / len(g):6.1%}  "
          f"boundary-only diffs {boundary:4d}  other diffs {other:4d}")
    return boundary, other, cov


def measure(lex=None):
    lex = lex if lex is not None else load_lexicon(exclude=("M121V",))
    path = next(p for p in (HERE / "raw" / "kcd").glob("*.tei.xml")
                if "stav_ms276" in p.name)
    chunks = X.witness_text(path, want_expan=True)
    gold = gold_tokens()
    print(f"gold: {len(gold):,} tokens from {GOLD.name}\n")
    b0, o0, c0 = score(gold, stream_from(chunks), "without join")
    joined, n = join_chunks(chunks, lex)
    b1, o1, c1 = score(gold, stream_from(joined), "with lexicon join")
    print(f"\njoins applied: {n:,}")
    print(f"boundary-only differences: {b0} -> {b1}  ({b0 - b1:+d})")
    print(f"other differences:         {o0} -> {o1}  ({o1 - o0:+d})")
    print(f"block coverage:            {c0 / len(gold):.1%} -> {c1 / len(gold):.1%}")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    measure()
