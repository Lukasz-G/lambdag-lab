# Gate G2, on the traditions whose witnesses do not share a verse numbering.
#
# Only the Parzival witnesses carry an edition's verse numbers, so only they
# could be compared verse by verse. Everywhere else each witness numbers its own
# lines, and one omitted verse puts every later verse against the wrong one --
# which is what made the first run report that two witnesses of the
# Kaiserchronik agree on one rhyme word in a hundred.
#
# The remedy is a collation. The two witnesses' verses are matched on their
# BODIES -- every token of the verse except the last -- so that the alignment
# owes nothing to the position whose stability is being measured, and the runs
# of matching verses are then used as anchors to carry the verses between them
# across as well. What is measured on the aligned pairs is the same profile as
# before: at each position, how often the two witnesses agree on the lemma, on
# the scribe's own form, and on the rhyme sound.
#
#   python experiments/run_rhyme_collation.py --source kcd
#   python experiments/run_rhyme_collation.py --source hvad --max-pairs 12
#
# Output: medieval/rhyme_collation_{source}.tsv and the profile on stdout

import argparse
import csv
import sys
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "medieval"))
from rime import rime  # noqa: E402
from run_rhyme_stability import read  # noqa: E402

POSITIONS = [("first", 0), ("2nd", 1), ("3rd", 2),
             ("3rd last", -3), ("2nd last", -2), ("last (the rhyme)", -1)]


def collate(a_verses, b_verses):
    """pairs of verse indices, matched on the bodies and never on the rhyme"""
    ab = [tuple(v[:-1]) for v in a_verses]
    bb = [tuple(v[:-1]) for v in b_verses]
    blocks = SequenceMatcher(None, ab, bb, autojunk=False).get_matching_blocks()
    pairs, prev = [], (0, 0)
    for i, j, n in blocks:
        # the stretch between two anchors: carried across only where the two
        # witnesses have the same number of verses in it, since anything else
        # would be guesswork
        gi, gj = i - prev[0], j - prev[1]
        if gi == gj:
            pairs += [(prev[0] + k, prev[1] + k) for k in range(gi)]
        pairs += [(i + k, j + k) for k in range(n)]
        prev = (i + n, j + n)
    return pairs


def kappa(hits, counts):
    n = sum(counts.values())
    p = sum((v / n) ** 2 for v in counts.values()) if n else 1.0
    return (float(np.mean(hits)) - p) / (1 - p) if p < 1 else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="kcd")
    ap.add_argument("--max-pairs", type=int, default=12)
    ap.add_argument("--min-verses", type=int, default=500)
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    with (ROOT / "masked" / "mhg_witnesses_lemma_full" /
          "MANIFEST.tsv").open(encoding="utf-8") as fh:
        man = [r for r in csv.DictReader(fh, delimiter="\t")
               if r["file"].startswith(args.source)
               and not r["file"].endswith("__reg")
               and int(r["units"]) >= args.min_verses]
    works = defaultdict(list)
    for r in man:
        works[r["work"]].append(r)

    acc = {k: defaultdict(list) for k in ("lemma", "surface", "rime")}
    ch = {k: defaultdict(Counter) for k in acc}
    rows, done = [], 0
    for work, ws in sorted(works.items()):
        if len(ws) < 2:
            continue
        ws.sort(key=lambda r: -int(r["tokens"]))
        ref = ws[0]["file"]
        RL = list(read("lemma_full", ref).items())
        RS = read("surface_full", ref)
        if not RL:
            continue
        rl = [v for _, v in RL]
        rs = [RS[k] for k, _ in RL if k in RS]
        if len(rs) != len(rl):
            continue
        for r in ws[1:]:
            if done >= args.max_pairs:
                break
            name = r["file"]
            AL = list(read("lemma_full", name).items())
            AS = read("surface_full", name)
            al = [v for _, v in AL]
            as_ = [AS[k] for k, _ in AL if k in AS]
            if not al or len(as_) != len(al):
                continue
            pairs = collate(al, rl)
            n_used = 0
            for ia, ib in pairs:
                x, y = al[ia], rl[ib]
                sx, sy = as_[ia], rs[ib]
                if len(x) < 4 or len(y) < 4 or len(sx) != len(x) \
                        or len(sy) != len(y):
                    continue
                n_used += 1
                for lab, idx in POSITIONS:
                    acc["lemma"][lab].append(x[idx] == y[idx])
                    ch["lemma"][lab][y[idx]] += 1
                    acc["surface"][lab].append(sx[idx] == sy[idx])
                    ch["surface"][lab][sy[idx]] += 1
                    acc["rime"][lab].append(rime(sx[idx]) == rime(sy[idx]))
                    ch["rime"][lab][rime(sy[idx])] += 1
            done += 1
            rows.append((work, name, ref, len(al), len(pairs), n_used))
            print(f"  {name:54} {len(pairs):6,} verses collated with "
                  f"{ref.split('__')[-1]}", flush=True)

    if not rows:
        print("no pairs")
        return
    n = len(acc["lemma"]["last (the rhyme)"])
    print(f"\n  {args.source}: {len(rows)} witness pairs, {n:,} aligned verses "
          f"of four tokens or more\n")
    print(f"  {'position':20}{'lemma':>9}{'surface':>9}{'rime':>9}"
          f"   (agreement above chance)")
    for lab, _ in POSITIONS:
        print(f"  {lab:20}" + "".join(
            f"{kappa(acc[k][lab], ch[k][lab]):>9.1%}"
            for k in ("lemma", "surface", "rime")))
    out = ROOT / "medieval" / f"rhyme_collation_{args.source}.tsv"
    with out.open("w", encoding="utf-8") as fh:
        fh.write("position\tlemma\tsurface\trime\tn\n")
        for lab, _ in POSITIONS:
            fh.write(f"{lab}\t" + "\t".join(
                f"{kappa(acc[k][lab], ch[k][lab]):.4f}"
                for k in ("lemma", "surface", "rime")) + f"\t{n}\n")
    print(f"  wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
