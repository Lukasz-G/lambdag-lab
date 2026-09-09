# Encode the persisted parse as a symbol stream, two ways, with a corruption knob.
#
# THE ALPHABET. Every token becomes `relation·symbol`, where the second half is
# exactly what POSNoise emits today -- the wordform for a function word, the
# class sigil for a content word. The relation is applied to EVERY token and not
# only to content words: German function words are the ambiguous ones, and the
# relation is what disambiguates them. *die* is a determiner or a relative
# pronoun in subject or object role, *es* is a subject or an expletive, *da* is
# an adverb or a causal subordinator, *zu* is a preposition or an infinitive
# marker. Those distinctions are syntactic style. Restricting the relation to
# content words would also leave roughly half of every window byte-identical to
# the POSNoise stream -- and, measured on these banks, it is the genre-STABLE
# half (function words hold at 46-49% of tokens in all three genres, while the
# content sigil rate swings 12.6 points), so a null result would be unreadable.
#
# THE TWO ORDERS, which is the actual experiment.
#   linear     surface order. The CONTROL: it changes the symbols and nothing
#              else, so it isolates what the relation alone is worth.
#   traversal  head first, then dependents recursively in canonical relation
#              order. Surface position is gone, so metre and line-breaking
#              cancel -- which is the claim. It also collapses a verb-final
#              subordinate clause and a V2 main clause onto the same string,
#              and Vorfeld choice is real style, so this is a cost as well as
#              the point. Running both is what tells the two apart.
# Both emit exactly one symbol per token, so stream length is unchanged and the
# per-token rates compare directly against the POSNoise and class-rank ladders.
#
# THE CORRUPTION KNOB. The parser is genre-sensitive, so parse error could
# manufacture a cross-genre penalty on its own. Relabelling or reattaching a
# known fraction of arcs calibrates the slope: degrade the cleaner genre by the
# fragmentation gap parse_dep_banks.py reports, re-run the within-genre
# direction, and see whether the trend survives. That bounds the confound with a
# number instead of an argument, and needs no gold-standard parse.
#
#   python experiments/encode_dep.py                        # both variants
#   python experiments/encode_dep.py --variants traversal --corrupt-attach 0.05
#
# Output: masked_dep/{ds}_{variant}[_cN]/bank/*.tsv -- the layout the ladder's
# --alphabet dispatch already reads.

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PARSED = HERE.parent / "parsed_dep"
OUT = HERE.parent / "masked_dep"
GENRES = {"prose": "german_tgproseall", "verse": "german_tgverseall",
          "drama": "german_tgdramaall"}
SEP = "·"
FLOOR = 20          # pairs rarer than this fold back to the bare relation


def records(genre):
    for f in sorted((PARSED / genre).glob("*.npz")):
        yield f.stem, np.load(f, allow_pickle=True)


def strings(d):
    """Local ids -> (relation, symbol) strings for one record.

    The parse stores per-record vocabularies so the harvest can be parsed in
    independent shards; the union happens here, in the one process that sees
    every record anyway.
    """
    return d["deps"][d["dep"]], d["vocab"][d["sym"]]


def sentences(d):
    """Slice the flat arrays into the masker's own sentence units."""
    cuts = np.concatenate([[0], np.cumsum(d["sent_len"])])
    for a, b in zip(cuts[:-1], cuts[1:]):
        yield int(a), int(b)


def corrupt(dep, head, a, b, rng, p_label, p_attach, ndep):
    """Relabel and/or reattach arcs inside one sentence, at known rates."""
    dep = dep.copy()
    head = head.copy()
    n = b - a
    if p_label > 0:
        m = rng.random(n) < p_label
        dep[m] = rng.integers(0, ndep, size=int(m.sum()), dtype=dep.dtype)
    if p_attach > 0:
        m = rng.random(n) < p_attach
        idx = np.flatnonzero(m)
        # reattach within the sentence, so corruption changes the tree's shape
        # without changing which tokens belong to which unit
        head[idx] = rng.integers(0, n, size=len(idx)) - idx
    return dep, head


def traverse(dep_s, head_s):
    """Head-first DFS, children in canonical relation order. Length-preserving.

    Heads may point outside the sentence -- the masker segments on punctuation
    and newlines while the parser follows its own sentence model -- and after
    corruption they may form a cycle. Both cases make the token a local root
    rather than being dropped, and any token a cycle still hides from the walk
    is appended in surface order, so the output is always exactly n symbols.
    """
    n = len(dep_s)
    kids = [[] for _ in range(n)]
    roots = []
    for j in range(n):
        h = j + int(head_s[j])
        if h == j or not (0 <= h < n):
            roots.append(j)
        else:
            kids[h].append(j)
    for j in range(n):
        if len(kids[j]) > 1:
            kids[j].sort(key=lambda c: (dep_s[c], c))
    out, seen = [], bytearray(n)
    stack = list(reversed(roots))
    while stack:
        j = stack.pop()
        if seen[j]:
            continue
        seen[j] = 1
        out.append(j)
        for c in reversed(kids[j]):
            if not seen[c]:
                stack.append(c)
    if len(out) < n:
        out.extend(j for j in range(n) if not seen[j])
    return out


def fold_map(genres, floor):
    """Pair -> emitted symbol, fitted on the POOLED genres.

    One map across all three genres, deliberately: a per-genre floor would let
    the same pair be spelled differently in prose and in verse, which is exactly
    the comparison the ladder makes. The map is a FITTED artefact, so what it
    was fitted on is printed and recorded.
    """
    c = Counter()
    for g in genres:
        for _, d in records(g):
            c.update(zip(*strings(d)))
    out = {}
    for (p, s), n in c.items():
        out[(p, s)] = (f"{p}{SEP}{s}" if n >= floor else p)
    kept = sum(1 for k in out if SEP in out[k])
    print(f"  fold map fitted on {'+'.join(genres)}: {len(c)} attested pairs, "
          f"{kept} kept above floor {floor}, {len(c) - kept} folded to the bare "
          f"relation")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", default="linear,traversal")
    ap.add_argument("--genres", default="prose,verse,drama")
    ap.add_argument("--fit-on", default="", help="genres the fold map is fitted "
                    "on (default: the encoded genres, pooled)")
    ap.add_argument("--floor", type=int, default=FLOOR)
    ap.add_argument("--corrupt-label", type=float, default=0.0)
    ap.add_argument("--corrupt-attach", type=float, default=0.0)
    ap.add_argument("--tag", default="", help="suffix for a corrupted build")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    genres = [g for g in args.genres.split(",") if g in GENRES]
    fit = [g for g in (args.fit_on.split(",") if args.fit_on else genres) if g in GENRES]
    fmap = fold_map(fit, args.floor)

    cor = args.corrupt_label > 0 or args.corrupt_attach > 0
    tag = args.tag or (f"_c{int(100 * max(args.corrupt_label, args.corrupt_attach)):02d}"
                       if cor else "")
    if cor:
        print(f"  CORRUPTION: label {args.corrupt_label:.3f}, "
              f"attach {args.corrupt_attach:.3f} -> tag '{tag}'")

    for variant in args.variants.split(","):
        for g in genres:
            d = OUT / f"{GENRES[g]}_{variant}{tag}" / "bank"
            d.mkdir(parents=True, exist_ok=True)
            ntok = nfile = 0
            for stem, rec in records(g):
                _, sstr = strings(rec)
                dep, head = rec["dep"], rec["head"]
                ndep = len(rec["deps"])
                rng = np.random.default_rng(abs(hash((stem, variant, tag))) % 2**32)
                lines = []
                for a, b in sentences(rec):
                    ds, hs = dep[a:b], head[a:b]
                    if cor:
                        ds, hs = corrupt(ds, hs, a, b, rng, args.corrupt_label,
                                         args.corrupt_attach, ndep)
                    dd = rec["deps"][ds]
                    idx = (range(b - a) if variant == "linear"
                           else traverse(dd, hs))
                    lines.append("\t".join(
                        fmap.get((dd[j], sstr[a + j]), dd[j]) for j in idx))
                    ntok += b - a
                (d / f"{stem}.tsv").write_text("\n".join(lines) + "\n",
                                               encoding="utf-8")
                nfile += 1
            print(f"  {variant:10s} {g:6s} {nfile:3d} files {ntok:9d} tokens "
                  f"-> {d.parent.name}", flush=True)


if __name__ == "__main__":
    main()
