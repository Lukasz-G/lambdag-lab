# Authorship verification across manuscript witnesses: the four cells.
#
# A manuscript text is the work of two hands, the author's and the scribe's,
# and the claim under test is that the masked lemma stream keeps the first
# and loses the second. Every case pairs a KNOWN passage with a QUESTIONED
# passage, and the pair falls into one of four cells:
#
#   same author, same scribe         two passages of one witness, or two works
#                                    of one author in one scribe's hand
#   same author, different scribe    two witnesses of one work, or two works
#                                    of one author in different hands
#   different author, same scribe    two authors copied by one scribe (the
#                                    Ambraser Heldenbuch, the Vorau codex)
#   different author, different scribe
#
# Identity. An author is the manifest's author; an anonymous work is its own
# identity. A scribe is the source and siglum of the witness, with the
# manuscripts known to be one codex merged: every Ambraser text and the
# Iwein d of the Hartmann edition are Hans Ried's; the Vorau codex holds the
# Kaiserchronik A1 and the Vorauer Alexander; ReM's Iwein B is the Giessen
# manuscript of the Hartmann edition's B, ReM's Parzival D the St. Gallen
# codex of the Bern edition's d. Texts that duplicate a fuller witness of
# the same manuscript (ReM's Iwein B, Parzival D and Kaiserchronik V excerpts,
# the Ambraser copy of the Iwein already present from the Hartmann edition)
# are excluded from the cases and the donors.
#
# Disjointness. Known passages are cut from the first 45 % of a work's
# verses by the edition's numbering, questioned passages from the last 45 %,
# in every witness alike, so that two passages of one work can never carry
# the same text however the recensions differ in length. Cross-work pairs
# are disjoint by construction; the same rule is applied to them for
# uniformity. The Kaiserchronik witnesses carry sequential verse numbers,
# which the ten-percent guard band covers.
#
# Scoring is the symmetrised grammar model of the journal paper: the known
# passage's grammar against r donor grammars fitted on size-matched passages
# of other authors' witnesses, the statistic the mean over donors, the
# per-donor lambdas persisted. Evidence units are --seg tokens (100 by the
# protocol); known and questioned passages are --L tokens alike.
#
#   python experiments/run_witness_cells.py --stream lemma --L 1000
#   python experiments/run_witness_cells.py --stream lemma --L 1000 --dry-run
#
# Output: scores/witness/{stream}__L{L}__w{seg}{tag}.jsonl, one row per case:
#   {"cell","known","quest","known_author","quest_author","known_scribe",
#    "quest_scribe","known_period","quest_period","kw","qw","lambda_G","n_q","lam_j"}

import argparse
import csv
import json
import random
import re
import sys
import time
import unicodedata
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
from lambdag import LambdaG  # noqa: E402

SCORES = ROOT / "scores" / "witness"
DUPLICATES = {"rem__M312", "rem__M325", "rem__M121V", "ambraser__hartmann_von_aue__iwein__d"}
SAME_SCRIBE = {
    "ambraser": ["ambraser__", "hvad__hartmann_von_aue__iwein__d_lc"],
    "vorau": ["kcd__anon__kaiserchronik__A1", "rem__M009"],
    "giessen97": ["hvad__hartmann_von_aue__iwein__B"],
    "stgallen857": ["parzival__wolfram_von_eschenbach__parzival__d"],
}
CELLS = ["SA-SS", "SA-DS", "DA-SS", "DA-DS"]
# The masking placeholders. They are not punctuation and both stages carry them,
# so they survive the punctuation filter a borrowed donor bank goes through.
PLACEHOLDER_SYMBOLS = set("#øØ@§©$¶¥")


def norm(s):
    s = unicodedata.normalize("NFKD", s.lower())
    s = "".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9]+", "_", s).strip("_")


def author_id(row):
    a = row["author"]
    if a in ("anon.", "-", ""):
        return norm(row["work"].replace("anon__", ""))
    if row["source"] == "rem":
        # ReM's author field for anonymous texts is the title's head
        a = re.sub(r"\s*\(.*$", "", a)
        a = re.sub(r"\s+(A|B|C)\s*$", "", a)
        a = re.sub(r"\s*\[.*$", "", a)
        if "Kaiserchronik" in a:
            return "kaiserchronik"
    return norm(a)


def scribe_id(name, row):
    for group, prefixes in SAME_SCRIBE.items():
        if any(name == p or name.startswith(p) for p in prefixes):
            return group
    if row["source"] == "rem":
        return "rem__" + row["siglum"]
    return row["source"] + "__" + row["siglum"]


def period_of(row):
    d = row.get("date", "-")
    m = re.search(r"1[2-6]\d{2}", d)
    y = int(m.group(0)) if m else None
    if y is None:
        m = re.search(r"(1[2-6])\.(?:.*?(1[2-6])\.)?\s*Jh", d)
        if m:
            y = (int(m.group(1)) - 1) * 100 + (100 if m.group(2) else 50)
        elif re.match(r"1[2-4],[12]", d):          # ReM's 12,2 style
            y = (int(d[:2]) - 1) * 100 + (75 if d[3] == "2" else 25)
    if y is None:
        return "?"
    return "early" if y <= 1350 else "late"


def verse_pos(vid):
    m = re.match(r"(\d+)", vid)
    return int(m.group(1)) if m else 0


def rechunk(tokens, w):
    return [tokens[i:i + w] for i in range(0, len(tokens), w) if tokens[i:i + w]]


def load(stream, encoding="posnoise"):
    # An alternative encoding of the same bank (experiments/encode_mhg_catranks.py)
    # keeps the manifest and the verse numbering, so a case drawn under one
    # encoding is the very same case under the other.
    name = f"mhg_witnesses_{stream}" if encoding == "posnoise" \
        else f"mhg_witnesses_{stream}_{encoding}"
    base = ROOT / "masked" / name
    man = {r["file"]: r for r in csv.DictReader(open(base / "MANIFEST.tsv", encoding="utf-8"), delimiter="\t")}
    banks = {}
    for f in sorted((base / "bank").glob("*.tsv")):
        name = f.stem
        # duplicates of a fuller witness, the regularised layer, and the few
        # prose texts among the reference corpus's gold (the manifest's genre
        # column) stay out: donors and distractors mirror the known's construction
        if name in DUPLICATES or name.endswith("__reg") or not man[name].get("genre", "V").startswith("V"):
            continue
        sents = [line.split("\t") for line in f.read_text(encoding="utf-8").splitlines() if line]
        vids = (base / "verse_ids" / (name + ".txt")).read_text(encoding="utf-8").split("\n")
        vids = [v for v in vids if v]
        if len(vids) != len(sents):
            vids = [str(i + 1) for i in range(len(sents))]
        r = man[name]
        banks[name] = dict(sents=sents, vids=vids, author=author_id(r), scribe=scribe_id(name, r),
                           work=norm(r["work"]), period=period_of(r), ntok=sum(map(len, sents)))
    # a work's verse range, over all its witnesses (edition numbering)
    vmax = {}
    for b in banks.values():
        vmax[b["work"]] = max(vmax.get(b["work"], 0), max(verse_pos(v) for v in b["vids"]))
    for b in banks.values():
        top = vmax[b["work"]]
        first = [t for s, v in zip(b["sents"], b["vids"]) if verse_pos(v) <= 0.45 * top for t in s]
        last = [t for s, v in zip(b["sents"], b["vids"]) if verse_pos(v) >= 0.55 * top for t in s]
        b["first"], b["last"] = first, last
    return banks


def load_donors(name, drop_punct=True):
    """A reference population from ANOTHER bank, for the borrowed condition.

    The donor grammars then come from a different language stage entirely, which
    only works because both sides are encoded in the shared CatRanks alphabet.
    Two differences from the case bank have to be reconciled here:

      punctuation  the medieval stream carries none by protocol, while a modern
                   bank keeps it -- 16 % of the modern mass. Left in place, every
                   donor grammar would spend that mass on symbols the questioned
                   text can never emit. Dropped by definition rather than by
                   comparison with the case bank, so nothing is fitted on the
                   test material.
      identity     one file is one author, and no author is shared with the case
                   bank, so the author-exclusion and period filters do not apply.
    """
    for root in ("masked", "masked_catsrank"):
        base = ROOT / root / name
        if (base / "bank").exists():
            break
    else:
        raise SystemExit(f"no donor bank {name} under masked/ or masked_catsrank/")
    # "P" is the encoder's catch-all for punctuation it does not keep verbatim,
    # so it is punctuation too and must go with the rest; an earlier version kept
    # it because the string is alphanumeric, leaving 1.95 % of the donor mass on a
    # symbol the medieval side cannot emit.
    keep_sym = (lambda t: t != "P"
                and (any(ch.isalnum() for ch in t) or t in PLACEHOLDER_SYMBOLS)) \
        if drop_punct else (lambda t: True)
    donors, dropped, total = {}, 0, 0
    for f in sorted((base / "bank").glob("*.tsv")):
        sents = []
        for line in f.read_text(encoding="utf-8").splitlines():
            if not line:
                continue
            toks = [t for t in line.split("\t") if t]
            total += len(toks)
            kept = [t for t in toks if keep_sym(t)]
            dropped += len(toks) - len(kept)
            if kept:
                sents.append(kept)
        if sents:
            donors[f.stem] = dict(sents=sents, author=f.stem, scribe=f.stem,
                                  period=None, work=f.stem,
                                  ntok=sum(map(len, sents)))
    print(f"  donor bank {base.relative_to(ROOT)}: {len(donors)} identities, "
          f"{sum(b['ntok'] for b in donors.values()):,} symbols"
          + (f", {dropped / max(total, 1):.1%} punctuation dropped"
             if drop_punct else ""), flush=True)
    return donors


def passages(tokens, L, seg, n, rng):
    """up to n non-overlapping L-token passages as lists of seg-token units"""
    blocks = len(tokens) // L
    if blocks == 0:
        return []
    picks = sorted(rng.sample(range(blocks), min(n, blocks)))
    return [(j, rechunk(tokens[j * L:(j + 1) * L], seg)) for j in picks]


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--stream",
                    choices=["lemma", "surface", "lemma_full", "rhyme",
                             "rime"],
                    default="lemma",
                    help="the masked streams, or the rhyme-word streams, whose "
                         "tokens are verses rather than words: one token per "
                         "verse, so a length in tokens is a length in verses "
                         "and the same passage is about a fifth as long")
    ap.add_argument("--encoding", default="posnoise",
                    help="posnoise (the masked tokens) or the suffix of an "
                         "encoded bank, e.g. catranks or catranks_leaky")
    ap.add_argument("--L", type=int, default=1000)
    ap.add_argument("--seg", type=int, default=100)
    ap.add_argument("--cells", default=",".join(CELLS))
    ap.add_argument("--max-known", type=int, default=6, help="known witnesses per author identity")
    ap.add_argument("--kper", type=int, default=2, help="known passages per known witness")
    ap.add_argument("--same", type=int, default=3, help="questioned passages per same-author witness")
    ap.add_argument("--nsame-wits", type=int, default=6, help="same-author questioned witnesses per known passage")
    ap.add_argument("--ncross", type=int, default=3, help="different-author questioned passages per known passage")
    ap.add_argument("--r-donors", type=int, default=15)
    ap.add_argument("--max-donors", type=int, default=60)
    ap.add_argument("--min-donors", type=int, default=8)
    ap.add_argument("--donor-period", choices=["any", "quest"], default="any",
                    help="quest: donors drawn from witnesses of the questioned text's period (early: to 1350, "
                         "late: after), so that the reference population mirrors the questioned hand's language")
    ap.add_argument("--donor-bank", default="",
                    help="draw the reference population from another bank "
                         "(the borrowed condition), e.g. "
                         "german_tgverseall_catranks_fitdracor; both sides must "
                         "already be encoded in the shared alphabet")
    ap.add_argument("--donor-keep-punct", action="store_true",
                    help="keep punctuation in a borrowed donor bank; off by "
                         "default because the medieval stream carries none")
    ap.add_argument("--tag", default="")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    L, seg = args.L, args.seg
    cells = args.cells.split(",")
    enc = "" if args.encoding == "posnoise" else f"__{args.encoding}"
    # the donor bank is part of the arm's identity, never of its sampling seed
    dbk = f"__borrowed" if args.donor_bank else ""
    tag = f"{args.stream}__L{L}__w{seg}{enc}{dbk}{args.tag}"
    # every sampling decision is seeded by stream, length and unit alone, so
    # that arms differing only in an option (donor period, encoding, tag) score
    # the very same cases and can be compared pair by pair
    seed = f"{args.stream}__L{L}__w{seg}"
    SCORES.mkdir(parents=True, exist_ok=True)
    fn = SCORES / (tag + ".jsonl")
    if fn.exists() and not args.dry_run:
        print(f"{tag}: exists, skipped"); return

    banks = load(args.stream, args.encoding)
    # passages per witness
    known_of = {n: passages(b["first"], L, seg, args.kper, random.Random(seed + n + "k")) for n, b in banks.items()}
    quest_of = {n: passages(b["last"], L, seg, args.same, random.Random(seed + n + "q")) for n, b in banks.items()}
    known_wits = [n for n in banks if known_of[n]]
    quest_wits = [n for n in banks if quest_of[n]]
    by_author = {}
    for n in known_wits:
        by_author.setdefault(banks[n]["author"], []).append(n)
    # the known witnesses per author identity, capped
    for a, ws in by_author.items():
        if len(ws) > args.max_known:
            by_author[a] = sorted(random.Random(seed + a).sample(ws, args.max_known))

    def cell_of(kn, qn):
        sa = banks[kn]["author"] == banks[qn]["author"]
        ss = banks[kn]["scribe"] == banks[qn]["scribe"]
        return ("SA" if sa else "DA") + "-" + ("SS" if ss else "DS")

    # enumerate cases: (known witness, known passage, questioned witness, questioned passage)
    cases = []
    for a, ws in sorted(by_author.items()):
        for kn in ws:
            for kj, kunits in known_of[kn]:
                # same-author questioned passages: the witness itself and up to
                # --nsame-wits other witnesses of this author
                own = [qn for qn in quest_wits if banks[qn]["author"] == a]
                others_own = [qn for qn in own if qn != kn]
                prng_s = random.Random(seed + kn + str(kj) + "same")
                chosen = ([kn] if kn in own else []) + sorted(prng_s.sample(others_own, min(args.nsame_wits, len(others_own))))
                for qn in chosen:
                    for qj, qunits in quest_of[qn]:
                        cases.append((cell_of(kn, qn), kn, kj, qn, qj))
                prng = random.Random(seed + kn + str(kj) + "cross")
                # different-author questioned passages, same scribe: every other
                # identity in the same hand, capped
                same_scribe = [qn for qn in quest_wits if banks[qn]["author"] != a and banks[qn]["scribe"] == banks[kn]["scribe"]]
                for qn in prng.sample(same_scribe, min(args.ncross, len(same_scribe))):
                    qj, qunits = prng.choice(quest_of[qn])
                    cases.append((cell_of(kn, qn), kn, kj, qn, qj))
                # different-author questioned passages, other scribe: one
                # negative MIRRORS each same-author other-scribe witness in
                # period, so that the period is not stacked against the
                # same-author side (a same-author pair may span three
                # centuries; its negative then spans them too)
                others = [qn for qn in quest_wits if banks[qn]["author"] != a and banks[qn]["scribe"] != banks[kn]["scribe"]]
                mirrored = [qn for qn in chosen if banks[qn]["scribe"] != banks[kn]["scribe"]]
                targets = [banks[qn]["period"] for qn in mirrored] or [banks[kn]["period"]] * args.ncross
                for per in targets:
                    pool = [qn for qn in others if banks[qn]["period"] == per] or others
                    if not pool:
                        break
                    qn = prng.choice(pool)
                    qj, qunits = prng.choice(quest_of[qn])
                    cases.append((cell_of(kn, qn), kn, kj, qn, qj))
    cases = [c for c in cases if c[0] in cells]
    counts = {c: sum(1 for x in cases if x[0] == c) for c in CELLS}
    print(f"{tag}: {len(banks)} witnesses, {len(known_wits)} with a known passage, {len(by_author)} identities; "
          f"cases per cell {counts}", flush=True)
    if args.dry_run:
        for c in CELLS:
            ex = [x for x in cases if x[0] == c][:4]
            for x in ex:
                print("  ", c, x[1], "->", x[3])
        return

    lg = LambdaG(N=10, r=1, engine="kn", random_state=0)
    if args.donor_bank:
        # the borrowed condition: the reference population comes from another
        # bank, so no identity is shared with the cases and the period of a
        # medieval witness has no counterpart on the donor side
        donor_src = load_donors(args.donor_bank, not args.donor_keep_punct)
        borrowed = True
    else:
        donor_src, borrowed = banks, False
    donors_all = {n: b for n, b in donor_src.items() if b["ntok"] >= L + seg}

    def score(q_units, k_units, exclude_authors, seed, period=None):
        if borrowed:
            exclude_authors, period = frozenset(), None
        pool = sorted(n for n, b in donors_all.items() if b["author"] not in exclude_authors
                      and (period is None or b["period"] == period))
        if len(pool) < args.min_donors:      # too few in that period: fall back to all
            pool = sorted(n for n, b in donors_all.items() if b["author"] not in exclude_authors)
        prng = random.Random(seed)
        if len(pool) > args.max_donors:
            pool = prng.sample(pool, args.max_donors)
        picks = prng.sample(pool, min(args.r_donors, len(pool)))
        lams, n_q = [], 0
        for dn in picks:
            toks = [t for s in donors_all[dn]["sents"] for t in s]
            start = prng.randrange(0, max(1, len(toks) - L))
            ref = rechunk(toks[start:start + L], seg)
            r = lg.score(q_units, k_units, ref_sentences=ref, r=1, with_details=False)
            lams.append(r.lambda_G); n_q = r.n_query_tokens
        return float(np.mean(lams)), n_q, lams

    if len(donors_all) < args.min_donors:
        print(f"{tag}: {len(donors_all)} donors, skipped"); return
    t0, rows = time.time(), []
    kcache = {(n, j): u for n in known_of for j, u in known_of[n]}
    qcache = {(n, j): u for n in quest_of for j, u in quest_of[n]}
    for i, (cell, kn, kj, qn, qj) in enumerate(cases):
        excl = {banks[kn]["author"], banks[qn]["author"]}
        per = banks[qn]["period"] if args.donor_period == "quest" else None
        lam, n_q, lams = score(qcache[(qn, qj)], kcache[(kn, kj)], excl, f"{seed}|{kn}|{kj}|{qn}|{qj}", per)
        rows.append(dict(cell=cell, known=kn, quest=qn, known_author=banks[kn]["author"], quest_author=banks[qn]["author"],
                         known_scribe=banks[kn]["scribe"], quest_scribe=banks[qn]["scribe"],
                         known_period=banks[kn]["period"], quest_period=banks[qn]["period"],
                         kw=kj, qw=qj, lambda_G=lam, n_q=n_q, lam_j=[round(x, 3) for x in lams]))
        if (i + 1) % 50 == 0:
            print(f"  {tag}: {i + 1}/{len(cases)} cases, {time.time() - t0:.0f}s", flush=True)
    with open(fn, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"{tag}: {len(rows)} rows in {time.time() - t0:.0f}s -> {fn}", flush=True)


if __name__ == "__main__":
    main()
