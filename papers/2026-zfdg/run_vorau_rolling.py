# The ground truth for a rolling profile: one codex, two authors, a known
# boundary between them.
#
# A rolling verification score along a long text is only worth reading if it is
# known what the score does where the author actually changes. Vorau,
# Stiftsbibliothek, Cod. 276 gives that case. The codex carries the
# Kaiserchronik on leaves 1ra-73vb and Pfaffe Lambrecht's Alexanderlied on
# leaves 109ra-115va, and the reference corpus annotates both, so the two texts
# come from ONE manuscript, in ONE annotation, and the boundary between them is
# not in doubt. Everything that usually confounds such a comparison -- scribe,
# period, region, editorial practice, tagger -- is therefore held fixed, and
# what is left between the two halves of the profile is the author.
#
# WHAT IS COMPUTED. A window of L tokens is rolled along the codex stream, the
# chronicle first and the Alexanderlied after it, and each window is scored as
# the questioned document against known passages of the chronicle, with the
# reference grammars drawn from other texts of the corpus, one grammar per donor
# and the mean taken over donors, as everywhere else in this work. Three things
# are then read off:
#
#   the BAND. The windows that lie inside the chronicle are by the chronicle's
#     own author, so their spread is the fluctuation a profile shows when
#     nothing has changed. A drop must leave that band to mean anything.
#   the STEP. The windows inside the Alexanderlied are by another author in the
#     same hand, and their distance from the band, in the band's own robust
#     standard deviations, is what the method can see.
#   the COHORT. Passages by other authors, scored against the same known
#     passages, locate the different-author hypothesis for this case, so the
#     Alexanderlied can be read against a known-different level and not only
#     against the chronicle.
#
# WHERE THE KNOWN PASSAGES COME FROM, and why it is asked twice:
#
#   --known edition   from the edition's text of the same Vorau manuscript,
#     excluding every stretch the reference annotation also covers, so the
#     known material is disjoint from the questioned and not adjacent to it.
#     The two sides then differ in their annotation pipeline -- but they differ
#     in it EQUALLY on both sides of the boundary, so the pipeline cannot
#     produce the step.
#   --known rem   from the head of the reference annotation itself, so that
#     known and questioned share manuscript, annotation and tagger exactly. The
#     price is that the known material is short and textually adjacent to the
#     first questioned windows.
#
# Both are run and both are reported; a step that only one of them shows is a
# property of the arrangement and not of the codex.
#
#   python experiments/run_vorau_rolling.py --known edition
#   python experiments/run_vorau_rolling.py --known rem --L 1000
#
# Output: scores/vorau_rolling/{stream}__L{L}__w{seg}__{known}.jsonl

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
import build_kc_reuse_mask as M  # noqa: E402
import run_witness_cells as W  # noqa: E402
from lambdag import LambdaG  # noqa: E402

OUT = ROOT / "scores" / "vorau_rolling"
CHRON = "rem__M121V"     # Kaiserchronik A (V), the reference corpus's excerpt
ALEX = "rem__M009"       # Pfaffe Lambrecht, Alexanderlied (V)
EDITION = "kcd__anon__kaiserchronik__A1"
K = 8                    # the shingle at which the excerpt is located in the edition


def read_bank(stream, name):
    """a text the bank loader leaves out, read from the same files it reads

    The reference corpus's annotation of the Vorau Kaiserchronik is registered
    as a duplicate of the edition's text and dropped from every experiment that
    draws cases, which is right there and wrong here: this experiment is about
    that manuscript.
    """
    p = ROOT / "masked" / f"mhg_witnesses_{stream}" / "bank" / f"{name}.tsv"
    return [t for line in p.read_text(encoding="utf-8").splitlines() if line
            for t in line.split("\t")]


def excerpt_mask(edition_toks, excerpt_toks, margin):
    """the positions of the edition that the excerpt also carries"""
    sh = {tuple(excerpt_toks[i:i + K]) for i in range(len(excerpt_toks) - K + 1)}
    hit = np.zeros(len(edition_toks), bool)
    for i in range(len(edition_toks) - K + 1):
        if tuple(edition_toks[i:i + K]) in sh:
            hit[max(i - margin, 0):i + K + margin] = True
    return hit


def spread_passages(free, n, L):
    """n passages of L tokens, as far apart as the free stretches allow"""
    runs, start = [], None
    for i, ok in enumerate(free):
        if ok and start is None:
            start = i
        elif not ok and start is not None:
            runs.append((start, i)); start = None
    if start is not None:
        runs.append((start, len(free)))
    runs = [r for r in runs if r[1] - r[0] >= L]
    if not runs:
        raise SystemExit("no stretch of the edition is free of the excerpt")
    # one passage from each free stretch before any stretch gives a second, so
    # that the known material is spread over the work rather than being one
    # long block with several names
    out, guard = [], 0
    while len(out) < n and runs and guard < 1000:
        guard += 1
        runs.sort(key=lambda r: -(r[1] - r[0]))
        nxt = []
        for a, b in runs:
            if len(out) < n:
                out.append(a)
                nxt.append((a + L, b))
            else:
                nxt.append((a, b))
        runs = [r for r in nxt if r[1] - r[0] >= L]
    return sorted(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stream", choices=["lemma", "surface"], default="lemma")
    ap.add_argument("--known", choices=["edition", "rem"], default="edition")
    ap.add_argument("--L", type=int, default=1000, help="window length")
    ap.add_argument("--step", type=int, default=250)
    ap.add_argument("--seg", type=int, default=100)
    ap.add_argument("--kpassages", type=int, default=4)
    ap.add_argument("--donors", type=int, default=15)
    ap.add_argument("--cohort", type=int, default=40)
    ap.add_argument("--margin", type=int, default=200,
                    help="tokens of the edition kept clear on either side of "
                         "every stretch the excerpt also carries")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    L, seg = args.L, args.seg
    rng = random.Random(args.seed)

    bank = W.load(args.stream)
    chron = read_bank(args.stream, CHRON)
    alex = [t for s in bank[ALEX]["sents"] for t in s]
    print(f"Vorau, Cod. 276: chronicle {len(chron):,} tokens "
          f"(reference annotation), Alexanderlied {len(alex):,}")

    # the known passages
    if args.known == "edition":
        ed = [t for s in bank[EDITION]["sents"] for t in s]
        hit = excerpt_mask(ed, chron, args.margin)
        print(f"  the excerpt covers {hit.mean():.1%} of the edition's text "
              f"(with a margin of {args.margin}); known passages are drawn "
              f"from the rest")
        starts = spread_passages(~hit, args.kpassages, L)
        known = [ed[s:s + L] for s in starts]
        kfrom = [f"{EDITION}@{s}" for s in starts]
        q_from = 0
    else:
        need = args.kpassages * L
        if need >= len(chron) - 2 * L:
            raise SystemExit("the excerpt is too short for that many passages")
        known = [chron[i * L:(i + 1) * L] for i in range(args.kpassages)]
        kfrom = [f"{CHRON}@{i * L}" for i in range(args.kpassages)]
        q_from = need
        print(f"  known passages are the excerpt's first {need:,} tokens; "
              f"the profile starts after them")

    # the codex stream, chronicle first as the codex has it
    stream = chron[q_from:] + alex
    nchron = len(chron) - q_from
    print(f"  profile over {len(stream):,} tokens, boundary at {nchron:,}")

    # donors and cohort: neither the chronicle nor either Alexander poem, and
    # no identity that either side of the boundary belongs to
    bad_author = {bank[EDITION]["author"], bank[ALEX]["author"]}
    pool = [n for n in bank
            if not M.is_kc(n, bank) and n not in (ALEX, "rem__M008")
            and bank[n]["author"] not in bad_author
            and bank[n]["ntok"] >= 2 * L]
    pool.sort()
    rng.shuffle(pool)
    dn, cn = pool[:args.donors], pool[args.donors:args.donors + args.cohort]
    print(f"  {len(pool)} texts eligible; {len(dn)} donors, {len(cn)} cohort")
    refs, donor_names = [], []
    for n in dn:
        t = [x for s in bank[n]["sents"] for x in s]
        st = rng.randrange(0, len(t) - L + 1)
        refs.append(W.rechunk(t[st:st + L], seg))
        donor_names.append(f"{n}@{st}")
    cohort = []
    for n in cn:
        t = [x for s in bank[n]["sents"] for x in s]
        st = rng.randrange(0, len(t) - L + 1)
        cohort.append((n, st, W.rechunk(t[st:st + L], seg)))

    wins = [(i, stream[i:i + L])
            for i in range(0, len(stream) - L + 1, args.step)]
    n_scores = (len(wins) + len(cohort)) * len(known) * len(refs)
    print(f"  {len(wins)} windows x {len(known)} known passages x "
          f"{len(refs)} donors, plus the cohort: {n_scores:,} scores")
    if args.dry_run:
        for i, s in enumerate(kfrom):
            print("   known", s)
        return

    lg = LambdaG(N=10, r=1, engine="kn", random_state=0)
    kunits = [W.rechunk(k, seg) for k in known]
    t0, rows = time.time(), []

    def score(q):
        """the donor-mean lambda_G of one passage against every known passage"""
        out = []
        for ku in kunits:
            lams = [lg.score(q, ku, ref_sentences=ref, r=1,
                             with_details=False).lambda_G for ref in refs]
            out.append(float(np.mean(lams)))
        return out

    for w, (start, toks) in enumerate(wins):
        q = W.rechunk(toks, seg)
        lam = score(q)
        share = float(np.mean([j < nchron for j in range(start, start + L)]))
        rows.append(dict(kind="window", w=w, start=start,
                         chronicle_share=round(share, 3),
                         text=("kaiserchronik" if share == 1 else
                               "alexanderlied" if share == 0 else "boundary"),
                         lambda_G=round(float(np.mean(lam)), 4),
                         per_known=[round(x, 4) for x in lam]))
        if (w + 1) % 10 == 0:
            el = time.time() - t0
            print(f"    {w + 1}/{len(wins)} windows, {el:.0f}s "
                  f"({el / (w + 1):.1f}s per window)", flush=True)
    for n, st, q in cohort:
        lam = score(q)
        rows.append(dict(kind="cohort", text=n, start=st,
                         author=bank[n]["author"],
                         lambda_G=round(float(np.mean(lam)), 4),
                         per_known=[round(x, 4) for x in lam]))

    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / f"{args.stream}__L{L}__w{seg}__{args.known}.jsonl"
    with p.open("w", encoding="utf-8") as fh:
        fh.write(json.dumps(dict(kind="meta", stream=args.stream, L=L, seg=seg,
                                 step=args.step, known=args.known,
                                 known_from=kfrom, donors=donor_names,
                                 boundary=nchron, n_chron=nchron,
                                 n_stream=len(stream))) + "\n")
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    # what the profile says
    ch = np.array([r["lambda_G"] for r in rows
                   if r.get("text") == "kaiserchronik"])
    al = np.array([r["lambda_G"] for r in rows
                   if r.get("text") == "alexanderlied"])
    co = np.array([r["lambda_G"] for r in rows if r["kind"] == "cohort"])
    med = float(np.median(ch))
    sd = 1.4826 * float(np.median(np.abs(ch - med))) or float(np.std(ch))
    auc = float(np.mean([(a < c) + 0.5 * (a == c)
                         for c in ch for a in al])) if len(al) else float("nan")
    print(f"\n  chronicle windows {len(ch)}: median {med:.1f}, "
          f"robust sd {sd:.1f}, 5th percentile {np.percentile(ch, 5):.1f}")
    print(f"  Alexanderlied windows {len(al)}: median {np.median(al):.1f} "
          f"= {(np.median(al) - med) / sd:+.1f} robust sd from the chronicle's")
    print(f"  cohort passages {len(co)}: median {np.median(co):.1f} "
          f"= {(np.median(co) - med) / sd:+.1f} robust sd")
    print(f"  chronicle against Alexanderlied windows: AUC {auc:.3f}; "
          f"{np.mean(al < np.percentile(ch, 5)):.0%} of the Alexanderlied "
          f"windows fall below the chronicle's 5th percentile")
    # where the profile falls most steeply, against where the boundary is. This
    # is the profile read as it would be read on an anonymous text, with no
    # label: the steepest fall between consecutive windows is the only thing an
    # examiner could point at.
    ws = [r for r in rows if r["kind"] == "window"]
    mid = np.array([r["start"] + L / 2 for r in ws])
    fall = int(np.argmin(np.diff([r["lambda_G"] for r in ws])))
    print(f"  the steepest fall lies at token {mid[fall]:,.0f}, "
          f"{abs(mid[fall] - nchron):,.0f} tokens from the boundary "
          f"({abs(mid[fall] - nchron) / L:.2f} of a window)")
    print(f"  wrote {p.relative_to(ROOT)} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
