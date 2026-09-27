# Kestemont's claim, stated in this corpus: are frequent RHYME words a steadier
# basis for authorship than frequent words anywhere?
#
# The argument (Kestemont 2012, Digital Philology 1; Kestemont, Daelemans &
# Sandra 2012, Journal of Quantitative Linguistics 19) is that a medieval
# copyist inserts and deletes inconspicuous words as he goes, which corrupts the
# frequency profile a stylometrist measures, while the words at the rhyme are
# held in place by the verse. The features are therefore the HIGHLY FREQUENT
# rhyme words -- frequent, so that they are grammatical rather than thematic,
# and well spread through a text -- and the method is a distance between
# frequency profiles.
#
# That is a claim about FREQUENCIES, and it is not the claim this repository
# tested earlier: the positional measurement (experiments/run_rhyme_collation.py)
# asked whether the same word stands at the same place in two copies, which a
# scribe can satisfy while still shifting the rate of `unde` by a tenth.
#
# Four feature sets, the same texts and the same distance for each:
#
#                    all positions        the rhyme position only
#   the scribe's form   surface/all          surface/rhyme
#   the canonical lemma lemma/all            lemma/rhyme
#
# Surfaces are what a stylometrist actually holds and carry the scribe's
# spelling; lemmas have had that taken out by the corpus's own convention. Both
# are run because the difference between them is the size of the spelling
# problem, which is what the rhyme features are supposed to sidestep.
#
# TWO DESIGNS, because the corpus will not support one of them alone.
#
# --design work is the strict one. A witness is a profile; a pair of witnesses of
# one work is the SCRIBE floor and is set aside; the remaining pairs are labelled
# by author, and the summary is the AUC of same author against different author.
# It is the design the philological argument wants, and it collapses on this
# corpus: only four identities have a second work at all, and cross-work with a
# different hand is structurally Hartmann's, because only his works survive in
# many manuscripts. Its same-author class is therefore one man.
#
# --design author drops the hand from the design and buys identities with it.
# Every identity contributes several disjoint samples of the same size, taken
# from different witnesses of that identity where it has them and from different
# stretches of its one witness where it has not; a pair of samples is labelled by
# author and nothing is set aside. That turns one identity into twenty, at the
# price that most positive pairs now share a work and a hand, so the number is an
# upper bound on what a stranger's text would give. The two kinds of positive
# pair are counted and their distances reported separately, so the size of that
# price can be read off rather than argued about.
#
#   python experiments/run_rhyme_mfw.py --design work   --top 100 --tokens 2000
#   python experiments/run_rhyme_mfw.py --design author --top 100 --tokens 1000
#
# Output: medieval/rhyme_mfw.tsv, medieval/rhyme_mfw_authors.tsv, and the table
# on stdout

import argparse
import csv
import random
import sys
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import run_witness_cells as W  # noqa: E402
sys.path.insert(0, str(ROOT / "medieval"))
from rime import rime as rime_of  # noqa: E402
OUT = ROOT / "medieval" / "rhyme_mfw.tsv"
OUT_AUTHORS = ROOT / "medieval" / "rhyme_mfw_authors.tsv"
SOURCES = ("kcd", "hvad", "parzival", "ambraser")


def load_texts(stream, min_verses, bank="mhg_witnesses", sources=SOURCES):
    """every text of the bank as (identity, verses)

    The witness banks are keyed by manuscript and the normalised-edition bank by
    work; both are read the same way, and which one is read decides what a
    same-author pair can be.
    """
    base = ROOT / "masked" / f"{bank}_{stream}"
    with (base / "MANIFEST.tsv").open(encoding="utf-8") as fh:
        man = [r for r in csv.DictReader(fh, delimiter="\t")
               if (not sources or r["file"].startswith(sources))
               and not r["file"].endswith("__reg")]
    out = {}
    for r in man:
        lines = [l.split("\t") for l in
                 (base / "bank" / f"{r['file']}.tsv").read_text(
                     encoding="utf-8").splitlines() if l]
        if len(lines) < min_verses:
            continue
        # the manifest's author field says "anon." for every anonymous work,
        # and taking it at its word makes the Nibelungenlied the same author as
        # the Kaiserchronik. The programme's own identity function maps an
        # anonymous text to its work instead.
        out[r["file"]] = (dict(year=int(r["date"]) if r["date"].isdigit()
                               else 0,
                               work=W.norm(r["work"]), author=W.author_id(r),
                              source=r["file"].split("__")[0],
                              named=r["author"] not in ("anon.", "-", "")
                              and "anonym" not in r["author"].lower()),
                          lines)
    return out


def stream_tokens(lines, which):
    """the tokens a feature set reads out of a stretch of verses

    all    every word of the verse
    rhyme  the word standing at the rhyme
    rime   that word's rime, from its last stressed vowel -- the part a copyist
           keeps when he does not keep the word. It is read off the text's own
           forms, since lemmas do not rhyme.
    """
    if which == "all":
        return [t for u in lines for t in u]
    last = [u[-1] for u in lines if u and u[-1]]
    return [rime_of(w) or w for w in last] if which == "rime" else last


def profile(tokens, vocab):
    c = Counter(tokens)
    n = max(sum(c.values()), 1)
    return np.array([c.get(w, 0) / n for w in vocab])


def delta(P):
    """Burrows's delta: the mean absolute difference of z-scored frequencies"""
    z = (P - P.mean(0)) / (P.std(0) + 1e-12)
    return np.abs(z[:, None, :] - z[None, :, :]).mean(-1)


def cut(tokens, ntok, rng):
    if ntok and len(tokens) > ntok:
        i = rng.randrange(0, len(tokens) - ntok)
        return tokens[i:i + ntok]
    return tokens


# ---------------------------------------------------------------- design work

def design_work(texts, which, top, ntok, repeats, seed):
    names = sorted(texts)
    pool = Counter()
    for n in names:
        pool.update(stream_tokens(texts[n][1], which))
    vocab = [w for w, _ in pool.most_common(top)]
    aucs, floors, ids = [], [], Counter()
    for rep in range(repeats):
        rng = random.Random(seed + rep)
        P = np.array([profile(cut(stream_tokens(texts[n][1], which), ntok, rng),
                              vocab) for n in names])
        D = delta(P)
        scribe, y, s = [], [], []
        for i, j in combinations(range(len(names)), 2):
            a, b = texts[names[i]][0], texts[names[j]][0]
            if a["work"] == b["work"]:
                scribe.append(D[i, j])
                continue
            same = a["author"] == b["author"]
            y.append(1 if same else 0)
            s.append(-D[i, j])
            if same and rep == 0:
                ids[a["author"]] += 1
        aucs.append(roc_auc_score(y, s) if len(set(y)) == 2 else float("nan"))
        floors.append(float(np.mean(scribe)))
    return dict(auc=float(np.mean(aucs)), auc_sd=float(np.std(aucs)),
                floor=float(np.mean(floors)), n_ids=len(ids),
                n_pos=sum(ids.values()), ids=ids)


# -------------------------------------------------------------- design author

def pick_samples(texts, ident_names, nverse, per_author, rng):
    """disjoint verse blocks for one identity, spread over its witnesses first

    A block is `nverse` verses long, which is what the rhyme feature set needs
    to reach `nverse` tokens. Witnesses are taken in turn before any witness is
    read twice, so an identity with several manuscripts is sampled ACROSS them.
    """
    ws = sorted(ident_names, key=lambda n: -len(texts[n][1]))
    free = {}
    for n in ws:
        k = len(texts[n][1]) // nverse
        free[n] = rng.sample(range(k), k)
    out = []
    while len(out) < per_author and any(free.values()):
        for n in ws:
            if free[n]:
                out.append((n, free[n].pop() * nverse))
                if len(out) >= per_author:
                    break
    return out


def shared_vocab(texts, top):
    """the frequent words of the whole text, whatever position they stand in

    The control the comparison needs: the rhyme vocabulary is content-bearing
    where the all-position vocabulary is not, so a rhyme profile built on its own
    top words differs from an all-position profile in TWO ways at once. Fixing
    the vocabulary here leaves only one -- where the words are counted.
    """
    pool = Counter()
    for n in texts:
        pool.update(stream_tokens(texts[n][1], "all"))
    return [w for w, _ in pool.most_common(top)]


def design_author(texts, which, top, nverse, per_author, repeats, seed,
                  cross_work=False, vocab_fixed=None):
    by = defaultdict(list)
    for n, (m, _) in texts.items():
        by[m["author"]].append(n)
    aucs, xaucs, dpos_in, dpos_x, dneg = [], [], [], [], []
    n_ids, n_pos, n_x, who = 0, 0, 0, []
    for rep in range(repeats):
        rng = random.Random(seed + rep)
        units = []
        for a in sorted(by):
            s = pick_samples(texts, by[a], nverse, per_author, rng)
            if len(s) >= 2:
                units += [(a, n, o) for n, o in s]
        pool = Counter()
        toks = []
        for a, n, o in units:
            t = stream_tokens(texts[n][1][o:o + nverse], which)[:nverse]
            toks.append(t)
            pool.update(t)
        vocab = vocab_fixed or [w for w, _ in pool.most_common(top)]
        P = np.array([profile(t, vocab) for t in toks])
        D = delta(P)
        y, s, xy, xs, xw, seen = [], [], [], [], 0, set()
        for i, j in combinations(range(len(units)), 2):
            (ai, ni, _), (aj, nj, _) = units[i], units[j]
            same = ai == aj
            # with works enough to choose from, a same-author pair out of ONE
            # work is not evidence about an author, so it is dropped rather
            # than counted on either side
            if cross_work and same and                     texts[ni][0]["work"] == texts[nj][0]["work"]:
                continue
            y.append(1 if same else 0)
            s.append(-D[i, j])
            if same:
                (dpos_x if ni != nj else dpos_in).append(D[i, j])
                xw += ni != nj
            else:
                dneg.append(D[i, j])
            # the same question asked of the positives that cross a manuscript:
            # a same-author pair read out of one witness shares a hand and a
            # work, and only these do not
            if not same or ni != nj:
                xy.append(1 if same else 0)
                xs.append(-D[i, j])
            seen.add(ai)
        aucs.append(roc_auc_score(y, s) if len(set(y)) == 2 else float("nan"))
        xaucs.append(roc_auc_score(xy, xs) if len(set(xy)) == 2
                     else float("nan"))
        n_ids, n_pos, n_x, who = len(seen), sum(y), xw, sorted(seen)
    return dict(auc=float(np.mean(aucs)), auc_sd=float(np.std(aucs)),
                auc_cross=float(np.mean(xaucs)),
                auc_cross_sd=float(np.std(xaucs)), who=who,
                n_ids=n_ids, n_pos=n_pos, n_cross=n_x,
                d_in=float(np.mean(dpos_in)) if dpos_in else float("nan"),
                d_cross=float(np.mean(dpos_x)) if dpos_x else float("nan"),
                d_neg=float(np.mean(dneg)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--design", choices=["work", "author"], default="work",
                    help="work: a witness is a profile, same-work pairs are the "
                         "scribe floor; author: several disjoint samples per "
                         "identity, the hand no longer in the design")
    ap.add_argument("--top", type=int, default=100,
                    help="how many of the most frequent words make the profile")
    ap.add_argument("--tokens", type=int, default=0,
                    help="cut every text to this many tokens before counting, "
                         "so that the rhyme profile is not judged on a fifth of "
                         "the evidence; 0 uses whole texts. In design author "
                         "this is also the sample's length in verses, and an "
                         "identity too short for two samples is left out.")
    ap.add_argument("--per-author", type=int, default=3,
                    help="design author: how many samples one identity gives")
    ap.add_argument("--min-verses", type=int, default=2000)
    ap.add_argument("--named-only", action="store_true",
                    help="keep only the works of NAMED authors. An anonymous "
                         "work is not an author: the identity function has to "
                         "fall back on the work's own title for it, so a pair "
                         "of samples from one anonymous poem is a same-TEXT "
                         "pair wearing an author's label. Seven identities "
                         "survive here (Wolfram, Hartmann, Stricker, Herrand "
                         "von Wildonie, Ulrich von Liechtenstein, Wernher der "
                         "Gartenaere, Priester Wernher), and the shortest of "
                         "them sets the sample length.")
    ap.add_argument("--repeats", type=int, default=1,
                    help="how many times the token window is re-drawn; one "
                         "draw is one sample and says nothing about its own "
                         "variability")
    ap.add_argument("--bank", default="mhg_witnesses",
                    help="mhg_witnesses for the manuscripts, mhdbdb for the "
                         "normalised editions, where a text is a WORK and there "
                         "is no manuscript dimension at all")
    ap.add_argument("--years", default="",
                    help="keep only the texts written inside this band, as "
                         "MIN-MAX. The language stage moves across the corpus's "
                         "three centuries, so a run over all of it may be "
                         "separating authors by their German; a narrow band is "
                         "the control for that.")
    ap.add_argument("--vocab", choices=["own", "shared"], default="own",
                    help="own: each feature set is profiled on its OWN most "
                         "frequent words, which is Kestemont's design; shared: "
                         "every feature set is profiled on the frequent words of "
                         "the whole text, so that the only difference left "
                         "between them is the position the words are counted in")
    ap.add_argument("--gate", action="store_true",
                    help="drop the texts whose printing scheme makes the last "
                         "word of a line something other than a rhyme word -- "
                         "an edition that prints the two halves of a long line "
                         "separately, and a text with no rhyme at all. The "
                         "verdicts come from experiments/run_rhyme_gate.py.")
    ap.add_argument("--cross-work-only", action="store_true",
                    help="design author: a same-author pair must come from two "
                         "different works, so that no positive is a pair of "
                         "samples out of one poem")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    rows = []

    for stream in ("surface_full", "lemma_full"):
        texts = load_texts(stream, args.min_verses, args.bank,
                           () if args.bank != "mhg_witnesses" else SOURCES)
        if args.named_only:
            texts = {n: t for n, t in texts.items() if t[0]["named"]}
        if args.years:
            lo, hi = (int(x) for x in args.years.split("-"))
            texts = {n: t for n, t in texts.items()
                     if lo <= t[0]["year"] <= hi}
        if args.gate:
            g = ROOT / "medieval" / (
                "rhyme_gate.tsv" if args.bank == "mhg_witnesses"
                else f"rhyme_gate_{args.bank}.tsv")
            with g.open(encoding="utf-8") as fh:
                bad = {r["file"] for r in csv.DictReader(fh, delimiter="	")
                       if r["verdict"] in ("HALF-LINES", "NO RHYME FOUND")}
            texts = {n: t for n, t in texts.items() if n not in bad}
            print(f"  {len(bad)} texts dropped by the printing-scheme gate")
        print(f"{stream}: {len(texts)} witnesses of {args.min_verses}+ verses"
              + (", named authors only" if args.named_only else ""))
        fixed = shared_vocab(texts, args.top) if args.vocab == "shared"             else None
        # the rime is the sound of the text's own form, so it is read off the
        # surface stream only; a lemma's ending is the dictionary's, not the
        # poet's
        # a rime is not a word, so a vocabulary of the text's frequent WORDS
        # cannot profile it: under the shared-vocabulary control the rime set
        # would be scored on features it never contains, and it is left out
        for which in (("all", "rhyme", "rime")
                      if stream == "surface_full" and args.vocab == "own"
                      else ("all", "rhyme")):
            if args.design == "work":
                r = design_work(texts, which, args.top, args.tokens,
                                args.repeats, args.seed)
                print(f"  {which:6} top {args.top:3}: scribe floor "
                      f"{r['floor']:5.3f}, AUC {r['auc']:5.3f} "
                      f"+- {r['auc_sd']:.3f} over {args.repeats} draws; "
                      f"same-author pairs {r['n_pos']} from {r['n_ids']} "
                      f"identities {dict(r['ids'].most_common(4))}")
                rows.append((stream, which, args.top, args.tokens,
                             round(r["floor"], 4), r["n_ids"], r["n_pos"],
                             round(r["auc"], 4), round(r["auc_sd"], 4)))
            else:
                r = design_author(texts, which, args.top, args.tokens,
                                  args.per_author, args.repeats, args.seed,
                                  args.cross_work_only, fixed)
                print(f"  {which:6} top {args.top:3}: AUC {r['auc']:5.3f} "
                      f"+- {r['auc_sd']:.3f}, across witnesses only "
                      f"{r['auc_cross']:5.3f} +- {r['auc_cross_sd']:.3f}; "
                      f"{r['n_ids']} identities, {r['n_pos']} same-author "
                      f"pairs ({r['n_cross']} across witnesses); delta "
                      f"{r['d_in']:.3f} same witness / {r['d_cross']:.3f} "
                      f"across / {r['d_neg']:.3f} different author")
                if which == "all" and stream == "surface_full":
                    print(f"    identities: {', '.join(r['who'])}")
                rows.append((stream, which, args.top, args.tokens,
                             args.per_author, r["n_ids"], r["n_pos"],
                             r["n_cross"], round(r["d_in"], 4),
                             round(r["d_cross"], 4), round(r["d_neg"], 4),
                             round(r["auc"], 4), round(r["auc_sd"], 4),
                             round(r["auc_cross"], 4),
                             round(r["auc_cross_sd"], 4)))

    out = OUT if args.design == "work" else OUT_AUTHORS
    head = ("stream\tpositions\ttop\ttokens\tscribe_floor\tn_identities\t"
            "n_same_pairs\tauc\tauc_sd\n" if args.design == "work" else
            "stream\tpositions\ttop\tverses\tper_author\tn_identities\t"
            "n_same_pairs\tn_cross_witness\td_same_witness\td_cross_witness\t"
            "d_diff_author\tauc\tauc_sd\tauc_cross\tauc_cross_sd\n")
    out.parent.mkdir(parents=True, exist_ok=True)
    new = not out.exists() or out.stat().st_size == 0
    with out.open("a", encoding="utf-8") as fh:
        if new:
            fh.write(head)
        for r in rows:
            fh.write("\t".join(str(x) for x in r) + "\n")
    print(f"\n  wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
