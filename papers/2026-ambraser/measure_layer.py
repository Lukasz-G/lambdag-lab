# What the layer costs: an editor's text against a scribe's, on the same poem.
#
# Both medieval studies here carry the same caveat. The scale that makes scores
# comparable has to come from normalised editions, because the manuscript corpus
# cannot supply enough authors of one period, while the cases themselves are
# manuscript text. The difference between the two -- an editor's normalised
# spelling and punctuation-free regularised text against a scribe's -- has so far
# been estimated by scoring two cohorts against the same known texts. It can be
# measured exactly instead, because five works survive in BOTH banks.
#
# The design is a single comparison with four questioned classes against one known
# text, so that every class is in the same units:
#
#   the same work, the same layer   the known text's own other half. The top of
#                                   the scale, and the thing the layer is to be
#                                   measured against.
#   the same work, the other layer  the edition of the very poem the known text
#                                   is. Same author, same work, same words --
#                                   everything but the layer is held constant, so
#                                   the gap to the class above IS the layer.
#   strangers, the witness layer    other poems of the codex, the scale as the
#                                   Ambraser cases use it.
#   strangers, the edition layer    other poems of the edition bank, the scale as
#                                   the Kaiserchronik case uses it. The gap
#                                   between the two stranger classes is the offset
#                                   a cohort of the wrong layer introduces.
#
#   python experiments/measure_layer.py
#
# Output: printed, plus medieval/kc_reuse/layer_offset.tsv

import csv
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
import run_calibration_cases as C  # noqa: E402
import run_witness_cells as W  # noqa: E402
from lambdag import LambdaG  # noqa: E402

OUT = ROOT / "medieval" / "kc_reuse" / "layer_offset.tsv"
L, SEG = 2000, 100

# the same poem in a scribe's copy and in an editor's text
PAIRS = [
    ("Hartmann, Erec", "ambraser__hartmann_von_aue__erec__d",
     "hartmann_von_aue__ER"),
    ("Hartmann, Iwein", "hvad__hartmann_von_aue__iwein__d_lc",
     "hartmann_von_aue__IW"),
    ("Ulrich von Liechtenstein, Frauenbuch",
     "ambraser__ulrich_von_liechtenstein__frauenbuch__d",
     "ulrich_von_liechtenstein__FB"),
    ("Der Stricker, Pfaffe Amis", "ambraser__stricker__pfaffe_amis__d",
     "der_stricker__AMI"),
    ("Der Mantel", "ambraser__anon__der_mantel__d",
     "heinrich_von_dem_turlin__MAN"),
    # The anonymous epics, now that the editions of them are tagged. These are
    # the poems the Ambraser study is ABOUT, so the layer cost measured on them
    # is the one that applies to its verdicts rather than a cost borrowed from
    # Hartmann. They also carry a second difference the named pairs above carry
    # less of: an editor works from a witness of his own choosing, and Ried's
    # text of the Nibelungenlied is not the text the standard edition prints. The
    # figure these pairs give is therefore the cost of the whole move from this
    # copy to that edition -- hand, orthography and redaction together -- which is
    # what the replication has to cross.
    ("Nibelungenlied", "ambraser__anon__nibelungenlied__d", "anonym__NBB"),
    ("Biterolf und Dietleib", "ambraser__anon__biterolf_und_dietleib__d",
     "anonym__BRF"),
    ("Kudrun", "ambraser__anon__kudrun__d", "anonym__KU"),
    ("Dietrichs Flucht", "ambraser__anon__dietrichs_flucht__d", "anonym__DFL"),
    ("Rabenschlacht", "ambraser__anon__rabenschlacht__d", "anonym__RAB"),
    ("Wolfdietrich A", "ambraser__anon__wolfdietrich_a__d", "anonym__WDA"),
    ("Ortnit", "ambraser__anon__ortnit__d", "anonym__ORT"),
    ("Nibelungenklage", "ambraser__anon__nibelungenklage__d", "anonym__KLA"),
    ("Moriz von Craun", "ambraser__anon__moriz_von_craun__d", "anonym__CR"),
]


def toks(e):
    return [t for s in e["sents"] for t in s]


def cut(t, n=None):
    out = [t[i:i + L] for i in range(0, len(t) - L + 1, L)]
    return out if n is None else out[:n]


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    rng = random.Random(0)
    wit = W.load("lemma", "posnoise")
    ed = C.load_editions("lemma", "1150-1300")
    lg = LambdaG(N=10, r=1, engine="kn", random_state=0)
    rows, t0 = [], time.time()

    for label, wname, ename in PAIRS:
        if wname not in wit or ename not in ed:
            print(f"  skipped {label}: not in both banks")
            continue
        wt, et = toks(wit[wname]), toks(ed[ename])
        klen = len(wt) // 2
        if klen < 4000:
            print(f"  skipped {label}: {len(wt):,} tokens, too short to halve")
            continue
        known, rest = wt[:klen], wt[klen:]
        author = wit[wname]["author"]

        # A text may be a grammar or a stranger only if it is a DIFFERENT poet's.
        # Two things make that test harder than a comparison of author fields.
        # The questioned edition is the same poem and must go whatever its field
        # says; and "anonymous" is not an author, so when the known text is
        # anonymous no other anonymous work can be certified to be somebody
        # else's -- the field would match them all to one another and to nobody.
        # Both were live: the edition bank calls these poems anonym and the codex
        # calls them anon, so the poem's own edition passed the field test and
        # could have been fitted as a grammar for itself.
        anon = "anon" in author
        def other_poet(n):
            return (n != ename and ed[n]["author"] != author
                    and not (anon and "anon" in ed[n]["author"]))

        # the grammars, of the known text's own length, one per edition author and
        # none of them this author
        pool = sorted(n for n in ed if ed[n]["ntok"] >= klen and other_poet(n))
        best = {}
        for n in sorted(pool, key=lambda n: -ed[n]["ntok"]):
            best.setdefault(ed[n]["author"], n)
        dn = sorted(best.values())[:12]
        refs = []
        for n in dn:
            t = toks(ed[n])
            st = rng.randrange(0, len(t) - klen + 1)
            refs.append(W.rechunk(t[st:st + klen], SEG))
        if len(refs) < 6:
            print(f"  skipped {label}: only {len(refs)} grammars at {klen:,}")
            continue

        # BOTH questioned classes must cover the same part of the poem. The known
        # text is its first half, so the edition is taken from ITS second half as
        # well: an edition whose opening is compared with the manuscript's opening
        # is a text against itself -- and for Erec, whose edition rests on this
        # very manuscript, that produced a score five times the same-layer one.
        quest = [("same work, same layer", q) for q in cut(rest, 6)]
        quest += [("same work, the other layer", q)
                  for q in cut(et[len(et) // 2:], 6)]
        # strangers, one passage per author, in each layer
        seen = set()
        for n in sorted(wit):
            a = wit[n]["author"]
            # the codex's anonymous epics are all called anon, so for an anonymous
            # known text they are one key and would collapse to a single stranger;
            # the work's name keys them instead, and the poem itself is excluded
            key = n if anon and "anon" in a else a
            if (not n.startswith("ambraser__") or n == wname or key in seen
                    or (a == author and not anon) or wit[n]["ntok"] < L):
                continue
            seen.add(key)
            quest.append(("strangers, the witness layer", toks(wit[n])[:L]))
        seen = set()
        for n in sorted(ed):
            if not other_poet(n) or ed[n]["ntok"] < L or n in dn:
                continue
            key = n if anon and "anon" in ed[n]["author"] else ed[n]["author"]
            if key in seen:
                continue
            seen.add(key)
            quest.append(("strangers, the edition layer", toks(ed[n])[:L]))

        ku = W.rechunk(known, SEG)
        by = {}
        for kind, q in quest:
            lam = float(np.mean([lg.score(W.rechunk(q, SEG), ku,
                                          ref_sentences=ref, r=1,
                                          with_details=False).lambda_G
                                 for ref in refs]))
            by.setdefault(kind, []).append(lam)
        base = np.array(by["strangers, the witness layer"])
        m = float(np.median(base))
        sd = max(1.4826 * float(np.median(np.abs(base - m))), 1e-9)
        print(f"\n{label}: known {klen:,} masked tokens, {len(refs)} grammars, "
              f"{time.time() - t0:.0f}s")
        for kind in ("same work, same layer", "same work, the other layer",
                     "strangers, the witness layer",
                     "strangers, the edition layer"):
            v = np.array(by.get(kind, []))
            if not len(v):
                continue
            z = (np.median(v) - m) / sd
            print(f"  {kind:30} n={len(v):3}  lambda_G {np.median(v):9,.0f}  "
                  f"z {z:+.2f}")
            rows.append((label, kind, len(v), float(np.median(v)), float(z)))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as fh:
        fh.write("work\tclass\tn\tlambda_G\tz\n")
        for r in rows:
            fh.write(f"{r[0]}\t{r[1]}\t{r[2]}\t{r[3]:.4f}\t{r[4]:.4f}\n")

    def med(kind):
        v = [r[4] for r in rows if r[1] == kind]
        return float(np.median(v)) if v else float("nan")

    print(f"\nover {len({r[0] for r in rows})} works, in deviations of the "
          f"witness-layer strangers:")
    print(f"  the same work in the same layer      {med('same work, same layer'):+.2f}")
    print(f"  the same work in the other layer     {med('same work, the other layer'):+.2f}")
    print(f"  strangers of the edition layer sit   "
          f"{med('strangers, the edition layer'):+.2f} against the witness layer's 0")

    # THE COMPARISON THAT MATTERS, and the one a difference of the two lines above
    # gets wrong. Those two are read against the strangers of the WITNESS layer,
    # so their difference charges the edition class for an offset that every
    # edition text carries, strangers included. A verdict is never read off an
    # absolute score: it is read off the distance between the questioned text and
    # the strangers it is scored beside, and in a design whose cohort is native to
    # the case that distance is all the layer can touch. Referring each class to
    # its own layer's strangers separates the two -- a location the design removes
    # by construction, and a loss of separation, which it cannot.
    per = {}
    for w, kind, _, _, z in rows:
        per.setdefault(w, {})[kind] = z
    own = [d["same work, the other layer"] - d["strangers, the edition layer"]
           for d in per.values()
           if {"same work, the other layer",
               "strangers, the edition layer"} <= d.keys()]
    keep = float(np.median(own)) if own else float("nan")
    print(f"\n  against the strangers of its OWN layer, the same work stands")
    print(f"    in the witness layer                {med('same work, same layer'):+.2f}")
    print(f"    in the edition layer                {keep:+.2f}")
    print(f"  -> the layer costs                    "
          f"{med('same work, same layer') - keep:.2f} deviations of separation,")
    print(f"     on top of an offset of             "
          f"{abs(med('strangers, the edition layer')):.2f} that a native cohort "
          f"removes")
    print(f"  wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
