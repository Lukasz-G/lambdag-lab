# Textual reuse in the Kaiserchronik, located and masked.
#
# A rolling grammar-model profile across the Kaiserchronik would mark any passage
# the chronicle shares verbatim with another work, whatever its style: the
# Annolied's Caesar block stands in the Kaiserchronik word for word, and a spike
# there is text reuse and not authorship. The profile is therefore computed only
# after the reused spans are masked, and this locates them.
#
# WHAT THE REUSE LOOKS LIKE, measured before this was written. It is NOT a copied
# block. The longest run the chronicle shares verbatim with the Annolied is 17
# masked tokens, about three verses, on either stream; an earlier version of this
# script required runs of thirty and consequently found nothing at all. The
# relation is DISTRIBUTED VERBAL ECHO, and at k=8 it is unmistakable: of 107 texts
# of the bank at 2,000 tokens or more, the Annolied shares 0.0135 of its 8-grams
# with the Vorau witness against 0.0039 for the next text, and its 6-gram share
# stands 4.2 standard deviations above the mean. So the instrument is DENSITY at a
# length that formula cannot reach, never span length.
#
# WHAT IS COMPARED. The bank's masked stream, in the bank's own coordinates, so
# that the mask applies directly to what the verifier reads and no position has to
# be mapped between streams. Middle High German verse is formulaic, so some
# sharing is expected between any two texts of the form; the threshold is
# therefore not guessed but measured, as the distribution of matched-token rate
# over EVERY other text of the bank, and a text counts as a partner only where it
# stands clear of that distribution.
#
# A LOCALISATION PROFILE accompanies each partner, because where the echo falls is
# the philological result: the Annolied's lies in the first twentieth of the
# chronicle, seventeen times the average density, which is where the Roman
# material stands; the Rolandslied's lies at the end.
#
# Other Kaiserchronik witnesses are excluded from the comparison: shared text
# between two witnesses of one work is transmission, not reuse.
#
# VALIDATION. The Annolied block is documented, so it is ground truth for the
# detector: if the Annolied does not come out among the partners of the long
# witnesses, the detector is wrong and the parameters need saying so.
#
#   python experiments/build_kc_reuse_mask.py
#   python experiments/build_kc_reuse_mask.py --k 12 --min-span 40 --stream lemma
#
# Output:
#   medieval/kc_reuse/{witness}.json   spans and masked unit indices per witness
#   medieval/kc_reuse/summary.tsv      witness x partner, with the baseline

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import run_witness_cells as W  # noqa: E402

OUT = ROOT / "medieval" / "kc_reuse"


def is_kc(name, bank):
    """Every witness of the chronicle, whichever edition it came from.

    The reference corpus's fragments are named rem__M121* and their normalised
    work strings read `kaiserchronik_a_fragment_w` and the like, so a test on the
    end of the string misses them; an earlier version did, and they came out as
    the top "reuse partners" of every long witness -- which is transmission
    between witnesses of one work, precisely what must be excluded.
    """
    return ("kaiserchronik" in bank[name]["work"]
            or name.startswith("kcd__")
            or re.match(r"rem__M121", name) is not None)


def flat(bank_entry):
    """tokens of a witness, with the bank line each one came from"""
    toks, unit = [], []
    for i, sent in enumerate(bank_entry["sents"]):
        for t in sent:
            toks.append(t)
            unit.append(i)
    return toks, unit


def shingles(toks, k):
    return {tuple(toks[i:i + k]) for i in range(len(toks) - k + 1)}


def matched_positions(toks, k, other):
    """token indices of every k-gram of `toks` that `other` also contains"""
    hit = np.zeros(len(toks), bool)
    for i in range(len(toks) - k + 1):
        if tuple(toks[i:i + k]) in other:
            hit[i:i + k] = True
    return hit


def spans_of(hit, gap, min_span):
    """contiguous runs of matched tokens, joined across short gaps"""
    idx = np.flatnonzero(hit)
    if not len(idx):
        return []
    out, start, prev = [], idx[0], idx[0]
    for i in idx[1:]:
        if i - prev > gap:
            out.append((int(start), int(prev)))
            start = i
        prev = i
    out.append((int(start), int(prev)))
    return [s for s in out if s[1] - s[0] + 1 >= min_span]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stream",
                    choices=["lemma", "surface", "lemma_full"],
                    default="lemma",
                    help="lemma and surface are POSNoise-masked; lemma_full is "
                         "the unmasked lemma stream, on which the shingle has "
                         "to be shorter because its alphabet is the language's")
    ap.add_argument("--k", type=int, default=8,
                    help="shingle length in masked tokens; 8 is where formula "
                         "stops reaching and the Annolied stands clear")
    ap.add_argument("--min-span", type=int, default=8,
                    help="a matched run shorter than this is dropped; at the "
                         "default it keeps the shingle matches themselves, since "
                         "the reuse is echo rather than block")
    ap.add_argument("--gap", type=int, default=4,
                    help="matched runs closer than this are one span")
    ap.add_argument("--z", type=float, default=3.0,
                    help="a partner must stand this many standard deviations "
                         "above the baseline of all comparison texts")
    ap.add_argument("--baseline-min-tokens", type=int, default=2000,
                    help="texts below this are compared but excluded from the "
                         "baseline, their rates being too noisy")
    ap.add_argument("--min-tokens", type=int, default=20000,
                    help="witnesses below this are reported but not masked, "
                         "their profiles being too short to roll")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    bank = W.load(args.stream)
    kc = sorted(n for n in bank if is_kc(n, bank))
    others = sorted(n for n in bank if n not in kc)
    print(f"reuse in the Kaiserchronik, {args.stream} stream, k={args.k}, "
          f"min span {args.min_span}, gap {args.gap}")
    print(f"  {len(kc)} witnesses of the chronicle, "
          f"{len(others)} other texts to compare against")

    # the comparison texts' shingle sets, built once
    other_sh = {}
    for n in others:
        toks, _ = flat(bank[n])
        if len(toks) >= args.k:
            other_sh[n] = shingles(toks, args.k)
    print(f"  shingle sets built for {len(other_sh)} texts")

    # every combination other than the masked stream at eight writes beside the
    # first one rather than over it, so two streams can be compared
    out = OUT if (args.stream, args.k) == ("lemma", 8) \
        else OUT.with_name(f"kc_reuse_{args.stream}_k{args.k}")
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    long_kc = [n for n in kc if bank[n]["ntok"] >= args.min_tokens]
    print(f"  {len(long_kc)} witnesses at or above {args.min_tokens:,} tokens "
          f"carry a rolling profile and are masked\n")

    for n in kc:
        toks, unit = flat(bank[n])
        if len(toks) < args.k:
            continue
        kc_sh = shingles(toks, args.k)
        # TWO coverages, because they answer different questions and only one of
        # them is length-fair. The share of the CHRONICLE matched says how much
        # there is to mask, but it rises with the comparison text's length: a
        # 43,000-token Rolandslied can cover far more of the chronicle than a
        # 4,500-token Annolied ever could, and judging partners on it buried the
        # Annolied under every long text. The share of the PARTNER's own k-grams
        # that occur in the chronicle is normalised by the partner's length, so
        # that is what the partner test uses; the first is what the mask uses.
        rates, cover, hits = {}, {}, {}
        for m, sh in other_sh.items():
            h = matched_positions(toks, args.k, sh)
            rates[m] = h.mean()
            mt, _ = flat(bank[m])
            tot = max(len(mt) - args.k + 1, 1)
            cover[m] = sum(1 for i in range(tot)
                           if tuple(mt[i:i + args.k]) in kc_sh) / tot
            if h.any():
                hits[m] = h
        # the formulaic baseline, on the length-fair coverage: every comparison
        # text long enough for its rate to mean something, and this witness's own
        # rates, so the threshold is local to the witness
        vals = np.array([cover[m] for m in cover
                         if bank[m]["ntok"] >= args.baseline_min_tokens])
        med = float(np.median(vals))
        # A ROBUST spread, because the mean and the standard deviation let a
        # strong partner raise its own bar: the Annolied covers the Vorau witness
        # at 0.0135 against a median of 0.0004, which inflated that witness's sd
        # enough to push the cut to 0.0047 and so exclude the Rolandslied, whose
        # coverage there (0.0033) sits mid-range across the tradition. The
        # apparent absence of the Rolandslied from the oldest witness was
        # therefore an artefact of the threshold, and this removes it.
        mad = float(np.median(np.abs(vals - med)))
        sd = 1.4826 * mad + 1e-12
        mu = med
        cut = med + args.z * sd
        partners = {}
        union = np.zeros(len(toks), bool)
        for m, c in sorted(cover.items(), key=lambda kv: -kv[1]):
            r = rates[m]
            if c <= cut or m not in hits:
                continue
            sp = spans_of(hits[m], args.gap, args.min_span)
            if not sp:
                continue
            # where the echo falls: density over twenty equal slices, since that
            # is the philological result and not a by-product
            nb = 20
            edges = np.linspace(0, len(toks), nb + 1).astype(int)
            dens = [float(hits[m][edges[j]:edges[j + 1]].mean())
                    for j in range(nb)]
            partners[m] = dict(share_of_chronicle=round(r, 5),
                               share_of_partner=round(c, 5),
                               z=round((c - mu) / sd, 2),
                               spans=sp,
                               tokens=int(sum(b - a + 1 for a, b in sp)),
                               profile=[round(d, 5) for d in dens],
                               densest_twentieth=int(np.argmax(dens)) + 1)
            for a, b in sp:
                union[a:b + 1] = True
        masked_units = sorted({unit[i] for i in np.flatnonzero(union)})
        rec = dict(witness=n, tokens=len(toks), units=len(bank[n]["sents"]),
                   k=args.k, min_span=args.min_span, gap=args.gap,
                   baseline_median=round(med, 5),
                   baseline_robust_sd=round(sd, 6),
                   partner_cut=round(cut, 5),
                   masked_tokens=int(union.sum()),
                   masked_share=round(float(union.mean()), 5),
                   masked_units=masked_units,
                   partners=partners)
        (out / f"{n}.json").write_text(
            json.dumps(rec, indent=1, ensure_ascii=False), encoding="utf-8")
        top = sorted(partners.items(), key=lambda kv: -kv[1]["tokens"])[:4]
        rows.append((n, len(toks), round(med, 5), round(cut, 5),
                     int(union.sum()), round(float(union.mean()), 5),
                     len(partners),
                     ";".join(f"{m}:{d['tokens']}" for m, d in top)))
        if n in long_kc:
            print(f"  {n:44} {len(toks):7,} tok  baseline mean {mu:.4f} "
                  f"sd {sd:.4f} cut {cut:.4f}  masked "
                  f"{union.sum():6,} ({union.mean():5.2%})  "
                  f"{len(partners)} partners")
            for m, d in top:
                avg = sum(d["profile"]) / len(d["profile"]) or 1e-9
                bars = "".join("#" if x > 3 * avg else ("+" if x > avg else ".")
                               for x in d["profile"])
                print(f"       {m:38} {d['tokens']:6,} tok  z {d['z']:6.1f}  "
                      f"|{bars}|  densest 1/20 #{d['densest_twentieth']}")

    with (out / "summary.tsv").open("w", encoding="utf-8") as fh:
        fh.write("witness\ttokens\tbaseline_median\tbaseline_quantile\t"
                 "masked_tokens\tmasked_share\tn_partners\ttop_partners\n")
        for r in rows:
            fh.write("\t".join(str(x) for x in r) + "\n")
    print(f"\n  wrote {out.relative_to(ROOT)}/ "
          f"({len(rows)} witnesses + summary.tsv)")

    # the documented case, as a check on the detector
    anno = [n for n in others if "M013" in n]
    print("\n  CHECK -- the Annolied is documented to stand in the chronicle:")
    for n in long_kc:
        rec = json.loads((out / f"{n}.json").read_text(encoding="utf-8"))
        got = {m: d for m, d in rec["partners"].items() if m in anno}
        if got:
            for m, d in got.items():
                print(f"    {n:40} {m}: {d['tokens']:,} tokens, "
                      f"{len(d['spans'])} span(s)")
        else:
            print(f"    {n:40} the Annolied is NOT among its partners")


if __name__ == "__main__":
    main()
