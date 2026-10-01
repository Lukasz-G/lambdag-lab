# What the layer of the cohort does to the scale.
#
# The cohort is not a sample of texts but the scale every score is expressed in,
# so its composition is a measurement decision. Two compositions are available
# and neither is free:
#
#   THE WITNESSES are the case's own layer -- a scribe's text, annotated as the
#   chronicle is -- but only fourteen authors survive in copies early enough, and
#   twelve of them are needed for the reference grammars, so one text per author
#   leaves six passages to carry the scale.
#
#   THE EDITIONS add thirteen authors of the poem's own period, one work each,
#   and none of them shares a work with a grammar. What they do not share with
#   the case is the layer: an editor's normalised text against a scribe's.
#
# A run made with --cohort-bank both scores both cohorts against every region, so
# the offset between the layers is a measured quantity. This reads that run and
# reports, per region and pooled: each layer's location and robust spread, the
# offset between them, and what the candidate and the control become under each
# scale. If the offset is a constant it cancels in the comparison that yields the
# verdict; if it varies by region it does not, and the witnesses stay.
#
#   python experiments/compare_cohort_layers.py --run regions__A1__r20__q2000__twolayer.jsonl
#
# Output: printed; no file, because it decides an option rather than a number in
# the paper.

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from calibrate_crosswork import spread  # noqa: E402

SCORES = ROOT / "scores" / "konrad_profile"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="regions__A1__r20__q2000__twolayer.jsonl")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    rows = [json.loads(l) for l in (SCORES / args.run).open(encoding="utf-8")]
    meta, rows = rows[0], rows[1:]
    by_region = defaultdict(lambda: defaultdict(list))
    for r in rows:
        by_region[r["region"]][r["kind"]].append(r["lambda_G"])
    print(f"{args.run}: {meta['regions']} regions of {meta['region_len']:,} "
          f"tokens, {len(meta['donors'])} grammars")
    kinds = sorted({r["kind"] for r in rows})
    print(f"kinds scored: {', '.join(kinds)}")

    print("\n  region   witness cohort      edition cohort     offset   "
          "candidate z         control z")
    print("            median  spread     median  spread    (in wit.  "
          "wit.   ed.        wit.   ed.")
    off, zc_w, zc_e, zt_w, zt_e = [], [], [], [], []
    for ri in sorted(by_region):
        d = by_region[ri]
        mw, sw = spread(np.array(d["cohort_witness"]))
        me, se = spread(np.array(d["cohort_edition"]))
        o = (me - mw) / sw
        cw = float(np.median([(v - mw) / sw for v in d["candidate"]]))
        ce = float(np.median([(v - me) / se for v in d["candidate"]]))
        tw = float(np.median([(v - mw) / sw for v in d["control"]]))
        te = float(np.median([(v - me) / se for v in d["control"]]))
        off.append(o)
        zc_w.append(cw)
        zc_e.append(ce)
        zt_w.append(tw)
        zt_e.append(te)
        print(f"  {ri:5}   {mw:8.1f} {sw:7.1f}   {me:8.1f} {se:7.1f}   "
              f"{o:+7.2f}   {cw:6.2f} {ce:6.2f}     {tw:6.2f} {te:6.2f}")

    print(f"\n  the edition cohort sits {np.median(off):+.2f} witness deviations "
          f"from the witness cohort (range {min(off):+.2f} to {max(off):+.2f}, "
          f"spread of the offset {np.std(off):.2f})")
    print(f"  candidate: median z {np.median(zc_w):.2f} on the witness scale, "
          f"{np.median(zc_e):.2f} on the edition scale")
    print(f"  control:   median z {np.median(zt_w):.2f} on the witness scale, "
          f"{np.median(zt_e):.2f} on the edition scale")
    gw = [c - t for c, t in zip(zc_w, zt_w)]
    ge = [c - t for c, t in zip(zc_e, zt_e)]
    print(f"  the gap that decides the case (candidate - control): "
          f"{np.median(gw):+.2f} witness, {np.median(ge):+.2f} edition; "
          f"the candidate leads in {sum(g > 0 for g in gw)}/{len(gw)} regions "
          f"under the witnesses and {sum(g > 0 for g in ge)}/{len(ge)} under the "
          f"editions")
    if "own" in kinds:
        def own_z(layer):
            out = []
            for ri in sorted(by_region):
                m, sd = spread(np.array(by_region[ri][layer]))
                out.append(float(np.median([(v - m) / sd
                                            for v in by_region[ri]["own"]])))
            return out

        ow, oe = own_z("cohort_witness"), own_z("cohort_edition")
        print(f"  the chronicle's own other regions, the top of the scale: "
              f"{np.median(ow):.2f} witness, {np.median(oe):.2f} edition")


if __name__ == "__main__":
    main()
