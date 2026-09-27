# The Konrad runs side by side, on a scale they can share.
#
# lambda_G is not comparable from one known text to another: how well a known
# passage models anything at all is a property of that passage, which is why the
# gauge of the journal paper locates every case inside itself. So the runs are
# compared here in units each of them supplies: the median of the run's own
# cohort of passages by other authors, and that cohort's robust spread. A text
# standing "eight cohort deviations above the cohort" means the same thing
# whichever known text produced it, and a raw lambda_G does not.
#
# Read the table by columns. What decides the question is not how high the
# chronicle stands against the Rolandslied, but whether it stands higher there
# than against other poems of the same milieu, and whether other poems of that
# milieu stand as high against the Rolandslied as the chronicle does.
#
#   python experiments/analyze_konrad_profile.py
#
# Output: the table on stdout, and medieval/kc_reuse/konrad_summary.tsv

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SCORES = ROOT / "scores" / "konrad_profile"
OUT = ROOT / "medieval" / "kc_reuse" / "konrad_summary.tsv"

RUNS = [("lemma__L2000__w100__A1__none", "known: Rolandslied (Konrad)"),
        ("lemma__L2000__w100__A1__reuse",
         "known: Rolandslied, shared wording removed"),
        ("lemma__L2000__w100__A1__none__lambrecht",
         "known: Alexanderlied (Lambrecht)"),
        ("lemma__L2000__w100__A1__none__annolied", "known: Annolied"),
        ("lemma__L2000__w100__A1__reuse__annolied",
         "known: Annolied, shared wording removed")]


def spread(v):
    m = float(np.median(v))
    mad = 1.4826 * float(np.median(np.abs(v - m)))
    p90 = (float(np.quantile(v, 0.9)) - m) / 1.2816
    return m, max(mad, p90, 1e-9)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    rows = []
    print(f"{'run':44} {'chronicle':>10} {'best':>7} {'ceiling':>9} "
          f"{'named points':>28} {'above all':>10}")
    print(f"{'':44} {'windows':>10} {'window':>7} {'':>9} {'':>28} "
          f"{'cohort':>10}")
    for name, lab in RUNS:
        p = SCORES / f"{name}.jsonl"
        if not p.exists():
            print(f"{lab:44} -- not run --")
            continue
        rec = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines()
               if l]
        meta = rec[0]
        g = {}
        for r in rec[1:]:
            g.setdefault(r["kind"] if r["kind"] != "named" else r["text"],
                         []).append(r["lambda_G"])
        coh = np.array(g["cohort"])
        m, s = spread(coh)
        def z(v):
            return (float(np.median(v)) - m) / s
        win = np.array(g["window"])
        named = " ".join(f"{k.split('__')[-1]} {z(v):+.1f}"
                         for k, v in g.items()
                         if k not in ("window", "cohort", "ceiling"))
        ceil_ = f"{z(g['ceiling']):+.1f}" if "ceiling" in g else "--"
        print(f"{lab:44} {z(win):+10.1f} {(win.max() - m) / s:+7.1f} "
              f"{ceil_:>9} {named:>28} {np.mean(win > coh.max()):9.0%}")
        rows.append((lab, meta["known"], round(z(win), 2),
                     round((win.max() - m) / s, 2), ceil_, named,
                     round(float(np.mean(win > coh.max())), 3)))
    print("\n  every figure is in robust deviations of that run's own cohort "
          "of passages by other authors")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as fh:
        fh.write("run\tknown\tchronicle_windows_z\tbest_window_z\tceiling_z\t"
                 "named_z\tshare_above_whole_cohort\n")
        for r in rows:
            fh.write("\t".join(str(x) for x in r) + "\n")
    print(f"  wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
