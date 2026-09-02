# Tempered-sqrt control (defence against the "bounded-range artefact"
# objection to Route B): is the studentised score's Cllr gain genuine
# calibration, or just range compression that Cllr punishes less?
# Label-free temperings of the sqrt score, per matched symmeter arm:
#   clip4    sqrt clipped to [-4, +4] (the t statistic's natural range)
#   zmatch   sqrt rescaled so its pooled (label-free) sd equals t's pooled sd
# If a tempering closes most of the sqrt->stud gap, the gain was compression;
# if stud still wins, the cohort genuinely supplies the slope.
#
#   python experiments/analyze_tempered.py

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SCORES = HERE / "scores"
sys.path.insert(0, str(HERE.parent))
from lambdag import cllr  # noqa: E402


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    med = {k: [] for k in ("sqrt", "clip4", "zmatch", "stud")}
    print(f"{'matched arm':28s} {'Cllr sqrt':>9s} {'clip4':>7s} {'zmatch':>7s} "
          f"{'stud':>7s}")
    for fn in sorted(SCORES.glob("*__symref-*__L2000.jsonl")):
        ds, refds = fn.name.replace("__L2000.jsonl", "").split("__symref-")
        if ds != refds:
            continue
        rows = [json.loads(l) for l in open(fn, encoding="utf-8")]
        if not rows:
            continue
        y = np.array([r["within"] for r in rows])
        sq = np.array([r["lambda_G"] / np.sqrt(r["n_q"]) for r in rows])
        st = np.array([float(np.mean(r["lam_j"]) /
                             (np.std(r["lam_j"], ddof=1) + 1e-9))
                       for r in rows])
        clip4 = np.clip(sq, -4.0, 4.0)
        zmatch = sq * (st.std(ddof=1) / (sq.std(ddof=1) + 1e-9))
        out = {}
        for name, s in (("sqrt", sq), ("clip4", clip4), ("zmatch", zmatch),
                        ("stud", st)):
            out[name] = float(cllr(s[y == 1], s[y == 0]))
            med[name].append(out[name])
        print(f"{ds:28s} {out['sqrt']:9.3f} {out['clip4']:7.3f} "
              f"{out['zmatch']:7.3f} {out['stud']:7.3f}")
    print("\nmedians: " + "  ".join(f"{k} {np.median(v):.3f}"
                                    for k, v in med.items()))
    wins = sum(1 for a, b in zip(med["stud"], med["zmatch"]) if a < b)
    print(f"stud < zmatch in {wins}/{len(med['stud'])} arms; "
          f"stud < clip4 in "
          f"{sum(1 for a, b in zip(med['stud'], med['clip4']) if a < b)}"
          f"/{len(med['stud'])}")


if __name__ == "__main__":
    main()
