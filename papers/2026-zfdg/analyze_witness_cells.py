# The four cells read out: does the masked stream keep the author and lose
# the scribe?
#
# Reads scores/witness/{stream}__L{L}__w{seg}.jsonl and reports, per stream
# and evidence length, the discrimination under each scribe condition --
# same-scribe cases (same author against different author, one hand) and
# different-scribe cases (two hands) -- with the statistic the mean over
# donors, and the calibrated cost of a logistic calibrator fitted in five
# folds grouped by the known passage's author identity, with its floor.
# The scribe toll is then the drop from the lemma stream to the surface
# stream, and from same-scribe to different-scribe, stated per length.
#
#   python experiments/analyze_witness_cells.py [--seg 100] [--tag __dq]
#
# With --tag the tagged run (an arm differing from the baseline in one
# option, e.g. donors drawn from the questioned text's period) is read out
# and set beside the baseline case by case, since both score the same cases.
#
# Writes medieval/tables/tab_witness_cells.tex.

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
from lambdag import LambdaGCalibrator, PAVCalibrator, cllr, cllr_min  # noqa: E402

SCORES = ROOT / "scores" / "witness"
TAB = ROOT / "medieval" / "tables" / "tab_witness_cells.tex"
LENGTHS = [100, 400, 500, 1000, 2000, 2080, 5000]
# the rhyme streams carry one token per verse, so their lengths are verse
# counts and 400 of them is the same stretch of text as 2,080 masked tokens
STREAMS = ("lemma", "surface", "rhyme", "rime")


def calibrated(x, y, groups):
    """the cost under two calibrations, and the floor: a logistic map, which is
    unbounded and so can be ruined by one case beyond the training range, and a
    monotone map bounded by what a calibration set of this size can support"""
    x = np.asarray(x, float)[:, None]; y = np.asarray(y, int); groups = np.asarray(groups)
    if len(set(y)) < 2 or len(set(groups)) < 5:
        return float("nan"), float("nan"), float("nan")
    out = []
    for cal in (LambdaGCalibrator, PAVCalibrator):
        LR = np.empty(len(y))
        for tr, te in GroupKFold(5).split(x, y, groups):
            if len(set(y[tr])) < 2:
                LR[te] = 0.0; continue
            LR[te] = cal().fit(x[tr], y[tr]).transform(x[te])
        out.append(cllr(LR[y == 1], LR[y == 0]))
    return out[0], out[1], cllr_min(LR[y == 1], LR[y == 0])


def work_of(name):
    p = name.split("__")
    return p[2] if len(p) > 3 else p[1]


# The runner falls back to the undivided donor pool whenever the questioned
# text's period holds fewer than --min-donors eligible witnesses, and it does
# so silently: the score file records no trace of it. So a tagged arm that
# reads identically to its baseline may not be evidence that mirroring the
# period changes nothing -- it may be an arm in which the mirroring never
# happened. The count is recomputed here from the same banks and the same
# eligibility rule the runner uses, and printed beside every delta.
MIN_DONORS = 8
_BANKS = {}


def period_pools(stream, L, seg, rows):
    """per case, how many donors its questioned period actually offered"""
    if stream not in _BANKS:
        from run_witness_cells import load  # noqa: E402  (same eligibility rule)
        _BANKS[stream] = {n: (b["author"], b["period"], b["ntok"])
                          for n, b in load(stream).items()}
    banks = _BANKS[stream]
    sizes = []
    for r in rows:
        excl = {r["known_author"], r["quest_author"]}
        sizes.append(sum(1 for a, p, nt in banks.values()
                         if nt >= L + seg and a not in excl and p == r["quest_period"]))
    return np.asarray(sizes)


def readout(rows):
    """per scribe condition: n same, n diff, AUC, Cllr, Cllr_min; the
    different-scribe condition also split by what the same-author side is,
    another witness of the same work or another work by the same author"""
    out = {}
    # A witness may be the work of SEVERAL scribes, so two passages of one
    # manuscript are in one hand only where that manuscript is documented to be
    # single-scribal. The converse is definitive: another witness is always
    # another hand. The same-scribe row is therefore reported split --
    #   same witness            hand status unknown unless documented
    #   same scribe, other witness   the codices merged by hand, which are the
    #                                documented ones (the Ambraser is Hans Ried
    #                                throughout, the Vorau codex one hand)
    # -- and the aggregate above them is kept only so the two can be seen to
    # differ. Never quote the aggregate as a same-hand result.
    conds = [("same scribe as labelled", "SA-SS", "DA-SS", None),
             ("  same witness (hand unknown)", "SA-SS", "DA-SS", "onewit"),
             ("  same scribe, other witness", "SA-SS", "DA-SS", "otherwit"),
             ("different scribe", "SA-DS", "DA-DS", None),
             ("  same work, other witness", "SA-DS", "DA-DS", "same"),
             ("  other work, same author", "SA-DS", "DA-DS", "other")]
    for cond, same_cell, diff_cell, split in conds:
        sel = [r for r in rows if r["cell"] in (same_cell, diff_cell)]
        if split == "same":
            sel = [r for r in sel if r["cell"] == diff_cell or work_of(r["known"]) == work_of(r["quest"])]
        elif split == "other":
            sel = [r for r in sel if r["cell"] == diff_cell or work_of(r["known"]) != work_of(r["quest"])]
        elif split == "onewit":
            sel = [r for r in sel if r["cell"] == diff_cell or r["known"] == r["quest"]]
        elif split == "otherwit":
            sel = [r for r in sel if r["cell"] == diff_cell or r["known"] != r["quest"]]
        y = [1 if r["cell"] == same_cell else 0 for r in sel]
        x = [r["lambda_G"] for r in sel]
        g = [r["known_author"] for r in sel]
        if len(set(y)) < 2:
            out[cond] = (sum(y), len(y) - sum(y)) + (float("nan"),) * 4; continue
        auc = roc_auc_score(y, x)
        c, cp, cm = calibrated(x, y, g)
        out[cond] = (sum(y), len(y) - sum(y), auc, c, cp, cm)
    return out


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    seg = int(sys.argv[sys.argv.index("--seg") + 1]) if "--seg" in sys.argv else 100
    tag = sys.argv[sys.argv.index("--tag") + 1] if "--tag" in sys.argv else ""
    if tag:
        print("%-8s %6s %-28s %6s %6s %7s  (baseline -> %s, same cases)" % ("stream", "L", "condition", "base", "tagged", "delta", tag))
        for stream in STREAMS:
            for L in LENGTHS:
                fb, ft = SCORES / f"{stream}__L{L}__w{seg}.jsonl", SCORES / f"{stream}__L{L}__w{seg}{tag}.jsonl"
                if not (fb.exists() and ft.exists()):
                    continue
                key = lambda r: (r["cell"], r["known"], r["kw"], r["quest"], r["qw"])
                base = {key(r): r for r in (json.loads(l) for l in fb.open(encoding="utf-8"))}
                tagged = {key(r): r for r in (json.loads(l) for l in ft.open(encoding="utf-8"))}
                common = sorted(set(base) & set(tagged))
                rb, rt = [base[k] for k in common], [tagged[k] for k in common]
                pools = period_pools(stream, L, seg, rt)
                fell = int((pools < MIN_DONORS).sum())
                for (cond, vb), (_, vt) in zip(readout(rb).items(), readout(rt).items()):
                    print("%-8s %6d %-28s %6.3f %6.3f %+7.3f   (%d common cases)" % (stream, L, cond, vb[2], vt[2], vt[2] - vb[2], len(common)))
                print("%-8s %6d %-28s %d of %d cases (%.0f%%) had fewer than %d donors in the "
                      "questioned period and were scored on the undivided pool; median pool %.0f"
                      % (stream, L, "  -- mirroring NOT applied:", fell, len(rt),
                         100.0 * fell / max(1, len(rt)), MIN_DONORS, float(np.median(pools))))
        return
    table = []
    print("%-8s %6s %-17s %5s %5s %6s %7s %7s %7s"
          % ("stream", "L", "condition", "same", "diff", "AUC", "Cllr", "bounded", "Cllrmin"))
    for stream in STREAMS:
        for L in LENGTHS:
            fn = SCORES / f"{stream}__L{L}__w{seg}.jsonl"
            if not fn.exists():
                continue
            rows = [json.loads(l) for l in fn.open(encoding="utf-8")]
            for cond, (ns, nd, auc, c, cp, cm) in readout(rows).items():
                table.append((stream, L, cond, ns, nd, auc, c, cp, cm))
                print("%-8s %6d %-17s %5d %5d %6.3f %7.3f %7.3f %7.3f"
                      % (stream, L, cond, ns, nd, auc, c, cp, cm))
    # per-cell mean scores, to see where the toll sits
    print("\nmean lambda_G per cell (same-author cells should sit above, different-author below):")
    for stream in STREAMS:
        for L in LENGTHS:
            fn = SCORES / f"{stream}__L{L}__w{seg}.jsonl"
            if not fn.exists():
                continue
            rows = [json.loads(l) for l in fn.open(encoding="utf-8")]
            by = defaultdict(list)
            for r in rows:
                by[r["cell"]].append(r["lambda_G"])
            print("  %-8s %6d  " % (stream, L) + "  ".join("%s %+7.1f (n=%d)" % (c, np.mean(v), len(v)) for c, v in sorted(by.items())))
    TAB.parent.mkdir(parents=True, exist_ok=True)
    with TAB.open("w", encoding="utf-8") as f:
        f.write("% Autogenerated by experiments/analyze_witness_cells.py -- do not edit.\n")
        f.write("\\begin{tabular}{llrrrrrrr}\n\\toprule\n")
        f.write("stream & tokens a side & scribe & same & different & AUC & $C_{\\mathrm{llr}}$ & "
                "$C_{\\mathrm{llr}}$ bounded & $C_{\\mathrm{llr}}^{\\min}$ \\\\\n\\midrule\n")
        last = None
        for stream, L, cond, ns, nd, auc, c, cp, cm in table:
            if last is not None and stream != last:
                f.write("\\midrule\n")
            f.write("%s & %d & %s & %d & %d & %.3f & %.3f & %.3f & %.3f \\\\\n"
                    % (stream, L, cond, ns, nd, auc, c, cp, cm))
            last = stream
        f.write("\\bottomrule\n\\end{tabular}\n")
    print("->", TAB)


if __name__ == "__main__":
    main()
