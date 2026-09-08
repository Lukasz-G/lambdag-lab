# Figure: what cohort studentisation does, and what it does not do.
#
# The statistic. LambdaG fits r reference grammars and keeps only the mean of
# their scores. Writing lam_j for the questioned text's score under the
# candidate's grammar against reference grammar j, LambdaG reports mean(lam_j);
# studentisation reports mean(lam_j) / sd(lam_j), the sample standard deviation
# across the same r models. No external constant, and the cohort is already
# computed and discarded, so the spread is free.
#
# The figure is a paired slope chart because the claim is PAIRED: every dataset
# is scored both ways on the same cases, so what matters is that each line moves,
# not where the clouds sit. Two panels, because the point is a contrast between
# two quantities -- C_llr moves and AUC does not -- which is what identifies the
# effect as calibration rather than discrimination. Both panels share the same
# two series colours; a second y-axis is never used.
#
#   python experiments/make_studentisation_fig.py
#
# Numbers are recomputed from the score files, never transcribed.

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from lambdag import cllr, cllr_min  # noqa: E402

IN = HERE / "scores"
OUT = HERE.parent / "figures"

# Two hues chosen for colour-vision separation, not decoration; text stays in
# ink colours so identity is carried by the marks alone.
C_GRID, C_DON, C_STUD = "#0173B2", "#949494", "#DE8F05"
INK, MUTED, GRID = "#22252A", "#6B7280", "#E5E7EB"


def load():
    """Recompute both arms from the score files, matched case by case.

    The baseline is the PUBLISHED grid score -- lambda_G with the pooled r=30
    reference, corrected by sqrt(N) -- read from the grid run and joined on case
    id, not the per-donor symmetrised variant, which is a different estimator and
    a different (worse) number. Comparing against the wrong baseline would
    overstate the gain.
    """
    from sklearn.metrics import roc_auc_score
    rows = {}
    for f in sorted(IN.glob("*__routebreal__L1200.jsonl")):
        ds = f.name.split("__")[0]
        gfn = IN / f"{ds}__kn__sent__L1200.jsonl"
        if not gfn.exists():
            continue
        grid = {r["id"]: r for r in
                (json.loads(l) for l in open(gfn, encoding="utf-8") if l.strip())}
        g, ss, s, y, nd = [], [], [], [], []
        for line in open(f, encoding="utf-8"):
            if not line.strip():
                continue
            d = json.loads(line)
            if d["id"] not in grid:
                continue
            assert grid[d["id"]]["label"] == d["label"], f"{ds} id {d['id']}"
            lj = np.asarray(d["lam_j"], dtype=float)
            g.append(float(grid[d["id"]]["sqrt"]))
            ss.append(float(np.mean(lj)) / np.sqrt(d["n_q"]))
            s.append(float(np.mean(lj) / (np.std(lj, ddof=1) + 1e-9)))
            y.append(bool(d["label"]))
            nd.append(len(lj))
        if not g:
            continue
        g, ss, s, y = (np.array(g), np.array(ss), np.array(s),
                       np.array(y, bool))
        rows[ds.replace("_novels", "")] = {
            "cllr_grid": cllr(g[y], g[~y]), "cllr_don": cllr(ss[y], ss[~y]),
            "cllr_stud": cllr(s[y], s[~y]),
            "auc_grid": roc_auc_score(y, g), "auc_don": roc_auc_score(y, ss),
            "auc_stud": roc_auc_score(y, s),
            "n": len(g), "R": int(np.median(nd))}
    return rows


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = load()
    if not rows:
        print(f"no score files under {IN}")
        return
    order = sorted(rows, key=lambda k: rows[k]["cllr_grid"] - rows[k]["cllr_stud"])
    ypos = np.arange(len(order))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True,
                             gridspec_kw={"width_ratios": [1.35, 1]})
    fig.patch.set_facecolor("white")

    # Three arms, because two changes are at stake and must not be conflated:
    # the reference models change (pooled r=30 -> one grammar per donor author)
    # AND the normaliser changes (sqrt(N) -> the cohort's own sd). The middle
    # arm holds the donors fixed and shows they alone make matters worse, so the
    # gain belongs to the spread rather than to the donor design.
    for ax, (keys, title, sub) in zip(axes, [
            (("cllr_grid", "cllr_don", "cllr_stud"),
             "$C_{llr}$  (cost — lower is better)",
             "studentisation moves this"),
            (("auc_grid", "auc_don", "auc_stud"), "AUC  (discrimination)",
             "and leaves this alone")]):
        for i, k in enumerate(order):
            xs = [rows[k][q] for q in keys]
            ax.plot([min(xs), max(xs)], [i, i], color=GRID, lw=1.4, zorder=1,
                    solid_capstyle="round")
            for x, c in zip(xs, (C_GRID, C_DON, C_STUD)):
                ax.plot([x], [i], "o", ms=7.5, color=c, zorder=3,
                        markeredgecolor="white", markeredgewidth=1.8)
        ax.set_title(title, fontsize=11, color=INK, pad=12, loc="left")
        ax.text(0, 1.015, sub, transform=ax.transAxes, fontsize=9,
                color=MUTED, va="bottom")
        ax.grid(axis="x", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for sp in ("top", "right", "left"):
            ax.spines[sp].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
        ax.tick_params(colors=MUTED, length=0, labelsize=9)

    # C_llr = 1 is the forensic threshold: above it the system costs more than
    # reporting nothing at all. Two datasets stay above it after studentisation.
    axes[0].axvline(1.0, color=INK, lw=1, ls=(0, (4, 3)), alpha=0.55, zorder=2)
    axes[0].text(1.03, -0.62, "→ worse than reporting nothing",
                 fontsize=8.5, color=INK, alpha=0.75, va="bottom")

    # The AUC panel is drawn on the FULL range from chance to perfect. Left to
    # autoscale it spans 0.75-0.90 and magnifies movements of a thousandth into
    # visible travel, which would contradict the very claim the panel makes.
    axes[1].set_xlim(0.5, 1.0)
    axes[1].set_xticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    axes[1].text(0.505, -0.62, "chance", fontsize=8.5, color=MUTED,
                 va="bottom")

    axes[0].set_yticks(ypos)
    axes[0].set_yticklabels(order, fontsize=9.5, color=INK)
    axes[0].set_ylim(-0.7, len(order) - 0.3)

    n = len(order)
    imp = sum(rows[k]["cllr_stud"] < rows[k]["cllr_grid"] for k in order)
    med_g = np.median([rows[k]["cllr_grid"] for k in order])
    med_d = np.median([rows[k]["cllr_don"] for k in order])
    med_s = np.median([rows[k]["cllr_stud"] for k in order])
    Rmed = int(np.median([rows[k]["R"] for k in order]))
    ncase = sum(rows[k]["n"] for k in order)
    fig.suptitle("Dividing by the reference cohort's spread", x=0.008, y=0.99,
                 ha="left", fontsize=13.5, color=INK, weight="semibold")
    fig.text(0.008, 0.915,
             f"$t$ = mean$_j(\\lambda_j)\\,/\\,$sd$_j(\\lambda_j)$, sample sd over "
             f"$R\\approx${Rmed} per-author donor grammars, no external constant"
             f"   ·   symmetric $L$=1200 tokens on BOTH sides"
             f"   ·   {ncase:,} evaluation cases"
             f"   ·   median $C_{{llr}}$ {med_g:.3f} → {med_d:.3f} → "
             f"{med_s:.3f}, improving on the published score in {imp}/{n}",
             ha="left", fontsize=9.5, color=MUTED)

    handles = [
        plt.Line2D([], [], marker="o", ls="", ms=8, color=C_GRID,
                   label="published: pooled $r$=30 reference, $\\div\\sqrt{N}$"),
        plt.Line2D([], [], marker="o", ls="", ms=8, color=C_DON,
                   label="per-author donors, $\\div\\sqrt{N}$  (donors alone: worse)"),
        plt.Line2D([], [], marker="o", ls="", ms=8, color=C_STUD,
                   label="per-author donors, $\\div$ cohort sd")]
    fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.008, -0.02),
               ncol=3, frameon=False, fontsize=9, labelcolor=INK)

    fig.tight_layout(rect=[0, 0.06, 1, 0.87])
    OUT.mkdir(exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"studentisation.{ext}", dpi=200,
                    facecolor="white", bbox_inches="tight")
    print(f"{imp}/{n} improved; median C_llr {med_g:.3f} -> {med_s:.3f}")
    print(f"wrote {OUT / 'studentisation.png'}")


if __name__ == "__main__":
    main()
