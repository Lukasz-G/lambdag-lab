# Three PNG exhibits for the Nini email, computed from the real score files.
#   1 fig_entrenchment.png   author scatter: self-consistency vs outsider band
#   2 fig_meter.png          the mismatch meter: 12 arms, matched vs mismatched
#   3 fig_cohort.png         cohort normalisation: real-case Cllr, 8 datasets
#   python experiments/make_email_figs.py

import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
SCORES = HERE / "scores"
OUT = HERE.parent / "journal" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(HERE.parent))
from lambdag import cllr  # noqa: E402

BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, MUTED, GRID, BG = "#33322f", "#6b6a66", "#e6e4e0", "#fcfbf9"

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "text.color": INK, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.axisbelow": True, "font.family": "DejaVu Sans",
})


def fig1():
    within, cross = defaultdict(list), defaultdict(list)
    fn = SCORES / "romanian_novels__kn__withink__L5000.jsonl"
    for line in open(fn, encoding="utf-8"):
        r = json.loads(line)
        (within if r["within"] else cross)[r["known"]].append(
            r["lambda_G"] / r["n_q"])
    auth = [a for a in within if len(within[a]) >= 6 and len(cross[a]) >= 6]
    x = np.array([np.mean(within[a]) for a in auth])
    y = np.array([np.mean(cross[a]) for a in auth])
    from scipy import stats as sps
    rho = sps.spearmanr(x, y).statistic
    b, a0 = np.polyfit(x, y, 1)
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.scatter(x, y, s=55, color=BLUE, alpha=0.85, edgecolor=BG, linewidth=1.5,
               zorder=3)
    xs = np.linspace(x.min(), x.max(), 10)
    ax.plot(xs, b * xs + a0, color=MUTED, linewidth=2, zorder=2)
    ax.set_xlabel("how consistent the author is with himself\n"
                  "(the author's held-out text scored against his own model)")
    ax.set_ylabel("where other authors land\nagainst the same model")
    ax.set_title("The more entrenched the author, the further outsiders fall",
                 loc="left", fontsize=12, color=INK, pad=12)
    ax.text(0.97, 0.95, f"Romanian novels, {len(auth)} authors\n"
            f"rank correlation −{abs(rho):.2f}",
            transform=ax.transAxes, ha="right", va="top", color=MUTED)
    fig.tight_layout()
    fig.savefig(OUT / "fig_entrenchment.png", dpi=160)
    print("fig_entrenchment.png", len(auth), "authors, rho", round(rho, 2))


def fig2():
    arms = []
    for fn in sorted(SCORES.glob("*__symref-*__L2000.jsonl")):
        rows = [json.loads(l) for l in open(fn, encoding="utf-8")]
        if not rows:
            continue
        ds, refds = fn.name.replace("__L2000.jsonl", "").split("__symref-")
        c = np.array([r["lambda_G"] / r["n_q"] for r in rows if not r["within"]])
        se = c.std(ddof=1) / np.sqrt(len(c))
        lang = ds.split("_")[0].capitalize()
        if ds == refds:
            label, kind = f"{lang} novels" + (" (poetry)" if "poetree" in ds else ""), 0
            if "poetree" in ds:
                label = f"{lang} poetry"
        else:
            g = {"poetree": "poetry", "dracor": "drama", "novels": "novel"}[
                refds.split("_")[-1]]
            src = {"poetree": "poetry", "dracor": "drama", "novels": "novels"}[
                ds.split("_")[-1]]
            label, kind = f"{lang} {src} ← {g} bank", 1
        arms.append((kind, label, float(c.mean()), 1.96 * se))
    arms.sort(key=lambda t: (t[0], t[2]))
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    for i, (kind, label, m, ci) in enumerate(arms):
        col = BLUE if kind == 0 else ORANGE
        ax.errorbar(m, i, xerr=ci, fmt="o", color=col, markersize=8,
                    capsize=3, elinewidth=1.6, zorder=3)
    ax.axvline(0, color=MUTED, linewidth=1.2, zorder=2)
    ax.set_yticks(range(len(arms)))
    ax.set_yticklabels([a[1] for a in arms], fontsize=9)
    ax.set_xlabel("average per-token score of OTHER authors against the known "
                  "author\n(zero = the reference corpus explains strangers as "
                  "well as strangers do)")
    ax.set_title("The reference-corpus gauge:\nmatched banks sit at zero, "
                 "wrong ones do not", loc="left", fontsize=12, pad=10)
    from matplotlib.lines import Line2D
    ax.legend(handles=[
        Line2D([], [], marker="o", ls="", color=BLUE, label="matched reference corpus"),
        Line2D([], [], marker="o", ls="", color=ORANGE, label="wrong-genre reference corpus")],
        loc="lower right", frameon=False, fontsize=9)
    ax.grid(axis="x")
    ax.grid(False, axis="y")
    fig.tight_layout()
    fig.savefig(OUT / "fig_meter.png", dpi=160)
    print("fig_meter.png", len(arms), "arms")


def fig3():
    rows = []
    for fn in sorted(SCORES.glob("*__routebreal__L1200.jsonl")):
        ds = fn.name.split("__")[0]
        recs = [json.loads(l) for l in open(fn, encoding="utf-8")]
        gfn = SCORES / f"{ds}__kn__sent__L1200.jsonl"
        grid = {r["id"]: r for r in (json.loads(l) for l in open(gfn, encoding="utf-8"))}
        y, g, st = [], [], []
        for r in recs:
            if r["id"] in grid:
                lj = np.asarray(r["lam_j"], dtype=float)
                y.append(r["label"]); g.append(grid[r["id"]]["sqrt"])
                st.append(float(np.mean(lj) / (np.std(lj, ddof=1) + 1e-9)))
        y, g, st = np.array(y), np.array(g), np.array(st)
        rows.append((ds.split("_")[0].capitalize(),
                     float(cllr(g[y == 1], g[y == 0])),
                     float(cllr(st[y == 1], st[y == 0]))))
    rows.sort(key=lambda t: -t[1])
    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    for i, (lang, a, b) in enumerate(rows):
        ax.plot([a, b], [i, i], color=GRID, linewidth=2.5, zorder=2)
        ax.scatter([a], [i], s=60, color=ORANGE, zorder=3)
        ax.scatter([b], [i], s=60, color=BLUE, zorder=3)
    ax.axvline(1.0, color=MUTED, linewidth=1.2, linestyle=":", zorder=1)
    ax.text(1.02, -0.45, "Cllr = 1: as costly as saying nothing",
            color=MUTED, fontsize=8.5, va="top")
    ax.set_ylim(-0.9, len(rows) - 0.5)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] + " novels" for r in rows], fontsize=9)
    ax.set_xlabel("Cllr on the real evaluation cases, 1200-token texts "
                  "(lower is better)")
    ax.set_title("Keeping the reference cohort's spread\nimproves the "
                 "reported evidence in 8 of 8 languages",
                 loc="left", fontsize=12, pad=10)
    from matplotlib.lines import Line2D
    ax.legend(handles=[
        Line2D([], [], marker="o", ls="", color=ORANGE, label="plain λ/√N"),
        Line2D([], [], marker="o", ls="", color=BLUE, label="cohort-normalised")],
        loc="upper right", frameon=False, fontsize=9)
    ax.grid(axis="x")
    ax.grid(False, axis="y")
    fig.tight_layout(rect=[0, 0.11, 1, 1])
    fig.text(0.055, 0.055,
             "The cohort: the ~12 reference authors whose models also score "
             "the questioned text.\nCohort-normalised: the suspect model's "
             "lead over them, divided by how much they disagree\namong "
             "themselves — the case's own yardstick, no external constant.",
             fontsize=8.5, color=MUTED, va="bottom")
    fig.savefig(OUT / "fig_cohort.png", dpi=160)
    print("fig_cohort.png", len(rows), "datasets")


if __name__ == "__main__":
    fig1(); fig2(); fig3()
