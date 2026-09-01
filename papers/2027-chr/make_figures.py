# Figures for the CHR 2027 paper (part one).
#   fig_lambdag_logic.png   schematic of what lambda_G computes
#   fig_pattern_lists.png   composition of the POSNoise pattern lists per language
#
#   python chr2027/make_figures.py
# Output: chr2027/ach-latex-template-v2/figures/

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

HERE = Path(__file__).resolve().parent
OUT = HERE / "ach-latex-template-v2" / "figures"; OUT.mkdir(parents=True, exist_ok=True)
LISTS = HERE.parent / "posnoise_lists"

# validated categorical slots (normal-vision dE 33.6, deuteranope dE 46.8)
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, MUTED, RULE = "#1a1a19", "#5c5b55", "#d8d7d0"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                     "axes.edgecolor": RULE, "text.color": INK})


def box(ax, x, y, w, h, label, sub=None, fc="#ffffff", ec=RULE, lw=1.2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.02",
                                linewidth=lw, edgecolor=ec, facecolor=fc, zorder=2))
    ax.text(x + w / 2, y + h / 2 + (0.018 if sub else 0), label, ha="center", va="center",
            fontsize=9.5, zorder=3, color=INK)
    if sub:
        ax.text(x + w / 2, y + h / 2 - 0.030, sub, ha="center", va="center",
                fontsize=7.6, color=MUTED, zorder=3)


def arrow(ax, p, q, color=RULE, lw=1.3, style="-|>"):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=11,
                                 linewidth=lw, color=color, zorder=1,
                                 shrinkA=2, shrinkB=2))


def fig_logic():
    """What lambda_G computes: one questioned text scored against a candidate's grammar
    and against r size-matched grammars drawn from a reference population."""
    # Layout follows the top-down flow of the companion DHD figure: inputs, one masking
    # bar spanning all three, the two model families side by side, then the ratio with
    # its two readings (similarity above, typicality below), calibration, and outcomes.
    # Taller canvas with shallower boxes: the connectors need visible length, otherwise
    # the arrowheads sit almost on the box edges and the flow reads as cramped.
    fig, ax = plt.subplots(figsize=(7.4, 7.8))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    L, M, R = 0.02, 0.355, 0.69           # three columns
    W = 0.29

    box(ax, L, 0.912, W, 0.058, "$D_A$", "texts of the known author", fc="#eaf1fb", ec=BLUE)
    box(ax, M, 0.912, W, 0.058, "$D_U$", "questioned text", fc="#eaf1fb", ec=BLUE)
    box(ax, R, 0.912, W, 0.058, "$D_{\\mathrm{ref}}$", "reference corpus", fc="#eaf1fb", ec=BLUE)

    box(ax, L, 0.782, 0.96, 0.050,
        "POSNoise masking: content words out, function words and punctuation kept",
        fc="#f2f1ea")

    box(ax, L, 0.632, W, 0.062, "grammar model $G_A$", "$n$-gram model of the author",
        fc="#e8f5ee", ec="#1baf7a", lw=1.6)
    box(ax, M, 0.632, W, 0.062, "sentences of $D_U$", "to be scored", fc="#f2f1ea")
    box(ax, R, 0.632, W, 0.062, "models $G_1 \\ldots G_r$", "samples of the population",
        fc="#fdf0e6", ec=ORANGE, lw=1.6)

    ax.add_patch(FancyBboxPatch((L, 0.355), 0.96, 0.205,
                                boxstyle="round,pad=0.012,rounding_size=0.02",
                                linewidth=1.2, edgecolor=INK, facecolor="#ffffff", zorder=2))
    ax.text(0.50, 0.532, "similarity: how familiar is this language use?",
            ha="center", fontsize=8, color=MUTED, style="italic", zorder=3)
    ax.text(0.50, 0.455,
            "$\\lambda_G=\\frac{1}{r}\\sum_j \\log_{10}\\ "
            "\\frac{P(D_U\\mid G_A)}{P(D_U\\mid G_j)}$",
            ha="center", va="center", fontsize=15, zorder=3, color=INK)
    ax.text(0.50, 0.378, "typicality: how ordinary is it in the population?",
            ha="center", fontsize=8, color=MUTED, style="italic", zorder=3)

    box(ax, 0.215, 0.245, 0.57, 0.052, "calibration to the likelihood ratio $\\Lambda_G$",
        fc="#f0eaf8", ec="#4a3aa7", lw=1.6)
    ax.text(0.50, 0.232, "logistic regression — or, with no calibration data,\n"
            "$\\lambda_G/\\sqrt{N(Q)}$   and   $\\lambda_G\\cdot V_1(Q)/N(Q)$",
            ha="center", va="top", fontsize=7.4, color=MUTED, zorder=4)

    box(ax, 0.10, 0.028, 0.35, 0.058, "$\\Lambda_G > 0$ : same author")
    box(ax, 0.55, 0.028, 0.35, 0.058, "$\\Lambda_G < 0$ : different author")

    for x in (L, M, R):
        arrow(ax, (x + W / 2, 0.908), (x + W / 2, 0.836))
        arrow(ax, (x + W / 2, 0.778), (x + W / 2, 0.698))
    arrow(ax, (L + W / 2, 0.628), (L + W / 2, 0.584), color="#1baf7a")
    arrow(ax, (M + W / 2, 0.628), (M + W / 2, 0.584))
    arrow(ax, (R + W / 2, 0.628), (R + W / 2, 0.584), color=ORANGE)
    arrow(ax, (0.50, 0.351), (0.50, 0.301))
    arrow(ax, (0.50, 0.178), (0.50, 0.135), color="#4a3aa7")
    arrow(ax, (0.275, 0.135), (0.275, 0.090))
    arrow(ax, (0.725, 0.135), (0.725, 0.090))
    ax.plot([0.275, 0.725], [0.135, 0.135], color=RULE, lw=1.3, zorder=1)
    

    fig.savefig(OUT / "fig_lambdag_logic.png", dpi=300, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)


RED_SA, BLUE_DA = "#d62728", "#1f77b4"    # same-author / different-author, as in the heat maps


def _gauss(x, mu, sd):
    import numpy as np
    return np.exp(-0.5 * ((x - mu) / sd) ** 2) / (sd * np.sqrt(2 * np.pi))


def fig_calibration():
    """Why the multiplicative corrections cannot calibrate a mislocated score, and
    what the bank calibration does about it: three panels, raw -> scaled -> located."""
    import numpy as np
    fig, axes = plt.subplots(1, 3, figsize=(10.6, 3.1), sharey=False)
    x = np.linspace(-6, 10, 600)

    def panel(ax, mu_d, sd_d, mu_s, sd_s, title, note):
        ax.plot(x, _gauss(x, mu_d, sd_d), color=BLUE_DA, lw=1.8)
        ax.plot(x, _gauss(x, mu_s, sd_s), color=RED_SA, lw=1.8)
        ax.fill_between(x, 0, _gauss(x, mu_d, sd_d), where=x > 0,
                        color=BLUE_DA, alpha=0.25)
        ax.axvline(0, color=INK, lw=1.0, ls=":")
        ax.text(0, ax.get_ylim()[1] * 0.02, " LR = 1", fontsize=7, color=INK,
                ha="left", va="bottom")
        ax.set_title(title, fontsize=9.5)
        ax.text(0.5, -0.24, note, transform=ax.transAxes, ha="center", va="top",
                fontsize=7.6, color=MUTED, wrap=True)
        ax.set_yticks([]); ax.set_xticks([])
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(RULE)

    # P1: raw lambda_G -- same shapes, but the whole axis scales with N(Q)
    panel(axes[0], 1.4, 2.4, 5.2, 2.4,
          "raw $\\lambda_G$",
          "spread and location both grow with $N(Q)$:\n"
          "magnitudes read as absurdly strong LRs")
    axes[0].annotate("", xy=(9.4, 0.055), xytext=(5.2, 0.055),
                     arrowprops=dict(arrowstyle="<->", color=MUTED, lw=1.0))
    axes[0].text(7.3, 0.066, "grows $\\propto N$", fontsize=7, color=MUTED,
                 ha="center")

    # P2: after /sqrt(N) -- spread normalised, location bias b*sqrt(N) untouched
    panel(axes[1], 1.6, 1.0, 4.6, 1.1,
          "after $\\lambda_G/\\sqrt{N(Q)}$",
          "scale is fixed, location is not: the different-author\n"
          "mass right of 0 is reported as evidence FOR same")
    axes[1].annotate("", xy=(1.6, 0.20), xytext=(0.0, 0.20),
                     arrowprops=dict(arrowstyle="->", color=INK, lw=1.2))
    axes[1].text(2.05, 0.20, "offset $b\\sqrt{N}$", fontsize=7.4, color=INK,
                 ha="left", va="center")

    # P3: bank calibration -- shift + scale estimated from pseudo-cases
    panel(axes[2], -2.2, 1.0, 2.6, 1.1,
          "after bank calibration",
          "shift and scale fitted on pseudo-cases the bank's\n"
          "author metadata supplies free: boundary restored to 0")

    handles = [plt.Line2D([], [], color=RED_SA, lw=1.8, label="same-author cases"),
               plt.Line2D([], [], color=BLUE_DA, lw=1.8, label="different-author cases")]
    fig.legend(handles=handles, loc="upper right", frameon=False, fontsize=8,
               bbox_to_anchor=(0.995, 1.02))
    fig.tight_layout(rect=[0, 0.05, 1, 0.97])
    fig.savefig(OUT / "fig_calibration_logic.png", dpi=300, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)


def fig_pattern_table():
    """DHD-style categorised table of pattern-list entries across six languages.
    Every cell is VERIFIED present in the shipped list before rendering; a dash
    means the item is genuinely absent (and the caption says why)."""
    langs = {"English": "En_v2.1", "German": "De_v3.0", "French": "Fr_v1.0",
             "Lithuanian": "Lt_v1.0", "Polish": "Pl_v1.0", "Hungarian": "Hu_v1.0"}
    lists = {}
    for name, ver in langs.items():
        lists[name] = {l.strip().lower() for l in
                       open(LISTS / f"POSNoise_PatternList_{ver}.txt",
                            encoding="utf-8") if l.strip()}
    rows = [
        ("auxiliary `to be'",
         ["be", "sein", "être", "būti", "być", "van"]),
        ("modal `can'",
         ["can", "können", "pouvoir", "galėti", "móc", "tud"]),
        ("modal `must'",
         ["must", "müssen", "devoir", "turėti", "musieć", "kell"]),
        ("degree `very'",
         ["very", "sehr", "très", "labai", "bardzo", "nagyon"]),
        ("frequency `always'",
         ["always", "immer", "toujours", "visada", "zawsze", "mindig"]),
        ("frequency `never'",
         ["never", "nie", "jamais", "niekada", "nigdy", None]),
        ("negation `not'",
         ["not", "nicht", "pas", "ne", "nie", "nem"]),
        ("quantifier `all'",
         ["all", "alle", "tout", "visas", "wszyscy", "minden"]),
        ("conjunct `however'",
         ["however", "jedoch", "cependant", "tačiau", "jednak", "azonban"]),
        ("multiword unit",
         ["as well as", "vor allem", "ainsi que", "iš viso",
          "przede wszystkim", None]),
    ]
    names = list(langs)
    for cat, cells in rows:                      # verify before rendering
        for name, w in zip(names, cells):
            if w is not None:
                assert w.lower() in lists[name], f"{w!r} not in {name} list"

    fig, ax = plt.subplots(figsize=(9.6, 4.4), layout="constrained")
    ax.axis("off")
    cell_text = [[w if w is not None else "—" for w in cells] for _, cells in rows]
    tbl = ax.table(cellText=cell_text,
                   rowLabels=[cat for cat, _ in rows],
                   colLabels=names, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(9); tbl.scale(1, 1.55)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor("#cfcfe0")
        if r == 0:
            cell.set_facecolor("#40466e")
            cell.set_text_props(color="white", fontweight="bold")
        elif c == -1:
            cell.set_facecolor("#eceaf4")
            cell.set_text_props(ha="right", fontsize=8.5)
        else:
            cell.set_facecolor("#f4f4fb" if r % 2 else "white")
            cell.set_text_props(style="italic")
    ax.set_title("POSNoise pattern lists: one validated exemption per category and language\n"
                 "(English and German: upstream surface-form lists; the rest: UD-mined, lemma-based)",
                 fontsize=10, pad=14)
    fig.savefig(OUT / "fig_pattern_table.png", dpi=300, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)


def list_stats():
    want = [("English", "En"), ("German", "De"), ("French", "Fr"),
            ("Polish", "Pl"), ("Lithuanian", "Lt"), ("Hungarian", "Hu")]
    rows = []
    for name, tag in want:
        f = sorted(LISTS.glob(f"POSNoise_PatternList_{tag}_*.txt"))
        if not f:
            continue
        entries = [l.strip() for l in open(f[-1], encoding="utf-8") if l.strip()]
        multi = sum(1 for e in entries if " " in e)
        rows.append((name, len(entries) - multi, multi, f[-1].name))
    return rows


def fig_lists():
    """Composition of the pattern lists: single-word vs multiword functional units."""
    rows = list_stats()
    rows.sort(key=lambda r: r[1] + r[2])
    names = [r[0] for r in rows]
    single = [r[1] for r in rows]
    multi = [r[2] for r in rows]
    y = range(len(rows))

    fig, ax = plt.subplots(figsize=(6.6, 3.0))
    h = 0.55
    ax.barh(y, single, height=h, color=BLUE, label="single-word entries", zorder=3)
    # 2px surface gap between stacked segments
    ax.barh(y, multi, height=h, left=[s + 6 for s in single], color=ORANGE,
            label="multiword units", zorder=3)

    for i, (n, s, m, _) in enumerate(rows):
        ax.text(s + m + 34, i, f"{s + m:,}", va="center", fontsize=8.5, color=INK)

    ax.set_yticks(list(y)); ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("entries in the POSNoise pattern list", fontsize=8.5, color=MUTED)
    ax.tick_params(axis="x", labelsize=8, colors=MUTED)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color=RULE, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(RULE)
    ax.set_xlim(0, max(s + m for _, s, m, _ in rows) * 1.16)
    ax.legend(frameon=False, fontsize=8, loc="lower right", ncol=1)

    ax.text(0, -0.30, "English and German are the upstream surface-form lists; the rest are "
            "lemma-based and UD-mined,\nwhich is why they are an order of magnitude smaller "
            "while covering the same inflectional ground.",
            transform=ax.transAxes, fontsize=7.2, color=MUTED, va="top")

    fig.savefig(OUT / "fig_pattern_lists.png", dpi=300, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    return rows


if __name__ == "__main__":
    fig_logic()
    fig_calibration()
    fig_pattern_table()
    rows = fig_lists()
    for n, s, m, f in rows:
        print(f"{n:12s} {s + m:5d} entries ({m:4d} multiword)  <- {f}")
    print("wrote ->", OUT)
