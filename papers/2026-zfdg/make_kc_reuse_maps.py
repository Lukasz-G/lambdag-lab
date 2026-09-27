# Textual reuse in the Kaiserchronik, drawn the way text-reuse projects draw it.
#
# An earlier figure plotted the STATISTICS of the detector -- a histogram of
# sharing rates, a distribution of run lengths, a rank correlation. Those answer
# "is the measurement sound", and they do not let a reader see the reuse. The
# projects that made this kind of evidence readable (KITAB and Passim on Arabic
# historical writing, eTRAP on Greek and Latin, impresso on newspapers) draw the
# TEXTS instead, in three forms, and this script produces all three:
#
#   1  kc_reuse_dotplot.png     position against position. One dot per shared
#      8-gram, the chronicle along x and the comparison text along y. A copied
#      passage is a DIAGONAL streak; a text that merely shares the formulaic
#      language of the verse is an even haze; an echo gathered in one part of the
#      chronicle is a vertical band. The first panel is a CONTROL -- two
#      witnesses of the chronicle itself -- so the reader sees what transmission
#      of one text looks like before reading the other two panels.
#
#   2  kc_reuse_connectors.png  two tracks and connectors, as KITAB draws a pair
#      of works: the chronicle above, the comparison text below, one ribbon per
#      shared passage, its width the number of tokens shared. Whether the
#      ribbons cross says whether the shared material keeps its order.
#
#   3  kc_reuse_corpus.png      the corpus view: every comparison text as a
#      column, ordered by the date of its witness, the chronicle's own course
#      down the rows, with a bar below for each text's share and a bar to the
#      right for how much of each part of the chronicle is touched.
#
# Everything is recomputed from the masked corpus, so no figure can drift from
# the numbers. Nothing is fitted: the threshold drawn in figure 3 is the one the
# detector measured from the corpus baseline.
#
#   python medieval/make_kc_reuse_maps.py
#
# Output: medieval/ZfdG_Kaiserchronik/figures/kc_reuse_{dotplot,connectors,corpus}.png

import argparse
import csv
import json
import sys
import textwrap
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import PowerNorm
from matplotlib.gridspec import GridSpec
from matplotlib.patches import PathPatch
from matplotlib.path import Path as MPath

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(HERE))
import build_kc_reuse_mask as M  # noqa: E402  (its is_kc rule, not a second copy)
import run_witness_cells as W  # noqa: E402
from kc_dates import GREY, INK, STYLE, date_label, display_name, year_of  # noqa: E402

A1 = "kcd__anon__kaiserchronik__A1"
CONTROL = "kcd__anon__kaiserchronik__H"
PARTNERS = ["rem__M013O", "rem__M205P"]
OUTDIR = HERE / "ZfdG_Kaiserchronik" / "figures"

# THE STREAM AND THE SHINGLE LENGTH GO TOGETHER and are set in main(). The
# masked stream keeps the function words and writes every content word as a
# placeholder, so its alphabet is small, it repeats, and eight tokens is where
# the formula of the verse stops reaching. The lemma stream keeps every word, so
# its alphabet is the language's and six lemmas already separate; at eight,
# ninety-seven per cent of the corpus shares nothing with the chronicle at all
# and there is no baseline left to measure against.
K = 8
STREAM = "lemma"
UNIT = "masked tokens"
STREAM_LABEL = "POSNoise-masked lemma stream"
REUSE = HERE / "kc_reuse"
SUFFIX = ""


def toks(entry):
    return [t for s in entry["sents"] for t in s]


def grams(t, k=None):
    k = K if k is None else k
    return [tuple(t[i:i + k]) for i in range(len(t) - k + 1)]


def index_of(gs):
    d = defaultdict(list)
    for i, g in enumerate(gs):
        d[g].append(i)
    return d


def pair_positions(gs_a, idx_b):
    """every (position in a, position in b) at which the two share a shingle"""
    xi, yj = [], []
    for i, g in enumerate(gs_a):
        for j in idx_b.get(g, ()):
            xi.append(i)
            yj.append(j)
    return np.array(xi), np.array(yj)


def covered(n, starts, k=None):
    """the tokens covered by a set of matched shingle starts"""
    k = K if k is None else k
    hit = np.zeros(n, bool)
    for i in np.unique(starts):
        hit[i:i + k] = True
    return hit


def runs_of(hit, gap=4):
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
    return out


def ordinal(n):
    return {1: "first", 2: "second", 3: "third"}.get(n, f"{n}th")


def frame(ax, spines=("top", "right")):
    for s in spines:
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=INK, labelsize=9, length=3)
    for s in ax.spines.values():
        s.set_color(GREY)


# ---------------------------------------------------------------- figure 1

def dotplot(fig, gs, col, xi, yj, nx, ny, colour, marker, title, stats, ylab):
    """one position-against-position panel with its two marginal densities"""
    main = fig.add_subplot(gs[1, col])
    top = fig.add_subplot(gs[0, col], sharex=main)
    # a dense panel needs small marks to show its structure, a sparse one needs
    # marks big enough to see at all; sixty dots at the dense panel's size would
    # read as an empty plot
    size, alpha = (2.2, 0.45) if len(xi) > 5000 else (26, 0.85)
    main.scatter(xi / 1000, yj / 1000, s=size, marker=marker, linewidths=0,
                 color=colour, alpha=alpha, rasterized=True)
    main.set_xlim(0, nx / 1000)
    main.set_ylim(0, ny / 1000)
    main.set_xlabel(f"position in the Vorau Kaiserchronik\n"
                    f"(thousands of {UNIT})", fontsize=9.5, color=INK)
    main.set_ylabel(f"position in {ylab}\n(thousands of {UNIT})",
                    fontsize=9.5, color=INK)
    main.grid(alpha=0.12, lw=0.6)
    frame(main)

    nb = 160
    edges = np.linspace(0, nx, nb + 1)
    h, _ = np.histogram(np.unique(xi), bins=edges)
    dens = h / np.diff(edges)
    top.fill_between(edges[:-1] / 1000, dens, step="post", color=colour,
                     alpha=0.75, lw=0)
    top.set_ylim(0, max(dens.max() * 1.15, 1e-9))
    top.set_yticks([])
    top.tick_params(labelbottom=False)
    top.set_ylabel("share\nmatched", fontsize=8, color=GREY, rotation=0,
                   ha="right", va="center", labelpad=6)
    frame(top, ("top", "right", "left"))
    top.set_title(title, fontsize=11, loc="left", color=INK, pad=34,
                  fontweight="bold")
    for row, line in enumerate(stats):
        top.text(0, 1.42 - 0.20 * row, line, transform=top.transAxes,
                 fontsize=8.6, color=GREY, ha="left", va="bottom")
    return main


def fig_dotplot(bank, man):
    a1 = toks(bank[A1])
    ga1 = grams(a1)
    rows = [(CONTROL, GREY, ".", "witness H", "the chronicle itself",
             "transmission of one work"),
            (PARTNERS[0], STYLE[PARTNERS[0]][0], STYLE[PARTNERS[0]][1],
             "Annolied", "an older poem", "reuse of an older poem"),
            (PARTNERS[1], STYLE[PARTNERS[1]][0], STYLE[PARTNERS[1]][1],
             "Rolandslied", "a contemporary poem",
             "reuse of a contemporary poem")]
    fig = plt.figure(figsize=(14.6, 6.4))
    gs = GridSpec(2, 3, height_ratios=[1, 4.4], hspace=0.05, wspace=0.24,
                  left=0.055, right=0.988, top=0.685, bottom=0.185)
    letters = "abc"
    # where the note sits and what it points at; what it SAYS is read off the
    # measurement below, since the same figure is drawn for two streams and the
    # shape of the echo is not the same on both
    at = {CONTROL: ((0.06, 0.80), (0.42, 0.42)),
          PARTNERS[0]: ((0.22, 0.80), (0.045, 0.45)),
          PARTNERS[1]: ((0.06, 0.95), (0.90, 0.72))}
    longest = 0
    for col, (name, colour, marker, short, what, kind) in enumerate(rows):
        t = toks(bank[name])
        xi, yj = pair_positions(ga1, index_of(grams(t)))
        title = f"{letters[col]}   {short} — {what}"
        stats = [f"{display_name(name, man.get(name))} · "
                 f"{len(t):,} {UNIT}",
                 f"{date_label(man[name]['date'])} · "
                 f"{len(np.unique(xi)):,} {K}-grams matched"]
        ax = dotplot(fig, gs, col, xi, yj, len(a1), len(t), colour, marker,
                     title, stats, short)
        hit = covered(len(a1), xi)
        # gap=1 joins adjacent tokens only, so a run is a literally contiguous
        # stretch; anything larger would merge two runs across a gap and report
        # a shared passage longer than the text has
        runs = [b - a + 1 for a, b in runs_of(hit, gap=1)]
        if name != CONTROL:
            longest = max(longest, max(runs, default=0))
        # the twentieth of the chronicle the echo falls densest in, and whether
        # it is gathered there or spread over the whole text
        nb = 20
        edges = np.linspace(0, len(a1), nb + 1).astype(int)
        share = np.array([hit[a:b].sum() for a, b in zip(edges[:-1], edges[1:])],
                         float)
        share /= max(share.sum(), 1)
        best = int(np.argmax(share)) + 1
        ordin = ordinal(best)
        if name == CONTROL:
            txt = "one unbroken diagonal:\nthe same text, copied"
        elif share.max() > 0.3:
            txt = (f"a vertical band: {share.max():.0%} of the echo\n"
                   f"falls in the chronicle's {ordin} twentieth")
        else:
            txt = (f"spread over the chronicle,\ndensest in its "
                   f"{ordin} twentieth ({share.max():.0%})")
        (tx, ty), (px, py) = at[name]
        ax.annotate(txt, xy=(px, py), xytext=(tx, ty),
                    xycoords="axes fraction", textcoords="axes fraction",
                    fontsize=9, color=INK, ha="left", va="center",
                    bbox=dict(fc="white", ec="none", alpha=0.85, pad=2.5),
                    arrowprops=dict(arrowstyle="->", color=GREY, lw=0.9,
                                    shrinkA=4, shrinkB=4))
        print(f"  {name:44} {len(t):7,} tok  {len(xi):8,} shared pairs  "
              f"longest run {max(runs, default=0):3}  densest 1/20 #{best} "
              f"({share.max():.0%})")
    fig.suptitle(f"Textual reuse in the Kaiserchronik, position against "
                 f"position: one dot per {K}-gram shared with the Vorau witness",
                 fontsize=14, x=0.055, ha="left", y=0.975, color=INK,
                 fontweight="bold")
    fig.text(0.055, 0.905,
             f"Middle High German verse, {STREAM_LABEL}, {len(a1):,} {UNIT} in "
             f"the Vorau witness.\nA copied passage draws a diagonal streak, as "
             f"in panel a; shared formulaic language draws an even haze; an "
             f"echo confined to one part\nof the chronicle draws a vertical "
             f"band, as in panel b. The three panels share their horizontal "
             f"axis and each keeps its own vertical one.",
             fontsize=9.6, color=INK, ha="left", va="top", linespacing=1.5)
    fig.text(0.055, 0.075,
             f"Panel a is the control: two witnesses of the chronicle itself, a "
             f"century apart, share whole stretches of text and the diagonal "
             f"runs from end to end.\nPanels b and c carry no diagonal of any "
             f"length — the longest run the chronicle shares with either poem "
             f"is {longest} {UNIT} — so the relation\nis distributed verbal "
             f"echo and not a copied block, and a detector that looks for "
             f"blocks finds nothing at all.",
             fontsize=9.2, color=GREY, ha="left", va="top", linespacing=1.5)
    out = OUTDIR / f"kc_reuse_dotplot{SUFFIX}.png"
    fig.savefig(out, dpi=190)
    plt.close(fig)
    print(f"  wrote {out.relative_to(ROOT)}")


# ---------------------------------------------------------------- figure 2

def ribbon(ax, x1, x2, y1, y2, w, colour):
    verts = [(x1, y1), (x1, (y1 + y2) / 2), (x2, (y1 + y2) / 2), (x2, y2)]
    p = MPath(verts, [MPath.MOVETO, MPath.CURVE4, MPath.CURVE4, MPath.CURVE4])
    ax.add_patch(PathPatch(p, fc="none", ec=colour, lw=w, alpha=0.55,
                           capstyle="round"))


def track(ax, y0, y1, dens, colour, left_label, right_label):
    ax.imshow(dens[None, :], extent=(0, 1, y0, y1), aspect="auto",
              cmap=colour, vmin=0, vmax=max(dens.max(), 1e-9), zorder=3)
    ax.add_patch(plt.Rectangle((0, y0), 1, y1 - y0, fc="none", ec=INK,
                               lw=0.8, zorder=4))
    ax.text(-0.012, (y0 + y1) / 2, left_label, fontsize=10, color=INK,
            ha="right", va="center")
    ax.text(1.012, (y0 + y1) / 2, right_label, fontsize=8.5, color=GREY,
            ha="left", va="center")


def fig_connectors(bank, man):
    a1 = toks(bank[A1])
    ga1 = grams(a1)
    fig, axes = plt.subplots(len(PARTNERS), 1, figsize=(13.4, 7.6))
    cmaps = {"rem__M013O": "Blues", "rem__M205P": "YlOrBr"}
    for ax, name in zip(axes, PARTNERS):
        colour, _, short = STYLE[name]
        t = toks(bank[name])
        xi, yj = pair_positions(ga1, index_of(grams(t)))
        hit_a = covered(len(a1), xi)
        hit_b = covered(len(t), yj)
        nb = 300
        da = np.array([hit_a[a:b].mean() for a, b in
                       zip(*_edges(len(a1), nb))])
        db = np.array([hit_b[a:b].mean() for a, b in
                       zip(*_edges(len(t), nb))])
        ax.set_xlim(-0.16, 1.16)
        ax.set_ylim(0, 1)
        ax.axis("off")
        track(ax, 0.80, 0.90, da, "Greys",
              "Kaiserchronik\nVorau, Cod. 276",
              f"{len(a1):,}\n{UNIT}")
        track(ax, 0.10, 0.20, db, cmaps[name],
              display_name(name, man.get(name)).replace(" (", "\n("),
              f"{len(t):,}\n{UNIT}")
        # one ribbon per shared passage: a run of matched tokens in the
        # chronicle, against the place in the poem the match comes from
        drawn = 0
        for a, b in runs_of(hit_a):
            sel = (xi >= a) & (xi <= b)
            js = np.sort(yj[sel])
            cut = np.flatnonzero(np.diff(js) > 200)
            for grp in np.split(js, cut + 1):
                if not len(grp):
                    continue
                w = 0.5 + 2.6 * (b - a + 1) / 40
                ribbon(ax, (a + b) / 2 / len(a1), float(grp.mean()) / len(t),
                       0.80, 0.20, min(w, 3.4), colour)
                drawn += 1
        ax.text(-0.16, 0.985, f"{short}: {drawn} shared passages, "
                f"{hit_a.sum():,} of the chronicle's tokens "
                f"({hit_a.mean():.2%}), {hit_b.sum():,} of the poem's "
                f"({hit_b.mean():.2%})",
                fontsize=10.5, color=INK, ha="left", va="top",
                fontweight="bold")
        print(f"  {name:44} {drawn:4} ribbons  "
              f"{hit_a.mean():.2%} of the chronicle")
    fig.suptitle("Where the shared passages sit in each text, and whether they "
                 "keep their order",
                 fontsize=13.5, x=0.035, ha="left", y=0.975, color=INK,
                 fontweight="bold")
    fig.text(0.035, 0.028,
             f"Each ribbon joins a run of tokens the Vorau Kaiserchronik shares "
             f"with the poem to the place in the poem it is shared with; the "
             f"ribbon's width grows with the run's length, and the shading "
             f"inside each bar is the density of shared tokens along that text. "
             f"Lengths are drawn to the same width, so the two scales differ. "
             f"Ribbons that run parallel would mean the shared material keeps "
             f"its order; ribbons that cross mean it does not. Measured on the "
             f"{STREAM_LABEL}, matches of {K} tokens.",
             fontsize=9, color=GREY, ha="left", va="bottom", wrap=True)
    fig.subplots_adjust(left=0.135, right=0.905, top=0.915, bottom=0.10,
                        hspace=0.22)
    out = OUTDIR / f"kc_reuse_connectors{SUFFIX}.png"
    fig.savefig(out, dpi=190)
    plt.close(fig)
    print(f"  wrote {out.relative_to(ROOT)}")


def _edges(n, nb):
    e = np.linspace(0, n, nb + 1).astype(int)
    return e[:-1], e[1:]


# ---------------------------------------------------------------- figure 3

def fig_corpus(bank, man):
    rec = json.loads((REUSE / f"{A1}.json").read_text(encoding="utf-8"))
    cut, med = rec["partner_cut"], rec["baseline_median"]
    a1 = toks(bank[A1])
    ga1 = grams(a1)
    a1set = set(ga1)
    others = [m for m in bank
              if not M.is_kc(m, bank) and bank[m]["ntok"] >= 2000]
    order = sorted(others, key=lambda m: (year_of(man[m]["date"]) is None,
                                          year_of(man[m]["date"]) or 0, m))
    nsl = 60
    lo, hi = _edges(len(ga1), nsl)
    mat = np.zeros((nsl, len(order)))
    cov = np.zeros(len(order))
    union = np.zeros(len(ga1), bool)
    print(f"  {len(order)} comparison texts at 2,000 {UNIT} or more")
    for c, m in enumerate(order):
        gs_m = grams(toks(bank[m]))
        sh = set(gs_m)
        hit = np.fromiter((g in sh for g in ga1), bool, len(ga1))
        mat[:, c] = [hit[a:b].mean() for a, b in zip(lo, hi)]
        cov[c] = sum(1 for g in gs_m if g in a1set) / max(len(gs_m), 1)
        if m in rec["partners"]:
            union |= hit
    ranked = np.argsort(-cov)

    # The texts the mask keeps as partners, named over their columns -- NOT
    # every text above the threshold. Clearing the threshold is necessary and
    # not sufficient: a partner must also share a run of min_span tokens, and on
    # the lemma stream, where a single shared shingle already clears a threshold
    # measured from a corpus that mostly shares nothing, that second condition
    # is what separates three partners from forty.
    # The earliest witnesses are also the strongest partners, so their columns
    # stand side by side and the names are staggered with a leader line; how
    # many levels that takes decides how much room the figure leaves above.
    named, levels, lv_of = [], [], {}
    for i, m in enumerate(order):
        if m not in rec["partners"]:
            continue
        named.append(i)
        for lv, last in enumerate(levels):
            if i - last > 11:
                levels[lv] = i
                break
        else:
            levels.append(i)
            lv = len(levels) - 1
        lv_of[i] = lv

    fig = plt.figure(figsize=(14.6, 9.4))
    gs = GridSpec(2, 2, width_ratios=[8.5, 1.5], height_ratios=[7, 2.0],
                  wspace=0.02, hspace=0.035, left=0.075, right=0.985,
                  top=min(0.85, 0.875 - 0.026 * len(levels)), bottom=0.235)
    ax = fig.add_subplot(gs[0, 0])
    show = np.ma.masked_where(mat <= 0, mat)
    vmax = float(np.quantile(mat[mat > 0], 0.995))
    # the strongest slice is twenty times the ordinary one, so a linear ramp
    # would leave every ordinary slice indistinguishable from white
    im = ax.imshow(show, aspect="auto", cmap="Blues", interpolation="nearest",
                   norm=PowerNorm(0.55, vmin=0, vmax=vmax),
                   extent=(-0.5, len(order) - 0.5, len(a1) / 1000, 0))
    ax.set_ylabel(f"the chronicle's course\n(thousands of {UNIT})",
                  fontsize=10, color=INK)
    ax.tick_params(labelbottom=False)
    frame(ax, ())
    for i in named:
        col = STYLE[order[i]][0] if order[i] in STYLE else INK
        ax.annotate(display_name(order[i], man[order[i]]),
                    xy=(i, 0), xytext=(i, -3 - 5.5 * lv_of[i]),
                    color=col, fontsize=8.4, rotation=34, ha="left",
                    va="bottom", annotation_clip=False,
                    arrowprops=dict(arrowstyle="-", lw=0.6, color=GREY,
                                    shrinkA=2, shrinkB=1))

    axr = fig.add_subplot(gs[0, 1], sharey=ax)
    ymid = (lo + hi) / 2 / 1000
    axr.barh(ymid, [union[a:b].mean() for a, b in zip(lo, hi)],
             height=(len(a1) / nsl) / 1000 * 0.9, color="#4a7fb5", lw=0)
    axr.set_xlabel("share of the\nchronicle masked", fontsize=9, color=INK)
    axr.tick_params(labelleft=False, labelsize=8)
    axr.xaxis.set_major_formatter(lambda v, p: f"{v:.0%}")
    frame(axr)
    axr.grid(axis="x", alpha=0.15, lw=0.6)

    axb = fig.add_subplot(gs[1, 0], sharex=ax)
    colours = [STYLE[m][0] if m in STYLE else "#9db8d2" for m in order]
    # a logarithmic axis, because the strongest partner stands ten times above
    # the threshold and a hundred times above the weakest text: on a linear
    # axis the threshold and the baseline both collapse onto the floor
    pos = cov[cov > 0] * 1000
    floor = float(pos.min()) * 0.7 if len(pos) else 0.01
    axb.bar(range(len(order)), np.maximum(cov * 1000, floor), color=colours,
            width=0.82, lw=0, bottom=0)
    axb.set_yscale("log")
    axb.set_ylim(floor, float(pos.max()) * 3 if len(pos) else 30)
    axb.yaxis.set_major_formatter(lambda v, p: f"{v:g}")
    axb.axhline(cut * 1000, color="#b02020", lw=1.1, ls="--")
    axb.axhline(med * 1000, color=GREY, lw=1.0, ls=":")
    axb.text(len(order) - 1, cut * 1000 * 1.12,
             "threshold measured from the corpus ",
             fontsize=8.5, color="#b02020", va="bottom", ha="right")
    axb.text(len(order) - 1, med * 1000 / 2.6, "corpus median ",
             fontsize=8.5, color=GREY, va="bottom", ha="right")
    axb.set_ylabel(f"share of the text's own\n{K}-grams in the chronicle\n"
                   f"(per thousand, log scale)", fontsize=9.5, color=INK)
    axb.set_xlabel(f"the {len(order)} comparison texts, ordered by the date of "
                   f"their witness (earliest at the left; the undated stand "
                   f"last)", fontsize=10, color=INK)
    axb.set_xticks([])
    axb.set_xlim(-0.5, len(order) - 0.5)
    frame(axb)
    nod = sum(1 for m in order if year_of(man[m]["date"]) is None)
    if nod:
        for a in (ax, axb):
            a.axvline(len(order) - nod - 0.5, color=GREY, lw=0.9, ls="-",
                      alpha=0.7)

    cax = fig.add_axes([0.075, 0.055, 0.24, 0.015])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    cb.set_label(f"share of the chronicle's {K}-grams in that slice which also "
                 f"stand in the comparison text", fontsize=8.5, color=INK)
    cb.ax.tick_params(labelsize=8)
    cb.ax.xaxis.set_major_formatter(lambda v, p: f"{v:.1%}")

    fig.suptitle("The whole comparison corpus against the Vorau Kaiserchronik",
                 fontsize=13.5, x=0.075, ha="left", y=0.978, color=INK,
                 fontweight="bold")
    n_above = int((cov > cut).sum())
    ratio = cov[ranked[0]] / cov[ranked[1]]
    sharing = float((cov > 0).mean())
    # where the strongest partner's sharing falls, measured rather than
    # remembered: the figure is drawn for two streams and only one of them was
    # looked at when this sentence was first written
    col = mat[:, ranked[0]]
    tw = int(np.argmax(col) * 20 / nsl) + 1
    conc = float(col[np.argmax(col)] / max(col.sum(), 1e-12))
    fig.text(0.075, 0.175, textwrap.fill(
        f"One column per comparison text, the chronicle running down the rows "
        f"in {nsl} slices, shaded where the two share "
        f"{'an' if K == 8 else 'a'} {K}-gram of the "
        f"{STREAM_LABEL}. {sharing:.0%} of the texts share something, because "
        f"Middle High German verse is formulaic, which is why the threshold is "
        f"measured from the corpus rather than chosen. {n_above} of the "
        f"{len(order)} texts clear it, and the {len(named)} that also share a "
        f"run of at least {rec['min_span']} tokens are named above their "
        f"columns; the strongest of them, "
        f"the {display_name(order[ranked[0]], man[order[ranked[0]]])}, stands "
        f"{ratio:.1f} times above the next, and its sharing is densest in the "
        f"chronicle's {ordinal(tw)} twentieth.", 152),
        fontsize=9.4, color=GREY, ha="left", va="top", linespacing=1.5)
    out = OUTDIR / f"kc_reuse_corpus{SUFFIX}.png"
    fig.savefig(out, dpi=190)
    plt.close(fig)
    print(f"  wrote {out.relative_to(ROOT)}")
    top = [(display_name(order[i], man[order[i]]), round(cov[i] * 1000, 2))
           for i in ranked[:6]]
    print(f"  cut {cut * 1000:.2f} per thousand; strongest: {top}")


def main():
    global K, STREAM, UNIT, STREAM_LABEL, REUSE, SUFFIX
    ap = argparse.ArgumentParser()
    ap.add_argument("--stream", choices=["lemma", "surface", "lemma_full"],
                    default="lemma")
    ap.add_argument("--k", type=int, default=None,
                    help="the shingle length; the default is 8 on a masked "
                         "stream and 6 on the lemma stream")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    STREAM = args.stream
    full = STREAM == "lemma_full"
    K = args.k or (6 if full else 8)
    UNIT = "lemmas" if full else "masked tokens"
    STREAM_LABEL = ("unmasked lemma stream" if full else
                    f"POSNoise-masked {STREAM} stream")
    # the mask built on this stream, for the threshold the corpus figure draws;
    # the first run keeps the plain directory, every other writes beside it
    REUSE = HERE / ("kc_reuse" if (STREAM, K) == ("lemma", 8)
                    else f"kc_reuse_{STREAM}_k{K}")
    SUFFIX = "" if (STREAM, K) == ("lemma", 8) else f"_{STREAM}_k{K}"
    if not (REUSE / f"{A1}.json").exists():
        raise SystemExit(
            f"no mask in {REUSE.relative_to(ROOT)}; run first:\n"
            f"  python experiments/build_kc_reuse_mask.py "
            f"--stream {STREAM} --k {K}")
    OUTDIR.mkdir(parents=True, exist_ok=True)
    print(f"{STREAM_LABEL}, shingles of {K}")
    bank = W.load(STREAM)
    with (ROOT / "masked" / f"mhg_witnesses_{STREAM}" / "MANIFEST.tsv").open(
            encoding="utf-8") as fh:
        man = {r["file"]: r for r in csv.DictReader(fh, delimiter="\t")}
    print("figure 1, position against position")
    fig_dotplot(bank, man)
    print("figure 2, tracks and connectors")
    fig_connectors(bank, man)
    print("figure 3, the corpus view")
    fig_corpus(bank, man)


if __name__ == "__main__":
    main()
