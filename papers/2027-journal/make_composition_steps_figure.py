# The composition score, one step per panel, with the real numbers of one
# author (Grillparzer: prose, drama and verse in TextGrid) so that nothing is
# left implicit: masking, windows, counting, standardising, the author's
# genre means and shifts, the shared shift direction, its removal, the score
# and its in-case calibration.
#
# Reads masked/german_tg{prose,drama,verse}all/bank directly.
# Output: journal/figures/composition_steps.png

import re
import sys
import textwrap
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyArrowPatch  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent          # journal/figures -> repository root
GENRES = ["prose", "drama", "verse"]
GCOL = {"prose": "#1f77b4", "drama": "#ff7f0e", "verse": "#2ca02c"}
W = 1000
FLOOR = 200
EXAMPLE = "grillparzer"
STRANGER = "hofmannsthal"
PLACE = {"#": "noun", "§": "proper noun", "Ø": "verb or auxiliary", "@": "adjective",
         "©": "adverb", "µ": "numeral", "$": "symbol", "¥": "other"}
SHOW = {"#": "NOUN", "§": "PROPN", "Ø": "VERB", "@": "ADJ", "©": "ADV",
        "µ": "NUM", "$": "SYM", "¥": "X"}


def namekey(stem):
    """full name, order-free, so the same person matches across the three
    banks even where the file names order the parts differently"""
    return "_".join(sorted(p for p in re.sub(r"^\d+_", "", stem).split("_") if p))


DISPLAY = {}     # name key -> surname (the first part of the file name)


def read_bank(d):
    out = {}
    for f in sorted(d.glob("*.tsv")):
        toks = []
        for line in f.read_text(encoding="utf-8").splitlines():
            if line:
                toks += line.split("\t")
        out[namekey(f.stem)] = toks
        DISPLAY[namekey(f.stem)] = re.sub(r"^\d+_", "", f.stem).split("_")[0]
    return out


banks = {g: read_bank(ROOT / "masked" / f"german_tg{g}all" / "bank") for g in GENRES}
authors = sorted(set.intersection(*(set(b) for b in banks.values())))
EXAMPLE = next(a for a in authors if DISPLAY[a] == EXAMPLE)
STRANGER = next(a for a in authors if DISPLAY[a] == STRANGER)
wins = {(a, g): [banks[g][a][i:i + W] for i in range(0, len(banks[g][a]) - W + 1, W)]
        for g in GENRES for a in authors}
total = Counter()
for ws in wins.values():
    for w in ws:
        total.update(w)
vocab = [s for s, n in total.most_common() if n >= FLOOR]
idx = {s: i for i, s in enumerate(vocab)}
X, A, G = [], [], []
for (a, g), ws in wins.items():
    for w in ws:
        c = Counter(w)
        v = np.zeros(len(vocab))
        for s, n in c.items():
            if s in idx:
                v[idx[s]] = n / W
        X.append(v); A.append(a); G.append(g)
X = np.array(X); A = np.array(A); G = np.array(G)
MU, SD = X.mean(0), X.std(0) + 1e-12
Z = (X - MU) / SD

# shifts and the leading genre direction (all authors, for the picture)
shift = []
for a in authors:
    gm = {g: Z[(A == a) & (G == g)].mean(0) for g in GENRES}
    am = np.mean(list(gm.values()), 0)
    for g in GENRES:
        shift.append((a, g, gm[g] - am))
S = np.array([s[2] for s in shift])
_, sv, vt = np.linalg.svd(S, full_matrices=False)
v1 = vt[0]
if v1[idx["du"]] > 0:      # orient: dialogue end negative, as in the other figure
    v1 = -v1
Zr = Z - np.outer(Z @ v1, v1)          # one direction removed, for the last panels

ex_prose = np.flatnonzero((A == EXAMPLE) & (G == "prose"))
ex_drama = np.flatnonzero((A == EXAMPLE) & (G == "drama"))
st_drama = np.flatnonzero((A == STRANGER) & (G == "drama"))
w0 = ex_prose[0]
window_tokens = wins[(EXAMPLE, "prose")][0]

plt.rcParams.update({"font.size": 9.5, "axes.titlesize": 9.8})
fig, axes = plt.subplots(3, 4, figsize=(21, 16.5))
fig.subplots_adjust(hspace=0.62, wspace=0.32, top=0.93, bottom=0.07, left=0.04, right=0.98)
fig.suptitle("How the composition score is built, step by step. Example: %s (German, TextGrid), "
             "with %d authors who wrote prose, drama and verse" % (DISPLAY[EXAMPLE].title(), len(authors)),
             fontsize=13, fontweight="bold")


def panel(ax, n, title):
    ax.set_title("%d   %s" % (n, title), loc="left", fontweight="bold")


# 1 ---------------------------------------------------------------- mask --- #
ax = axes[0, 0]
panel(ax, 1, "Before step 1: mask the text (POSNoise)")
ax.axis("off")
txt = " ".join(window_tokens[:60])
ax.text(0.0, 0.98, textwrap.fill(txt, 46), va="top", family="monospace", fontsize=8.6)
key = "\n".join("%s  %s" % (k, v) for k, v in PLACE.items())
ax.text(0.0, 0.66, "the first 60 tokens of %s's first prose window" % DISPLAY[EXAMPLE].title(),
        va="top", fontsize=8, color="0.4")
ax.text(0.0, 0.56, "content words become a class placeholder:\n" + key
        + "\n\nfunction words and punctuation stay as they are",
        va="top", fontsize=8.6)

# 2 ------------------------------------------------------------- windows --- #
ax = axes[0, 1]
panel(ax, 2, "Before step 1: cut each bank into %d-token windows" % W)
for i, g in enumerate(GENRES):
    n = len(banks[g][EXAMPLE]); m = len(wins[(EXAMPLE, g)])
    ax.barh(i, n, color=GCOL[g], alpha=0.8, height=0.55)
    if m <= 40:
        for k in range(1, m):
            ax.plot([k * W, k * W], [i - 0.27, i + 0.27], color="white", lw=1.2)
    ax.text(n + 800, i, "%s tokens → %d windows" % (format(n, ","), m), va="center", fontsize=8.6)
ax.set_yticks(range(3)); ax.set_yticklabels(["%s (%s)" % (DISPLAY[EXAMPLE].title(), g) for g in GENRES])
ax.set_xlim(0, max(len(banks[g][EXAMPLE]) for g in GENRES) * 1.55)
ax.invert_yaxis()
ax.set_xlabel("tokens   (every window is one observation; the 17 authors give %d windows)" % len(A))

# 3 --------------------------------------------------------------- count --- #
ax = axes[0, 2]
panel(ax, 3, "Step 1: count the symbols in one window")
c = Counter(window_tokens)
top = c.most_common(18)
ax.barh(range(len(top)), [n for _, n in top], color="0.55")
ax.set_yticks(range(len(top))); ax.set_yticklabels([SHOW.get(s, s) for s, _ in top], fontsize=8.6)
ax.invert_yaxis(); ax.set_xlabel("occurrences in the window")
ax.text(0.98, 0.02, "window: %d tokens, %d distinct symbols\nvocabulary used: the %d symbols that occur\n"
        "≥ %d times in all windows of the panel;\nrarer symbols are ignored"
        % (W, len(c), len(vocab), FLOOR), transform=ax.transAxes, ha="right", va="bottom", fontsize=8.4,
        bbox=dict(boxstyle="round", fc="white", ec="0.7"))

# 4 ----------------------------------------------------------- standardise --- #
ax = axes[0, 3]
panel(ax, 4, "Step 1: a window is a point (standardised rates)")
syms = ["hatte", "war", "der", "und", "#", "du", "?", "ja"]
z_prose = Z[w0]
z_drama = Z[ex_drama[0]]
xs = np.arange(len(syms))
ax.bar(xs - 0.18, [z_prose[idx[s]] for s in syms], width=0.36, color=GCOL["prose"], label="the prose window above")
ax.bar(xs + 0.18, [z_drama[idx[s]] for s in syms], width=0.36, color=GCOL["drama"],
       label="one drama window of the same author")
ax.axhline(0, color="0.3", lw=0.8)
ax.set_xticks(xs); ax.set_xticklabels([SHOW.get(s, s) for s in syms])
ax.set_ylabel("z = (rate − mean rate) / sd of rate")
ax.legend(fontsize=8.2, loc="upper right")
ax.text(0.02, 0.03, "rate = occurrences / %d; mean and sd over all %d windows\n"
        "so a window is a point with %d coordinates" % (W, len(A), len(vocab)),
        transform=ax.transAxes, fontsize=8.4, va="bottom")

# 5 -------------------------------------------------- author's genre means --- #
sx, sy = "hatte", "du"
ix, iy = idx[sx], idx[sy]
ax = axes[1, 0]
panel(ax, 5, "Step 2: genre shift = genre mean − the author's mean")
for g in GENRES:
    m = (A == EXAMPLE) & (G == g)
    ax.plot(Z[m, ix], Z[m, iy], "o", ms=3.5, color=GCOL[g], alpha=0.45)
gm = {g: Z[(A == EXAMPLE) & (G == g)].mean(0) for g in GENRES}
am = np.mean(list(gm.values()), 0)
ax.plot(am[ix], am[iy], "*", ms=16, color="black", zorder=5, label="his mean over the three genres")
for g in GENRES:
    ax.plot(gm[g][ix], gm[g][iy], "s", ms=9, color=GCOL[g], markeredgecolor="black", zorder=5)
    ax.add_patch(FancyArrowPatch((am[ix], am[iy]), (gm[g][ix], gm[g][iy]), arrowstyle="->",
                                 mutation_scale=16, color=GCOL[g], lw=2, zorder=6))
ax.set_xlabel("z(%s)" % sx); ax.set_ylabel("z(%s)" % sy)
ax.legend(fontsize=8.2, loc="upper right")
ax.text(0.0, -0.17, "square = genre mean; arrow = genre mean − his mean:\nhis shift vector for that genre "
        "(shown in 2 of the %d coordinates)" % len(vocab), transform=ax.transAxes, fontsize=8.4, va="top")

# 6 ---------------------------------------------------- shared direction --- #
ax = axes[1, 1]
panel(ax, 6, "Step 3: genre direction = what all shifts share")
for a, g, d in shift:
    ax.add_patch(FancyArrowPatch((0, 0), (d[ix], d[iy]), arrowstyle="->", mutation_scale=10,
                                 color=GCOL[g], lw=1.2, alpha=0.7))
lim = max(abs(S[:, ix]).max(), abs(S[:, iy]).max()) * 1.15
u = np.array([v1[ix], v1[iy]]); u = u / np.linalg.norm(u) * lim
ax.plot([-u[0], u[0]], [-u[1], u[1]], color="black", lw=2.5, label="genre direction 1")
ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
ax.set_xlabel("shift in z(%s)" % sx); ax.set_ylabel("shift in z(%s)" % sy)
ax.legend(fontsize=8.2, loc="upper right")
ax.text(0.0, -0.17, "%d shift vectors (%d authors × 3 genres), all drawn from the origin;\n"
        "the direction is the leading singular vector of that stack,\ncomputed in all %d coordinates "
        "(here its projection on two)" % (len(shift), len(authors), len(vocab)),
        transform=ax.transAxes, fontsize=8.4, va="top")

# 7 ------------------------------------------------------------- removal --- #
ax = axes[1, 2]
panel(ax, 7, "Step 3: remove the genre component")
wv = Z[ex_drama[0]]
comp = wv @ v1
wr = wv - comp * v1
u2 = np.array([v1[ix], v1[iy]]); u2 = u2 / np.linalg.norm(u2)
lim = max(abs(wv[ix]), abs(wv[iy]), abs(wr[ix]), abs(wr[iy])) * 1.4 + 0.5
ax.plot([-u2[0] * lim, u2[0] * lim], [-u2[1] * lim, u2[1] * lim], color="black", lw=2, label="genre direction 1")
ax.add_patch(FancyArrowPatch((0, 0), (wv[ix], wv[iy]), arrowstyle="->", mutation_scale=16,
                             color=GCOL["drama"], lw=2.2))
ax.text(wv[ix], wv[iy], "  window w", color=GCOL["drama"], fontsize=9, va="bottom")
proj2 = comp * np.array([v1[ix], v1[iy]])
ax.add_patch(FancyArrowPatch((0, 0), (proj2[0], proj2[1]), arrowstyle="->", mutation_scale=12,
                             color="0.4", lw=1.5, linestyle="--"))
ax.text(proj2[0], proj2[1], "  (w·v) v", color="0.4", fontsize=9, va="top")
ax.add_patch(FancyArrowPatch((0, 0), (wr[ix], wr[iy]), arrowstyle="->", mutation_scale=16,
                             color="#d62728", lw=2.2))
ax.text(wr[ix], wr[iy], "  w′ = w − (w·v) v", color="#d62728", fontsize=9, va="bottom")
ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
ax.set_xlabel("z(%s)" % sx); ax.set_ylabel("z(%s)" % sy)
ax.legend(fontsize=8.2, loc="upper right")
ax.text(0.0, -0.17, "for this drama window w·v = %.1f; with ten directions,\nthe same subtraction is "
        "done for each of them" % comp, transform=ax.transAxes, fontsize=8.4, va="top")

# 8 ---------------------------------------------------- score and calibrate --- #
ax = axes[1, 3]
panel(ax, 8, "Use: score against the known bank, standardise")
B10 = vt[:10]                              # the method removes ten directions
Zr10 = Z - (Z @ B10.T) @ B10
K = Zr10[ex_prose]
kmean = K.mean(0)


def cos(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


anchors = [cos(K[j], (K.sum(0) - K[j]) / (len(K) - 1)) for j in range(len(K))]
mu_a, sd_a = np.mean(anchors), np.std(anchors)
same = [cos(Zr10[i], kmean) for i in ex_drama]
other = [cos(Zr10[i], kmean) for i in st_drama]
s_same, s_other = float(np.median(same)), float(np.median(other))
ax.hist(anchors, bins=15, color=GCOL["prose"], alpha=0.5,
        label="known windows against the mean of the others\n(%d prose windows of %s)" % (len(K), DISPLAY[EXAMPLE].title()))
ax.axvline(mu_a, color=GCOL["prose"], lw=1.5)
ax.axvline(s_same, color=GCOL["drama"], lw=2.5,
           label="median of %d drama windows of %s: z = %+.1f"
           % (len(same), DISPLAY[EXAMPLE].title(), (s_same - mu_a) / sd_a))
ax.axvline(s_other, color="0.2", lw=2.5, ls="--",
           label="median of %d drama windows of %s: z = %+.1f"
           % (len(other), DISPLAY[STRANGER].title(), (s_other - mu_a) / sd_a))
ax.set_xlabel("cosine to the known mean, ten genre directions removed")
ax.set_ylabel("known windows")
ax.legend(fontsize=8, loc="upper left")
ax.text(0.0, -0.17, "z = (score − mean of anchors) / sd of anchors; the gap a same-author\n"
        "text shows in another genre, and where strangers sit, come from the\nother %d authors "
        "of the same genre pair and turn z into a likelihood ratio" % (len(authors) - 1),
        transform=ax.transAxes, fontsize=8.2, va="top")

# ------------------------------------------- third row: the grammar side --- #
# A grammar cannot have directions projected out of it. What it can have is a
# known text that has been MOVED along those directions before the grammar is
# fitted: weights on the known text's 100-token units such that their mean
# composition sits where the other authors say this author's questioned genre
# would sit, then a grammar fitted on the units drawn by those weights.
sys.path.insert(0, str(ROOT / "experiments"))
from genre_adapt import adapt_units, tilt_weights  # noqa: E402


class _Space:                     # what the adaptation needs: a unit's standardised rates
    def z(self, toks):
        v = np.zeros(len(vocab))
        for s_, n in Counter(toks).items():
            if s_ in idx:
                v[idx[s_]] = n / len(toks)
        return (v - MU) / SD


space = _Space()
SEG, KT = 100, 30000
ktoks = banks["prose"][EXAMPLE][:KT]
units = [ktoks[i:i + SEG] for i in range(0, len(ktoks) - SEG + 1, SEG)]
others = [a for a in authors if a != EXAMPLE]
vecs, dvec = [], []
for a in others:
    gm = {g: Z[(A == a) & (G == g)].mean(0) for g in GENRES}
    am = np.mean(list(gm.values()), 0)
    vecs += [gm[g] - am for g in GENRES]
    dvec.append(gm["drama"] - gm["prose"])
_, _, vt16 = np.linalg.svd(np.array(vecs), full_matrices=False)
B10, d12 = vt16[:10], np.mean(dvec, 0)
Cfull = np.array([space.z(u) @ B10.T for u in units])
sign = 1.0 if (vt16[0] @ d12) > 0 else -1.0            # show the drama end on the right
c1 = sign * Cfull[:, 0]
c1_known, c1_target = c1.mean(), c1.mean() + sign * (vt16[0] @ d12)
c1_drama_true = sign * (Z[(A == EXAMPLE) & (G == "drama")] @ vt16[0]).mean()
w_full, _ = tilt_weights(Cfull, Cfull.mean(0) + B10 @ d12)
fl_units, fl_info = adapt_units(units, space, B10, d12, seed=0, ess_floor=0.25)
w_fl, _ = tilt_weights(Cfull, Cfull.mean(0) + fl_info["scale"] * (B10 @ d12))
ess = lambda w: 1.0 / np.sum(w ** 2)

# 9 -------------------------------------------------------------------------- #
ax = axes[2, 0]
panel(ax, 9, "Grammar side, A: where the known text must move")
ax.hist(c1, bins=25, color=GCOL["prose"], alpha=0.5,
        label="%d known units of 100 tokens\n(%s, prose, first %s tokens)"
        % (len(units), DISPLAY[EXAMPLE].title(), format(KT, ",")))
ax.axvline(c1_known, color=GCOL["prose"], lw=2, label="their mean")
ax.axvline(c1_target, color="#d62728", lw=2.5,
           label="target: mean + the prose→drama shift\nof the other %d authors" % len(others))
ax.axvline(c1_drama_true, color=GCOL["drama"], lw=2, ls=":",
           label="where his own drama really sits\n(never used; a check)")
ax.set_xlabel("coordinate on genre direction 1 (drama end to the right)")
ax.set_ylabel("known units")
ax.legend(fontsize=7.8, loc="upper left")

# 10 ------------------------------------------------------------------------- #
ax = axes[2, 1]
panel(ax, 10, "B: maximum-entropy weights on the units")
o = np.argsort(c1)
ax.plot(c1[o], (len(units) * w_full)[o], "-", color="0.35", lw=1.8,
        label="take the full shift: effective sample %.0f%% of the units" % (100 * ess(w_full) / len(units)))
ax.plot(c1[o], (len(units) * w_fl)[o], "-", color="#d62728", lw=2.2,
        label="floor of 25%%: move %.2f of the shift, effective sample %.0f%%"
        % (fl_info["scale"], 100 * ess(w_fl) / len(units)))
ax.axhline(1, color="0.6", lw=1, ls="--")
ax.set_yscale("log")
ax.set_xlabel("coordinate on genre direction 1")
ax.set_ylabel("weight, relative to uniform (log scale)")
ax.legend(fontsize=7.8, loc="upper left")
ax.text(0.0, -0.17, "w_i ∝ exp(λ·c_i) on the ten genre coordinates, λ chosen so that the weighted\n"
        "mean hits the target; a floor on the effective sample scales the move back,\n"
        "so the grammar keeps enough distinct text to be a grammar",
        transform=ax.transAxes, fontsize=8.4, va="top")

# 11 ------------------------------------------------------------------------- #
ax = axes[2, 2]
panel(ax, 11, "C: the known text the grammar is fitted on")
c1_new = sign * np.array([space.z(u) @ vt16[0] for u in fl_units])
bins = np.linspace(min(c1.min(), c1_new.min()), max(c1.max(), c1_new.max()), 26)
ax.hist(c1, bins=bins, color=GCOL["prose"], alpha=0.45, label="units as they are")
ax.hist(c1_new, bins=bins, color="#d62728", alpha=0.45,
        label="units drawn by the weights (same count;\nheavy ones repeated, light ones dropped)")
ax.axvline(c1_known, color=GCOL["prose"], lw=2)
ax.axvline(c1_new.mean(), color="#d62728", lw=2)
ax.set_xlabel("coordinate on genre direction 1")
ax.set_ylabel("units")
ax.legend(fontsize=7.8, loc="upper left")
ax.text(0.0, -0.17, "the Kneser-Ney grammar of the known author is fitted on the red multiset;\n"
        "donors come from the questioned genre and are size-matched as before;\n"
        "λG itself is unchanged", transform=ax.transAxes, fontsize=8.4, va="top")

# 12 ------------------------------------------------------------------------- #
ax = axes[2, 3]
panel(ax, 12, "D: what it does to λG (mean over donors)")
try:
    from analyze_xgenre_adapt import read_arm as _read_arm, PAIRS as _PAIRS  # noqa: E402
    from sklearn.metrics import roc_auc_score as _auc
    XA = ROOT / "scores" / "xgenre_adapt"
    res = {}
    for pair in _PAIRS:
        for arm in ("adapt0", "adapt10f25"):
            fs = sorted(XA.glob("%s__%s__*.jsonl" % (pair, arm)))
            if fs:
                v = _read_arm(fs[0])
                res[(pair, arm)] = _auc(v["y"], v["m"])
    if res:
        xs = np.arange(len(_PAIRS))
        a0 = [res.get((p, "adapt0"), np.nan) for p in _PAIRS]
        a1 = [res.get((p, "adapt10f25"), np.nan) for p in _PAIRS]
        ax.bar(xs - 0.2, a0, width=0.4, color="0.55", label="known grammar as it is")
        ax.bar(xs + 0.2, a1, width=0.4, color="#d62728", label="known grammar genre-adapted (floor 25%)")
        ax.set_xticks(xs); ax.set_xticklabels([p.replace("2", " → ") for p in _PAIRS], rotation=30, ha="right", fontsize=8)
        ax.set_ylim(0.5, 1.0); ax.set_ylabel("AUC, same author vs. other authors")
        ax.legend(fontsize=7.8, loc="upper left")
        ax.text(0.0, -0.32, "17 authors; known 30 000 tokens; questioned 1000-token windows;\n"
                "donors from the questioned genre; 100-token units",
                transform=ax.transAxes, fontsize=8.4, va="top")
    else:
        raise FileNotFoundError
except Exception:
    ax.axis("off")
    ax.text(0.0, 0.9, "results pending: run_xgenre_adapt.py", va="top", fontsize=9.5)

out = HERE / "composition_steps.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print("->", out)
