# Cross-genre verification with a composition score: what a long known side
# and the removal of genre directions do to AUC and to calibrated cost.
#
# Data: journal/figures/genre_rescue_{w1000,w500,w1000_k3}.json, written by
#   python experiments/analyze_genre_symbols.py --dump ... [--window 500]
#                                                          [--known-windows 3]
# Output: journal/figures/genre_rescue.png

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

HERE = Path(__file__).resolve().parent
D1000 = json.load(open(HERE / "genre_rescue_w1000.json", encoding="utf-8"))
D500 = json.load(open(HERE / "genre_rescue_w500.json", encoding="utf-8"))
DK3 = json.load(open(HERE / "genre_rescue_w1000_k3.json", encoding="utf-8"))

CROSS = ["prose->drama", "drama->prose", "prose->verse", "verse->prose",
         "drama->verse", "verse->drama"]
WITHIN = ["prose->prose", "drama->drama", "verse->verse"]
PAIRS = CROSS + WITHIN
XPOS = list(range(6)) + [7, 8, 9]
LABEL = {p: p.replace("->", " →\n") for p in PAIRS}

plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10})
fig = plt.figure(figsize=(15, 10))
gs = fig.add_gridspec(2, 2, height_ratios=[1.15, 1], hspace=0.42, wspace=0.22)

fig.suptitle("Cross-genre authorship verification with a composition score, "
             "German (TextGrid): 17 authors who wrote prose, drama and verse",
             fontsize=13, fontweight="bold", y=0.985)
fig.text(0.5, 0.945,
         "Texts masked by POSNoise (content words replaced by class placeholders, function words kept), "
         "%d symbols. Score = cosine between standardised symbol-rate vectors.\n"
         "Known = the author's whole bank in the known genre (median %s tokens, minimum %s); "
         "questioned = one window of 1000 tokens (or 500 where marked). "
         "Genre directions are estimated on the other 16 authors, never on the case author."
         % (D1000["n_symbols"], format(D1000["known_tokens"]["median"], ","),
            format(D1000["known_tokens"]["min"], ",")),
         ha="center", va="top", fontsize=9.5)

# ---------------------------------------------------------------- AUC ----- #
ax = fig.add_subplot(gs[0, :])
series = [
    ("known side 3 000 tokens, no genre removal", DK3["nap"], 0, "0.45", "o", "full", -0.28),
    ("known whole bank, no genre removal", D1000["nap"], 0, "#1f77b4", "s", "full", -0.14),
    ("known whole bank, 2 genre directions removed", D1000["nap"], 2, "#ff7f0e", "^", "full", 0.0),
    ("known whole bank, 10 genre directions removed", D1000["nap"], 10, "#d62728", "D", "full", 0.14),
    ("known whole bank, 10 removed, questioned 500 tokens", D500["nap"], 10, "#d62728", "D", "none", 0.28),
]
for name, src, k, col, mk, fill, dx in series:
    ys = [src["%s|k%d" % (p, k)] for p in PAIRS]
    ax.plot([x + dx for x in XPOS], ys, linestyle="none", marker=mk, ms=8,
            color=col, markerfacecolor=col if fill == "full" else "white",
            markeredgewidth=1.6, label=name)
ax.set_xticks(XPOS)
ax.set_xticklabels([LABEL[p] for p in PAIRS], rotation=0, fontsize=9.5)
ax.axvline(6.5, color="0.8", lw=1)
ax.text(2.5, 0.995, "known genre → questioned genre", ha="center", va="top", fontsize=9.5, color="0.3")
ax.text(8, 0.995, "same genre (reference)", ha="center", va="top", fontsize=9.5, color="0.3")
ax.set_ylim(0.6, 1.0)
ax.set_ylabel("AUC (same author vs. other authors)")
ax.set_title("Discrimination: two separate levers, and they add up", loc="left")
ax.grid(axis="y", color="0.9")
ax.legend(loc="lower right", fontsize=9, frameon=True, ncol=1)
# annotate the two levers on the first pair
p = "prose->drama"
y_k3 = DK3["nap"]["%s|k0" % p]; y_full = D1000["nap"]["%s|k0" % p]; y_10 = D1000["nap"]["%s|k10" % p]
ax.annotate("", xy=(-0.5, y_full), xytext=(-0.5, y_k3),
            arrowprops=dict(arrowstyle="->", color="#1f77b4", lw=1.5))
ax.text(-0.62, (y_k3 + y_full) / 2, "longer\nknown side", ha="right", va="center", fontsize=8.5, color="#1f77b4")
ax.annotate("", xy=(0.45, y_10), xytext=(0.45, y_full),
            arrowprops=dict(arrowstyle="->", color="#d62728", lw=1.5))
ax.text(0.55, (y_full + y_10) / 2, "genre\nremoved", ha="left", va="center", fontsize=8.5, color="#d62728")
ax.set_xlim(-1.2, 9.6)

# --------------------------------------------------------------- Cllr ----- #
ax = fig.add_subplot(gs[1, 0])
cal = D1000["calib"]
cseries = [
    ("floor (perfect calibration), no genre removal", "cmin", 0, "#1f77b4", "s", "none", -0.21),
    ("calibrated cost, no genre removal", "cllr", 0, "#1f77b4", "s", "full", -0.07),
    ("floor, 10 genre directions removed", "cmin", 10, "#d62728", "D", "none", 0.07),
    ("calibrated cost, 10 directions removed", "cllr", 10, "#d62728", "D", "full", 0.21),
]
for name, key, k, col, mk, fill, dx in cseries:
    ys = [cal["%s|k%d" % (p, k)][key] for p in PAIRS]
    ax.plot([x + dx for x in XPOS], ys, linestyle="none", marker=mk, ms=8, color=col,
            markerfacecolor=col if fill == "full" else "white", markeredgewidth=1.6, label=name)
ax.axhline(1.0, color="0.5", lw=1, ls="--")
ax.text(-0.5, 1.02, "no information", ha="left", va="bottom", fontsize=8.5, color="0.4")
ax.axvline(6.5, color="0.8", lw=1)
ax.set_xticks(XPOS)
ax.set_xticklabels([p.replace("->", "\n→\n") for p in PAIRS], rotation=0, fontsize=9)
ax.set_ylim(0, 1.1)
ax.set_xlim(-0.6, 9.6)
ax.set_ylabel("Cllr (lower is better)")
ax.set_title("Calibration: known bank gives the score's location and spread;\n"
             "genre gap and stranger location come from the other 16 authors of the pair",
             loc="left")
ax.grid(axis="y", color="0.9")
ax.legend(loc="upper right", fontsize=8.5, frameon=True)

# ------------------------------------------------------------- space ----- #
inner = gs[1, 1].subgridspec(1, 2, wspace=0.25)
sc = D1000["scatter"]
authors = sorted({s["author"] for s in sc})
cols = plt.get_cmap("tab10")
acol = {a: cols(i) for i, a in enumerate(authors)}
gcol = {"prose": "#1f77b4", "drama": "#ff7f0e", "verse": "#2ca02c"}
gmark = {"prose": "o", "drama": "^", "verse": "s"}
import random
random.Random(0).shuffle(sc)
per = {}
keep = []
for s_ in sc:            # at most 30 windows per author and genre
    kk = (s_["author"], s_["genre"])
    if per.get(kk, 0) < 30:
        per[kk] = per.get(kk, 0) + 1
        keep.append(s_)


def surname(key):
    return key.title()          # the data file already carries the surname


axs = fig.add_subplot(inner[0, 0])
for s_ in keep:
    x, y = s_["before"]
    axs.plot(x, y, marker=gmark[s_["genre"]], ms=5, linestyle="none", color=gcol[s_["genre"]],
             alpha=0.7, markeredgewidth=0.4, markeredgecolor="0.3")
axs.set_title("Along the two strongest genre directions\n(colour = genre): windows sort by genre", fontsize=9)
axs.set_xticks([]); axs.set_yticks([])
ld = D1000.get("loadings")
if ld:
    lab = lambda d: "%s  ←  direction %d  →  %s" % (
        ", ".join(ld[d]["negative"][:4]), d + 1, ", ".join(ld[d]["positive"][:4]))
    axs.set_xlabel(lab(0), fontsize=7.5)
    axs.set_ylabel(lab(1), fontsize=7.5)
else:
    axs.set_xlabel("genre direction 1"); axs.set_ylabel("genre direction 2")
axs.legend(handles=[Line2D([], [], marker=m, ls="none", color=gcol[g], label=g) for g, m in gmark.items()],
           loc="lower left", fontsize=8, frameon=False)

axs = fig.add_subplot(inner[0, 1])
for s_ in keep:
    x, y = s_["after"]
    axs.plot(x, y, marker=gmark[s_["genre"]], ms=5, linestyle="none", color=acol[s_["author"]],
             alpha=0.7, markeredgewidth=0.4, markeredgecolor="0.3")
axs.set_title("After removing ten genre directions\n(colour = author): windows group by author", fontsize=9)
axs.set_xticks([]); axs.set_yticks([])
axs.set_xlabel("principal direction 1"); axs.set_ylabel("principal direction 2")
axs.legend(handles=[Line2D([], [], marker="o", ls="none", color=acol[a], label=surname(a)) for a in authors],
           loc="lower right", fontsize=7.5, frameon=False, ncol=2)
fig.text(0.74, 0.03, "markers: ● prose   ▲ drama   ■ verse   (30 windows per author and genre shown)",
         ha="center", fontsize=8.5, color="0.3")

out = HERE / "genre_rescue.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print("->", out)
