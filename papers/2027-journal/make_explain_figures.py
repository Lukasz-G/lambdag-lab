# Exhibits for the explainability chapter: one case, two verifiers.
#
# For each case the Julia explain pipeline dumped (phase4/explain.jl ->
# phase4/explain_cases/<pid>_*.tsv) this draws
#   fig_explain_tm_<pid>.png   the supervised verifier's account: the clause
#                              waterfall (every clause's signed activation,
#                              the margin as their running sum), the occlusion
#                              attributions (change of margin when one masked
#                              word type is deleted from both documents), and
#                              the questioned text coloured by attribution;
#   fig_heatmap_<pid>.png      the grammar model's account of the same pair:
#                              the questioned text coloured by each token's
#                              signed contribution to lambda_G.
# Both use one palette: red for evidence of the same author, blue for
# evidence of a different one. The grammar model's heat map is drawn by the
# CHR machinery (chr2027/make_heatmaps.py) on the raw German texts of the
# pair, retagged and masked exactly as in the evaluation.
#
#   python journal/figures/make_explain_figures.py [pid ...] [--tm-only]
#
# Data: German novels, the five hundred evaluation pairs (phase3/pairs500.tsv,
# pair ids = record ids of german/av_test_novels_de.jsonl; known = the longer
# text), reference population data/german/av_reference_novels_de.jsonl.

import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "chr2027"))
from make_heatmaps import tag, heatmap, REF_AUTHORS, REF_TOKENS, _rgba, UD_MEANING  # noqa: E402
from lambdag import LambdaG, POSNoiseMasker, DEFAULT_ABBREV_POS_TAGS  # noqa: E402

GLYPHS = {}
for _cat, _sym in DEFAULT_ABBREV_POS_TAGS.items():
    GLYPHS.setdefault(_sym, []).append(_cat)
LEGEND = ("placeholder glyphs (masked word classes):   "
          + "   ".join(sym + " = " + "/".join(UD_MEANING.get(c, c) for c in cats) for sym, cats in GLYPHS.items())
          + "   -- kept function words appear verbatim")

CASES = ROOT / "phase4" / "explain_cases"
TEST = ROOT / "german" / "av_test_novels_de.jsonl"
REFS = ROOT / "data" / "german" / "av_reference_novels_de.jsonl"
MAN = ROOT / "phase3" / "pairs500.tsv"
KNOWN_TOKENS, QUERY_TOKENS = 5000, 1000
TOP = 18          # occlusion bars shown
LINES = 10        # questioned sentences shown in the coloured text
WRAP = 22


def read_case(pid):
    meta = list(csv.DictReader(open(CASES / f"{pid}_meta.tsv", encoding="utf-8"), delimiter="\t"))[0]
    cl = np.loadtxt(CASES / f"{pid}_clauses.tsv", delimiter="\t", skiprows=1)
    att = [(r["token"], float(r["delta_margin"]))
           for r in csv.DictReader(open(CASES / f"{pid}_attrib.tsv", encoding="utf-8"), delimiter="\t")]
    q = [line.rstrip("\n").split("\t") for line in open(CASES / f"{pid}_qtext.tsv", encoding="utf-8") if line.strip()]
    return meta, cl, att, q


def manifest():
    return {int(r["id"]): r for r in csv.DictReader(open(MAN, encoding="utf-8"), delimiter="\t")}


# the corpus keys are alphabetised name tokens; the exhibits use the names as read
DISPLAY = {"maria rainer rilke": "Rainer Maria Rilke", "arthur schnitzler": "Arthur Schnitzler",
           "gräfin hahnhahn ida von": "Ida von Hahn-Hahn", "carl spitteler": "Carl Spitteler",
           "conrad ferdinand meyer": "Conrad Ferdinand Meyer", "freytag gustav": "Gustav Freytag"}


def cap(name):
    return DISPLAY.get(name, " ".join(w.capitalize() for w in name.split()))


def tm_figure(pid, meta, cl, att, q, ka, qa, out_png):
    label = int(meta["label"])
    margin = float(meta["margin"])
    members = [float(x) for x in meta["members"].split(",")]
    pos, neg = cl[:, 1], cl[:, 2]
    votes = np.concatenate([pos[pos > 0], -neg[neg > 0]])
    order = np.argsort(-np.abs(votes))
    votes = votes[order]
    fig = plt.figure(figsize=(11, 8.2))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.15], width_ratios=[1.35, 1])

    # (a) clause waterfall
    ax = fig.add_subplot(gs[0, 0])
    cum = np.cumsum(votes)
    x = np.arange(len(votes))
    ax.bar(x, votes, width=1.0, color=["#d62728" if v > 0 else "#1f77b4" for v in votes], lw=0)
    ax.step(x, cum, where="mid", color="#222", lw=1.2)
    ax.axhline(0, color="#999", lw=0.6)
    ax.set_xlabel("clauses that fired, largest first (%d of %d; one ensemble member)" % (len(votes), len(pos)))
    ax.set_ylabel("clause activation, signed by polarity")
    ax.set_title("(a) the vote, clause by clause; line = running sum\nmember margins "
                 + ", ".join("%+.0f" % m for m in members) + ", mean %+.1f" % margin, fontsize=9.5, loc="left")

    # (b) occlusion attributions
    ax = fig.add_subplot(gs[0, 1])
    top = att[:TOP][::-1]
    vals = np.array([v for _, v in top])
    ax.barh(np.arange(len(top)), vals, color=["#d62728" if v > 0 else "#1f77b4" for v in vals], height=0.72)
    ax.set_yticks(np.arange(len(top)))
    ax.set_yticklabels([w for w, _ in top], fontsize=7.8, family="monospace")
    ax.axvline(0, color="#999", lw=0.6)
    ax.set_xlabel("change of margin if the word type is deleted from both texts", fontsize=8.5)
    ax.set_title("(b) occlusion attributions, %d most influential of %d probed" % (len(top), len(att)),
                 fontsize=9.5, loc="left")

    # (c) the questioned text coloured by attribution (word-type level)
    ax = fig.add_subplot(gs[1, :])
    amap = dict(att)
    scale = max(abs(v) for _, v in att) or 1.0
    rows = []
    for s in q[:LINES]:
        for i in range(0, len(s), WRAP):
            rows.append(s[i:i + WRAP])
    y = 0.0
    for row in rows:
        for j, w in enumerate(row):
            v = amap.get(w, 0.0)
            ax.add_patch(Rectangle((j, y), 1, 1, facecolor=_rgba(v, scale), edgecolor="white", lw=0.4))
            ax.text(j + 0.5, y + 0.5, w, ha="center", va="center", fontsize=6.4, color="#222")
        y -= 1.3
    ax.set_xlim(0, WRAP)
    ax.set_ylim(y + 0.3, 1.1)
    ax.set_axis_off()
    ax.set_title("(c) the questioned text (POSNoise-masked, first %d sentences) coloured by attribution" % LINES,
                 fontsize=9.5, loc="left")
    verdict = "same author" if margin > 0 else "different authors"
    truth = "same author" if label == 1 else "different authors"
    fig.suptitle("Supervised verifier, German novels, case %d: known %s (N(K) = %s tokens), questioned %s (N(Q) = %s)"
                 "\nverdict %s, truth %s" % (pid, cap(ka), meta["klen"], cap(qa), meta["qlen"], verdict, truth),
                 fontsize=10, y=0.995)
    fig.text(0.5, 0.022, "red = evidence for the same author, blue = evidence for different authors; "
             "the same palette as the grammar model's heat map of this case",
             ha="center", fontsize=7.5, color="#555")
    fig.text(0.5, 0.004, LEGEND, ha="center", fontsize=6.6, color="#555")
    fig.tight_layout(rect=[0, 0.035, 1, 0.94])
    fig.savefig(out_png, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote ->", out_png)


def lambdag_figure(pid, rec, ka, qa, label, lg, masker, nlp, out_png):
    t0, t1 = rec["pair"]
    known, q = (t1, t0) if len(t1) >= len(t0) else (t0, t1)
    heatmap(lg, masker, tag(nlp, q, QUERY_TOKENS), tag(nlp, known, KNOWN_TOKENS),
            cap(qa), cap(ka), label == 1, out_png)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    tm_only = "--tm-only" in sys.argv
    pids = [int(a) for a in sys.argv[1:] if a.isdigit()] or sorted(int(p.name.split("_")[0]) for p in CASES.glob("*_meta.tsv"))
    man = manifest()
    recs = {}
    for line in open(TEST, encoding="utf-8"):
        r = json.loads(line)
        if int(r["id"]) in pids:
            recs[int(r["id"])] = r
    lg = masker = nlp = None
    if not tm_only:
        import spacy
        nlp = spacy.load("de_core_news_lg")
        masker = POSNoiseMasker(language="de", require_tagger=False)
        ref_masked, n = [], 0
        for line in open(REFS, encoding="utf-8"):
            ref_masked += masker.mask_tagged(tag(nlp, json.loads(line)["text"], REF_TOKENS))
            n += 1
            if n >= REF_AUTHORS:
                break
        lg = LambdaG(N=10, r=30, engine="kn", random_state=0)
        lg.set_reference(ref_masked)
    for pid in pids:
        meta, cl, att, q = read_case(pid)
        ka, qa = man[pid]["known_author"], man[pid]["q_author"]
        tm_figure(pid, meta, cl, att, q, ka, qa, HERE / f"fig_explain_tm_{pid}.png")
        if not tm_only and pid in recs:
            lambdag_figure(pid, recs[pid], ka, qa, int(meta["label"]), lg, masker, nlp,
                           HERE / f"fig_heatmap_{pid}.png")


if __name__ == "__main__":
    main()
