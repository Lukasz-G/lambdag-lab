# Explainability heat maps for the CHR 2027 paper (part one), in the format of the
# DHD MHG paper's exhibits (medieval/lambdag_mhd_verification.ipynb): per-token
# lambda_G contributions for one same-author and one different-author case,
# red = entrenched for the known author D_A, blue = typical of the reference
# population. English novels (ELTeC-en), sentence segmentation -- the same
# configuration as the paper's experiments; the header reports lambda_G and its
# sqrt correction (no calibrated LLR, since part one's regimes are calibration-free).
#
#   python chr2027/make_heatmaps.py
# Output: chr2027/ach-latex-template-v2/figures/fig_heatmap_{same,diff}.png

import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = HERE / "ach-latex-template-v2" / "figures"
sys.path.insert(0, str(ROOT))
from lambdag import LambdaG, POSNoiseMasker, DEFAULT_ABBREV_POS_TAGS  # noqa: E402

KNOWN_TOKENS = 5000      # D_A, as in the paper's datasets
QUERY_TOKENS = 1000      # D_U likewise; only the first SHOW_TOKENS are displayed
SHOW_TOKENS = 170
WRAP = 26                # display wrap: tokens per drawn row (long sentences fold)
REF_AUTHORS = 24         # reference pool, authors x REF_TOKENS tokens
REF_TOKENS = 6000

UD_MEANING = {"NOUN": "noun", "PROPN": "proper noun", "VERB": "verb",
              "AUX": "auxiliary / modal", "ADJ": "adjective", "ADV": "adverb",
              "NUM": "numeral", "SYM": "symbol", "X": "other"}
_SYM2CATS = {}
for _cat, _sym in DEFAULT_ABBREV_POS_TAGS.items():
    _SYM2CATS.setdefault(_sym, []).append(_cat)


def tag(nlp, text, n_tokens):
    """First n_tokens of text as sentences of (surface, upos, lemma)."""
    sents, count = [], 0
    for doc in nlp.pipe([text[:n_tokens * 12]]):     # ~chars per token upper bound
        for s in doc.sents:
            row = [(t.text, t.pos_, t.lemma_) for t in s if not t.is_space]
            if not row:
                continue
            sents.append(row)
            count += len(row)
            if count >= n_tokens:
                return sents
    return sents


def mask_verbose(m, sent):
    """Per token (surface, emitted, ud_pos, is_masked), with the masker's decisions."""
    raw = [t[0] for t in sent]
    lem = [t[2] if len(t) > 2 else "" for t in sent]
    pos = [m._resolve_pos(t[1]) for t in sent]
    keys = [m._keys_from(raw[i], lem[i]) for i in range(len(sent))]
    keep = (np.ones(len(sent), bool) if m.mode == "none"
            else m._safe_mask(keys, pos, raw))
    rows = []
    for i in range(len(sent)):
        emitted = m._emit_surface(raw[i], pos[i], bool(keep[i]), lem[i])
        rows.append((raw[i], emitted, pos[i],
                     (not keep[i]) and pos[i] in m.abbrev_pos_tags))
    return rows


def _rgba(v, scale):
    x = max(-1.0, min(1.0, v / scale))
    return ((214/255, 39/255, 40/255, 0.08 + 0.62 * x) if x >= 0
            else (31/255, 119/255, 180/255, 0.08 + 0.62 * -x))


def render(rows_lam, title, subtitle, out_png):
    """rows_lam: list of (label, [(surface, emitted, pos, is_masked, lam), ...])
    already wrapped to <= WRAP tokens per drawn row."""
    flat = np.array([t[4] for _, row in rows_lam for t in row])
    scale = float(np.percentile(np.abs(flat), 98)) if flat.size else 1.0
    scale = scale or 1.0
    W = max(len(row) for _, row in rows_lam)
    fig, ax = plt.subplots(figsize=(max(8, W * 0.52), 1.9 + len(rows_lam) * 0.78))
    ch, gap, y = 1.0, 0.5, 0.0
    for label, row in rows_lam:
        if label:
            ax.text(-0.35, y + ch / 2, label, ha="right", va="center",
                    fontsize=6.4, color="#999")
        for j, (surf, emitted, pos, is_masked, lv) in enumerate(row):
            ax.add_patch(Rectangle((j, y), 1, ch, facecolor=_rgba(lv, scale),
                                   edgecolor="white", lw=0.4))
            disp = UD_MEANING.get(pos, pos) if is_masked else emitted
            ax.text(j + 0.5, y + ch * 0.71, surf, ha="center", va="center",
                    fontsize=4.6, color="#777")
            ax.text(j + 0.5, y + ch * 0.27, disp, ha="center", va="center",
                    fontsize=5.6, color="#222",
                    style="italic" if is_masked else "normal",
                    fontweight="normal" if is_masked else "bold")
        y -= ch + gap
    ax.set_xlim(-3.2, W)
    ax.set_ylim(y + gap - 0.1, ch + 0.2)
    ax.set_axis_off()
    fig.suptitle(title, fontsize=12, y=0.995)
    ax.set_title(subtitle, fontsize=7.6, loc="left", pad=6)
    legend = ("placeholder glyphs (the real LambdaG input):   "
              + "   ".join(f"{s} = " + "/".join(c) + f" ({UD_MEANING.get(c[0], '')})"
                           for s, c in _SYM2CATS.items())
              + "   -- kept function words appear verbatim.")
    fig.text(0.5, 0.006, legend, ha="center", va="bottom", fontsize=6.4, color="#555")
    fig.tight_layout(rect=[0, 0.035, 1, 0.955])
    fig.savefig(out_png, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote ->", out_png)


def heatmap(lg, masker, u_sents, a_sents, u_label, a_label, same, out_png):
    U = masker.mask_tagged(u_sents)
    A = masker.mask_tagged(a_sents)
    res = lg.score(U, A, with_details=True)
    assert len(res.token_lambda) == len(u_sents), "sentence alignment mismatch"

    # wrap sentences into display rows of <= WRAP tokens, sentence lambda on row 1
    rows, shown = [], 0
    for si, sent in enumerate(u_sents):
        verbose = mask_verbose(masker, sent)
        lam = res.token_lambda[si]
        cells = [(s, e, p, m, float(lam[j]) if j < len(lam) else 0.0)
                 for j, (s, e, p, m) in enumerate(verbose)]
        for k in range(0, len(cells), WRAP):
            label = f"S{si} {res.sentence_lambda[si]:+.1f}" if k == 0 else ""
            rows.append((label, cells[k:k + WRAP]))
        shown += len(cells)
        if shown >= SHOW_TOKENS:
            break

    verdict = "same" if same else "different"
    n_q = res.n_query_tokens
    subtitle = (f"$D_U$ (questioned) = {u_label}   vs   $D_A$ (known) = {a_label}\n"
                f"truth: {verdict} author  (first {shown} of {n_q} tokens)   |   "
                f"$\\lambda_G$ = {res.lambda_G:+.2f}   ->   "
                f"$\\lambda_G/\\sqrt{{N(Q)}}$ = {res.lambda_sqrt:+.2f} "
                f"(evidence for {'same' if res.lambda_sqrt > 0 else 'different'})   |   "
                "red = entrenched for $D_A$,  blue = typical of the reference population")
    title = "Same-author case" if same else "Different-author case"
    render(rows, title, subtitle, out_png)
    return res


LANGS = [
    # (folder, test/ref jsonl stem suffix, spaCy model, masker language, out suffix)
    ("english", "en", "en_core_web_lg", "en", ""),
    ("polish", "pl", "pl_core_news_lg", "pl", "_pl"),
    ("lithuanian", "lt", "lt_core_news_lg", "lt", "_lt"),
]


def run_language(folder, code, model, mlang, suffix):
    import spacy
    nlp = spacy.load(model)

    # real evaluation cases: pair = [questioned ~1000 tok, known ~5000 tok],
    # authors = [questioned author, known author]
    same_case = diff_case = None
    for line in open(ROOT / "data" / folder / f"av_test_novels_{code}.jsonl",
                     encoding="utf-8"):
        r = json.loads(line)
        if r["label"] == 1 and same_case is None:
            same_case = r
        if r["label"] == 0 and diff_case is None:
            diff_case = r
        if same_case and diff_case:
            break
    print(f"[{folder}] same-author case :", same_case["authors"][1])
    print(f"[{folder}] diff-author case :", diff_case["authors"][0],
          "(questioned)  vs ", diff_case["authors"][1], "(known)")

    masker = POSNoiseMasker(language=mlang, require_tagger=False)
    a1_known = tag(nlp, same_case["pair"][1], KNOWN_TOKENS)
    a1_query = tag(nlp, same_case["pair"][0], QUERY_TOKENS)
    a2_known = tag(nlp, diff_case["pair"][1], KNOWN_TOKENS)
    a2_query = tag(nlp, diff_case["pair"][0], QUERY_TOKENS)

    ref, nref = [], 0
    for line in open(ROOT / "data" / folder / f"av_reference_novels_{code}.jsonl",
                     encoding="utf-8"):
        r = json.loads(line)
        ref.append(r["text"])
        nref += 1
        if nref >= REF_AUTHORS:
            break
    ref_masked = []
    for t in ref:
        ref_masked += masker.mask_tagged(tag(nlp, t, REF_TOKENS))
    print(f"[{folder}] reference: {len(ref)} authors, "
          f"{sum(len(s) for s in ref_masked)} masked tokens")

    lg = LambdaG(N=10, r=30, engine="kn", random_state=0)
    lg.set_reference(ref_masked)

    cap = lambda name: name.title()
    heatmap(lg, masker, a1_query, a1_known,
            cap(same_case["authors"][0]), cap(same_case["authors"][1]), True,
            OUT / f"fig_heatmap_same{suffix}.png")
    heatmap(lg, masker, a2_query, a2_known,
            cap(diff_case["authors"][0]), cap(diff_case["authors"][1]), False,
            OUT / f"fig_heatmap_diff{suffix}.png")


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    for folder, code, model, mlang, suffix in LANGS:
        run_language(folder, code, model, mlang, suffix)


if __name__ == "__main__":
    main()
