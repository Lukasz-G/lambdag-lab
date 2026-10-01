# Every number in the article comes from here, and none is typed into the prose.
#
# The script reads the frozen result files the experiments wrote, prints the
# digest each chapter quotes, and regenerates the tables the chapters input. Run
# it before reading a draft: where a number in the text and a number here
# disagree, the text is wrong.
#
#   python tools/make_tables.py
#
# Reads   medieval/kc_reuse/edition_provenance.tsv          (chapter 4)
#         medieval/kc_reuse/layer_offset.tsv                (chapter 4)
#         medieval/kc_reuse/kc_regions_crosswork__amb*.tsv  (chapter 5)
#         medieval/kc_reuse/kc_regions_field__amb*.tsv      (chapter 5)
#         medieval/kc_reuse/ambraser_panel.tsv              (chapter 6)
#         medieval/kc_reuse/nibelungen_collation.tsv        (chapter 7)
#         masked/mhg_witnesses_lemma/MANIFEST.tsv           (chapter 2)
# Writes  tables/tab_codex.tex, tab_provenance.tex, tab_layer.tex,
#         tab_questions.tex, tab_arms.tex, tab_panel.tex, tab_redaction.tex

import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PAPER = HERE.parent
ROOT = PAPER.parent.parent
REUSE = ROOT / "medieval" / "kc_reuse"
TAB = PAPER / "tables"
EOL = " \\\\"

# the four relational questions and the two tests of verse form, as they are run
# and as the chapters name them
QUESTIONS = [
    ("nibelungen", "the Klage", "the Nibelungenlied"),
    ("dietrich", "the Rabenschlacht", "Dietrichs Flucht"),
    ("ortnit", "Ortnit", "Wolfdietrich"),
    ("kudrun", "the Nibelungenlied", "Kudrun"),
]
FORM = [
    ("formdf", "the Klage", "Dietrichs Flucht"),
    ("formbi", "the Klage", "Biterolf"),
]
# the four readings of each question, and what each one changes
ARMS = [
    ("codex", "the codex's own scale, questioned passages of 4,000"),
    ("short", "the same at 2,000, the floor for a question about a hand"),
    ("borrowed", "a modern verse population, ordering only"),
    ("edition", "the whole case in the editors' layer"),
]


def rows(path, delim="\t"):
    with open(path, encoding="utf-8") as fh:
        return list(csv.DictReader(fh, delimiter=delim))


def tex(name, body):
    (TAB / name).write_text(body, encoding="utf-8")
    print(f"    wrote tables/{name}")


def esc(s):
    return s.replace("&", "\\&").replace("_", "\\_")


# ---------------------------------------------------------------- chapter 2
def codex():
    """what the book contains, and which texts clear the floors"""
    print("\nTHE CODEX (chapter 2)")
    man = [r for r in rows(ROOT / "masked" / "mhg_witnesses_lemma"
                           / "MANIFEST.tsv") if r["source"] == "ambraser"]
    anon = [r for r in man if r["author"].startswith("anon")]
    named = [r for r in man if not r["author"].startswith("anon")]
    tot = sum(int(r["masked_tokens"]) for r in man)
    print(f"  {len(man)} texts, {tot:,} masked tokens, one hand; "
          f"{len(anon)} anonymous, {len(named)} under a name")
    authors = {r["author"] for r in named}
    print(f"  the named pieces give {len(authors)} authors: "
          f"{', '.join(sorted(a.split('__')[0] for a in authors))}")

    def line(r):
        return (f"{esc(r['work'].split('__')[-1].replace('_', ' '))} & "
                f"{int(r['masked_tokens']):,}{EOL}")
    body = ["\\begin{tabular}{lr}", "\\hline",
            f"anonymous & masked tokens{EOL}", "\\hline"]
    for r in sorted(anon, key=lambda r: -int(r["masked_tokens"])):
        body.append(line(r))
    body += ["\\hline", f"under an author's name & {EOL}", "\\hline"]
    for r in sorted(named, key=lambda r: -int(r["masked_tokens"])):
        body.append(line(r))
    body += ["\\hline", "\\end{tabular}"]
    tex("tab_codex.tex", "\n".join(body) + "\n")
    return man


# ---------------------------------------------------------------- chapter 4
def provenance():
    """which printed text can rest on a book other than the codex"""
    print("\nPROVENANCE (chapter 4)")
    rs = rows(REUSE / "edition_provenance.tsv")
    by = {r["siglum"]: r for r in rs}
    for r in rs:
        print(f"  {r['siglum']:4} {r['work'][:22]:23} {r['codices']:>3} codices"
              f"  {r['independent']:>7}   {r['grounds']}")
    body = ["\\begin{tabular}{llrl}", "\\hline",
            f"poem & in the codex as & copies & the printed text{EOL}",
            "\\hline"]
    for r in rs:
        if not r["work"]:
            continue
        body.append(f"{esc(r['work'])} & {r['codex_siglum'] or '--'} & "
                    f"{r['codices']} & {esc(r['grounds'])}{EOL}")
    body += ["\\hline", "\\end{tabular}"]
    tex("tab_provenance.tex", "\n".join(body) + "\n")

    print("\n  what each question's second reading crosses:")
    out = []
    for tag, cand, known in QUESTIONS:
        sig = {"nibelungen": ("NBB", "KLA"), "dietrich": ("DFL", "RAB"),
               "ortnit": ("WDA", "ORT"), "kudrun": ("KU", "NBB")}[tag]
        ok = [by[s]["independent"] == "yes" for s in sig]
        what = ("the layer, the hand and the manuscript" if all(ok) else
                "the layer and the hand alone" if not any(ok) else
                "the layer throughout, the manuscript on one side")
        print(f"    {cand} against {known:20} {what}")
        out.append((tag, what))
    return dict(out)


def layer():
    """what an editor's text costs, measured on the works in both banks"""
    print("\nTHE LAYER (chapter 4)")
    rs = rows(REUSE / "layer_offset.tsv")
    per = defaultdict(dict)
    for r in rs:
        per[r["work"]][r["class"]] = float(r["z"])

    def med(kind):
        v = [d[kind] for d in per.values() if kind in d]
        return float(np.median(v)) if v else float("nan")
    own = [d["same work, the other layer"] - d["strangers, the edition layer"]
           for d in per.values()
           if {"same work, the other layer",
               "strangers, the edition layer"} <= d.keys()]
    keep = float(np.median(own))
    same = med("same work, same layer")
    off = abs(med("strangers, the edition layer"))
    print(f"  {len(per)} works in both banks")
    print(f"  the same work, against the strangers of its OWN layer:")
    print(f"    in the scribe's layer   {same:+.2f}")
    print(f"    in the editor's layer   {keep:+.2f}")
    print(f"  -> the layer costs        {same - keep:.2f} deviations of "
          f"separation")
    print(f"     on top of an offset of {off:.2f}, which a native cohort removes")
    body = ["\\begin{tabular}{lrr}", "\\hline",
            f"the same poem, against the strangers of & its own layer & "
            f"the other{EOL}", "\\hline",
            f"in a scribe's copy & {same:+.2f} & {0.0:+.2f}{EOL}",
            f"in an editor's text & {keep:+.2f} & "
            f"{med('same work, the other layer'):+.2f}{EOL}",
            "\\hline", "\\end{tabular}"]
    tex("tab_layer.tex", "\n".join(body) + "\n")
    return dict(works=len(per), same=same, keep=keep, cost=same - keep,
                offset=off)


# ---------------------------------------------------------------- chapter 5
def arm_result(tag):
    """(regions, tops, P, candidate at bound, control at bound, bound) or None"""
    f = REUSE / f"kc_regions_field__{tag}.tsv"
    if not f.exists():
        return None
    fr = rows(f)
    tops = sum(int(r["tops_field"]) for r in fr)
    c = REUSE / f"kc_regions_crosswork__{tag}.tsv"
    cw = rows(c) if c.exists() else []
    hi = float(cw[0]["elub_hi"]) if cw else float("nan")
    at = sum(float(r["llr"]) >= hi - 1e-9 for r in cw)
    atc = sum(float(r["llr_control"]) >= hi - 1e-9 for r in cw)
    return dict(n=len(fr), tops=tops, at=at, at_control=atc, bound=hi,
                regions=len(cw))


def binom_tail(k, n, p):
    from math import comb
    return sum(comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(k, n + 1))


def questions(crosses):
    """the four questions, each read four ways"""
    print("\nTHE FOUR QUESTIONS (chapter 5)")
    body = ["\\begin{tabular}{llrrl}", "\\hline",
            f"question & reading & tops the field & $P$ & at the bound"
            f"{EOL}", "\\hline"]
    digest = {}
    for tag, cand, known in QUESTIONS + FORM:
        print(f"\n  {cand} against {known}"
              + (f"   [{crosses[tag]}]" if tag in crosses else ""))
        first = True
        for arm, _ in ARMS:
            r = arm_result(f"amb{arm}{tag}")
            if not r:
                continue
            # the null: one poem of the milieu among the field of candidate
            # plus controls tops it with probability one over the field
            p = binom_tail(r["tops"], r["n"], 1 / 5)
            digest[(tag, arm)] = dict(r, p=p)
            # the borrowed population is admissible for an ordering and not for
            # odds, so that arm is deliberately not calibrated and has no bound
            bound = ("--" if not r["regions"] else
                     f"{r['at']}/{r['regions']} against "
                     f"{r['at_control']}/{r['regions']}")
            print(f"    {arm:9} {r['tops']:2}/{r['n']:<2} regions  "
                  f"P = {p:8.2e}   at the bound {bound}"
                  + ("" if not r["regions"]
                     else f"   (bound {10 ** r['bound']:.0f}:1)"))
            body.append(f"{esc(cand + ' / ' + known) if first else ''} & "
                        f"{arm} & {r['tops']}/{r['n']} & {p:.1e} & "
                        f"{bound}{EOL}")
            first = False
        body.append("\\hline")
    body += ["\\end{tabular}"]
    tex("tab_questions.tex", "\n".join(body) + "\n")

    body = ["\\begin{tabular}{ll}", "\\hline",
            f"reading & what it changes{EOL}", "\\hline"]
    for arm, what in ARMS:
        body.append(f"{arm} & {esc(what)}{EOL}")
    body += ["\\hline", "\\end{tabular}"]
    tex("tab_arms.tex", "\n".join(body) + "\n")
    return digest


# ---------------------------------------------------------------- chapter 6
def panel():
    """the short pieces, and the controls that say the panel cannot attribute"""
    print("\nTHE SHORT PIECES (chapter 6)")
    rs = rows(REUSE / "ambraser_panel.tsv")
    cols = [c for c in rs[0] if c != "piece"]
    body = ["\\begin{tabular}{l" + "r" * len(cols) + "}", "\\hline",
            "known text & " + " & ".join(esc(c) for c in cols) + EOL,
            "\\hline"]
    for r in rs:
        print(f"  {r['piece'][:26]:27} " +
              "  ".join(f"{c}={float(r[c]):+.2f}" for c in cols))
        body.append(esc(r["piece"]) + " & " +
                    " & ".join(f"{float(r[c]):+.2f}" for c in cols) + EOL)
    body += ["\\hline", "\\end{tabular}"]
    tex("tab_panel.tex", "\n".join(body) + "\n")


# ---------------------------------------------------------------- chapter 7
def redaction():
    """what a redactor changes, on ten copies of one poem"""
    f = REUSE / "nibelungen_collation.tsv"
    if not f.exists():
        print("\nTHE REDACTIONS (chapter 7): not collated yet")
        return
    print("\nTHE REDACTIONS (chapter 7)")
    rs = [r for r in rows(f) if r["share"]]
    same = [float(r["share"]) for r in rs
            if r["first_redaction"] == r["second_redaction"]]
    other = [float(r["share"]) for r in rs
             if r["first_redaction"] != r["second_redaction"]]
    cov = np.median([float(r["coverage"]) for r in rs])
    ms = float(np.median(same))
    mo = float(np.median(other))
    closest = min(rs, key=lambda r: float(r["share"]))
    print(f"  {len({r['first'] for r in rs})} copies, median coverage "
          f"{cov:.0%}")
    print(f"  two copies of one redaction    {ms:.1%} of lemmas differ "
          f"({len(same)} pairs)")
    print(f"  copies of different redactions {mo:.1%} of lemmas differ "
          f"({len(other)} pairs)")
    print(f"  the closest pair in the matrix is {closest['first']} against "
          f"{closest['second']} at {float(closest['share']):.1%}, and they are "
          f"{closest['first_redaction']} against {closest['second_redaction']}")
    body = ["\\begin{tabular}{lr}", "\\hline",
            f"over the strophes a collation pairs & lemmas differing{EOL}",
            "\\hline",
            f"two copies of one redaction & {ms:.1%}{EOL}".replace("%", "\\%"),
            f"copies of different redactions & {mo:.1%}{EOL}".replace("%",
                                                                     "\\%"),
            "\\hline", "\\end{tabular}"]
    tex("tab_redaction.tex", "\n".join(body) + "\n")


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    TAB.mkdir(parents=True, exist_ok=True)
    codex()
    crosses = provenance()
    layer()
    questions(crosses)
    panel()
    redaction()
    print("\nno number in the prose that is not printed above")


if __name__ == "__main__":
    main()
