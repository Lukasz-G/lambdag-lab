# The panel read two ways: down the columns, and across them.
#
# DOWN A COLUMN is the attribution question. Each known text gives every
# questioned piece a score in the units of that text's own cohort of strangers, so
# a column says which of the short pieces this author's grammar claims. Where the
# codex carries a second work by the known author -- Hartmann's Klage under Erec,
# the Stricker's Frauenehre under the Pfaffe Amis -- that piece is a positive
# control and its score is what a genuine attribution looks like here.
#
# ACROSS A ROW is the question no pair of short texts can be asked directly. Two
# pieces of four thousand tokens cannot be scored against each other, because both
# sides would then be grammars of four thousand tokens; but if one poet wrote two
# of them, the two should rank the panel's authors alike. The rows are therefore
# correlated with each other, which is the statistic that showed the Rolandslied's
# two witnesses agreeing at r = 0.94 across the chronicle's regions.
#
# A correlation over six knowns is a weak instrument and is reported as such: the
# positive controls give the scale against which to read it, since a pair that is
# genuinely one author should agree at least as well as they do.
#
#   python experiments/panel_matrix.py
#
# Output: printed, plus medieval/kc_reuse/ambraser_panel.tsv

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SCORES = ROOT / "scores" / "konrad_profile"
OUT = ROOT / "medieval" / "kc_reuse" / "ambraser_panel.tsv"

KNOWN = [("erec", "Hartmann, Erec", "hartmann_von_aue"),
         ("iwein", "Hartmann, Iwein", "hartmann_von_aue"),
         ("amis", "the Stricker, Pfaffe Amis", "stricker"),
         ("frauenbuch", "Ulrich von Liechtenstein", "ulrich_von_liechtenstein"),
         ("helmbrecht", "Wernher der Gartenaere", "wernher_der_gartenaere"),
         ("nibelungenlied", "the Nibelungenlied", None),
         ("dietrichs_flucht", "Dietrichs Flucht", None)]
SHORT = {"moriz_von_craun": "Moriz von Craun", "der_mantel": "Der Mantel",
         "von_dem_uebelen_wibe": "Von dem uebelen Wibe",
         "zweites_buechlein": "Zweites Buechlein",
         "klage": "Hartmann's Klage", "frauenehre": "the Stricker's Frauenehre",
         "titurel": "Wolfram's Titurel",
         "der_nackte_kaiser": "Herrand, Der nackte Kaiser"}


def short_name(text):
    for k, v in SHORT.items():
        if text.endswith(k + "__d"):
            return k
    return text


def spread(v):
    m = float(np.median(v))
    return m, max(1.4826 * float(np.median(np.abs(v - m))),
                  (float(np.quantile(v, 0.9)) - m) / 1.2816, 1e-9)


def author_of(text):
    """the author a questioned piece is known to belong to, or None"""
    for _, _, a in KNOWN:
        if a and f"__{a}__" in text:
            return a
    if "wolfram_von_eschenbach" in text:
        return "wolfram_von_eschenbach"
    if "herrand_von_wildonie" in text:
        return "herrand_von_wildonie"
    return None


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    z = defaultdict(dict)          # piece -> known -> median z
    for tag, label, author in KNOWN:
        hits = sorted(SCORES.glob(f"regions__*__panel_{tag}.jsonl"))
        if not hits:
            print(f"  (no run yet for {label})")
            continue
        rows = [json.loads(l) for l in hits[0].open(encoding="utf-8")]
        meta, rows = rows[0], rows[1:]
        by = defaultdict(lambda: defaultdict(list))
        for r in rows:
            by[r["region"]][r["kind"]].append((r["text"], r["lambda_G"]))
        per = defaultdict(list)
        for ri in sorted(by):
            coh = [v for _, v in by[ri]["cohort"]]
            if not coh:
                continue
            m, sd = spread(np.array(coh))
            for kind in ("candidate", "control"):
                for t, v in by[ri][kind]:
                    per[short_name(t)].append((v - m) / sd)
        for piece, vals in per.items():
            z[piece][tag] = float(np.median(vals))
        z["__meta__"][tag] = (label, meta["regions"], meta["region_len"],
                              author)

    meta = z.pop("__meta__")
    tags = [t for t, _, _ in KNOWN if t in meta]
    order = ["moriz_von_craun", "der_mantel", "von_dem_uebelen_wibe",
             "zweites_buechlein", "klage", "frauenehre", "titurel",
             "der_nackte_kaiser"]
    order = [p for p in order if p in z]

    print("THE PANEL: each short piece against each known text, in deviations of "
          "that text's own cohort")
    print(f"  {'':28}" + "".join(f"{meta[t][0][:13]:>15}" for t in tags))
    print(f"  {'':28}" + "".join(f"{str(meta[t][1]) + ' regions':>15}"
                                 for t in tags))
    for piece in order:
        row = "".join(f"{z[piece].get(t, float('nan')):>15.2f}" for t in tags)
        star = ""
        for t in tags:
            if meta[t][3] and meta[t][3] in piece:
                star = "   <- the same author, by attribution"
        print(f"  {SHORT[piece][:26]:28}{row}{star}")

    # the positive controls: a piece whose author IS the known text's author
    print("\n  the positive controls, which say what an attribution looks like "
          "here:")
    for piece in order:
        for t in tags:
            a = meta[t][3]
            if a and (a in piece or (a == "hartmann_von_aue" and piece == "klage")
                      or (a == "stricker" and piece == "frauenehre")):
                print(f"    {SHORT[piece]:28} under {meta[t][0]:26} "
                      f"{z[piece].get(t, float('nan')):+.2f}")

    print("\nACROSS THE PANEL: do two short pieces rank the known texts alike?")
    print("  (a pair by one poet should agree; six knowns make this a weak test)")
    common = [p for p in order if len(z[p]) == len(tags)]
    print(f"  {'':28}" + "".join(f"{SHORT[p][:11]:>13}" for p in common))
    corr = {}
    for a in common:
        va = np.array([z[a][t] for t in tags])
        line = ""
        for b in common:
            vb = np.array([z[b][t] for t in tags])
            r = float(np.corrcoef(va, vb)[0, 1]) if a != b else 1.0
            corr[(a, b)] = r
            line += f"{r:>13.2f}"
        print(f"  {SHORT[a][:26]:28}{line}")
    anon = [p for p in common if p in
            ("moriz_von_craun", "der_mantel", "von_dem_uebelen_wibe",
             "zweites_buechlein")]
    pairs = [(a, b, corr[(a, b)]) for i, a in enumerate(anon)
             for b in anon[i + 1:]]
    if pairs:
        pairs.sort(key=lambda x: -x[2])
        print("\n  the anonymous pieces ranked by agreement with each other:")
        for a, b, r in pairs:
            print(f"    {SHORT[a]:26} and {SHORT[b]:26} r = {r:+.2f}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as fh:
        fh.write("piece\t" + "\t".join(tags) + "\n")
        for piece in order:
            fh.write(SHORT[piece] + "\t"
                     + "\t".join(f"{z[piece].get(t, float('nan')):.4f}"
                                 for t in tags) + "\n")
    print(f"\n  wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
