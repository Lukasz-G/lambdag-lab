# Witness texts for tagging: one file per witness, one verse per line.
#
# Gathers the verse records of the four witness sources extracted so far
# (Kaiserchronik digital, Hartmann von Aue digital, Parzival digital, the
# Ambraser Heldenbuch) into a single directory in the form the tagger takes:
# a header line naming the witness and its size, then one verse per line,
# words separated by single spaces, no punctuation. The diplomatic layer is
# written for every source; the regularised layer of the Hartmann editions is
# written beside it under its own suffix so that both can be tagged and
# compared.
#
# A manifest lists every file with its source, work, author, siglum,
# manuscript, size, licence and, where the reference corpus already carries
# a gold-annotated excerpt of the same manuscript, that excerpt's identifier.
#
#   python medieval/make_hydra_inputs.py [--min-verses 50]
#
# Output: medieval/hydra_in/<source>__<work>__<siglum>[__reg].txt
#         medieval/hydra_in/MANIFEST.tsv

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "hydra_in"

SOURCES = [
    # (label, path, licence)
    ("kcd", HERE / "raw" / "kcd" / "kcd_verses.jsonl", "CC BY-SA 4.0"),
    ("hvad", HERE / "raw" / "hvad" / "hvad_verses.jsonl", "CC BY-SA 4.0"),
    ("parzival", HERE / "raw" / "parzival" / "parzival_verses.jsonl", "CC BY-NC-SA 4.0"),
    ("ambraser", HERE / "raw" / "ambraser" / "ambraser_verses.jsonl", "CC BY 4.0"),
]

# the reference corpus's own gold-annotated excerpts of the same manuscripts
GOLD = {
    ("hvad", "hartmann_von_aue__iwein", "B"): "M312 (Gießen Hs. 97, 15,231 tokens)",
    ("parzival", "wolfram_von_eschenbach__parzival", "d"): "M325 (St. Gallen 857, 15,202 tokens)",
    ("kcd", "anon__kaiserchronik", "A1"): "M121Y (Vorau 276, 87,455 tokens)",
}
# the same manuscript transcribed by two projects: one witness, two transcriptions
SAME_MS = {
    ("hvad", "hartmann_von_aue__iwein", "d"): "ambraser__hartmann_von_aue__iwein__d",
    ("ambraser", "hartmann_von_aue__iwein", "d"): "hvad__hartmann_von_aue__iwein__d",
}


def records():
    for label, path, licence in SOURCES:
        if not path.exists():
            print("missing:", path, file=sys.stderr)
            continue
        for line in path.open(encoding="utf-8"):
            r = json.loads(line)
            if label == "kcd":
                r = dict(work="anon__kaiserchronik", author="anon.", siglum=r["siglum"],
                         title=r.get("title", "Kaiserchronik"), manuscript=r.get("title", "-"),
                         verses=r["verses"], verse_ids=[str(i + 1) for i in range(len(r["verses"]))],
                         recension=r.get("recension", "-"))
            elif label == "ambraser":
                r = dict(work=r["key"], author=r["author"], siglum="d", title=r["title"],
                         manuscript="Wien, ÖNB, Cod. ser. nova 2663 (Ambraser Heldenbuch)",
                         verses=r["verses"], verse_ids=r["verse_ids"])
            yield label, licence, r


def dating():
    """witness -> (date, period) from collect_dates.py; the period names the
    reference corpus whose span covers the manuscript"""
    p = HERE / "raw" / "witness_dates.json"
    d = json.load(p.open(encoding="utf-8")) if p.exists() else {}

    def look(name):
        e = d.get(name) or (d.get("ambraser__*") if name.startswith("ambraser__") else None)
        if not e:
            return "-", "-"
        y = e.get("year")
        if y is None:
            return e["date"], "-"
        return e["date"], ("ReM period, to 1350" if y <= 1350 else "ReF period, 1350-1650")
    return look


def write(path, name, verses):
    with path.open("w", encoding="utf-8") as f:
        f.write("### %s (%d verses)\n" % (name, len(verses)))
        for v in verses:
            f.write(" ".join(v) + "\n")


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-verses", type=int, default=50)
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    recs = [(label, licence, r) for label, licence, r in records() if r["verses"]]
    # sigla differ by case only (Iwein A and a are different manuscripts) and
    # this filesystem does not: where two sigla of one work collide, the
    # lowercase one is written with the suffix _lc; the manifest keeps the siglum
    seen = {}
    for label, _, r in recs:
        seen.setdefault((label, r["work"], r["siglum"].lower()), set()).add(r["siglum"])
    rows = []
    look = dating()
    for label, licence, r in recs:
        verses = r["verses"]
        if isinstance(verses[0], str):
            verses = [v.split() for v in verses]
        if len(verses) < args.min_verses:
            continue
        sig = r["siglum"]
        fsig = sig + "_lc" if (len(seen[(label, r["work"], sig.lower())]) > 1 and sig[0].islower()) else sig
        name = "%s__%s__%s" % (label, r["work"], fsig)
        write(OUT / (name + ".txt"), name, verses)
        ntok = sum(map(len, verses))
        key = (label, r["work"], sig)
        date, period = look("%s__%s__%s" % (label, r["work"], sig))
        rows.append((name, label, r["work"], r["author"], sig, r.get("manuscript", "-")[:80], date, period,
                     len(verses), ntok, "diplomatic", licence, GOLD.get(key, "-"), SAME_MS.get(key, "-")))
        if r.get("verses_reg"):
            write(OUT / (name + "__reg.txt"), name + "__reg", r["verses_reg"])
            rows.append((name + "__reg", label, r["work"], r["author"], sig,
                         r.get("manuscript", "-")[:80], date, period, len(r["verses_reg"]),
                         sum(map(len, r["verses_reg"])), "regularised", licence, "-", "-"))
    rows.sort(key=lambda x: (x[3], x[2], -x[8]))
    with (OUT / "MANIFEST.tsv").open("w", encoding="utf-8") as f:
        f.write("file\tsource\twork\tauthor\tsiglum\tmanuscript\tdate\tperiod\tverses\ttokens\tlayer\tlicence"
                "\tgold_in_ReM\tsame_manuscript_as\n")
        for row in rows:
            f.write("\t".join(str(x) for x in row) + "\n")
    print("%d files -> %s" % (len(rows), OUT))
    # summary per author and work: witnesses and tokens (diplomatic layer)
    from collections import defaultdict
    agg = defaultdict(lambda: [0, 0, []])
    for row in rows:
        if row[10] != "diplomatic":
            continue
        k = (row[3], row[2])
        agg[k][0] += 1; agg[k][1] += row[9]; agg[k][2].append("%s:%dk" % (row[4], round(row[9] / 1000)))
    print("%-26s %-42s %4s %8s  %s" % ("author", "work", "wit", "tokens", "witness:ktok"))
    for (a, w), (n, t, sig) in sorted(agg.items()):
        print("%-26s %-42s %4d %8d  %s" % (a[:26], w[:42], n, t, " ".join(sig[:14]) + (" ..." if len(sig) > 14 else "")))


if __name__ == "__main__":
    main()
