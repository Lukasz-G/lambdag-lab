# The reference corpus's gold-annotated witnesses of works by named authors,
# and everything else in it that carries a multi-witness work: the part of
# the witness inventory that needs no tagging at all.
#
# Reads the headers of the ReM 2.1 CorA-XML files and writes one row per
# text: identifier, author, title, library and shelfmark, date, tokens, and
# whether the text is one of the August attribution set under mhd_test/.
# Texts of one work are grouped by the numeric part of the identifier
# (M205A, M205P, M205S are three witnesses of the Rolandslied); an identifier
# ending in Y is the same manuscript annotated a second time in full, not a
# second witness, and is marked so.
#
#   python medieval/list_rem_gold.py
#
# Output: medieval/hydra_in/REM_GOLD.tsv

import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REM = Path(r"D:\Corpora\ReM-v2.1_coraxml\ReM-v2.1_coraxml\cora-xml")
OUT = HERE / "hydra_in" / "REM_GOLD.tsv"
MHD_TEST = HERE / "mhd_test"

# authors whose works the demonstration draws on; matched against text-author
# and against the title's "Author: Work" prefix
AUTHORS = ["Hartmann von Aue", "Wolfram von Eschenbach", "Gottfried von Straßburg",
           "Rudolf von Ems", "Ulrich von Liechtenstein", "Ulrich von Türheim",
           "Hugo von Trimberg", "Hugo von Langenstein", "Herbort von Fritzlar",
           "Bruder Hermann", "Hermann von Veldenz", "Heinrich der Glîchezære", "Heinrich (Reinhart",
           "Pfaffe Lambrecht", "Pfaffe Konrad", "Priester Wernher", "Priester Konrad",
           "Eilhart von Oberg", "Freidank", "Konrad von Würzburg", "Der Stricker", "Stricker",
           "Thomasin", "Wirnt", "Heinrich von Veldeke", "Reinmar von Zweter", "Walther",
           "Neidhart", "Konrad Fleck", "Wernher der Gartenaere", "Herrand", "Kaiserchronik",
           "Nibelungenlied", "Annolied", "Alexander", "Rolandslied"]


def field(s, t):
    m = re.search("<%s>(.*?)</%s>" % (t, t), s, re.S)
    return m.group(1).strip() if m else "-"


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    test_ids = set()
    if MHD_TEST.is_dir():
        # the August files are named by author and work, not by ReM id; map back
        # through the title later
        pass
    rows = []
    for f in sorted(REM.glob("M*.xml")):
        s = f.read_text(encoding="utf-8")
        head = s[:s.find("<layoutinfo")] if "<layoutinfo" in s else s[:20000]
        tid = re.search(r'<text id="([^"]+)"', s).group(1)
        title, author = field(head, "title"), field(head, "text-author")
        hit = any(a.lower() in (author + " " + title).lower() for a in AUTHORS)
        if not hit:
            continue
        work = re.match(r"M\d+", tid).group(0)
        rows.append(dict(id=tid, work=work, author=author if author != "-" else title.split(":")[0],
                         title=title, library=field(head, "library"), shelfmark=field(head, "library-shelfmark"),
                         time=field(head, "time"), date=field(head, "date"), tokens=s.count("<token "),
                         genre=field(head, "genre"),
                         note="same manuscript re-annotated in full" if tid.endswith("Y") or tid[-2:-1] == "y" else ""))
    by_work = {}
    for r in rows:
        by_work.setdefault(r["work"], []).append(r)
    OUT.parent.mkdir(exist_ok=True)
    with OUT.open("w", encoding="utf-8") as f:
        f.write("id\twork_group\tauthor\ttitle\tlibrary\tshelfmark\ttime\tdate\ttokens\tgenre\twitnesses_of_work\tnote\n")
        for r in sorted(rows, key=lambda r: (r["author"], r["work"], -r["tokens"])):
            nw = len([x for x in by_work[r["work"]] if not x["note"]])
            f.write("\t".join(str(x) for x in (r["id"], r["work"], r["author"], r["title"], r["library"],
                                               r["shelfmark"], r["time"], r["date"], r["tokens"], r["genre"],
                                               nw, r["note"])) + "\n")
    print("%d gold texts -> %s" % (len(rows), OUT))
    print("%-7s %-26s %-48s %7s %5s  %s" % ("id", "author", "title", "tokens", "time", "note"))
    for r in sorted(rows, key=lambda r: (r["author"], r["work"], -r["tokens"])):
        print("%-7s %-26s %-48s %7d %5s  %s" % (r["id"], r["author"][:26], r["title"][:48], r["tokens"],
                                               r["time"], r["note"]))


if __name__ == "__main__":
    main()
