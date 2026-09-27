# Tagged witnesses -> masked banks in the layout the experiment runners read.
#
# Input: the tagger's four-column files (surface, lemma, POS, morphology; one
# token per line, the witness header passed through as an '@' line) in
# medieval/hydra_out/, and the same witnesses' one-verse-per-line originals in
# medieval/hydra_in_txt/, from which the verse boundaries are restored by
# token count -- the tagger tokenises on whitespace exactly as the originals
# are written, so the counts match line for line. The reference corpus's own
# gold-annotated texts (medieval/hydra_in/REM_GOLD.tsv) enter the same banks
# from their tab files; they carry no verse structure and are cut into
# twenty-five-token units, which the runners re-tile anyway.
#
# Masking is POSNoise for Middle High German on the pre-tagged stream, the
# HiTS tags bridged to UD as for ReM, matched on lemma as well as surface
# (the lists are in normalised orthography, the witnesses in the scribes'),
# lowercased. Two emissions are written, as two datasets:
#   mhg_witnesses_lemma     kept function words emitted as LEMMAS -- the
#                           stream of the whole medieval programme
#   mhg_witnesses_surface   kept function words emitted as the scribe wrote
#                           them -- the stream that still carries the scribe,
#                           for the scribe-toll comparison
# ONE LEMMA CONVENTION. The tagger was trained on four corpora and lemmatises
# each token in the convention of the corpus it resembles, so 'vnd' comes
# back as 'unte' in a thirteenth-century witness and as 'und' in a
# sixteenth-century one, 'ist' as 'sîn' there and 'sein' here. Left as it is,
# the period would masquerade as a difference between witnesses of one work.
# Every lemma outside the reference corpus's own inventory is therefore
# mapped to the reference convention through the surface forms the two
# conventions share: the lemma's surfaces in the tagged witnesses are looked
# up in the reference corpus's gold annotation, and the reference lemma that
# accounts for most of that surface mass is taken. The map is written beside
# the banks (lemma_canon.tsv) so that every mapping can be read and refused.
# Tokens carrying no analysis ('[!]' and '[!!]', the second of which stands
# for an unannotated stretch and can run to most of a witness) and
# foreign-material tokens are dropped before masking, from both emissions
# alike, so that the two streams stay unit for unit the same text. Each dataset holds bank/<witness>.tsv (one
# unit per line, masked tokens tab-separated), verse_ids/<witness>.txt (the
# edition's verse number of every unit, for disjointness bookkeeping) and a
# MANIFEST.tsv with source, author, work, siglum, layer and token counts.
#
#   python medieval/build_masked_banks.py [--only NAME_SUBSTRING]
#
# Output: masked/mhg_witnesses_{lemma,surface}/

import csv
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
from lambdag import POSNoiseMasker  # noqa: E402

TAGGED = HERE / "hydra_out"
ORIG = HERE / "hydra_in_txt"
MAN = HERE / "hydra_in" / "MANIFEST.tsv"
REM_GOLD = HERE / "hydra_in" / "REM_GOLD.tsv"
REM_TAB = Path(r"D:\Corpora\ReM-v2.1_tab")
REM_VOCAB = Path.home() / "github" / "hydra_august25" / "hydra" / "runs" / "s_crf" / "vocab.json"
VERSES = {"kcd": HERE / "raw" / "kcd" / "kcd_verses.jsonl", "hvad": HERE / "raw" / "hvad" / "hvad_verses.jsonl",
          "parzival": HERE / "raw" / "parzival" / "parzival_verses.jsonl",
          "ambraser": HERE / "raw" / "ambraser" / "ambraser_verses.jsonl"}
OUT = {"lemma": ROOT / "masked" / "mhg_witnesses_lemma", "surface": ROOT / "masked" / "mhg_witnesses_surface"}
DROP_POS = {"FM"}              # foreign material
DROP_LEMMA = {"[!]", "[!!]"}   # the placeholders for material carrying no analysis
REM_UNIT = 25


def read_tagged(path):
    """[(surface, pos, lemma)] and the header line"""
    header, toks = "", []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("@"):
            header = line[1:].strip(); continue
        c = line.split("\t")
        if len(c) < 4:
            continue
        toks.append((c[0], c[2], c[1]))
    return header, toks


def verse_counts(path):
    return [len(l.split()) for l in path.read_text(encoding="utf-8").splitlines()[1:] if l.strip()]


def verse_ids_of(name):
    """the edition's verse numbers for a witness, from the extraction records"""
    src, rest = name.split("__", 1)
    rest = rest.replace("__reg", "")
    work, sig = rest.rsplit("__", 1)
    sig = sig[:-3] if sig.endswith("_lc") else sig
    path = VERSES.get(src)
    if not path or not path.exists():
        return None
    for line in path.open(encoding="utf-8"):
        r = json.loads(line)
        key = r.get("key") or r.get("work")
        if src == "kcd":
            if r["siglum"] == sig:
                return [str(i + 1) for i in range(len(r["verses"]))]
        elif key == work and r.get("siglum", "d") == sig:
            return [str(v) for v in r["verse_ids"]]
    return None


def fold(s):
    return s.lower().replace("ſ", "s").strip("[]")


def head_pos(pos):
    return pos.split("+", 1)[0]


def rem_surface_lemma():
    """(surface folded, head POS) -> {lemma: count} over the reference corpus's
    gold, plus the POS-free marginal under the key (surface, '*')"""
    from collections import Counter, defaultdict
    d = defaultdict(Counter)
    for f in REM_TAB.glob("M*.txt"):
        for line in f.read_text(encoding="utf-8").splitlines():
            c = line.split("\t")
            if len(c) >= 3 and c[0].strip("[]"):
                sf = fold(c[0])
                d[(sf, head_pos(c[2]))][c[1]] += 1
                d[(sf, "*")][c[1]] += 1
    return d


def build_canon_map(files, min_count=5):
    """(lemma outside the reference inventory, head POS) -> reference lemma, by
    the surfaces the two conventions share under that POS; the POS separates
    readings the surface alone conflates (the conjunction and the
    demonstrative 'daz', the copula and the possessive 'sein')"""
    from collections import Counter, defaultdict
    # a lemma counts as the reference convention only if the reference corpus
    # uses it at least ten times: 'der', 'er', 'da' and 'an' occur there once
    # to four times as lemmas and would otherwise pass as its own
    counts = json.load(REM_VOCAB.open(encoding="utf-8"))["lemma_counts"]
    inventory = {l for l, c in counts.items() if c >= 10}
    surf_of = defaultdict(Counter)          # (model lemma, POS) -> the surfaces it was given to
    for f in files:
        for line in f.read_text(encoding="utf-8").splitlines():
            c = line.split("\t")
            if len(c) >= 4:
                surf_of[(c[1], head_pos(c[2]))][fold(c[0])] += 1
    gold = rem_surface_lemma()
    cmap, report = {}, []
    for (lemma, pos), surfs in surf_of.items():
        n = sum(surfs.values())
        if n < min_count:
            continue
        votes = Counter()
        for sf, k in surfs.items():
            g_exact = gold.get((sf, pos))
            g = g_exact or gold.get((sf, "*"))
            if g:
                tot = sum(g.values())
                for gl, gk in g.items():
                    votes[gl] += k * gk / tot
            # a frequent surface decides for itself, whatever the lemma's
            # standing: the reference lemmatises 'ist' and 'was' differently
            # although the model gave both one lemma, and it lemmatises the
            # surface 'er' as 'ër' where the model wrote 'er'; only the exact
            # (surface, POS) evidence may overrule, never the marginal
            if g_exact and k >= 20:
                b, gk = g_exact.most_common(1)[0]
                tot_e = sum(g_exact.values())
                if b != lemma and b in inventory and gk >= 0.5 * tot_e:
                    cmap[(lemma, pos, sf)] = b
                    report.append((k, lemma + "  [" + sf + "]", pos, b, gk / tot_e))
        if lemma in inventory or not votes:
            continue
        best, w = votes.most_common(1)[0]
        if best in inventory and w >= 0.5 * n:
            cmap[(lemma, pos)] = best
            report.append((n, lemma, pos, best, w / n))
    return cmap, sorted(report, reverse=True)


def canon(cmap, surface, pos, lemma):
    p = head_pos(pos)
    return cmap.get((lemma, p, fold(surface))) or cmap.get((lemma, p)) or lemma


def keep(t):
    return t[2] not in DROP_LEMMA and not (set(t[1].split("+")) & DROP_POS)


def write_bank(ds, name, units, ids, masker):
    masked = masker.mask_tagged(units)
    bank = OUT[ds] / "bank"; vdir = OUT[ds] / "verse_ids"
    bank.mkdir(parents=True, exist_ok=True); vdir.mkdir(parents=True, exist_ok=True)
    kept_ids, lines = [], []
    for u, vid in zip(masked, ids):
        if u:
            lines.append("\t".join(u)); kept_ids.append(vid)
    (bank / (name + ".tsv")).write_text("\n".join(lines) + "\n", encoding="utf-8")
    (vdir / (name + ".txt")).write_text("\n".join(kept_ids) + "\n", encoding="utf-8")
    return sum(len(u) for u in masked)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    only = sys.argv[sys.argv.index("--only") + 1] if "--only" in sys.argv else None
    maskers = {ds: POSNoiseMasker.pretagged("gmh", segment="sentence", emit=ds, lowercase=True)
               for ds in OUT}
    print("gmh list:", maskers["lemma"].pattern_list_path.name)
    man = {r["file"]: r for r in csv.DictReader(open(MAN, encoding="utf-8"), delimiter="\t")}
    files = [f for f in sorted(TAGGED.glob("*.txt")) if not only or only in f.stem]
    cmap, report = build_canon_map(files)
    for ds in OUT:
        OUT[ds].mkdir(parents=True, exist_ok=True)
        with (OUT[ds] / "lemma_canon.tsv").open("w", encoding="utf-8") as fh:
            fh.write("tokens\tmodel_lemma\tpos\treference_lemma\tshare_of_surface_mass\n")
            for n, l, p_, b, sh in report:
                fh.write("%d\t%s\t%s\t%s\t%.2f\n" % (n, l, p_, b, sh))
    print("%d (lemma, POS) pairs mapped to the reference convention; the twenty most frequent:" % len(cmap))
    for n, l, p_, b, sh in report[:20]:
        print("   %7d  %-12s %-7s -> %-12s (%.2f)" % (n, l, p_, b, sh))
    rows = []
    for f in files:
        name = f.stem
        header, toks = read_tagged(f)
        toks = [(s_, p_, canon(cmap, s_, p_, l_)) for s_, p_, l_ in toks]
        counts = verse_counts(ORIG / f.name)
        if sum(counts) != len(toks):
            print("TOKEN COUNT MISMATCH", name, sum(counts), len(toks)); continue
        ids = verse_ids_of(name) or [str(i + 1) for i in range(len(counts))]
        if len(ids) != len(counts):
            ids = [str(i + 1) for i in range(len(counts))]
        units, i = [], 0
        for n in counts:
            units.append([t for t in toks[i:i + n] if keep(t)]); i += n
        m = man.get(name, {})
        sizes = {ds: write_bank(ds, name, units, ids, maskers[ds]) for ds in OUT}
        rows.append((name, m.get("source", name.split("__")[0]), m.get("author", "-"), m.get("work", "-"),
                     m.get("siglum", "-"), m.get("layer", "diplomatic"), m.get("date", "-"), len(units),
                     len(toks), sizes["lemma"], "tagged", "V"))
        print("%-56s %6d verses %7d tokens -> %7d masked" % (name, len(units), len(toks), sizes["lemma"]))
    # the reference corpus's gold texts, from their tab files
    if REM_GOLD.exists() and not only:
        for r in csv.DictReader(open(REM_GOLD, encoding="utf-8"), delimiter="\t"):
            if r["note"]:
                continue                  # the re-annotated duplicate of a manuscript already present
            tab = REM_TAB / (r["id"] + ".txt")
            if not tab.exists():
                print("missing ReM tab file", r["id"]); continue
            toks = []
            for line in tab.read_text(encoding="utf-8").splitlines():
                c = line.split("\t")
                if len(c) >= 3 and c[0].strip("[]"):
                    toks.append((c[0].strip("[]"), c[2], c[1]))
            toks = [t for t in toks if keep(t)]
            units = [toks[i:i + REM_UNIT] for i in range(0, len(toks), REM_UNIT)]
            ids = [str(i + 1) for i in range(len(units))]
            name = "rem__" + r["id"]
            sizes = {ds: write_bank(ds, name, units, ids, maskers[ds]) for ds in OUT}
            rows.append((name, "rem", r["author"], r["title"], r["id"], "diplomatic (gold)", r["time"],
                         len(units), len(toks), sizes["lemma"], "gold", r["genre"]))
            print("%-56s %6d units  %7d tokens -> %7d masked" % (name, len(units), len(toks), sizes["lemma"]))
    for ds in OUT:
        with (OUT[ds] / "MANIFEST.tsv").open("w", encoding="utf-8") as f:
            f.write("file\tsource\tauthor\twork\tsiglum\tlayer\tdate\tunits\ttokens\tmasked_tokens\tannotation\tgenre\n")
            for row in rows:
                f.write("\t".join(str(x) for x in row) + "\n")
    print("%d banks -> %s" % (len(rows), ", ".join(str(p) for p in OUT.values())))


if __name__ == "__main__":
    main()
