# Split the German prose harvest into DOMAINS, by the library's own shelf marks.
#
# WHY. The harvest classifies genre from TEI markup, which separates drama and
# verse but files every paragraph text as "prose" -- so the prose banks mix
# novels with essays, memoirs, letters and, in one case, the Luther Bible
# (audit_domain_purity.py, 2026-09-10). For within-genre work that mixture was
# tolerable; for domain experiments it is the variable under test. This script
# rebuilds the prose side as three datasets keyed on the Digitale Bibliothek's
# own collection labels, which its editors assigned per work:
#
#   german_tgbell     belletristic prose (Romane, Erzaehlungen, Maerchen, ...)
#   german_tgessay    essays, treatises, criticism, aphorisms
#   german_tgautob    autobiography, memoirs
#
# WHAT IS DELIBERATELY LEFT OUT, and why:
#   scripture-translation  the Luther Bible: a translation, not idiolect prose.
#   letters                one author clears 20k words; not a dataset.
#   travel                 two authors; folded into neither essay nor prose.
#   dialogic essays        theoretical writings in dialogue form carry <sp>
#                          markup (Gottsched, Gerstenberg, Lessing's Aesthetische
#                          Schriften). Their register is essay but their FORM is
#                          dramatic; admitting them would leak drama structure
#                          into the essay bank, so markup==prose is required.
#   non-DigiBib objects    ELTeC novels and almanacs lack the shelf mark, and
#                          their author attribution lives in the harvest query,
#                          not the file (the first unscanned object is an
#                          anthology "hrsg. von ..."). Excluding them keeps every
#                          domain bank single-provenance -- one edition
#                          tradition -- at the price of smaller belletristic
#                          banks for ELTeC-heavy authors.
#
# The existing german_tgprose/verse/drama datasets and their banks are NOT
# touched: results are committed against them, and the point of this split is
# comparison, not revision.
#
#   python data_prep/split_domains.py             # write the three JSONLs
#   python data_prep/split_domains.py --build     # + full banks + masking
#
# Output: data_prep/raw/textgrid/german_tg{bell,essay,autob}_preprocessed.jsonl
#         (with --build) data/german/av_*_tg*all_de.jsonl and
#         masked/german_tg{bell,essay,autob}all/bank/*.tsv

import argparse
import json
import sys
import time
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from audit_domain_purity import DOMAIN  # noqa: E402
from fetch_textgrid import (TEI, TEIC, dupkeys, load_exclusions,  # noqa: E402
                            slug, text_of)

RAW = HERE / "raw" / "textgrid"
SCAN = HERE / "domain_scan.jsonl"
STEMS = {"belletristic": "german_tgbell", "essay": "german_tgessay",
         "autobiography": "german_tgautob"}
MIN_WORDS = 8_000          # the harvest's own per-record floor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true",
                    help="also build full banks and mask them")
    ap.add_argument("--min-words", type=int, default=MIN_WORDS)
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    excl = load_exclusions()
    rows = [json.loads(l) for l in open(SCAN, encoding="utf-8")]
    skipped = Counter()
    picked = defaultdict(list)             # (author_slug, domain) -> scan rows
    names = {}
    for r in rows:
        dom = DOMAIN.get(r["coll"].lower(), "other")
        if r["markup"] != "prose":
            if dom in ("belletristic", "essay", "autobiography"):
                skipped[f"{dom}, {r['markup']} markup"] += r["words"]
            continue
        if dom not in STEMS:
            skipped[dom] += r["words"]
            continue
        a = slug(r["author"])
        pat = excl.get(a)
        if pat and (pat.search(r["title"]) or pat.search(r["coll"])):
            skipped["excluded title"] += r["words"]
            continue
        names[a] = r["author"]
        picked[(a, dom)].append(r)

    print("skipped (words): "
          + ", ".join(f"{k} {v:,}" for k, v in skipped.most_common(10)))

    out = {dom: {} for dom in STEMS}
    t0 = time.time()
    for (a, dom), rs in sorted(picked.items()):
        rs.sort(key=lambda r: (r["coll"], r["title"], r["uri"]))
        seen, parts, words = set(), [], 0
        dropped = 0
        for r in rs:
            f = TEIC / f"{r['uri']}.xml"
            if not f.exists():
                continue
            try:
                root = ET.fromstring(f.read_text(encoding="utf-8",
                                                 errors="replace"))
            except ET.ParseError:
                continue
            txt = text_of(root)
            if len(txt.split()) < 20:
                continue
            keys = dupkeys(txt)
            if any(k in seen for k in keys):
                dropped += 1
                continue
            seen.update(keys)
            parts.append(txt)
            words += len(txt.split())
        if words >= args.min_words:
            out[dom][a] = {"author_id": a, "author_name": names[a],
                           "id_source": f"textgrid-digibib:{dom}",
                           "text": "\n\n".join(parts)}
        elif words:
            skipped[f"below floor ({dom})"] += words

    for dom, stem in STEMS.items():
        p = RAW / f"{STEMS[dom]}_preprocessed.jsonl"
        with open(p, "w", encoding="utf-8") as fh:
            for a in sorted(out[dom]):
                fh.write(json.dumps(out[dom][a], ensure_ascii=False) + "\n")
        tot = sum(len(v["text"].split()) for v in out[dom].values())
        big = sorted(out[dom], key=lambda a: -len(out[dom][a]["text"].split()))
        print(f"\n{stem}: {len(out[dom])} authors, {tot:,} words "
              f"({time.time() - t0:.0f}s)")
        for a in big[:10]:
            print(f"    {a[:34]:36s} {len(out[dom][a]['text'].split()):9,d}")

    if not args.build:
        return

    # ---- full banks (the *all convention build_av established) -------------
    import build_av
    for dom, stem in STEMS.items():
        corpus = stem.split("_", 1)[1]                    # tgbell
        texts = {a: v["text"] for a, v in out[dom].items()}
        if texts:
            build_av.build_full_bank(texts, corpus, "de", "german")

    # ---- masking, driven directly so no existing dataset is re-touched -----
    import mask_corpora
    lang, model = mask_corpora.SPACY["german"]
    tg = mask_corpora.Tagger(lang, "spacy", model)
    for dom, stem in STEMS.items():
        corpus = stem.split("_", 1)[1] + "all"
        if out[dom]:
            mask_corpora.do_dataset("german", corpus, "de", tg)


if __name__ == "__main__":
    main()
