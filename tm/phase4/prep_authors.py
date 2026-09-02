# Phase 4: mask ALL reference authors into per-author TSVs — the raw material for MASS
# pair generation (the reference corpus used as SCD-style supervised training examples).
# Also verifies train/test author disjointness (reference authors vs av_test pair authors).
#
#   python phase4/prep_authors.py          # P4_CHARS chars/author, default 400k

import json, os, re, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from lambdag import POSNoiseMasker

REF = HERE.parent / "german" / "av_reference_novels_de.jsonl"
TESTMAN = HERE.parent / "phase3" / "pairs500.tsv"
OUT = HERE / "authors"; OUT.mkdir(parents=True, exist_ok=True)
CHARS = int(os.environ.get("P4_CHARS", "400000"))

def slug(a): return re.sub(r"[^a-z0-9]+", "_", a.lower()).strip("_")

def main():
    recs = []
    with open(REF, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line); recs.append((r["author"], r["text"]))
    ref_authors = {a for a, _ in recs}

    test_authors = set()
    with open(TESTMAN, encoding="utf-8") as f:
        next(f)
        for line in f:
            pid, lab, ka, qa = line.rstrip("\n").split("\t")
            test_authors.add(ka); test_authors.add(qa)
    overlap = ref_authors & test_authors
    print(f"reference authors: {len(ref_authors)}, test-pair authors: {len(test_authors)}, "
          f"OVERLAP: {len(overlap)} {sorted(overlap) if overlap else '(disjoint — clean protocol)'}")

    m = POSNoiseMasker(language="de", spacy_model="de_core_news_lg", lowercase=True)
    t0 = time.time()
    with open(HERE / "authors.tsv", "w", encoding="utf-8") as man:
        man.write("idx\tauthor\tfile\tsents\ttokens\n")
        for i, (au, tx) in enumerate(sorted(recs, key=lambda r: r[0])):
            sents = [s for s in m.mask(tx[:CHARS]) if s]
            fn = f"{i:02d}_{slug(au)}.tsv"
            with open(OUT / fn, "w", encoding="utf-8") as f:
                for s in sents:
                    f.write("\t".join(s) + "\n")
            ntok = sum(len(s) for s in sents)
            man.write(f"{i}\t{au}\t{fn}\t{len(sents)}\t{ntok}\n")
            if (i + 1) % 7 == 0:
                print(f"  {i+1}/{len(recs)} authors ({time.time()-t0:.0f}s)")
    print(f"done in {time.time()-t0:.0f}s -> phase4/authors/ + authors.tsv")

if __name__ == "__main__":
    main()
