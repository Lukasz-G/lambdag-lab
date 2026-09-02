# Pretraining corpus for HDC atoms: POSNoise-mask a large slice of the German reference
# novels into one sentence stream (phase3/pretrain.tsv). Unsupervised — no verification
# labels touched. Low min_count is applied later (Julia) so rare inflected function-word
# forms survive as their own tokens and get atoms.
#
#   python phase3/prep_pretrain.py
#   P3_CHARS=300000 python phase3/prep_pretrain.py

import os, sys, time
from pathlib import Path
from collections import Counter

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from lambdag import POSNoiseMasker

REF = HERE.parent / "german" / "av_reference_novels_de.jsonl"
CHARS = int(os.environ.get("P3_CHARS", "250000"))       # chars masked per author

# paradigm probes: do these inflected function-word families survive masking (get atoms)?
PROBES = ["sein", "ist", "war", "bin", "sind", "waren", "gewesen",
          "haben", "hat", "hatte", "hätte", "habe", "hast",
          "müssen", "muss", "musste", "müsste",
          "können", "kann", "konnte", "könnte",
          "werden", "wird", "wurde", "würde",
          "der", "die", "das", "dem", "den", "des",
          "ich", "mich", "mir", "mein", "meine",
          "und", "oder", "aber", "nicht", "kein", "keine"]

def main():
    import json
    recs = []
    with open(REF, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line); recs.append((r["author"], r["text"]))
    print(f"{len(recs)} reference authors; masking {CHARS} chars each ...")

    m = POSNoiseMasker(language="de", spacy_model="de_core_news_lg", lowercase=True)
    vocab = Counter(); nsent = ntok = 0
    t0 = time.time()
    with open(HERE / "pretrain.tsv", "w", encoding="utf-8") as out:
        for i, (au, tx) in enumerate(recs):
            for s in m.mask(tx[:CHARS]):
                if s:
                    out.write("\t".join(s) + "\n"); nsent += 1; ntok += len(s)
                    vocab.update(s)
            if (i + 1) % 5 == 0:
                print(f"  {i+1}/{len(recs)} authors ({time.time()-t0:.0f}s)")

    print(f"\n{nsent} sentences, {ntok} tokens, vocab {len(vocab)} (min_count=1) in {time.time()-t0:.0f}s")
    print("\nparadigm-probe coverage (count in masked stream; 0 = masked away, no atom):")
    for w in PROBES:
        c = vocab.get(w, 0)
        print(f"  {w:10} {c:6}" + ("" if c else "   <- MASKED"))

if __name__ == "__main__":
    main()
