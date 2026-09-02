# Mask ~500 balanced av_test pairs for the supervised TM verifier (phase3/pairs500/).
# Emits the QUESTIONED author per pair so CV can be author-disjoint (group = questioned author).
#
#   python phase3/prep_test500.py          # P3_NPAIRS default 500

import json, os, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from lambdag import POSNoiseMasker

TEST = HERE.parent / "german" / "av_test_novels_de.jsonl"
OUT = HERE / "pairs500"; OUT.mkdir(parents=True, exist_ok=True)
N = int(os.environ.get("P3_NPAIRS", "500"))

def write_tsv(path, sents):
    with open(path, "w", encoding="utf-8") as f:
        for s in sents:
            if s:
                f.write("\t".join(s) + "\n")

def main():
    half = N // 2; picked = []; nsame = ndiff = 0
    with open(TEST, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line); lab = int(r["label"])
            if lab == 1 and nsame >= half: continue
            if lab == 0 and ndiff >= half: continue
            t0, t1 = r["pair"]; a0, a1 = r["authors"]
            # known = larger text (trains G_A) ; questioned = smaller (scored)
            if len(t1) >= len(t0):
                known, q, ka, qa = t1, t0, a1, a0
            else:
                known, q, ka, qa = t0, t1, a0, a1
            picked.append((r["id"], lab, ka, qa, known, q))
            nsame += (lab == 1); ndiff += (lab == 0)
            if nsame >= half and ndiff >= half:
                break

    print(f"masking {len(picked)} pairs ({nsame} same / {ndiff} diff) ...")
    m = POSNoiseMasker(language="de", spacy_model="de_core_news_lg", lowercase=True)
    t0 = time.time()
    with open(HERE / "pairs500.tsv", "w", encoding="utf-8") as man:
        man.write("id\tlabel\tknown_author\tq_author\n")
        for i, (pid, lab, ka, qa, known, q) in enumerate(picked):
            write_tsv(OUT / f"{pid}_known.tsv", m.mask(known))
            write_tsv(OUT / f"{pid}_q.tsv", m.mask(q))
            man.write(f"{pid}\t{lab}\t{ka}\t{qa}\n")
            if (i + 1) % 50 == 0:
                print(f"  {i+1}/{len(picked)}  ({time.time()-t0:.0f}s)")
    print(f"done in {time.time()-t0:.0f}s -> phase3/pairs500/ + pairs500.tsv")

if __name__ == "__main__":
    main()
