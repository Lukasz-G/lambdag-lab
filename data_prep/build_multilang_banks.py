# Build and mask the full banks for the English, French and Polish harvests.
#
# One dataset per (language, genre), named by the conventions the German banks
# established: {folder}_{corpus}all with a reference file holding EVERY author,
# because cross-genre work matches identity across banks and a 50/50 split
# would thin the pool quadratically (see build_av.build_full_bank). Masking is
# the standard POSNoise pass with each language's own spaCy model; measured on
# the German banks it runs near 24k tokens/s in one process, so the ~106M
# words here are an hour's local work sharded by dataset, not a rental.
#
#   python data_prep/build_multilang_banks.py --dataset en_pgprose
#   python data_prep/build_multilang_banks.py --list
#
# Output: data/{folder}/av_reference_{corpus}all_{iso}.jsonl (+ test stub)
#         masked/{folder}_{corpus}all/bank/*.tsv

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import build_av  # noqa: E402
import mask_corpora  # noqa: E402

# dataset key -> (jsonl path, folder, corpus tag, iso)
SETS = {}
for g in ("prose", "verse", "drama"):
    SETS[f"en_pg{g}"] = (HERE / "raw" / "gutenberg" / f"en_pg{g}_preprocessed.jsonl",
                         "english", f"pg{g}", "en")
    SETS[f"fr_pg{g}"] = (HERE / "raw" / "gutenberg" / f"fr_pg{g}_preprocessed.jsonl",
                         "french", f"pg{g}", "fr")
    SETS[f"pl_wl{g}"] = (HERE / "raw" / "wolnelektury" /
                         f"polish_wl{g}_preprocessed.jsonl",
                         "polish", f"wl{g}", "pl")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="", help="one key from --list")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    if args.list or not args.dataset:
        for k, (p, f, c, i) in SETS.items():
            print(f"  {k:12s} {'ok' if p.exists() else 'MISSING'}  -> "
                  f"masked/{f}_{c}all")
        return
    p, folder, corpus, iso = SETS[args.dataset]
    texts = {}
    for line in open(p, encoding="utf-8"):
        d = json.loads(line)
        texts[d["author_id"]] = d["text"]
    print(f"{args.dataset}: {len(texts)} authors, "
          f"{sum(len(t.split()) for t in texts.values()):,} words", flush=True)
    build_av.build_full_bank(texts, corpus, iso, folder)
    lang, model = mask_corpora.SPACY[folder]
    tg = mask_corpora.Tagger(lang, "spacy", model)
    mask_corpora.do_dataset(folder, corpus + "all", iso, tg)


if __name__ == "__main__":
    main()
