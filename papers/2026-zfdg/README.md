# ZfdG — the Kaiserchronik article: reproduction kit

Scripts and frozen outputs behind the article's tables and figures.
Paper link: *added upon publication.*

The corpus is Middle High German verse in manuscript: the Kaiserchronik in 47
witnesses (Kaiserchronik digital), Hartmann von Aue in 60, Parzival in 89, the
Ambraser Heldenbuch, and the gold-annotated reference corpus. Alongside them a
second corpus of 222 normalised editions by 25 named authors supplies the axis
the manuscripts cannot: one author's several works.

## Table/figure ↔ script map

| Exhibit | Producer | Inputs |
|---|---|---|
| Reuse maps (dotplot, connectors, corpus scatter) | `make_kc_reuse_maps.py` | `kc_reuse/*.json` from `build_kc_reuse_mask.py` |
| Text-against-text reuse matrix, shingle-length sweep | `kc_reuse_pairs.py`, `kc_reuse_k_sweep.py` | the same masks |
| Rolling profile across a documented author boundary | `make_vorau_rolling_figure.py` | `scores/vorau_rolling/*.jsonl` from `run_vorau_rolling.py` |
| What is compared with what (schematic) | `make_vorau_design_figure.py` | — |
| Candidates on the oldest witness | `make_paper_tables.py::candidates` | `kc_reuse/konrad_summary.tsv` from `analyze_konrad_profile.py` ← `run_konrad_profile.py` |
| A verdict for each fragment | `make_paper_tables.py::verdicts` | `kc_reuse/kc_regions_verdicts.tsv` from `calibrate_kc_regions.py`, `kc_reuse/kc_regions_crosswork.tsv` from `calibrate_crosswork.py` ← `run_calibration_cases.py`, both over `run_kc_regions.py` |
| The rhyme-word baseline | `make_paper_tables.py::rhyme_baseline` | `rhyme_mfw.tsv`, `rhyme_mfw_authors.tsv` from `run_rhyme_mfw.py` |
| Author-by-scribe cells, the in-case gauge | `analyze_witness_cells.py`, `analyze_witness_gauge.py` | `scores/witness_cells/*`, `scores/witness_gauge/*` |

## Building the corpus

1. `make_hydra_inputs.py` gathers the witness verses for tagging; the tagged
   output is banked by `build_masked_banks.py` (POSNoise-masked) and by
   `build_lemma_full_bank.py` / `build_surface_full_bank.py` (unmasked).
2. `build_rhyme_bank.py` cuts the verse-final word from either bank, as its
   canonical lemma or as its rime (`rime.py`).
3. `fetch_mhdbdb.py` → `extract_mhdbdb.py` → `build_mhdbdb_banks.py` fetch and
   bank the normalised editions. Their part of speech is the source's own
   tagset, so the masked stream needs a tag map before it can read them; the
   unmasked lemma, rhyme and rime streams need none.
4. `run_rhyme_gate.py` is the gate that must pass before any rhyme stream is
   used: it measures whether the last word of a printed line stands at the
   rhyme, and it refuses the texts whose editions print half-lines separately.

## Rerunning

These scripts ran from an `experiments/` directory (the medieval builders from a
`medieval/` directory) with `masked/` and `lambdag.py` in the parent; paths are
kept verbatim for provenance. The masked corpora and the per-case score files
are heavy and live on the Zenodo deposit, as elsewhere in this repository; the
`.tsv` and `.json` files here are the frozen results the tables read, so every
number in the article can be checked without rebuilding anything.

## Licences of the sources

Kaiserchronik digital CC BY-SA 4.0; Hartmann von Aue digital CC BY-SA 4.0;
Parzival digital CC BY-NC-SA 4.0 (analysis only); the reference corpus CC BY-SA;
the normalised editions CC BY-NC-SA 4.0. No source text is redistributed here.
