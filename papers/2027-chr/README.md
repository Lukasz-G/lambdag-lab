# CHR 2027 — "Grammar on Trial" reproduction kit

Scripts and frozen outputs behind the paper's tables and figures.
Paper link: *added upon publication.*

## Table/figure ↔ script map

| Exhibit | Producer | Inputs |
|---|---|---|
| Table 1 (datasets) | `make_paper_tables.py` (coverage) | `masked/*/DONE` |
| Table 2 (calibration, full length) | `make_paper_tables.py::write_calibration` + `_cllrmin_full` | `scores/*__kn__sent__L0.jsonl` |
| Table 3 (evidence lengths) | `write_lengths` | `primary.log` / score files |
| Table 4 (long texts + bank cal.) | `write_longtexts` | `scores/*__kn__long__L*.jsonl`, `run_bankcal.py` outputs |
| Table 5/6 (segmentation) | `write_segmentation` | `segmentation.log`, `scores/*__kn__w20__*` |
| Figures (logic, calibration, pattern table, lists) | `make_figures.py` | shipped lists |
| Heat maps (EN/PL/LT) | `make_heatmaps.py` | av_test JSONL + spaCy models |

## Rerunning

These scripts ran from an `experiments/` directory with `masked/` and
`lambdag.py` in its parent (paths are kept verbatim for provenance). To rerun:

1. `pip install -e ".[numba,spacy,dev]"` from the repo root.
2. Rebuild the masked corpora with `data_prep/` (fetch_eltec / fetch_dracor /
   fetch_poetree → mask_corpora), or download them from the Zenodo deposit
   (DOI added at release).
3. Recreate the layout the scripts expect: place `masked/` next to `lambdag.py`
   and run the `run_*.py` stages from a sibling directory, e.g.
   `python run_kn_grid.py --stage primary`.

Frozen here: `results_primary.csv`, `results_segmentation.csv` (partial-run
remnants — the score files are canonical), and the run logs (`*.log`) as
provenance for every number in the paper.

Defaults per the paper: order-10 KN, r=30, symmetric truncation, sentences as
the unit (windows in the segmentation study), 38 datasets analysed of 44
constructed (≥10 reference authors).
