# tm/ — the HDC–Tsetlin verifier (companion to the journal paper, experimental)

The supervised complement to λ_G: POSNoise-masked streams → hyperdimensional
rate sketches (the local `hdc/` Julia package) → a Fuzzy-Pattern Tsetlin
Machine verifier, plus the XGBoost adjudication of what the supervised signal
consists of. Every number in the journal paper's TM chapter regenerates from
the scripts here; result caches (per-pair margins, score files) are committed
so the tables reproduce without retraining.

**Status: experimental / pre-publication.** Single-language scope (German
novels, 500 author-disjoint evaluation pairs). APIs and layouts may change
until the journal paper is submitted.

## Dependencies

- **Julia ≥ 1.9** (campaign ran on 1.11).
- **Tsetlin.jl** (third-party, not vendored here): clone
  [BooBSD/Tsetlin.jl](https://github.com/BooBSD/Tsetlin.jl) into this
  directory as `Tsetlin.jl-main/` — the scripts include it by relative path
  (`include("../Tsetlin.jl-main/src/Tsetlin.jl")`). Its licence is its own.
- **Python** with `numpy scikit-learn xgboost` for the adjudication/tuning.
- The `hdc/` package (ours, Apache-2.0, stdlib-only) is activated by the
  scripts via `Pkg.activate`.

## Data

Committed: result caches (`kn500*.jsonl` KN scores per evaluation pair,
`tune_results/` margins of every tuned configuration), the pre-trained atom
codebooks (`*.jls`), pair manifests (`pairs500.tsv`), and run logs as
provenance. **Not committed** (rebuildable, Zenodo deposit to follow with the
release): the masked author pools (`phase4/authors*/`), the evaluation pair
texts (`pairs500/`), their enriched/windowed variants, and the exported
continuous feature matrices (`phase4/exportz/`, ~1.2 GB — regenerate with
`P4_EXPORTZ`, deterministic). Rebuild path: fetch + mask the German corpus
with the root `data_prep/`, then `phase4/prep_authors.py` and
`phase3/prep_test500.py`.

## Map (script → paper claim)

| Script | What it established |
|---|---|
| `phase2/compare.py`, `readout.jl`, `multires.jl` | the TM-as-language-model failure: softmax-of-votes is a lossy density; signal at α=1 is the unigram profile |
| `phase3/verify*.jl`, `fuse*.py`, `lensweep.*` | the direct-verifier reframe; unsupervised Delta-in-HDC cosine; first fusion gains |
| `phase4/tm_scale.jl` | the scaled verifier: mass pair generation, thermometer pair features, ensembles; every `P4_*` knob is an ablation from the paper (stylo, surprisal, disentangled, hard mining, stacking, enrichment, `P4_L`/`P4_LF` clause budget, `P4_EXPORTZ` feature dump) |
| `phase4/kn_sym*.py`, `kn_lens.py` | symmetric-length protocol; the KN collapse at 150 tokens; the w20 window rescue |
| `phase4/adjudicate_xgb.py`, `xgb_tune.py` | the interaction adjudication (linear vs trees on identical features; depth as interaction order) |
| `phase4/explain.jl` | the forensic reporting layer: clause waterfall, occlusion attributions, shared heat-map palette (`explain_report.html`) |
| `phase4/contrastive.jl`, `dep_probe.jl`, `remask.py` | measured nulls: atom geometry, dependency channel, morphological enrichment |

Headline results (see the journal paper's TM chapter for context): compact
clauses (`P4_L=256 P4_LF=64`) lift the standalone verifier to ~0.93 AUC,
matching Kneser–Ney λ_G on the same pairs; fused, KN + one such TM reaches
0.954 AUC / 0.400 Cllr under author-grouped cross-validation.
