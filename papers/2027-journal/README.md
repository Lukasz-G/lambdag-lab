# Journal long paper — reproduction kit (in progress)

Post-submission experiments feeding the long paper. Grows with the paper.

## The within-K campaign: the range of variation around K

| Script | Role |
|---|---|
| `run_withink.py` | symmetric within-author vs cross-author pseudo-pairs over the reference banks (L ∈ {600, 1200, 2000, 5000}) |
| `run_withink_test.py` | the same statistics from each *test case's own known document* (the casework protocol) |
| `analyze_withink.py` | the correlation: within-author self-fit predicts the H_d location (median Spearman −0.38/−0.55 at L=2000/5000) |
| `fit_bhat.py` | pooled b̂ regression (R² ≈ 0.3–0.4, transfers leave-one-dataset-out) + calibration-rescue test on long-text cases |
| `analyze_decisive.py` | the decisive matched-L rescue test on ill-calibrated cells — negative: even oracle per-author location does not beat λ/√N |

## The symmetrised estimator and the corpus-adequacy gauge

| Script | Role |
|---|---|
| `run_symmeter.py` | per-author, size-matched reference (donor) models; matched vs deliberately mismatched reference-corpus arms |
| `analyze_routeb.py` | cohort studentisation t = mean(λ_j)/sd(λ_j) on the symmeter arms; matched-arm Cllr 1.04 → 0.62; mismatch arms NOT rescued |
| `run_routeb_real.py` + `analyze_routeb_real.py` | the same on *real* grid evaluation cases at L=1200: Cllr improves 8/8 datasets (median 1.38 → 1.07) |

## Defensive controls

| Script | Objection addressed |
|---|---|
| `analyze_tempered.py` | "the studentisation gain is just range compression" — per-dataset variance rescaling recovers most; the cohort matches it from a single case's ingredients |
| `analyze_ttr_partial.py` | "entrenchment or just model sharpness?" — correlation unmoved by partialling out TTR/hapax rate |

## Aggregation functionals & engines

| Script | Role |
|---|---|
| `run_details.py` + `analyze_functionals.py` | per-token detail dumps + the functional sweep (sum survives; tail-concentration helps short texts, trimming helps full length) |
| `analyze_wkfeature.py` | within-K statistics as an additional feature (median +0.004 AUC, 23/24 cells) |
| `analyze_engines.py` | KN vs HPY vs PPMd baseline, 8 languages × 5 symmetric lengths |
| `analyze_hpy_sweep.py` | HPY θ × table-estimator × discount sweep (no variant meaningfully beats KN; `hpy_t0_min` ≡ KN confirms the oracle at scale) |
| `make_journal_tables.py` | regenerates the paper's calibration-chapter tables |
| `make_email_figs.py` | the three exhibit figures (entrenchment scatter, corpus gauge, cohort normalisation) |

Headline of the campaign: typicality around K is *predictable* from the known
author's material but does not convert into a case-internal calibrator; the
reference cohort's spread does calibrate (8/8 real-case datasets), and the
residual offset of the symmetrised estimator audits reference-corpus adequacy.
Engine grids run with the extended `ENGINES` dict in
`../2027-chr/run_kn_grid.py` (`--stage engines`).

Score files land on Zenodo with the release; the remote-execution pattern is
documented in `docs/remote-compute.md`.
