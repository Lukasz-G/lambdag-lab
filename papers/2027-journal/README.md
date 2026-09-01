# Journal long paper — reproduction kit (in progress)

Post-submission experiments feeding the long paper. Grows with the paper.

## The within-K campaign (2026-09)

Tests Nini's "range of variation around K" hypothesis: is the typicality band
around a known author predictable from the author's own material?

| Script | Role |
|---|---|
| `run_withink.py` | symmetric within-author vs cross-author pseudo-pairs over the reference banks (L ∈ {600, 1200, 2000, 5000}) |
| `run_withink_test.py` | the same statistics computed from each *test case's own known document* (the honest casework protocol) |
| `analyze_withink.py` | the correlation result: within-K self-fit predicts the H_d location (median Spearman −0.38/−0.55 at L=2000/5000, artefact-controlled) |
| `fit_bhat.py` | pooled b̂ regression (R² ≈ 0.3–0.4, LODO-transferable) + calibration-rescue test on long-text cases |
| `analyze_decisive.py` | the decisive matched-L rescue test on ill-calibrated cells — negative: even oracle per-author location does not beat λ/√N; the binding term is the population slope |

Headline: typicality is *predictable* from K (entrenchment → typicality), but
does not convert into a case-internal calibrator — location was the recoverable
half of the calibration problem; the slope is irreducibly a population
property. Full narrative in the paper's calibration chapter.

Score files land on Zenodo with the release; the remote-execution pattern used
for these runs is documented in `docs/remote-compute.md`.
