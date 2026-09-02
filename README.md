# lambdag-lab

Multilingual, grammar-based authorship verification for literary and historical
texts: POSNoise topic masking in 23 languages (incl. Middle High and Middle Low
German), a single-file, Numba-accelerated implementation of **LambdaG** with
three interchangeable probability engines (Kneser–Ney, Hierarchical Pitman–Yor,
PPMd), forensic calibration (log-likelihood ratios, Cllr and its decomposition),
and token-level explainability heat maps.

> **Positioning.** This is an *independent research implementation and
> extension suite* accompanying the papers below. For the **reference
> implementation** of the LambdaG method by its author, see
> [AndreaNini/LambdaG](https://github.com/AndreaNini/LambdaG) (also on PyPI as
> `LambdaG`); the R implementation lives in
> [idiolect](https://andreanini.github.io/idiolect/). The method is due to
> Nini et al.; the POSNoise masking approach to Halvani & Graner (2021).

## The papers this repo accompanies

| Paper | Where | Reproduction kit |
|---|---|---|
| Stylometrie mit LambdaG für die mittelhochdeutsche Literatur (2026) | DHd | [`papers/2026-mhg/`](papers/2026-mhg/) *(backfill in progress)* |
| *Grammar on Trial: Forensic Authorship Verification across Twenty-Three Languages and Three Genres* (CHR 2027, submitted) | CHR | [`papers/2027-chr/`](papers/2027-chr/) |
| Journal long paper (in preparation) | TBD | [`papers/2027-journal/`](papers/2027-journal/) |

Each paper directory holds the exact scripts and frozen result files behind the
paper's tables and figures; links to the published papers are added as they
appear. Heavy artefacts (masked corpora, per-case score files) are deposited on
Zenodo per release — see the paper READMEs.

## Install

```bash
git clone https://github.com/Lukasz-G/lambdag-lab
cd lambdag-lab
pip install -e ".[numba,dev]"       # editable install is REQUIRED (pattern
                                    # lists resolve relative to lambdag.py)
python -m pytest tests/             # incl. the KN==HPY regression oracle
```

Optional extras: `[spacy]` for masking raw text (plus a per-language spaCy
model), `[stanza]` for the Stanza tagger path, `[viz]` for figures.

## Quickstart

```python
from lambdag import POSNoiseMasker, LambdaG, cllr, heatmap_html

masker = POSNoiseMasker("de")                  # needs [spacy] + de_core_news_lg
S_known = masker.mask(open("known.txt").read())
S_quest = masker.mask(open("questioned.txt").read())
S_ref   = [s for doc in reference_texts for s in masker.mask(doc)]

lg = LambdaG(N=10, r=30, engine="kn", random_state=0).set_reference(S_ref)
res = lg.score(S_quest, S_known)               # res.lambda_G = log10 LR score
html = heatmap_html(res)                       # token-level evidence exhibit
```

Pre-tagged corpora (e.g. ReM/ReN for medieval German) go through
`POSNoiseMasker.pretagged(...)` / `mask_tagged(...)` — no tagger required.
A minimal CLI is included: `lambdag-lab info | mask | score`.

## Layout

```
lambdag.py          THE library — one auditable file, no __main__ (by design)
cli.py              thin CLI wrapper
posnoise_lists/     19 pattern lists + HiTS→UD map (provenance: docs/posnoise_lists.md)
tools/              list-building machinery (gmh/gml builders)
data_prep/          corpus fetchers + masking drivers (ELTeC/DraCor/PoeTree; spaCy/Stanza)
papers/             per-paper reproduction kits (see above)
tm/                 the HDC-Tsetlin verifier (experimental; journal paper)
docs/               design decisions, list provenance, remote-compute recipe
tests/              incl. the load-bearing KN==HPY(θ=0) oracle
```

## Design commitments

- **Likelihood-ratio semantics end to end** — anything that silently breaks LR
  semantics is treated as wrong-by-default; calibration quality is always
  reported as Cllr beside its discrimination floor Cllr_min.
- **The KN==HPY oracle**: HPY with zero concentration and minimal tables equals
  interpolated Kneser–Ney to ~1e-12, token by token. This is the regression
  test that guards every refactor.
- **Every pattern-list entry is treebank-derived and machine-validated** —
  never written from memory. See `docs/posnoise_lists.md` before using any
  non-en/de language.

## Citing

See `CITATION.cff`. Please cite the accompanying paper for the method variant
you use, and Nini et al. for LambdaG itself.

## License

Apache-2.0 (also the licence of the upstream Halvani/POSNoise repository from
which the En/De pattern lists are taken, with attribution).
