# The anonymous poems of the Ambraser Heldenbuch

Reproduction kit for the article on relational authorship verification inside
Vienna, Österreichische Nationalbibliothek, Cod. Ser. nova 2663. Companion to
[`../2026-zfdg/`](../2026-zfdg/), which asks a named question of the
*Kaiserchronik*; this one asks four unnamed questions of one codex.

## What the article claims

Hans Ried copied twenty-five texts into the book between 1504 and 1516, twelve
anonymous and thirteen under an author's name, so a case, the cohort of
strangers that scales it and the pairs of works that calibrate its odds can all
be in one hand. Four relational questions are asked — is this anonymous poem by
the poet of that one? — each of them four ways.

| question | result |
|---|---|
| *Ortnit* against *Wolfdietrich* | one poet, 4/4 regions, P = 1.6e-03 |
| *Rabenschlacht* against *Dietrichs Flucht* | one poet, 8/8 regions, P = 2.6e-06 |
| *Nibelungenlied* against *Kudrun* | shared tradition; the controls clear the same threshold |
| *Klage* against the *Nibelungenlied* | not the same poet, 1/10 regions, P = 0.89 |

Four short poems — *Moriz von Craûn*, *Der Mantel*, *Von dem übelen Wîbe*, the
*Zweites Büchlein* — cannot be attributed at any known-side length, and the
panel's own failed positive controls are what establish that.

## Running it

The pipeline has four stages. Corpora are not redistributed here; the fetchers
take them from their official sources.

```bash
# 1. corpora -> banks
python fetch_mhdbdb.py                      # and --sigles for the anonymous works
python extract_ambraser.py
python build_mhdbdb_banks.py
python make_hydra_inputs_mhdbdb.py
python tag_mhdbdb.py --batch 1              # the same checkpoint as the witnesses
python build_mhdbdb_masked_banks.py

# 2. the four questions, four readings each
python run_ambraser_pairs.py --arm codex
python run_ambraser_pairs.py --arm short
python run_ambraser_pairs.py --arm borrowed
python run_ambraser_pairs.py --arm edition

# 3. the chapters that are not about the four questions
python edition_provenance.py                # which edition rests on which manuscript
python measure_layer.py                     # what an editor's text costs
python run_ambraser_panel.py && python panel_matrix.py

# 4. the redaction control
python extract_nibelungen.py
python make_hydra_inputs_nibelungen.py
python tag_mhdbdb.py --inputs ../hydra_in_nibelungen --outputs ../hydra_out_nibelungen
python collate_nibelungen.py

python make_tables.py                       # every number the prose quotes
```

## Frozen results

`kc_reuse/` holds the files the article's tables are generated from, so
`make_tables.py` reproduces every figure in the prose without rerunning the
scoring.

| file | what it carries |
|---|---|
| `kc_regions_crosswork__amb*.tsv` | the calibrated ratio per region, per question, per reading |
| `kc_regions_field__amb*.tsv` | the field test per region |
| `edition_provenance.tsv` | each edition's base manuscript and the surviving witness counts |
| `layer_offset.tsv` | an editor's text against a scribe's, on 13 works in both banks |
| `ambraser_panel.tsv` | the short pieces against seven known texts |
| `nibelungen_collation.tsv` | ten copies of one poem, aligned by content |
| `redaction_inventory.tsv` | divergence by strophe number, surface and lemma |

## Three things worth knowing before reusing the scripts

**The exclusion rule is load-bearing and easy to break.** No text the case is
about — known text, candidate, or control — may stand in the reference
population or the cohort. A control that is also a reference grammar is
predicted by the denominator it belongs to, so its score is depressed and the
candidate's margin over it is manufactured. `run_kc_regions.py --exclude`
enforces it, and the filter has to reach *every* pool, including ones drawn from
a different bank than the case.

**The cohort is a scale, not a sample**, so it holds one text per author
(`--cohort-per-author 1`). An author with several surviving works otherwise sets
the scale for everyone else.

**A bounded ratio saturates.** The calibrated odds are capped at what the
calibration class supports, and where candidate and control are both clipped
the ratio discriminates nothing. Read the distance between candidate and
control, and let `rank_candidate_vs_field.py` — which has no ceiling — carry the
verdict.

## Corpus note

MHDBDB texts are CC BY-NC-SA 4.0 and are fetched, not redistributed. The
Ambraser transcription and the Nibelungenwerkstatt transcriptions come from
their publishers; `extract_*.py` consume their own formats.
