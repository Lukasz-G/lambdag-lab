# Per-case scores behind the CHR 2027 paper

Score files for Łukasz Gągała, "Grammar on Trial: Forensic Authorship
Verification across Twenty-Three Languages and Three Genres", Computational
Humanities Research Conference 2027.

`scores_chr2027.tar.gz` holds the per-case scores behind every number in the
paper, so that each reported AUC, $C_{llr}$ and $C_{llr}^{min}$ can be
recomputed without rebuilding the corpora or rerunning the grid. The scripts
that produced and consume these files are in the parent directory; the method
implementation is `lambdag.py` at the repository root.

## Verifying the headline figure

```
mkdir -p scores && tar -xzf scores_chr2027.tar.gz -C scores/
python recompute_auc.py scores/
```

This reports a median AUC of 0.9315 over the 38 analysed datasets, which is the
0.931 of the abstract. The script needs only the Python standard library.

## Score file format

One JSON object per verification case, one case per line:

```json
{"id": 0, "label": 1, "lambda_G": 98.956, "n_q": 1197, "v1_q": 74,
 "sqrt": 2.860, "hapax": 6.118}
```

| Field | Meaning |
|---|---|
| `id` | case index within the dataset |
| `label` | 1 same-author, 0 different-author |
| `lambda_G` | the raw score of the paper's Equation 4, base-10 |
| `n_q` | length of the questioned document in masked tokens, $N(Q)$ |
| `v1_q` | masked types occurring once in the questioned document, $V_1(Q)$ |
| `sqrt` | the square-root-corrected LLR, $\lambda_G/\sqrt{N(Q)}$ |
| `hapax` | the hapax-corrected LLR, $\lambda_G \cdot V_1(Q)/N(Q)$ |

The files hold aggregate statistics only. No token, text, title or author name
appears in them, and no source text can be reconstructed from them, so they
carry no licence obligation from the corpora the cases were built on (ELTeC,
DraCor, PoeTree). Fetchers for those corpora are in `data_prep/`.

## File naming

```
<dataset>__kn__<unit>__L<length>.jsonl
```

`kn` is the probability engine, interpolated Kneser–Ney at order 10, the only
engine used in the paper. `unit` is the segmentation unit: `sent` for the
tagger's own sentences, `w10`/`w20`/`w30` for fixed token windows of that size,
`long` for the long-text control, `bankmain`/`bank*` for the bank-calibration
arms. `L0` is full length; `L1200`, `L600`, `L300` and `L150` are the symmetric
truncations, applied to both sides of the case.

## Which files produce which exhibit

| Exhibit | Source |
|---|---|
| Table 1, treebanks and pattern lists | the shipped pattern lists in `posnoise_lists/` |
| Table 2, calibration at full length | `*__kn__sent__L0.jsonl` |
| Table 3, symmetric evidence lengths | `*__kn__sent__L{1200,600,300,150}.jsonl`, `../primary.log` |
| Table 4, bank calibration by length | `*__kn__bank*.jsonl`, `../bankcal*.log` |
| Table 5, long-text control | `*__kn__long__L*.jsonl`, `../longtexts.log` |
| Tables 6–8, segmentation | `*__kn__w{10,20,30}__L*.jsonl`, `../segmentation.log` |

`../make_paper_tables.py` reads these files and emits the paper's tables
verbatim.

## Licence

The score files are released under CC0-1.0. The code in this repository is
Apache-2.0.
