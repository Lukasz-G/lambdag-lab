# Design decisions and load-bearing invariants

Curated public summary of the project's internal design notes. These are the
decisions a contributor must not silently revert; several tempting
"improvements" have been tried and rejected with measurements.

## Architecture (one case c = (D_U, D_A))

```
raw text
  -> POSNoiseMasker.mask()   -> List[List[str]]  (sentences of function tokens)
  -> Vocabulary.encode()     -> SentenceStore    (flat int32 + offsets, CSR)
  -> LambdaG.score(S_U, S_A):
       G_A = engine.fit(S_A)                       numerator model
       G_j = engine.fit(sample(S_ref, |S_A|)) x r  r reference models
       lambda_G = sum_tokens(logP(t;G_A) - mean_j logP(t;G_j)) / ln(base)
  -> LambdaGCalibrator        -> calibrated log10 LR
  -> cllr / cllr_min                               forensic cost
```

The three engines share one skeleton and differ on two axes only: lower-order
counts and how orders combine (KN: continuation counts + interpolation; HPY:
table counts + interpolation; PPMd: raw counts + backoff-with-exclusion).

## Invariants (CI-guarded where possible)

1. **HPY(concentration=0, table_estimator="minimal", discount=0.75) == KN,
   exactly** (~1e-12, token-by-token and end-to-end). The regression oracle:
   if a refactor breaks the equality, the refactor is wrong. (`tests/test_oracle.py`)
2. **Padding**: each sentence gets n `<BOS>` and one `<EOS>` at order n.
   Normalisation depends on it; easy to get subtly wrong. (`tests/test_padding.py`)
3. **Log base 10** for lambda_G (matching the R `idiolect` package); sign and
   ranking are base-invariant but the base is kept.
4. **Per-model dictionaries** by default (`vocab_mode="per_model"`), matching
   `idiolect`; r=30 reference models; KN discount D=0.75.
5. **Adaptive PPM defaults to `reset="sentence"`** — document-level reset
   dilutes lambda_G badly (asymmetric adaptation inside a ratio). Measured.
6. **Sentences are the unit of independence** (per-sentence decomposition and
   the adaptive-PPM reset rely on it). Under `segment="window"` the window
   takes over that role, with BOS/EOS only at window edges.
7. **Punctuation survives masking verbatim** (PUNCT is not in the placeholder
   map) and `. ! ? …` double as sentence delimiters in sentence mode.

## Numba

Two hot paths are JIT-compiled: rolling 64-bit SplitMix hashing of n-gram
windows at fit time, and the per-token smoothing recursion at query time.
n-grams are hashed into sorted arrays and looked up by binary search; the rest
is vectorised NumPy. The module imports and runs (slowly) without Numba;
`warmup_jit()` pays the ~3 s compile cost up front.

## Tried and rejected (do not re-propose without new evidence)

- Cython port — no win over the Numba+NumPy design.
- Exact HPY Gibbs sampling — cost without benefit for this use.
- Bidirectional (forward+backward) fusion for accuracy.
- Writing pattern-list entries from an LLM's memory — the treebank validator
  repeatedly caught real errors; all entries are UD/ReM/ReN-derived and
  machine-validated.

## Single-file requirement

`lambdag.py` stays one module with no `__main__` — the entire method in one
auditable file is a forensic-transparency feature, not an accident. Usage
examples live in the commented block at the end of the file; orchestration
belongs in `cli.py` or the paper scripts.
