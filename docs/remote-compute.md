# Remote-compute recipe (CPU marketplaces)

The grid experiments are embarrassingly parallel across (dataset, length) cells
and CPU-only. The following pattern ran the 2026 campaigns (123k + 73k scores)
on rented vast.ai boxes for cents; it applies to any SSH-reachable Linux host.

1. **Payload**: tar.gz of `lambdag.py` + the needed scripts + the masked data
   (never raw corpora); record its `sha256sum`. Masking is done locally first,
   so remote deps are only `numpy numba scikit-learn tqdm` — no spaCy models.
2. **Rent**: pin offer AND image tag in one step (offers race away between
   search and create). A ~$0.10/h 24–48-core box does a full grid pass in under
   an hour; prefer geographically close hosts for the upload.
3. **Transfer**: scp the payload (or stage once on a share link for repeated
   re-rents); **verify the sha256 on the box before unpacking** — never run
   unverified code on untrusted hardware, and never put credentials on it.
4. **Run detached**: `setsid nohup env NJOBS=$CORES bash bootstrap.sh > run.log
   2>&1 < /dev/null &` — the bootstrap creates a venv with PINNED versions
   (reproducibility) and fans jobs out with `xargs -P`.
5. **Monitor from the local side**: poll every few minutes for a DONE marker,
   worker liveness (`pgrep -fc 'run_scri[p]t'` — note the bracket trick so the
   grep does not match itself) and failure signatures (Traceback/Killed).
6. **Verify, then destroy**: pull the score files, check row counts against the
   plan, and only then destroy the instance — destroy deletes everything.
   Billing runs until destroy; confirm the instance list is empty afterwards.

Gotchas that cost time once: CRLF line endings in file lists and shell scripts
written on Windows (force LF); `bash -c "cd X && cmd &"` backgrounds the whole
chain including the cd (use absolute paths in the follow-up commands); an ssh
session that launched a background job may linger holding the channel (wrap the
launching ssh in a local `timeout`).
