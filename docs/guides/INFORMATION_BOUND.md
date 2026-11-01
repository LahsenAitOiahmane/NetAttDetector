# Information-bound operations

Read the [proof and registered scope](../research/INFORMATION_BOUND.md). This computes an
optimistic ceiling using known observed labels; it does not train a detector or choose a
threshold. A passing ceiling does not demonstrate a learnable or transferable operating point.

For existing verified private core/statistical databases:

```bash
uv run --no-sync netatt information-bound --cohort data/processed/statistical-core \
  --statistics data/processed/statistical-sidecar --output reports/local/my-information-bound
```

The analysis imports NumPy and lightweight provenance utilities, not pandas/model libraries.
It reads only the small calibration/validation partitions and uses exact byte equality for
feature sets. Rational arithmetic prevents rounding up the false-positive allowance. Ten
million dynamic-program updates and 250k rows/partition are hard algorithmic screens; sampled
RSS/wall screens are diagnostics rather than OS limits.

To recreate private inputs and run the bound on Linux:

```bash
uv sync --frozen --extra dev --extra measurement
uv run --no-sync python scripts/research/run_statistical_study.py --information-bound
gh workflow run measurements.yml --ref YOUR_BRANCH -f information_bound=true
```

Use one of those final two commands, with fresh paths or the script's path overrides and
optional private `--captures` map. The reused workflow artifact is named
`statistical-study-evidence-RUN_ID`; its explicit aggregate allowlist includes
`information-bound.json`, `configuration.json` and input verification/manifests. No signatures,
per-UID decisions, raw feature rows, model or private database is uploaded.

Inspect each representation separately. `target_not_ruled_out_by_relaxation: false` proves a
limit only for the exact observed accepted inputs and UID-max family; true merely means this
optimistic relaxation does not rule the target out. Precision and cross-signature shared-vector
constraints were ignored, making the bound more permissive than the actual model family.
