# Statistical model comparison operations

Read the [registered study](../research/STATISTICAL_STUDY.md). This compares core controls
and additional verified measurements on the same accepted-label candidate cohort. It does
not promote labels or enable deployment/public model release.

From a working Linux checkout, using fresh default directories:

```bash
uv sync --frozen --extra dev --extra measurement
uv run --no-sync python scripts/research/run_statistical_study.py
```

Options `--data-root`, `--cohort`, `--statistics`, `--study` and `--evidence` choose fresh
locations. An optional private `--captures path/to/map.json` reuses the three pinned input
bundles. The script checks independent fixtures, recreates/reverifies core and statistical
databases, requires both registered logical digests, then fits sequentially with one thread.

```bash
gh workflow run measurements.yml --ref YOUR_BRANCH -f statistical_study=true
```

The reusable workflow runs twelve predefined candidates, one dummy and two repeat controls.
Core-only controls use the same 27-column external input, selecting eleven fields inside their
pipeline. Both weighting policies and all threshold/selection rules are unchanged from the
native baseline. Additional field values are validated; metadata cannot enter model inputs.

`study.json` includes all metrics, family support, float64/float32 pattern overlap, selected
error bins, warnings, reload/repeat checks and resource samples. `freeze.json` records the
chosen diagnostic's hash/threshold. A zero qualifying-candidate count is a completed negative
experiment even when reproducibility/resource checks pass and the workflow is green.

Only explicitly named aggregate JSON files enter CI artifacts. Core/statistical databases and
`.joblib` models remain private on the ephemeral runner; they are not downloaded to Windows.
Reproduce on private working Linux storage when retaining research artifacts is necessary.
The package's UQ `predict` command does not accept this model format. All native/statistical
artifacts remain experimental, with joblib loaded only from trusted local sources.

The one-hour/1.5 GiB screens are checked between bounded fits and through sampled RSS, not
enforced process guarantees. Inference probes exclude capture/export/I/O. Connection-level
UID-max predictions are not incident aggregation or an approved 50-alert/day pilot budget.
