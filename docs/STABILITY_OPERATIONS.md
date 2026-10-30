# Reproduce the development-only stability study

This is an offline diagnostic experiment. It does not produce a deployable model bundle.
The [registered protocol](STABILITY_PROTOCOL.md) fixes the grid, budget and acceptance screen.
Use the corrected version-2 UNSW/ToN cache. No CSE archive is needed or read.

```powershell
uv sync --frozen --extra dev --extra boosting
# Only if the corrected cache does not already exist:
uv run --no-sync netatt prepare-multisource --output data/processed/multisource
uv run --no-sync netatt study-stability --prepared data/processed/multisource --output artifacts/stability
python scripts/report_stability.py --study artifacts/stability --output reports/local/STABILITY.md
```

For the existing laptop cache, pass `--prepared data/processed/multisource-v2` instead.
Each output directory must be fresh. Keep the Git tree clean and commit the protocol/source
before fits. Do not run concurrent training jobs on the laptop. Two threads and a one-hour
start-new-fit deadline cap the study; the active fit is allowed to finish. Partial studies
are recorded and cannot pass the three-seed screen. Histogram boosting uses no internal
early-stopping split; each threshold is selected on calibration and evaluated on validation.

Run **Research dataset benchmark** in GitHub Actions with `experiment=stability` on the
branch being reproduced. It installs frozen dependencies, tests the code, downloads only
UNSW/ToN, prepares the bounded development cohort, and retains JSON/NPZ evidence for 14 days.
Download the artifact before expiration. Neither raw archives nor score arrays belong in Git.
The workflow's `multisource` mode reproduces the earlier already-observed CSE experiment;
it is not a new independent test and should not be used for tuning this study.

After downloading the Linux artifact:

```powershell
uv run --no-sync netatt compare-stability artifacts/stability artifacts/cloud/artifacts/stability --output reports/local/stability-comparison.json
python scripts/report_stability.py --study artifacts/stability --cloud artifacts/cloud/artifacts/stability --comparison reports/local/stability-comparison.json --output reports/local/STABILITY.md
```

The comparison refuses mismatched raw-source/configuration hashes, ordered feature/label/
weight payloads, post-preprocessing matrices, validation rows or incomplete control grids.
It compares numeric scores on the same rows. ZIP/NPZ container bytes can differ due to metadata;
canonical payload digests establish semantic equality. A score difference cannot be blamed
on a particular library/runtime cause without further evidence.

`study.json` includes all per-source/seed metrics and conditional intervals, feature-family
recall, overlap counts, complete/incomplete screens, signatures, versions and hardware.
`measurements.json` describes protocol/TTL/rate/duration patterns without repairing them.
`progress.json` checkpoints completed ablations. `control-scores.npz` contains eight
registered control score arrays and signed row identities; these are development data only.
Reported RSS samples the training process, not total computer memory or children; runtime
excludes cache loading and includes partitioning/measurement/fit/scoring within the monitor.
The original phase-2 frozen artifacts and holdout receipts are never modified.
