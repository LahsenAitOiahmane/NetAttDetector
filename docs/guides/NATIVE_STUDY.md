# Native diagnostic model operations

Read the [registered scope and selection rules](../research/NATIVE_STUDY.md) before fitting.
This is offline research on candidate-labeled captures, with unverified device independence.
It does not enable the UQ model predictor, live sensing or blocking.

## Reproduce the entire study

On working Linux, from the repository root:

```bash
uv sync --frozen --extra dev --extra measurement
uv run --no-sync python scripts/research/run_native_study.py
```

All default acquisition/cohort/model/evidence directories must be fresh. Add `--captures` with
a private capture-ID/local-path JSON map to reuse the three pinned expansion bundles without
downloading. `--cohort`, `--study` and `--evidence` select fresh output directories. Existing
verified private cohorts can be used directly:

```bash
uv run --no-sync netatt study-native --cohort data/processed/native-candidate \
  --output data/processed/my-native-models
```

The study reopens/verifies every cohort before fitting and requires the preregistered logical
row digest. It uses twelve predefined candidate specifications, a dummy, and two controls.
Models fit sequentially with at most two threads; prediction uses one thread. Each model
includes native feature transforms/preprocessing and a calibration-only threshold.

The one-hour limit is checked between fits; estimator iterations/depth/trees are bounded.
The 1.5 GiB memory screen is sampled, not OS-enforced. A local Windows machine with about
1 GiB currently available RAM should use the cloud job rather than run the complete study.

```bash
gh workflow run measurements.yml --ref YOUR_BRANCH -f native_study=true
```

The reusable `native-study.yml` recreates the cohort and fits in the same runner. Only an
explicit aggregate JSON allowlist enters CI artifacts: cohort manifest/verification, study
configuration/results, freeze and progress report. Databases, fitted `.joblib` bundles and
feature/UID rows remain private under `data/` and expire with the runner. A saved freeze
records an artifact hash; it does not mean a model binary was downloaded to Windows. Use
private local Linux storage when a retained research artifact is required. Dataset/public
model-release terms remain unresolved despite the owner's MIT choice for project code.

## Read the output

`study.json` records all candidate metrics, feature overlap, inference probes, saved/reloaded
prediction checks, two repeat/thread controls and resource screens. `freeze.json` identifies
the selected **diagnostic** artifact, threshold and input hashes. Validation selected that
specification, so its metrics are not an untouched final test. Zero qualifying candidates is
a valid completed negative experiment; resource/repeatability failure additionally exits with
an error after retaining aggregate evidence.

Flow metrics and UID-max connection metrics answer different questions. A connection is flagged
when any accepted segment crosses the threshold; that is not a real incident or analyst alert.
Intervals assume independent accepted UIDs and do not cover capture dependence or selection.
Report exclusions, per-family support and the DDoS coverage gap beside any model metric.

Never pass a native bundle to `netatt predict`: that command intentionally accepts only UQ
contracts. joblib loads trusted local artifacts only. A later native inference interface must
validate its separate contract and research-only/deployment flags before any product use.
