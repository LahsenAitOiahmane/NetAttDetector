# Development, training and recovery runbook

## Reproduce locally

Use an isolated environment from the repository root. `uv sync --frozen --extra dev --extra boosting`
installs the committed resolution. The benchmark uses Python 3.13; CI also checks 3.11. Optional
XGBoost is omitted from `configs/laptop.json`, included in `configs/benchmark.json`.
The lock records the selected package versions; `requirements.txt` is a convenience editable
installation, not a second hand-maintained freeze. Review upgrades through lockfile changes.

Run `uv run --no-sync netatt doctor` before a large job. Defaults are two threads, 50,000-row
chunks, and a 200,000-row sample. On a busy laptop reduce `max_rows` to 100,000 or 50,000,
`chunk_rows` to 10,000, trees to 50, and boosting iterations to 75 in a copied config.
Increasing the data sample changes the experiment and must use a new output directory.
Only one real training run should execute on the laptop at a time. No GPU is needed.

```powershell
uv run --no-sync netatt download nf-unsw-nb15-v2 --max-mib 512
uv run --no-sync netatt download nf-ton-iot-v2 --max-mib 512
uv run --no-sync netatt train --data data/raw/nf-unsw-nb15-v2.zip --external data/raw/nf-ton-iot-v2.zip --config configs/benchmark.json --output artifacts/run-001
```

Archive downloads remain compressed on disk. Repeated training scans/decompresses them again;
the bounded sample reduces memory, not scan work. Store future sampled split manifests or
partitioned Parquet only after provenance and split strategy are settled. Do not silently use the
first 200,000 rows to speed up a scientific benchmark.

## GitHub workflows

`Quality and smoke training` runs on pushes and PRs, checks formatting/lint/tests under
Python 3.11 and 3.13, generates deterministic synthetic flows, and saves smoke artifacts for
7 days. Those model scores are not real detection results.

`Research dataset benchmark` uses a standard Ubuntu runner, 45-minute timeout, two native
threads, real UQ archives with independent 512 MiB transfer caps, four candidates, and a
second-source frozen-threshold evaluation. It uploads model, metrics, profiles, configuration,
hardware snapshot and download manifests for 14 days. It does not upload raw archives.
Each workflow has read-only repository permissions, SHA-pinned actions and concurrency limits.
No secrets, deployment keys or paid GPU runner are required.

The initial modernization branch has an automatic bounded training trigger so that the workflow
can be exercised before merge. After merge, run it manually from Actions, or:

```powershell
gh workflow run train.yml --ref main -f profile=benchmark
gh run list --workflow train.yml --limit 5
gh run view RUN_ID
gh run download RUN_ID --dir artifacts/github-RUN_ID
```

Replace `RUN_ID` with an actual run ID. The branch trigger is intentionally limited; after the
foundation merges, remove it if no longer needed. Do not enable daily full retraining until the
data update process and a monitoring need exist. A green run means execution completed;
`validation_constraints_met` and `test_constraints_met` still govern research interpretation.

The repository was public at audit time. GitHub documents free use of standard hosted runners
for public repositories; private repositories and larger runners can have different limits/costs.
Review [current billing rules](https://docs.github.com/en/actions/concepts/billing-and-usage)
before changing runner type or repository visibility. Artifact expiry is not long-term model
archiving: preserve approved release evidence separately with its provenance.

## Files to preserve

For an experiment, keep source revision/dirty-state note, `uv.lock`, config, input manifests and
hashes, `data-profile.json`, `metrics.json`, `summary.md`, and the model artifact. New run outputs
are ignored by Git. The checked-in baseline report includes compact JSON evidence; large
models/raw inputs remain local or workflow artifacts. Original tracked models/logs are historical.
They are neither the active model registry nor trustworthy new benchmark evidence.

Changing a feature, unit, label mapping or exporter definition requires a new schema and retraining.
Changing dependency versions requires inference/replay revalidation. Future deployments promote
a reviewed immutable model version and retain a known working version for rollback.

## Failures and recovery

| Symptom | Action |
|---|---|
| Missing `netatt` / module | Sync the environment and run through `uv run --no-sync`; avoid mixing global Python |
| Download timeout | Rerun; block retries are bounded, failed invocation removes the partial archive |
| 403 on temporary URL | Resolve a new public record link; do not persist expired signed URLs |
| Download over byte budget | Inspect permanent record/size and free space before intentionally increasing the cap |
| Existing checksum mismatch | Preserve the suspect file for investigation, move it aside, and redownload; do not rewrite the manifest to fit it |
| Multiple CSVs in archive | Set `member` in config or `--member` for profiling to the actual data CSV |
| Schema mismatch | Verify exporter and units; train a compatible contract rather than inventing zeros |
| A split lacks benign/attack support | Increase sample or choose defensible grouping; do not silently switch to a convenient split |
| Output directory already populated | Choose a fresh run directory; review partial evidence before discarding it |
| High accuracy but poor attack recall | Inspect confusion matrix, family slices and threshold constraints; majority-class accuracy can be misleading |
| External performance poor | Keep result; audit shift/shortcuts and establish a new independent source before adapting |
| Memory pressure | Close unnecessary work, reduce sample/chunk/model sizes, or use the bounded GitHub workflow |

The downloader validates ZIP CRC but does not authenticate against a publisher hash when none
is supplied. It is not designed as a general untrusted-archive ingestion service. Batch inference
refuses existing output paths and writes through a temporary file; input/model files are preserved.
