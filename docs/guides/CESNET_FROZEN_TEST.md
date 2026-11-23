# Reproduce the independent frozen C&C test

Read the [registered protocol](../research/CESNET_FROZEN_TEST.md). This test scores the
existing NAD-053 diagnostic at its frozen threshold. It does not train or recalibrate a model.
The source supports positive-flow recall only; it cannot establish false-positive rate or
production readiness. Cite Oškera et al., *Botnet Detection Through Periodic Patterns in
Command-and-Control Network Traffic*, CNSM 2025, and the
[dataset DOI](https://doi.org/10.5281/zenodo.16752462). Data are CC-BY-4.0, separately from MIT code.

## Prepare on GitHub Linux

The public recipient must match the locally retained private key. From PowerShell:

```powershell
$studyBranch = git branch --show-current
gh workflow run measurements.yml --ref $studyBranch -f cesnet_recall=true
gh run list --branch $studyBranch
```

The manual workflow downloads only the 43.4 MB raw archive, verifies publisher size/MD5,
selects the 32 unsplit released captures, verifies the 235-value packet fixture, and exports
all supported flows with their causal history. The private numeric SQLite table includes
the endpoint-eligibility flag, but no raw IP addresses, payloads or real UID identifiers.
The runner uploads aggregate JSON plus recipient-encrypted measurements. It never receives
the private model or private key. Every artifact path is explicitly listed in the workflow.

Download the artifact using its run ID:

```powershell
gh run download RUN_ID --name cesnet-measurements-RUN_ID --dir artifacts/cesnet-download
```

The artifact retains source-relative subdirectories because its files have a shared repository
root. Inspect `data/raw/cesnet-cc25/acquisition.json`,
`data/processed/cesnet-cc25/{fixture,manifest,progress}.json` and the workflow commit identity.
Compare the acquisition inventory/checksums to the already committed acquisition evidence.
Retain and commit the inspected aggregate `manifest.json` under
`reports/evidence/cesnet-frozen-test/` **before** recovery/scoring. An inner encrypted manifest
alone does not establish who produced the data.

## Recover privately and score the frozen model on Windows

```powershell
uv sync --frozen --extra dev --extra artifacts
uv run --no-sync python scripts/research/run_cesnet_test.py recover --envelope artifacts/cesnet-download/artifacts/cesnet-measurements.nadenc
uv run --no-sync python scripts/research/run_cesnet_test.py evaluate
```

Recovery checks the separately trusted manifest, exact bounded ZIP member and database hash
before opening SQLite. Use a fresh `--measurements` directory if the default already exists.
It creates an owner-restricted private directory. The evaluation command verifies the measured
contract, byte/logical digests and all capture counts, then verifies the independently frozen
model hash before joblib loading. Score batches contain at most 1,024 rows at one thread.
The privately recovered NAD-053 model must already exist at the default `--model` path;
see [model recovery](../research/PRIVATE_ARTIFACTS.md). Never use an untrusted person's freeze
or model: joblib can execute code. Both recovered data and model stay ignored by Git.

Review `artifacts/cesnet-frozen-test/evaluation.json`. Preserve every capture, including samples
with no eligible endpoint-matched flows, all unknown-flow coverage, family summaries, capture
macro/pooled recall, conditional capture-bootstrap uncertainty and resource scope. Metrics
requiring benign labels remain null. Publish aggregate evidence only; keep the archive, numeric
database, encrypted artifact, private model and key private. The GitHub artifact expires after
30 days, so retain a secure local backup when needed.

## Working Linux host

With a working Linux environment and the locked measurement extra, the same bounded stages
can run locally without GitHub:

```bash
uv sync --frozen --extra dev --extra measurement
uv run --no-sync python scripts/research/run_cesnet_test.py acquire
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  uv run --no-sync python scripts/research/run_cesnet_test.py prepare
```

The existing Windows acquisition can be reused from WSL if its files and registration still
verify. Do not overwrite an incomplete measurement directory; inspect the failure and select
a fresh path. A platform launch failure is not evidence of successful fixture/real export.
