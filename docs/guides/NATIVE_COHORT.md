# Native candidate cohort operations

This is offline experimental dataset preparation, with no model fitting or live capture. Read
the [registered protocol](../research/NATIVE_COHORT.md) and the
[packet-label limitations](../../reports/PACKET_LABELS.md) first.

## Linux preparation

Use the pinned environment and fresh output directories. Only already registered IoT-23
development bundles are acquired; the external test remains untouched.

```bash
uv sync --frozen --extra dev --extra measurement
uv run --no-sync python scripts/research/prepare_native_cohort.py \
  --data-root data/raw/native-cohort \
  --cohort data/processed/native-candidate \
  --evidence artifacts/native-cohort
```

For existing verified captures, provide a private JSON map containing exactly the three capture
IDs and their local directory paths, then append `--captures path/to/map.json`. This skips all
downloads. The lower-level CLI accepts the same map:

```bash
uv run --no-sync netatt prepare-native-cohort --captures path/to/map.json \
  --output data/processed/native-candidate
uv run --no-sync netatt verify-native-cohort data/processed/native-candidate \
  --output reports/local/native-verification.json
```

The Windows installation can verify a compatible private database without a capture driver.
Actual NFStream export is Linux-only. Do not map the eleven native features onto the old UQ
model inputs. On this laptop, use cloud preparation when local Linux is unavailable; do not
install drivers or repair existing WSL distributions as part of dataset preparation.

## GitHub workflow

```bash
gh workflow run measurements.yml --ref YOUR_BRANCH -f native_cohort=true
```

The reusable `native-cohort.yml` runs sequentially with two native threads. Only the explicit
aggregate `manifest.json`, `verification.json` and `reproducibility.json` allowlist enters CI artifacts. Raw inputs and
`cohort.sqlite` stay in ignored `data/` directories on the ephemeral runner and are not retained.
To train later, recreate the cohort and consume it in the same job, or use private local Linux
storage. A saved manifest proves a measured run occurred; it does not mean a database was
downloaded to the Windows machine.

The workflow adds `--repeat-check`: a second complete build reuses the same pinned inputs and
compares ordered logical rows and partition audits. It also records whether SQLite bytes match.
This checks repeatability on one runner/version, not cross-platform or model-fit determinism.

## Storage, verification and interpretation

`cohort.sqlite` contains every exported flow, including exclusions. `status='matched'` identifies
candidate labeled rows. Exclusions have null `label`, `detail`, `uid_hash` and `group_hash`.
The only potential model inputs are the manifest's eleven `features`; partition, capture,
timestamps, hashes, status, labels and detail must never enter preprocessing/model features.

`group_hash` joins matched segments from the same capture-scoped UID. Its source-scoped UID
pseudonym is also retained to count accidental reuse across captures. No raw UID, endpoint
address or payload is stored. Hashed identifiers still belong in private storage.

The audit reports full export time bounds, class support by **connections**, feature collisions
and pairwise split overlap. Repeated feature patterns and conflicting labels remain present.
They are not removed to improve metrics; all measurements describe the accepted subset and
must be accompanied by exclusion coverage.

Completed preparations have `manifest.json`, written after the database is closed and reopened
for validation. Its byte hash and logical row digest bind the audit to that exact output. The
verifier checks current configuration hashes, database bytes, row values, grouping, support and
chronology, then compares the rebuilt audit to the saved audit and replay counters. Run from
the repository root at the recorded configuration revision. Fresh output paths prevent silent
overwrites. An interrupted directory has no completed manifest; retain it for diagnosis and
choose a fresh path to retry.

The 512 MiB cap applies to the main SQLite file, with an 8 MiB cache target. Journals, temporary
sorting files, simultaneous label/packet indexes and OS caches need additional space. Preparation
checks a 2 GiB reserve plus the main-file budget before starting. Sampled RSS is diagnostic, not
a hard memory limit. Passing preparation never enables training or promotes the label rule.
