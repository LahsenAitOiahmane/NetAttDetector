# Reproduce the causal-context diagnostic

Read the [registered study](../research/CONTEXT_STUDY.md),
[context clock and measurement rules](../research/CAUSAL_CONTEXT.md) and
[private model retention protocol](../research/PRIVATE_ARTIFACTS.md) first. This remains
candidate-label offline research, with no real-network capture or deployment enabled.

## GitHub runner

The public recipient in `configs/security/model-recipient.json` must correspond to your
locally retained private key. A GitHub runner receives only the public key. From PowerShell:

```powershell
$studyBranch = git branch --show-current
gh workflow run measurements.yml --ref $studyBranch -f context_study=true
gh run list --branch $studyBranch
```

The workflow recreates the pinned core/statistical/context cohorts in private runner storage,
checks the known packet fixture, every original value/identity and causal replay, and reopens
the complete joins and logical digests before fitting. It compares twelve candidates and a
dummy and repeats the selected model twice, all at one thread. It retains an explicit allowlist
of aggregates and the selected `.nadenc` ciphertext, not plaintext models or real input rows.
The 30-day artifact lifetime is not a backup. Download the encrypted artifact before expiry.

Review the config/commit/run identity, complete `study.json`, family support, repeat/resource
outcomes and `freeze.json`, then retain those aggregate files under `reports/evidence/`.
Only use that independently trusted freeze for local recovery; decryption does not authenticate
the sender. The [recovery command](../research/PRIVATE_ARTIFACTS.md) checks the model SHA before
joblib loading and tests 45 synthetic probes on the local platform. Keep the private key backed
up securely: losing it prevents recovery of retained ciphertext.

## Private Linux reproduction

Use a working Linux host with enough disk for fresh pinned captures, three private databases
and a 2 GiB reserve. NFStream is pinned to 6.6.0; Windows can recover/score the frozen model
but cannot reproduce this exporter directly.

```bash
uv sync --frozen --extra dev --extra measurement --extra artifacts
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  uv run --no-sync python scripts/research/run_context_study.py
```

Optional `--captures` accepts an ignored JSON map of the three registered capture IDs to
directories containing their original pinned PCAP/Zeek/README inputs. All output directories
must be fresh; the runner refuses overwrites. Input data checks remain in force when reusing
captures. Encryption targets the configured owner public key regardless of execution platform.

## Interpretation

Calibration selects thresholds; validation ranks diagnostics. The binary screen and the new
per-family guard are reported separately. A failed quality screen is a valid negative result,
not a workflow exception. Resource/repeat failures do fail the workflow after retaining evidence.
Even a qualifying development result needs a frozen, independently sourced test and further
label/device evidence. The owner has explicitly deferred the later real-network pilot.
