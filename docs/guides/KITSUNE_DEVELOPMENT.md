# Reproduce Kitsune development admission

Read the [registered protocol](../research/KITSUNE_DEVELOPMENT.md) and
[measured report](../../reports/KITSUNE_DEVELOPMENT.md) before interpreting counts as training
support. These stages acquire and audit packets; they do not authorize model fitting.

## Local Windows or Linux

Only the ordinary locked package is needed for the independent structural audit:

```powershell
uv sync --frozen --extra dev
uv run --no-sync python scripts/research/prepare_kitsune.py acquire
uv run --no-sync python scripts/research/prepare_kitsune.py audit
```

The default source is `data/raw/kitsune-mirai`; choose a fresh `--source` path when reproducing
an acquisition. Existing files/directories are never overwritten. The download follows the
public author-mirror GET route, restricts redirects and byte counts, and checks the committed
input hashes. No login, capture driver or malware execution is involved. The audit detects
PCAPNG by magic bytes despite the misleading filename, preserves original packet order, and
requires exactly one binary label per original packet. The output defaults to
`artifacts/kitsune-packet-audit.json`; choose a fresh `--output` for each repeat.

If labels/bytes have changed, an unsupported format is encountered, or packet counts differ,
the stage fails. Do not remove an initial numeric label, infer an index/header, omit unsupported
packets, sort the capture, or relax committed hashes merely to make alignment pass.

## GitHub reproduction

With the current branch pushed and an authenticated GitHub CLI:

```powershell
$studyBranch = git branch --show-current
gh workflow run measurements.yml --ref $studyBranch -f kitsune_audit=true
gh run list --branch $studyBranch
```

The workflow reacquires only the two registered files, tests the parser/downloader, verifies
input pins and performs the full packet-label audit. It retains only aggregate receipts and
audit JSON; raw packets, labels and source-derived real rows stay private on the runner.
Compare the two input SHA256 values and all non-resource audit fields. Acquisition receipt
hashes can differ because the receipt includes observed resource timings; the source input
hashes and semantic counts must match.

Inspect binary counts by structural status, truncation, clock regressions and phase ranges.
Most malicious packet labels are non-IP and cannot directly become native flow examples.
An exact packet/label count is necessary but does not prove exporter-to-label alignment.
Any native cohort requires a separately registered and tested linkage policy, complete packet
coverage accounting and leakage-safe capture/group handling. No per-packet labels, phase
boundaries or dataset identities may enter model inputs.
