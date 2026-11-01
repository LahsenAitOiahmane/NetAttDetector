# Causal context measurement operations

The [registered lab](../research/CAUSAL_CONTEXT.md) adds nine observed flow-start history
fields to the 27 statistical fields. This is offline measurement/replay, not model fitting or
an authorized live sensor. All interfaces remain rejected by the exporter.

## Known fixture and current inputs

On Linux, from the checkout root:

```bash
uv sync --frozen --extra dev --extra measurement
uv run --no-sync netatt verify-context-fixture --output reports/local/context-fixture.json
uv run --no-sync python scripts/research/audit_causal_context.py
```

The script first verifies the five-flow initialization history, then recreates the pinned
core/statistical cohorts and exports the same captures through a one-meter NFPlugin. Use fresh
`--data-root`, `--cohort`, `--statistics`, `--context` and `--evidence` paths, or add a private
`--captures path/to/map.json` to reuse existing verified input bundles.

```bash
gh workflow run measurements.yml --ref YOUR_BRANCH -f causal_context=true
```

The reusable workflow keeps model/row databases private and retains an explicit aggregate
JSON allowlist. No temporary endpoint replay targets are uploaded. Windows can test the state
and SQL reference routines without a driver; actual NFStream callback export is Linux-only.

## Interpret the clock and values

Each snapshot uses past **observed flow starts**, including excluded-label flows and timeout
segments. It does not count ground-truth connections or packets. Context time is a prefix
watermark over flow-start timestamps in actual initialization order. Windows are inclusive at
the lower bound and exclusive at the watermark, so every same-watermark peer stays out of the
current snapshot. Pending peers commit only when the clock advances. Late timestamps keep
arrival order and are counted in metadata; original timestamps remain untouched.

Warm-up indicates at least 60 seconds of watermark span since the first observed start, not
guaranteed physical uptime during clock jumps. Warm-up rows remain present. Initialization
sequence, watermark, late/warm-up flags and peak-cache metadata are excluded from model inputs.
Only nine counts/distinct counts join the measured feature contract. Endpoint addresses remain
in private temporary/state grouping and are absent from retained sidecars/model inputs.

The active plus pending cache refuses more than 100k events and becomes terminal after an
overflow. It never fills absent history with invented zeros or silently resumes after loss.
Per-source/destination zero counts mean zero **observed** eligible earlier starts. Per-capture
state resets prevent cross-capture history contamination.

## Acceptance evidence

Core/count/time fields must match the existing cohort exactly; statistical means/deviations
use their existing 1e-9 absolute/relative measurement tolerance, with any nonexact values
reported. Retained context joins use the original verified statistical columns. Full callback
arrival replay checks every row, sequence, watermark and history value. A separate direct-SQL
reference checks 100 evenly spaced starts/capture; full replay itself shares the tested window
routine and is not an independent algorithm.

The numeric sidecar is reopened to validate identity joins, hashes, prefix metadata, warm-up
counts and feature-pattern ambiguity. Only each newly created temporary endpoint DB is removed;
raw captures, old artifacts and policy-blocked historical files remain untouched.

The 512 MiB main-file caps apply separately to sidecar and temporary targets, with 8 MiB cache
targets. A 2 GiB reserve plus both caps is checked first. Journals/temp files need extra space;
RSS samples every 50 ms do not enforce OS limits. Callback behavior under real live loss,
load, NAT changes and sensor failure remains unvalidated; the owner-deferred pilot stays separate.
