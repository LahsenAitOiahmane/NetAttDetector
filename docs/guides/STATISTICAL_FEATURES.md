# Statistical measurement operations

This [registered lab](../research/STATISTICAL_FEATURES.md) measures an exporter contract, not
detector performance. `nfstream-statistical-v1` has the eleven core fields plus sixteen
bidirectional size/interarrival/TCP-flag fields. The original core/UQ contracts stay separate.

## Independent fixture

On Linux, from the checkout root:

```bash
uv sync --frozen --extra dev --extra measurement
uv run --no-sync python scripts/data/make_statistical_fixture.py --check
uv run --no-sync netatt export-statistical-pcap --pcap tests/fixtures/pcap/core.pcap \
  --output artifacts/my-statistical-fixture.json
uv run --no-sync netatt verify-statistical --export artifacts/my-statistical-fixture.json \
  --output reports/local/my-statistical-check.json
```

Expected values come from explicit packet-size/interval/flag lists independent of NFStream.
Five flows cover 165 feature/metadata values. Min/max/count fields compare exactly; statistical
means/sample standard deviations use 1e-9 absolute/relative tolerance. IPv6, both directions,
single packets, TCP handshake/data/reset and idle tuple reuse are included. This does not
validate fragmentation, checksum policy, live loss or every possible TCP state.

Windows can run expectation checks and verify a saved synthetic JSON export; actual NFStream
export remains Linux-only. Do not install capture drivers or repair WSL for this lab.

## Private development sidecar

```bash
uv run --no-sync python scripts/research/audit_statistical_features.py
```

This first checks the independent fixture, acquires the three pinned expansion bundles,
recreates/verifies the core cohort, then streams a fresh statistical export for each capture.
An optional private `--captures path/to/map.json` reuses existing capture-ID/path inputs. Use
fresh `--data-root`, `--cohort`, `--statistics` and `--evidence` directories when overriding
defaults. All statistical core fields and metadata must match the original row exactly; no
new/missing/reordered identity is accepted. The verified join preserves the old label/group
without using any statistical field to decide a label.

`statistics.sqlite` contains only ordinal/flow identities, the sixteen extra values and exact
full/no-port pattern hashes. Labels/exclusions stay in the original private core database.
The sidecar audit reopens both read-only, validates every join/value/hash, and reports accepted
feature collisions, per-detail flag presence and flow-weighted size/timing summaries. Single-
packet intervals and non-TCP flags must be zero; invalid values fail rather than being imputed.

```bash
gh workflow run measurements.yml --ref YOUR_BRANCH -f statistical_features=true
```

The manual reusable workflow retains only fixture verification, core manifest/verification,
statistical manifest and aggregate progress. No database, real row, UID or model is uploaded.
The 512 MiB sidecar cap/8 MiB cache target exclude journals/temp files; a 2 GiB reserve plus
the main-file budget is checked before writing. Per-capture samples cover statistical export
and inserts; final index/audit passes are separate. Resource samples are not hard OS limits.

Passing these checks establishes measured compatibility/information, not improved recall or
precision. A subsequent model study needs its own registered scope and independent quality
checks. Device isolation, label truth/DDoS coverage, external testing and the deferred pilot
remain separate requirements.
