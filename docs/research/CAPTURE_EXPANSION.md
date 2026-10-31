# Registered development capture expansion

Registered 31 October 2026 on top of PR #6. The first three captures exposed sparse attack
support and extensive timeout segmentation. This phase expands **development** data using
publisher file sizes and label tables. It is adaptive development, with no model selection or
untouched-test claim. CICIDS2017 remains reserved/unacquired; CSE remains observed evaluation.

## Inputs and bounded processing

[iot23-expansion.json](../../configs/data/iot23-expansion.json) registers complete scenarios
8-1 (Hakai), 34-1 (Mirai) and 3-1 (Muhstik) before their acquisition/label inspection. Original
publisher locations are in the configuration; only PCAP, labelled connection log and README
are acquired. Pins record observed bytes, not a publisher per-file checksum. Keep the original
[three-capture profile](../../configs/data/iot23-development.json) reproducible.

| Limit | Original profile | Expansion profile |
|---|---:|---:|
| PCAP input | 16 MiB | 128 MiB |
| Label-log input | 16 MiB | 32 MiB |
| Label records | 100,000 | 250,000 |
| Exported-flow hard stop | 100,000 | 250,000 |
| Downloaded bundle | 20 MiB | 160 MiB |
| Free-disk reserve before acquisition | 2 GiB | 2 GiB |
| Download deadline checked between reads | 300 seconds | 1,200 seconds |
| SQLite cache / main database bound | 8 MiB / 512 MiB | 8 MiB / 512 MiB |

The materializing fixture exporter retains its 16 MiB/10,000-flow defaults. Expanded bounds
require an explicit registered profile. Downloads hash/write in chunks of at most 64 KiB and
publish the target only after byte-count/checksum verification; failures remove their partial
file. Atomic destination creation refuses to replace existing inputs.
Each blocking network operation also uses a 45-second timeout. The larger download deadline
was added after the first local 34-1 transfer timed out; it does not relax the audit resource gate.

Label indexes use local SQLite with exact int64 nanoseconds, a tuple/start-time index, unique
capture-scoped UIDs, an 8 MiB cache and a 512 MiB main-database cap. Journals/temporary index
build files also consume disk; the cap is not a bound on total process memory or total disk use.
Each match queries at most two overlapping candidates: two suffice to declare ambiguity.
Unknown ends remain blockers. No timestamp rounding, matching-rule relaxation or benign default
is introduced. Sub-nanosecond/out-of-range timestamps fail explicitly. The private temporary
index is closed and its single file removed after the audit.

Feature rows are streamed. Bounded identity-key/UID sets and counters remain in memory; memory
is not constant in flow count. The exporter still uses one meter, two-second idle timeout,
1,800-second active timeout, IP bytes, and the original eleven features. The label matching
policy remains `iot23-interval-v1`. New diagnostics count missing durations by publisher class.

## Acceptance fixed before processing the new captures

1. Verify every observed input pin on a repeat and original synthetic measurement values for
   both the original and expanded exporter settings. Disk and in-memory interval matching
   must agree, including exact boundaries, unknown ends, duplicates and reversed direction.
2. Replay the original three captures and compare linkage counts with their saved evidence.
   Additional diagnostic fields/resource timings may differ; label decisions must agree.
3. Process each new complete input sequentially. Record all statuses, matched segment counts,
   distinct matched Zeek UIDs, event ranges, index size, elapsed time and parent/child RSS sum.
   Fail on incomplete flow/label limits; do not label a prefix as a complete capture.
4. Screen each capture for at least 25 distinct matched benign and 25 attack connections.
   These are distinct UIDs, not proof of independent attacks. All segments of one UID must
   stay together in future partitions. Report attack-detail coverage and excluded records.
5. Resource acceptance for this expansion is sampled process-tree RSS sum below 1 GiB and
   audit time below ten minutes per capture on the GitHub runner. Include parsing, imports,
   index building, matching and cleanup; exclude downloads. Fifty-millisecond sampling may
   miss brief peaks, and RSS sums can double-count shared pages. These are screening limits,
   not OS-enforced guarantees or live-sensor throughput/loss claims.

Passing a support/resource screen never authorizes training. Metadata/device identities, label
semantics, capture/time/host isolation, attack diversity, terms and a compatible labelled
training contract still require review. The existing global capture plan stays disabled. Do not
weaken the precision/FPR/recall requirements to manufacture a new champion.

Implementation amendments: capture 34-1 uses the explicit `det_label` header alias; both aliases
together are rejected. Per-detail matched UID counts expose selective label coverage. An ordered
flow/status/UID/label digest supports a fresh-process in-memory/SQLite comparison on the largest
registered label log. Resource results are one trial per mode. See the
[completed sprint evidence](../../reports/CAPTURE_EXPANSION.md).

## Reproduce

```bash
uv sync --frozen --extra dev --extra measurement
uv run --no-sync python scripts/research/audit_iot23.py --config configs/data/iot23-expansion.json --output artifacts/expansion --data-root data/raw/expansion
```

For the first registered acquisition only, append `--record-first`, then review/commit observed
pins before repeating. Existing pins are checked even in first-record mode. Use fresh output
directories. The manual measurement workflow accepts `capture_label_audit=true`,
`capture_profile=expansion`, and `record_first=false` for a pinned repeat. Raw captures, logs,
endpoint/feature rows and private SQLite indexes are excluded from artifacts; retain aggregates,
receipts, frozen configuration and source commit identity. Dataset terms/attribution remain in
the [capture runbook](../guides/CAPTURE_LABELS.md).
