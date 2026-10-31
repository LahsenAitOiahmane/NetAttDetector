# Packet witness audit operations

This is a development-data validation tool. Read the
[registered protocol](../research/PACKET_LABELS.md) and [measured report](../../reports/PACKET_LABELS.md)
before interpreting coverage. It uses existing offline files, not network interfaces.

## Windows: independent packet and label inspection

The three complete bundles are already downloaded. The expansion report's
[local inventory](../../reports/evidence/capture-expansion/local/acquisition.json) records their
actual directories, including verified retry directories.

```powershell
uv sync --frozen --extra dev
uv run --no-sync netatt audit-packet-labels --capture-dir data/raw/iot23-expansion-complete/CTU-IoT-Malware-Capture-34-1 --output reports/local/packet34.json
```

Use a fresh output path. This verifies pinned file bytes, scans classic PCAP headers, builds
private bounded packet/label indexes and checks exact point witnesses. It produces aggregates
without a capture driver or NFStream. It does not export features, infer attacks or modify source
labels. Normal and exceptional exits close/remove their newly created temporary indexes.

## Linux: compare the unchanged baseline and point candidate

```bash
uv sync --frozen --extra dev --extra measurement
uv run --no-sync netatt audit-packet-labels --capture-dir data/raw/iot23-expansion-complete/CTU-IoT-Malware-Capture-34-1 --output artifacts/packet34-comparison.json --compare-flows
```

Both matchers see the same NFStream 6.6.0 stream and settings. Baseline replay checks compare
saved counts and the ordered assignment digest. Losing/changing an old accepted match or
disagreeing with the frozen baseline writes the report and exits with an error. Gains remain
candidate labels; no command enables training or rewrites the old interval rule.

To acquire pinned inputs and run the full experiment sequentially:

```bash
uv run --no-sync python scripts/research/audit_packet_labels.py --data-root data/raw/packet-labels --output artifacts/packet-labels
```

Both directories must be fresh. The manual **Packet label evidence** workflow performs the
same steps. Before its first merge to main, dispatch **Offline flow measurement lab** on the
review branch with `packet_label_audit=true`; leave the other real-data audit disabled to avoid
duplicating acquisition/analysis. Ordinary PR fixture CI does not download these captures.

## Evidence and limits

The source configuration and policy pin source roles, byte/row limits, exact timestamp matching,
flow settings and source hashes. The reader supports classic PCAP 2.4 Ethernet, micro/nanosecond
timestamps, IPv4/IPv6 base headers, up to two VLAN tags and bounded TCP/UDP structure. It reports
fragments/extensions/other protocols rather than reassembling or guessing. It does not validate
transport checksums, capture loss, application content or the truth of the publisher's class label.

Main packet-database cap: 512 MiB; cache target: 8 MiB; PCAP cap: 128 MiB; record cap: two million.
Journals/temp files and bounded UID/identity sets also consume resources. Sampled RSS is not a
hard OS limit. Do one capture at a time on the laptop. Raw packets, endpoint rows, UID claims and
SQLite indexes are excluded from Git/artifacts; retain aggregate JSON, configuration and receipts.

To redraw the saved figure without acquiring data or fitting a model:

```bash
uv sync --frozen --extra analysis
uv run --no-sync python scripts/reports/plot_packet_labels.py --summary reports/evidence/packet-labels/cloud/summary.json --output reports/local/packet-labels.png
```

Inspect `native_partition_readiness` in the summary: capture/time order and support are checked,
while device isolation, a native cohort and independent test readiness remain explicitly false.
The [proposed plan](../../configs/data/native-dataset-plan.json) is not training authorization.
