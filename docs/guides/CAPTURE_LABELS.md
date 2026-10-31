# Development capture and label audit

The [registered IoT-23 inputs](../../configs/data/iot23-development.json) are development data.
The role was committed before inspecting PCAP bytes or label logs. This explicitly expands the
development lineages after the original UNSW/ToN publisher download links redirected to sign-in.
It does not repurpose the reserved CICIDS2017 source or create an untouched validation result.

The bounded pilot covers scenarios 20-1, 21-1 and 42-1. Acquisition fetches exactly the PCAP,
`bro/conn.log.labeled` and README for each scenario, sequentially. Each file is limited to 16 MiB,
each capture bundle to 20 MiB, with a 2 GiB free-disk reserve. Publisher HTTP response lengths and
previously observed SHA-256 pins are checked; no publisher per-member checksum is available.
It does not fetch binaries, recursively crawl folders or unpack the full 21.5 GB release.

The explicit [expansion profile](../research/CAPTURE_EXPANSION.md) adds scenarios 8/34/3 with
128 MiB PCAP and 32 MiB label-log limits, a 250,000-flow stop and an 8 MiB SQLite cache target.
The original profile and fixture defaults retain their limits. See the
[results and actual local directories](../../reports/CAPTURE_EXPANSION.md).

## Local acquisition and Linux audit

Windows can download and parse labels without capture drivers:

```powershell
uv run --no-sync netatt download-capture CTU-IoT-Malware-Capture-20-1 --output data/raw/iot23-local/CTU-IoT-Malware-Capture-20-1
```

Inspect an already downloaded expansion bundle on Windows or Linux:

```powershell
uv run --no-sync netatt inspect-capture-labels --config configs/data/iot23-expansion.json --capture-dir data/raw/iot23-expansion-verified/CTU-IoT-Malware-Capture-3-1 --output reports/local/capture3-labels.json
```

This verifies pins and builds a temporary private SQLite label index; it produces log statistics,
not flow exports. The recorded 156,103-row laptop inspection used 45.8 MiB sampled parent RSS.

Use a fresh output directory. Partial acquisition may leave downloaded files; no receipt is
written until all three files succeed. `--record-first` is for an explicitly registered new
acquisition, not a way to accept changed pinned bytes: existing pins are still checked.
Review and commit new observed pins before a repeat. Acquisition receipts retain source URLs,
byte counts, hashes, time, configuration hash and the unresolved terms note.

On Linux, process all three registered inputs with the locked optional exporter:

```bash
uv sync --frozen --extra dev --extra measurement
uv run --no-sync python scripts/research/audit_iot23.py --output artifacts/iot23-audit --data-root data/raw/iot23-audit
```

To reuse an already downloaded pinned bundle on Linux, run
`netatt audit-pcap-labels --capture-dir data/raw/iot23-development/CTU-IoT-Malware-Capture-20-1 --output artifacts/capture20-audit.json`.
The separate `audit-capture-labels --export ... --capture-dir ... --output ...` command checks
an existing JSON lab export; it retains that exporter's 10,000-flow/settings restrictions.

The manual **Development capture label audit** workflow runs this command. Until its first
merge, use **Offline flow measurement lab** with `capture_label_audit=true` and
`record_first=false` on the review branch. Automatic fixture CI does not acquire real traffic.
Artifacts retain aggregate counts, pinned configurations and receipts; raw PCAPs, label logs,
endpoint records and joined feature rows are excluded. Successful acquisition on a GitHub
runner does not mean the files have also been downloaded onto the laptop.

## Registered matching rule: iot23-interval-v1

Parse the publisher's `#fields` header; IoT-23 combines tab delimiters with spaces in its last
three fields. Preserve decimal Unix timestamps. Require unique Zeek UIDs and valid endpoints,
ports and nonnegative finite intervals. Map only explicit `Benign`/`Malicious` labels to 0/1.

Match TCP/UDP using an orientation-independent five-tuple and the entire exported time interval.
Exactly one overlapping Zeek record must contain the interval, allowing **one millisecond** for
timestamp quantization. Multiple candidates remain ambiguous even if their labels agree. A
partial interval, missing duration, unknown label or absent match remains unlabelled. An unknown
end can block later same-tuple candidates. ICMP is unsupported because exporter port conventions
have not been reconciled. There is no nearest-flow guess or benign default.

The original real-capture profile uses the fixture exporter's settings with a **100,000-flow**
hard stop and consumes one flow at a time. The original JSON fixture exporter retains its 10,000
limit. Input PCAPs remain capped at 16 MiB. The audit retains bounded identity keys, label indexes,
UID sets and counters, rather than every feature row. Hitting the limit fails instead of claiming
a complete audit. No training CSV or model is produced.

The expansion profile's larger bounds require `--config configs/data/iot23-expansion.json`.
`--label-index-backend memory` on `audit-pcap-labels` is a bounded reference implementation
for diagnostics; the default is SQLite. The expansion workflow compares both backends in fresh
processes on the largest label log, checking the ordered assignment digest and resource use.
`det_label` is accepted as an explicit publisher alias for `detailed-label`; both names together
are rejected. No binary class labels are inferred from the detailed label.

NFStream's two-second idle timeout can split one Zeek connection into many segments. Report
matched exported-flow counts and distinct matched Zeek-connection counts separately. Segments
from one UID are not independent examples and must stay together in any later split. A label
candidate match is not proof that every packet of a long connection shares its label, that the
exporters agree on all semantics, or that this data represents a small office.

Resource evidence samples the parent and child-process RSS every 50 ms during parsing,
exporter imports, iteration and matching. The process-tree RSS sum can count shared pages more
than once and can miss brief peaks; it is not a hard OS memory bound. Fits remain disabled.

## Terms, attribution and acceptance

Citation: Sebastian Garcia, Agustin Parmisano and Maria Jose Erquiaga (2020), *IoT-23: A labeled
dataset with malicious and benign IoT network traffic*, version 1.0.0,
[DOI record](https://doi.org/10.5281/zenodo.4743746).
The [Zenodo metadata](https://zenodo.org/api/records/4743746) lists CC-BY-4.0; individual
mirror READMEs also request authorization. Record this discrepancy rather than asserting that
the mirror or a derived model is commercially cleared. Raw/derived rows are not redistributed
by this workflow; resolve terms before a public data/model release.

See [the measured report](../../reports/DEVELOPMENT_CAPTURES.md) for coverage, failure history
and blockers. Adequate distinct attack connections, verified device identities, capture/time
isolation, additional protocol fixtures and reviewed label semantics are still required before
registering a new training experiment. The global capture plan continues to reject training.
