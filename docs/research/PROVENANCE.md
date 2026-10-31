# Source lineage and capture-aware split preparation

The [registered source-role map](../../configs/data/source_roles.json) distinguishes development,
observed evaluation, excluded merged data and a reserved/unacquired source. Its hash is recorded
in each header-inspection and split-plan report. Do not silently change source roles to reuse a test.

| Source | Role now | Capture/time evidence |
|---|---|---|
| NF-UNSW-NB15-v2 | Development | Local CSV has IPv4 endpoint columns; no event time, row-linked capture ID or session ID |
| NF-ToN-IoT-v2 | Development | Same metadata limitation; matching headers do not establish compatible exporters |
| NF-CSE-CIC-IDS2018-v2 | Observed evaluation | Already scored after the previous freeze; cannot become an untouched holdout |
| NF-UQ-NIDS-v2 | Excluded merged collection | Contains constituent source lineages; cannot supply independent testing of those constituents |
| IoT-23 | Development, registered 31 October before acquisition | Three pinned raw PCAP/Zeek-label bundles acquired; candidate linkage measured; sparse attack support and a README metadata discrepancy block a benchmark |
| CICIDS2017 | Reserved, not acquired | Publisher documents day-specific PCAPs, labeled flow timestamps and endpoints; a distinct adapter/re-export is required |

The actual development archive headers and hashes are in
[UNSW evidence](../../reports/evidence/provenance/unsw-header.json) and
[ToN evidence](../../reports/evidence/provenance/ton-header.json). Only headers were inspected;
no benchmark attack rows or scores were accessed in this sprint. `FLOW_DURATION_MILLISECONDS`
is not an event timestamp. A ZIP/CSV filename is not a row-linked capture identifier. Endpoint
columns alone cannot recover timestamps, capture boundaries, NAT identity or a stable device mapping.

Primary source records: [UQ publisher](https://www.cyber.uq.edu.au/project/machine-learning-based-nids-datasets)
and [CICIDS2017 publisher](https://www.unb.ca/cic/datasets/ids-2017.html). The latter describes
five capture days, including a benign-only Monday. A day split that omits attacks cannot be
treated as a complete detection-quality fold. No large PCAP collection was downloaded here.
Reservation is not label-quality validation or commercial-use clearance.

The [next sprint](../../reports/DEVELOPMENT_CAPTURES.md) explicitly added IoT-23 development
data after the original UNSW/ToN raw download links redirected to sign-in. The role change was
committed before reading traffic/labels; historical header reports retain the old role-map hash.
All three IoT-23 bundles are now downloaded locally and checked against pinned hashes. The 42-1
README conflicts with its PCAP on capture name/year, and distinct matched attack support is only
8/8/5 connections. Do not copy exported-segment counts into a supposedly independent split.

## Manifest and validation

[capture_plan.json](../../configs/data/capture_plan.json) deliberately contains no invented captures.
The [saved validator result](../../reports/evidence/provenance/capture-plan.json) is blocked.
The next benchmark needs raw captures and trustworthy row-to-capture/label linkage before fitting.

Each future capture declaration needs `source_id`, `capture_id`, `split`, existing local `path`,
`sha256`, timezone-aware `start_utc`/`end_utc`, explicit `host_groups`, `benign_rows` and `attack_rows`.
Paths and the source-role map resolve from the repository working directory. Host groups must
identify the documented held-out client/device units, using namespaced IDs rather than assuming
private IP addresses identify the same device across sources. State whether shared servers are
excluded from that grouping; the validator cannot discover undeclared endpoint overlaps.

The validator checks source-role and lineage separation, repeated capture/input bytes, actual
file hashes, host-group separation, within-source chronological ordering and a 300-second purge
between partitions, and at least 25 declared benign/attack rows per partition and source/partition.
This is a registered preparation policy, not an experimentally optimized gap/support level.
Capture start/end dates must describe actual event times. Different independent sources are not
required to share a calendar; a source-transfer test is not a claim of future-time prediction.

Even a structurally valid manifest does **not** authorize training. Declared label counts, time
semantics and host IDs need raw-data verification; row-to-capture linkage and a feature-compatible
labelled dataset must be audited. The preparation CLI always reports `training_authorized: false`.
It does not retrofit missing metadata into the existing cached cohort.

```powershell
uv run --no-sync netatt inspect-provenance --data data/raw/nf-unsw-nb15-v2.zip --source nf-unsw-nb15-v2 --output reports/local/unsw-provenance.json
uv run --no-sync netatt check-capture-plan --output reports/local/capture-plan.json
```

The second command currently writes the blocker report and exits with code 2. This is the
expected gate outcome, not a broken training run. Use fresh report paths. Tests exercise valid
synthetic declarations and reject chronology, host, checksum, lineage, role and support failures.
