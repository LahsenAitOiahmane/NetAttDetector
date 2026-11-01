# Statistical measurements preserve the native cohort

Measured 1 November 2026. **The 27-feature statistical export preserves every core value,
flow identity, order and label/group join for all 250,819 flows.** Five independent synthetic
flows pass 165 value checks. This validates measurement compatibility, not improved detection.

Registered [the protocol](../docs/research/STATISTICAL_FEATURES.md) at `66a5606` before export.
[Run 41313480823](https://github.com/LahsenAitOiahmane/NetAttDetector/actions/runs/41313480823)
at `f82832c` passed fixture, complete-core preservation, value/range, private-sidecar read-only
audit and resource screens. No model was fitted, no labels repaired and no external data used.

## Contract and storage

`nfstream-statistical-v1` retains the eleven measured core fields and adds sixteen
bidirectional statistics: min/mean/sample-standard-deviation/max IP packet size; equivalent
interarrival statistics in milliseconds; and SYN/CWR/ECE/URG/ACK/PSH/RST/FIN counts. Only
`statistical_analysis` changes in NFStream 6.6.0 settings. Direction, IP accounting, idle/active
timeouts, one meter, no dissection/tunnels and the existing byte/flow limits stay fixed.

Expected values use explicit independent packet-size/time/flag lists. The six TCP IP lengths
are 40/40/40/45/47/40, giving mean 42 and sample deviation sqrt(10); its five intervals give
mean 100 ms and deviation sqrt(950). SYN/ACK/PSH/RST counts are 2/5/2/1. Single packets have
zero intervals and non-TCP flows have zero flag counters. All 165 values pass with exact
integer comparisons and 1e-9 float tolerance. The original five-flow/85-value lab also passes.

The private sidecar has 250,819 entries and **118,968,320 bytes (113.5 MiB)**. It contains
only row/flow identities, sixteen extra values and exact pattern hashes; labels, exclusions
and eleven core fields remain in the original 173.4 MiB private database. Reopening both
read-only verifies a complete identity join and recomputes every stored pattern hash.

Sidecar logical row digest: `a4e6d0ff836814acadd4cc778b0b63320fe5dc38cc080f040caaeb95db189259`.
Sidecar byte digest: `6b9b4c289db495ae7c5c592d91b4c27781b62b21d1746b9749384dca27cb50f9`.
Core logical/byte digests remain the previously measured values. Only the
[aggregate manifest](evidence/statistical-features/manifest.json) and
[fixture verification](evidence/statistical-features/fixture-verification.json) are retained.
The private database was not downloaded to Windows.

## Additional information is limited

Exact-pattern analysis uses the 248,997 accepted rows; all repeated/conflicting rows remain.

| Input representation | Unique patterns | Conflicting-label patterns | Rows in conflicting patterns |
|---|---:|---:|---:|
| Eleven core fields | 128,407 | 16 | 50 |
| Core without ports + deterministic ratios/rates (earlier study) | 11,427 | 58 | 206,404 |
| Twenty-seven statistical fields | 128,801 | 16 | 50 |
| Statistical fields without ports | 12,416 | 63 | 206,075 |

The richer fields distinguish some patterns, but do not remove the large behavioral ambiguity
after ports are excluded. A higher number of distinct vectors does not prove better transfer.
Exact float64 comparisons also do not describe estimator binning/quantization.

Accepted DDoS data has 114 rows/connections: **113 contain SYN** and none contain ACK/PSH/RST/
FIN/URG/CWR/ECE; one has no TCP flags. Accepted scanning data has 145,699 UIDs, all with SYN
and none with the other reported flags. Benign data also has 2,504 UIDs with SYN. Therefore a
SYN flag alone does not establish an attack or solve benign/DDoS/scan separation. C&C has
14,936 accepted UIDs with SYN, but only 1,648 with ACK, further illustrating that publisher
family names do not imply completed application-level sessions in every accepted row.

These are observed counters, not checksum/reassembly validation or independent attack truth.
The earlier missing-duration/zero-counter DDoS contradictions and excluded protocols remain.
No new statistical field is used to manufacture a label or claim broader DDoS coverage.

## Resource evidence and next step

| Capture | Rows preserved | Statistical export/inserts seconds | Sampled tree RSS sum MiB |
|---|---:|---:|---:|
| 3-1 | 224,388 | 14.07 | 233.9 |
| 8-1 | 12,459 | .91 | 207.2 |
| 34-1 | 13,972 | 1.28 | 203.0 |

The separate joined audit took 13.60 seconds and sampled 114.6 MiB. Downloads, prior core
preparation/verification and final index construction are outside per-capture samples. SQL
cache/main-file caps exclude journals/temp files; 50 ms RSS samples are not OS limits and can
miss brief peaks/double-count child pages. Every registered screen passed.

Local focused checks passed 41 tests with three Linux-only skips before remote measurement.
Python 3.11/3.13 CI passed 159 tests with six skips each; the Linux measurement environment
runs the actual optional exporter instead. The new fixture is checked automatically in CI.

Next register a bounded comparison of core controls versus statistical/no-port models, using
fixed-thread fitting, unchanged capture partitions and calibration-only thresholds. If richer
fields still fail, prioritize causal traffic context, representative development captures and
label semantics rather than increasing model size. Preserve reserved independent data for a
later frozen evaluation. The owner-deferred pilot and deployment/release gates remain separate.
See the [runbook](../docs/guides/STATISTICAL_FEATURES.md).
