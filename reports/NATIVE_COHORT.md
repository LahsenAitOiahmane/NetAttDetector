# Native cohort preparation and partition audit

Measured 1 November 2026. **All 250,819 exported flows are materialized in a private candidate
cohort, including 1,822 exclusions.** The 248,997 matched rows retain exactly the assignments
from PR #8. This is dataset preparation, not a detector accuracy result or training approval.

## Registered scope and reproduction

PR #8 merged into main at `c537900`. Registered the
[native-cohort protocol](../docs/research/NATIVE_COHORT.md) and
[bounded configuration](../configs/data/native-cohort.json) at `3ba79b2`, before creating rows.
The [first measured cloud run](https://github.com/LahsenAitOiahmane/NetAttDetector/actions/runs/41308516328)
at `eb1455e` passed exporter fixtures, archived baseline/candidate replay, all previous-match
preservation checks, complete row storage and a fresh read-only database audit/verification.
No new source or holdout was acquired. No model was fitted.

The [follow-up run](https://github.com/LahsenAitOiahmane/NetAttDetector/actions/runs/41308841070)
at `2d51251` rebuilt the entire cohort twice from the same pinned captures. Both builds and
the original run produced the **same logical rows and complete SQLite byte hash**:
`ba66937547a1b5e22bbf665ef42366e1b47f5a539f87a158856bc449a26778f3`.
See the [first manifest](evidence/native-cohort/first/manifest.json),
[repeated-run manifest](evidence/native-cohort/repeated/manifest.json),
[independent verification](evidence/native-cohort/repeated/verification.json) and
[rebuild comparison](evidence/native-cohort/repeated/reproducibility.json).
This establishes repeatability for these inputs on the recorded Linux environment, not across
all platforms/versions or for model fitting.

The cohort stores the eleven integer `nfstream-core-v1` features, separate event/counter
metadata, capture/partition IDs, linkage status and accepted class/detail. SHA-256 pseudonyms
identify flow keys, exact feature vectors and connection groups. Raw endpoint addresses, raw
Zeek UIDs and payload bytes are absent. Hashes remain private pseudonyms, not anonymization.

## Actual stored partitions

| Partition / capture | All rows | Matched benign / attack rows | Distinct benign / attack connections | Excluded rows |
|---|---:|---:|---:|---:|
| Train / 3-1, May 2018 | 224,388 | 3,968 / 218,713 | 3,169 / 151,346 | 1,707 |
| Calibration / 8-1, July–August 2018 | 12,459 | 2,181 / 10,278 | 2,181 / 8,222 | 0 |
| Validation / 34-1, December 2018 | 13,972 | 2,090 / 11,767 | 1,904 / 6,923 | 115 |

The observed event ranges match the earlier proposal. Capture/time ordering, 300-second purging
and at least 25 distinct accepted connections per class/partition pass. There are zero repeated
flow identities, zero inconsistent/cross-partition connection groups and zero source-scoped UID
pseudonyms reused across these captures. All **matched** segments of each UID share one group;
excluded flows do not acquire guessed UIDs or benign labels.

These are three captures from one source, not independent datasets or verified independent
devices. Class and attack-family distributions differ sharply: training has only eight matched
C&C UIDs, calibration contains 8,222 C&C attack UIDs, and validation contains C&C, scanning and
114 matched DDoS UIDs. Connection counts are not proof of independent attack events. Training
is heavily attack-skewed, so a high overall accuracy alone would be misleading.

## Repeated patterns and uncertainty retained

Among accepted rows, there are 128,407 distinct exact eleven-feature vectors; 43,433 vectors
repeat. **Sixteen vectors have conflicting binary labels, affecting 50 rows.** They remain
present rather than being deleted to improve reported performance.

| Partition pair | Shared exact feature vectors | Left / right affected rows | Shared vectors with conflicting labels |
|---|---:|---:|---:|
| Train / calibration | 9 | 272 / 29 | 0 |
| Train / validation | 84 | 632 / 1,113 | 0 |
| Calibration / validation | 17 | 2,179 / 1,315 | 0 |

Repeated measured patterns do not establish a repeated physical connection or device. They
are disclosed as a benchmark limitation; they are not silently removed, resampled or used to
claim independent device generalization. Later preprocessing/feature policies require their own
overlap diagnostics because dropping ports or deriving ratios can merge additional patterns.

All excluded statuses are retained: 1,325 unsupported-protocol flows, 333 partial intervals,
69 ambiguous flows, 93 unknown-label/duration flows and two unmatched flows. The contradictory
14,260 zero-counter DDoS labels remain unrepaired and the packet-tail/fragment limitations in
[PACKET_LABELS.md](PACKET_LABELS.md) still apply. The candidate rule is not promoted.

## Resource measurements and artifact boundaries

The first completed SQLite database is **181,800,960 bytes (173.4 MiB)**, below its 512 MiB
main-file cap. Ordered logical row SHA-256:
`9be2a2d5c39b3cb78a31fec40118971ba03b0e62b398e1151f7480ce8c312d59`.

| Capture | Preparation seconds | Sampled process-tree RSS sum MiB |
|---|---:|---:|
| 3-1 | 38.43 | 260.9 |
| 8-1 | 3.23 | 205.6 |
| 34-1 | 7.42 | 220.9 |

The fresh database audit took 8.76 seconds and sampled 122.0 MiB; a separate verification took
8.63 seconds with the same rebuilt audit. Preparation measurements include packet/label parsing,
matching and row inserts, excluding downloads and final index construction. Audit/verification
measure their separate read-only passes. RSS is sampled every 50 ms, may miss peaks or count
shared child pages twice, and is not an OS memory guarantee. SQLite cache targets/main-file caps
do not cover all temporary files or journals. One run does not establish a speed improvement.

The follow-up first build used 59.56/4.91/10.98 seconds for captures 3/8/34, with maximum sampled
tree RSS of 260.5 MiB. Runner variation changes elapsed time despite identical output bytes;
the saved rebuild comparison also retains resources from its second build.

Only aggregate manifests and verification evidence are retained as CI artifacts. The database
was materialized privately on the ephemeral Linux runner and **was not downloaded to Windows**.
The Windows laptop retains the already downloaded raw development bundles; its Ubuntu WSL
registration points to an unavailable `E:` virtual disk. That environment was inspected, not
modified. Reproduce on working Linux or in the bounded cloud workflow; no driver installation
or WSL repair is required for this sprint.

## Exit and next study

NAD-046 now has an actual row-level candidate, verified grouping/chronology/support, immutable
hashes and explicit overlap/exclusion accounting. Read the
[operations guide](../docs/guides/NATIVE_COHORT.md) for preparation and verification commands.

This does not change the training-disabled global capture plan or approve deployment. The next
study must explicitly register its use of candidate labels, capture/time-only scope, feature
policies, connection weighting, calibration/selection rules and resource budgets before fitting.
Device independence, label contradictions, an untouched compatible external test, site operating
requirements and public-release terms remain open. No broader detector claim follows from
passing data-preparation checks.
