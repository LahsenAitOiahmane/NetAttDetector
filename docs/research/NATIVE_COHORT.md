# Registered native-cohort preparation

Registered 1 November 2026, before row-level materialization. This implements NAD-046 after
PR #8 merged into main at `c537900`. It prepares an **experimental candidate**, not an approved
training set or a promoted label policy. No fitting, threshold selection or external test occurs.

## Inputs and partition rules

Use only the pinned IoT-23 expansion PCAP/label/README bundles and the unchanged
`iot23-packet-point-candidate-v1` rule. Reuse the reviewed May capture 3 / July capture 8 /
December capture 34 train/calibration/validation proposal. Verify actual exported time ranges,
300-second inter-partition purging and at least 25 distinct matched UIDs per binary class in
each partition. Reserved CICIDS2017 remains unacquired and CSE remains observed evaluation.
Device identities are unverified: passing capture/time checks must not imply host isolation.

## Stored contract

Stream the existing Linux NFStream 6.6.0 exporter, one capture at a time, into a private SQLite
database. Persist exactly the eleven `nfstream-core-v1` model fields as integers, separate
event times/counters, capture/partition metadata, label status and accepted class/detail.
Unknown, ambiguous, partial and unsupported flows remain present with null labels/groups.
Never infer benign labels for exclusions or repair the contradictory zero-packet DDoS records.

Use SHA-256 canonical JSON identities for flow keys, feature vectors and source-scoped UID
pseudonyms. Group matched rows by source + capture + UID pseudonym. All **matched** segments
of a UID stay together; excluded flows cannot be assigned a UID by guessing. These hashes
are private pseudonyms, not anonymization. No endpoint addresses, raw UIDs or payload bytes
are stored in the cohort. Only the eleven named features may later become model inputs.

## Acceptance before any new experiment

- Replay archived v1 and candidate assignment counts/digests and preserve all old matches.
- Reject repeated flow identities, inconsistent labels within a connection group, cross-partition
  connection groups, invalid features/counters, insufficient class support or bad chronology.
- Report exact feature-vector repetitions, conflicting binary labels and overlap between splits.
  Repeated traffic patterns alone are not independent evidence of leakage. Do not delete them
  or rebalance validation data to make performance look better.
- Write input/config hashes, a database hash and an ordered logical row digest. Independently
  reopen the completed database, check its schema/integrity and recompute the aggregate audit.
  An interrupted preparation must not produce a completed manifest.
- Use the registered byte/flow/cache bounds. Measure time and sampled process-tree RSS. Publish
  aggregate evidence only; never upload the cohort database or feature/group rows to CI artifacts.

Candidate label promotion, packet-tail/DDoS semantics, device provenance, public release terms
and an independently registered model/calibration protocol remain separate work. Do not change
the blocked global capture plan or enable training just because preparation passes.
