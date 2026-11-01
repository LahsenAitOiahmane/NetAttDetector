# Registered causal flow-start context lab

Registered 1 November 2026 before context extraction/fitting. NAD-051 proved that no-port
inputs cannot reach the observed recall/FPR target in the present UID-max family. NAD-052
adds genuinely observed traffic history, not larger models, labels or endpoint identities.

## Nine additional model fields

Start with the verified 27 statistical fields. Add counts of prior flow starts for the current
observed source and destination IP over 1/10/60 seconds (six fields), and prior 60-second
distinct destination IPs/destination ports for that source, plus distinct source IPs for that
destination (three fields). The separate contract is `nfstream-context-v1`, 36 features.
These are **flow starts**, including timeout segments, not ground-truth connections/incidents.

Every exported flow contributes history regardless of label or linkage status. Packed endpoint
addresses are used only in private state/temporary replay targets, never as model features.
Retained sidecars contain identity pseudonyms and numeric values only. No payload/DPI,
timestamp, source/capture ID or UID becomes a model input. Reset context per capture/meter.

## Causal clock, ties, warm-up and bounds

Compute snapshots in NFStream 6.6.0 `NFPlugin.on_init`, before recording the current start.
Use one meter. Keep an initialization sequence and a watermark equal to the prefix maximum
of **observed flow-start timestamps**. Preserve original first/last times unchanged. Windows
are `[watermark - window, watermark)`, excluding all same-watermark peers. Peers enter pending
state and commit when the watermark advances. Late starts remain in arrival order at the
current watermark and are reported; do not retrospectively reorder timestamps or alter labels.

This clock describes observed flow-start activity, not guaranteed wall-clock seconds under
clock jumps. Record late starts, watermark and initialization sequence as metadata. Warm-up
metadata indicates a 60-second watermark span since the first start, not independently verified
physical uptime. Keep warm-up rows and report them; do not discard difficult starts for scores.
Neither clock nor warm-up fields enter models. Snapshots cannot depend on final flow bytes/
duration, future events, labels, or later segments from the same connection.

Bound all active plus pending flow events to 100k. Delete empty counters on expiry. Exceeding
state or row/byte bounds fails the capture; never silently replace missing context with zeros.
The offline adapter accepts files only; no interfaces, packet transmission or live pilot occur.

## Independent checks and full preservation

Hand-calculated event cases cover current-peer exclusion, inclusive lower window boundaries,
expiry, distinct counters, late clocks, large gaps, future-event invariance and state refusal.
On the known five-flow PCAP, only the final reused UDP source/destination has prior history:
one start in 10/60 seconds, zero in 1 second and one distinct peer/port. No attack labels apply.

On real captures, compare all core/statistical values and flow identities to the verified
cohorts. Require complete labels/groups by the same identity join, with no new rule. Record
core/count/metadata values exactly; compare statistical means/deviations with their existing
1e-9 absolute/relative fixture tolerance and disclose nonexact values. The sidecar joins retain
the original verified statistical columns, never overwrite them with a new export.
Record
actual callback initialization order privately, then replay every event in that order through
the tested window routine. Validate contiguous sequence/prefix watermarks and identical values.
Additionally check 100 evenly spaced starts/capture with direct, independent SQL count/distinct
queries over prior watermark bins and initialization sequence. Full replay shares the window
routine; only the SQL sample and independent fixtures are algorithmically separate checks.

Persist a bounded numeric context sidecar, remove only its own temporary endpoint target DB,
and publish aggregate checks, warm-up/late/cache statistics and feature-pattern conflicts.
No raw rows/endpoints or models enter artifacts. The registered resource screen measures the
offline capture path; sampling does not establish a hard OS bound or live loss/throughput.
No fit, label promotion, independent-source access or production acceptance occurs here.

Primary context: [NFPlugin/NFPacket API](https://www.nfstream.org/docs/api) and pinned
[flow callback implementation](https://github.com/nfstream/nfstream/blob/v6.6.0/nfstream/flow.py).
The clock/event policy above is this project's explicit contract, not a claim that NFStream
already implements those nine features or resolves source capture-clock truth.
