# Causal context: preserved measurements and substantially fewer input conflicts

Measured 1 November 2026. **All 250,819 flows retain their core/statistical values and verified
label joins while receiving causal flow-start context.** Full callback replay and 300
independent SQL snapshots agree. Removing ports now leaves **190 rows in conflicting patterns**,
down from 206,075 with statistical fields alone. This is an information/measurement result,
not detector accuracy; no model was fitted in this phase.

## Registered clock and experiment

Registered [the protocol](../docs/research/CAUSAL_CONTEXT.md) at `3b51c79` before extraction;
the pre-run implementation/documentation states the existing statistical float tolerance.
[Run 41321748817](https://github.com/LahsenAitOiahmane/NetAttDetector/actions/runs/41321748817)
at `039a361` passed fixture, preservation, complete replay/sample, joined audit and resources.

The separate `nfstream-context-v1` contract has 36 fields: 27 existing statistical features
plus source/destination prior flow-start counts over 1/10/60 seconds and three 60-second
distinct peer/port counts. All exported flows contribute history, including excluded labels.
The callback reads no label, UID, finished-flow bytes/duration or future event to form context.

Snapshots occur in `NFPlugin.on_init`. The context clock is a prefix maximum of observed
flow-start timestamps in actual initialization order; raw timestamps stay unchanged. Windows
include the lower boundary and exclude every same-watermark peer. Late starts retain arrival
order at the current watermark. This is an explicitly observed activity clock, not guaranteed
physical time during clock jumps. One meter/capture owns state; it resets between captures.

Metadata records initialization order, watermark, late/warm-up flags and peak-cache size, but
none of those fields enters models. Endpoint addresses occur only in private state/temporary
replay targets; the retained context sidecar contains pseudonyms and numeric values. Cache
overflow is terminal rather than silently resuming with incomplete history.

## Preservation and independent checks

The known five-flow/twelve-packet fixture passes **235 values**: 165 existing measurements and
70 context/metadata checks. Only the final idle-reused UDP pair has prior same-endpoint history:
one start in 10/60 seconds, none in 1 second, and one distinct peer/port. Separate event tests
cover ties, expiry boundaries, late clocks, distinct cleanup, future invariance and state refusal.

| Capture | Rows preserved/replayed | Independent SQL snapshots | Late starts | Warm-up rows | Peak active + pending events |
|---|---:|---:|---:|---:|---:|
| 3-1 | 224,388 | 100 | 0 | 18 | 306 |
| 8-1 | 12,459 | 100 | 0 | 4 | 14 |
| 34-1 | 13,972 | 100 | 0 | 19 | 135 |

All core/counter/time values match exactly. Statistical means/deviations also compare exactly
in this measured run (zero nonexact fields, maximum difference zero), despite allowing the
previously declared 1e-9 measurement tolerance. Retained joins keep the original verified
statistical columns. No label, exclusion or threshold rule changes.

Full arrival replay shares the tested window routine; it checks every sequence, prefix clock,
metadata and snapshot. The 100 evenly spaced SQL checks per capture use independently written
count/distinct range queries over prior bins/initializations. Temporary endpoint target DBs are
closed/removed after verification. Joined numeric audits remain possible without those targets.

Warm-up rows remain present and separately counted; warm-up means watermark span, not physical
sensor uptime. Some UIDs have segments on both sides of warm-up, so counts across those slices
must not be summed as if the UID sets were disjoint.

## Additional observed information

Exact accepted-row patterns use the same 248,997 candidate-labeled rows:

| Inputs | Unique vectors | Conflicting-label vectors | Rows in conflicting vectors |
|---|---:|---:|---:|
| Statistics27 | 128,801 | 16 | 50 |
| Statistics25 without ports | 12,416 | 63 | 206,075 |
| Context36 | 237,427 | 0 | 0 |
| Context34 without ports | 102,419 | 39 | 190 |

History adds observed distinctions that deterministic ratios alone could not create. These
finite-pattern counts do not prove trainability, transfer, independent devices or correct
publisher labels. The previous no-port ceiling applies to its old representation, not to
these new context fields. The DDoS join still accepts only 114/14,394 publisher UIDs.

## Resources and next study

The numeric context database is **118,947,840 bytes (113.4 MiB)**. Ordered row digest:
`7a5960b15ce59d56955c3975289bbef6e9cacbbad16c78c8336109589092c897`.
Its byte digest is `e437358785003dcee07f7911f81a56bd7bd6bdbebda28e84d1ddc16174268e7f`.

| Capture | Callback export + replay/SQL seconds | Sampled tree RSS sum MiB |
|---|---:|---:|
| 3-1 | 81.31 | 257.2 |
| 8-1 | 4.85 | 224.2 |
| 34-1 | 7.25 | 220.5 |

The final joined audit took 19.79 seconds / 121.8 MiB. Per-capture samples include context
inserts, target indexes and replay/SQL checks, excluding downloads, prior base preparation/
verification and final sidecar index/audit. Scope differs from the previous statistics-only
timing, so no speed ratio is claimed. RSS is sampled every 50 ms and is not an OS-enforced
bound. The 100k-event cap has headroom on these captures, not a general live-load guarantee.

Only [aggregate manifest](evidence/causal-context/manifest.json) and
[fixture verification](evidence/causal-context/fixture-verification.json) are retained. No real
rows/endpoints or databases are uploaded. [Operations](../docs/guides/CAUSAL_CONTEXT.md)
document exact clock/state semantics and reproduction.

Next register fixed-thread statistics/context/no-port model comparisons on these exact rows,
including supported minority-family recall and repeated prediction checks. Live capture/loss,
external-source testing, device independence, public model terms and the owner-deferred pilot
remain separate. No diagnostic is promoted by passing this measurement lab.
