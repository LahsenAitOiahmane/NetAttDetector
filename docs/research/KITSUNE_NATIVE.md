# Registered Kitsune native-flow phase-label lab

NAD-057, registered 23 November 2026 after complete packet admission and **before native
extraction, native labeling or fitting**. This is a candidate development-label experiment,
not independent verification of publisher attack truth. The 764,137 packet annotations have
one benign prefix followed by an attack suffix; complete per-label timestamp ranges do not
overlap even after millisecond quantization. The capture is not sorted to create that property.

## Label proof and callbacks

Pin both original file hashes, packet/class counts, and complete per-label time ranges in
`configs/data/kitsune-native.json`. Before export, repeat the full independent packet-label
audit against those values. Require the single phase transition and strictly disjoint rounded
millisecond ranges. If that proof fails, do not label a native cohort.

A new NFPlugin subclass computes the existing nine causal context fields at flow initialization
exactly as before. A separate private label accumulator reads every initialization/update packet's
event time. A timestamp within the proven benign range contributes benign; within the proven
attack range contributes attack; outside both contributes unknown. No phase, label, event time,
ordinal, endpoint or capture identity enters model features or changes context history. The
policy depends on the observed capture's publisher annotations, not a general live labeling rule.

Retain packet counts and a bit mask per finished segment. Require callback packet counts to
equal NFStream's complete bidirectional packet count for every row. Uniform-known segments
receive a candidate binary label; segments with both labels stay mixed/excluded, and any
unknown callback packet prevents a binary label. Preserve all rows, exclusions, protocol and
warm-up/late coverage. Record whether native first/last times equal callback min/max under
the observed timestamp regressions; do not assume monotonic packets within a flow.

## Unchanged measurements and independent checks

The feature contract stays `nfstream-context-v1`: 36 existing measurements. Use locked
NFStream 6.6.0, one meter, original packet order, idle 2 s / active 1800 s, IP-byte accounting,
and the same statistical/context settings. Refuse 250k-flow, 128 MiB-input, 100k-context-event
or 512 MiB-database overflow. Retain a 2 GiB free-disk reserve. No model or threshold is fitted.

Before real extraction, verify the known twelve-packet fixture: first six packets benign,
last six attack, yielding five segments with uniform benign, uniform attack and one mixed
TCP flow. Existing core/statistical/context values and event metadata must match the previous
known fixture, and callback label counts/masks/min/max must match independent hand counts.
Test unknown clocks, quantized overlap refusal and phase state independently of NFStream.

After real extraction, repeat the complete capture with the old context-only plugin and
compare every original measurement, event identity and context metadata in export order.
Require exact equality; publish only comparison counts/digests. Reopen SQLite read-only,
check integrity/schema, row identities, logical digest, counts and label/mask invariants.
Report per-class packet/segment counts and conflicting feature patterns without deleting them.

Private grouping hashes identify capture-scoped bidirectional native endpoint/port/protocol
tuples, **not Zeek UIDs, sessions, physical devices or independent attacks**. Reused tuples may
span both label phases; report those groups. Any future split must keep appropriate capture/
conversation groups together and cannot claim a pure chronological both-class holdout from
this single phased capture. All raw endpoints remain private and absent from retained rows.

## Coverage and reporting

Compare callback/export packet totals with the full publisher-label census, retaining the
non-IP/truncation/other-transport gap. Most source attack labels are non-IP, so a native subset
must never be presented as detection coverage of all 642,516 malicious packets. TCP/UDP-only
structural decoding does not define NFStream's protocol support. Retain unknown/mixed rows
and complete protocol counts. The labels' time-phase nature remains an unresolved target-
validity limitation even if mechanical callback parity succeeds.

Report acquisition/extraction/replay/audit resource intervals separately, with sampled RSS
scope and no live-throughput claim. Private numeric SQLite rows stay on the ephemeral Linux
runner; retain only aggregate manifests, fixture/control checks and source pins. Reproduction
recreates the cohort from pinned originals. A later fitting protocol and compatible benign/
missing-family coverage/fresh evaluation plan are still required. The pilot remains deferred.
