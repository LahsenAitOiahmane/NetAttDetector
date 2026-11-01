# Registered statistical flow measurement lab

Registered 1 November 2026 before statistical export or fitting. NAD-048 found HTTP-port,
single-packet/zero-duration errors and a stable fixed-one-thread logistic setting. NAD-049 now
tests **additional measured information**, not a larger model or a relaxed quality requirement.

Keep NFStream 6.6.0, all core settings, byte/flow bounds, eleven raw features and six metadata
fields. Change only `statistical_analysis` to true and add sixteen bidirectional fields:
minimum/mean/sample-standard-deviation/maximum IP packet size; the corresponding interarrival
statistics in milliseconds; and SYN/CWR/ECE/URG/ACK/PSH/RST/FIN packet counts. The separate
contract is `nfstream-statistical-v1` (27 features). No TTL, endpoint identities, payload/DPI,
labels or capture metadata become model features. The old `nfstream-core-v1` contract stays
unchanged and neither contract becomes equivalent to UQ.

## Fixture and preservation gates

Use the existing twelve known generated packets, grouped into five manually specified flows.
Compute expected size and time distributions from explicit packet-size/time lists, independently
of the exporter. Examples: the six TCP IP sizes are 40/40/40/45/47/40 bytes (mean 42, sample
standard deviation sqrt(10)); intervals are 100/100/120/130/50 ms (mean 100, sample standard
deviation sqrt(950)). Its flags give SYN=2, ACK=5, PSH=2, RST=1 and zero other counts. Single-
packet interval statistics and non-TCP flags are zero under this declared contract. Integer
fields compare exactly; floats use 1e-9 absolute/relative tolerance. Fail discrepancies and
investigate semantics rather than replacing independent expected values with exporter output.

On the three pinned development captures, prepare/reverify the exact core cohort and create
a private sidecar keyed by the same flow identity digest. Require every eleven-feature value,
event/counter metadata value, capture membership and row count to match the core database;
reject duplicate/missing/new identities. The unchanged identity/core join carries the verified
candidate label and group, so no new statistical field participates in label matching.

Store only the sixteen extra statistics, ordered row identity, full/no-port transformed-pattern
digests and existing labels/groups needed for aggregate diagnostics. Keep all excluded flows.
Publish finite/range checks, exact feature-pattern repetitions/conflicts, per-detail TCP-flag
presence and first/last size/interval summaries as aggregate distributions. No raw sidecar,
endpoint/UID/feature rows or models enter artifacts. Resource bounds match the registered
configuration; no model is fitted or label policy promoted.

More fields may still leave single packets indistinguishable or lack training support for DDoS.
Use the result to choose the next registered feature/data/context study. Preserve all previous
failed metrics and the reserved external source; the owner continues offline research.

Primary context: [NFStream API](https://www.nfstream.org/docs/api), pinned implementation
[flow statistics](https://github.com/nfstream/nfstream/blob/v6.6.0/nfstream/flow.py) and
[native accumulator](https://github.com/nfstream/nfstream/blob/v6.6.0/nfstream/engine/lib_engine.c).
Source inspection confirms sample denominators; actual fixtures, not current-version API prose,
determine acceptance for the pinned runtime.
