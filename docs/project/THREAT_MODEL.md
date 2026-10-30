# Small-office/home detector scope

Recorded 30 October 2026 from the user's choices: **small office/home network**, capacity to
review **up to 50 alerts/day**. These are product inputs, not a measured pilot acceptance result.
The machine remains a CPU-only i5/16 GiB development target. No GPU or network blocking is planned.
The machine-readable planning record is [small-office.json](../../configs/deployment/small-office.json).

The first intended use is passive assistance for reviewing suspicious outbound/inbound activity
and internal connection behavior. Candidate evaluation slices include scanning, brute-force
behavior, unusual connections and flood activity where labels support those categories. A
flow-based detector cannot promise to identify encrypted exploit contents or every malware type.
Attack-family support and uncertainty must be reported rather than claiming universal coverage.

Inputs should come from a documented network observation point with visibility into both flow
directions. NAT, encryption, dropped packets, mirrored-port duplication, sampling and offload can
change observations. Local device/capture identity and timestamps remain metadata outside model
features. The PCAP lab uses generated documentation addresses and no captured personal traffic.

## Alert budget and missing measurements

The 50/day ceiling applies to the analyst's reviewed alerts, including real attacks. The event
aggregation/suppression policy and typical benign flow volume are unknown. Consequently we
cannot yet turn this ceiling into an approved per-flow false-positive SLO.

As a conservative arithmetic illustration **only if every flagged benign flow became one alert**,
50 false alerts/day would correspond to these FPR ceilings, before reserving review capacity for
real attacks:

| Benign flows/day (illustrative, not measured) | FPR producing 50 false alerts |
|---:|---:|
| 10,000 | 0.5% |
| 100,000 | 0.05% |
| 1,000,000 | 0.005% |

The old 0.5% research FPR target could therefore consume the budget many times over. Measure
traffic volume and incident aggregation before approving thresholds. Repeated flows to one
destination may become one reviewed event; that behavior must be implemented and evaluated,
not assumed when reporting model precision.

## Exit criteria for a future shadow pilot

Verify observation-point authorization, feature/exporter parity, bounded capture/queue behavior,
retention/access handling, repeatable replay, investigation context and a local benign baseline.
Agree attack support, observation period and detection/false-alert confidence requirements with
the operator. Record who reviews alerts and how false/missed detections are labeled. No blocking,
remote traffic upload, deployment or live capture is enabled by this planning document.

Remaining product inputs are observation point/topology, flow volume, device inventory, retention,
alert aggregation, review ownership and an operator-approved pilot SLO.
