# Registered native error and fixed-thread diagnostic

Registered 1 November 2026 after the failed native baseline, before fitting these controls.
This advances NAD-048 with three fits of the already selected logistic/native11/connection-
class-balanced specification. It does not rerun candidate selection or change the failed
baseline's score tolerance, operating point or acceptance result.

Recreate the exact pinned cohort. Refit at two training threads once, then at one thread twice,
always using the existing one-thread prediction path. Compare score differences, decisions at
the **original frozen calibration threshold**, recalculated calibration thresholds, validation
metrics and coefficients. Require the two single-thread fits to agree within the original 1e-7
score tolerance and zero decision differences. The two-thread replay is reported separately;
do not imply a stable fixed-thread result proves cross-thread or cross-platform reproducibility.
Keep the one-hour/1.5 GiB sampled diagnostic resource screens from the baseline.

Publish only aggregate error analysis from the two-thread replay at the old frozen threshold.
For each accepted UID, use its highest-scoring matched segment (first stored segment on ties)
as context; the UID is flagged if that score crosses the threshold. Cross-tabulate confusion
counts by transport protocol, predefined observed destination-service buckets, source-port range,
packet-count range and zero/nonzero duration. These describe the segment's **first-observed
direction**, not a verified client/service role. Include score quantiles for TP/FP/TN/FN groups
and family support. No raw endpoints, UIDs, individual scores or row-level features are published.

The old threshold remains frozen for these validation error slices. Calibration-only thresholds
for each repeat are reported as diagnostics and do not replace the old freeze. No validation
threshold search, new features, source acquisition beyond the pinned development bundles, or
external-test access occurs. Improving the model needs a later registered feature/data study;
this experiment identifies causes and a reproducible training setting without claiming that it
fixes FPR, missing attack families, label truth, device isolation or production readiness.
