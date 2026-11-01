# Native error slices and fixed-thread repeat evidence

Measured 1 November 2026. **Two fixed-one-thread logistic fits reproduce scores, coefficients
and calibration thresholds exactly.** This identifies a stable setting for this diagnostic on
one runner; it does not fix model quality or establish cross-thread/platform reproducibility.

Registered [the three-fit protocol](../docs/research/NATIVE_ERRORS.md) at `48029e9` before
these refits. [Run 41311537200](https://github.com/LahsenAitOiahmane/NetAttDetector/actions/runs/41311537200)
at `0d5f854` passed its fixed-thread/resource screens. It uses the same pinned accepted subset,
selected logistic/native11/connection-class-balanced specification and **old frozen threshold
0.9921028048109991** for validation error analysis. No search or new validation threshold occurs.

## What repeats, and what does not

| Comparison | Maximum score difference | Frozen validation decision differences | Result |
|---|---:|---:|---|
| Two one-thread fits | 0 | 0 | Pass; coefficients and selected calibration thresholds also identical |
| Two-thread versus one-thread fit | 1.41358e-6 | 0 | Cross-thread variation persists above the original 1e-7 tolerance |

The two single-thread fits choose calibration threshold **0.9921027806705783**. That is a
diagnostic observation, not a replacement for the original freeze. Two-versus-one coefficient
maximum difference is 7.68312e-6. All three runs have the same validation confusion matrix at
the original threshold: TN1,849 / FP55 / FN113 / TP6,810.

The new two-thread replay's full metrics are not bit-identical to the earlier runner: flow and
connection Brier scores differ by approximately 6.1e-13 and 1.4e-12 respectively. All other
stored validation metrics, family results and confusion counts agree. The evidence retains
`two_thread_validation_metrics_replay_identical: false`; no exact cross-run claim is made.

## Where the errors are

Context uses each UID's highest-scoring accepted segment, taking the first stored segment on
ties. Ports/directions are observed flow fields, not independently verified application roles.

| Observed destination bucket | Benign connections | False positives | Attack connections | Missed attacks |
|---|---:|---:|---:|---:|
| DNS / 53 | 223 | 0 | 0 | 0 |
| HTTP / 80 | 56 | 53 | 113 | 113 |
| NTP / 123 | 1,618 | 0 | 0 | 0 |
| Other | 7 | 2 | 6,810 | 0 |

All 55 false positives are TCP. Independently, 53 of their highest-scoring segments contain
one packet and zero duration; the remaining two have 21+ packets and positive duration. These
are marginal slices, not a claim that separate 53-count bins necessarily contain identical UIDs.
All 113 false negatives are publisher DDoS UIDs with a one-packet, zero-duration, observed HTTP
top segment. C&C/scanning accepted-family results remain 6,706/6,706 and 103/103 detected.

False-positive scores range **.99697–.99999**, above the old .99210 threshold. False-negative
scores range **.00565–.92022**. They are not simply rounding errors at the decision boundary.
Do not add a broad HTTP allowlist or adjust the validation threshold to conceal these failures.
These aggregates do not establish why a label is benign/malicious or whether publisher labels
are correct. TCP-state, timing and contextual traffic measurements require a separate audit.

## Resources and next step

The three fits plus frame loading/error aggregation took **5.18 seconds**, with **380.8 MiB**
sampled process-tree RSS. This excludes imports, cohort construction, input downloads and
initial verification. RSS is sampled every 50 ms and is not an OS-enforced limit. The fitted
fixed-thread artifact stays private on the runner. See the
[full aggregate evidence](evidence/native-errors/diagnostics.json) and
[configuration](evidence/native-errors/diagnostics-config.json).

The real Linux job passed 49 focused tests; Python 3.11/3.13 quality checks passed separately.
No raw rows, individual scores, endpoint/UID values or model coefficients are published.
Only scalar coefficient differences and aggregate error tables leave the runner.

Next validate a richer NFStream contract on independent synthetic packet fixtures and check
that statistical export preserves core counters, flow identities and label assignments. Measure
actual packet-length/interarrival/TCP-flag information before another fit; richer fields may
still be insufficient without causal context, more representative captures or label review.
The original baseline remains rejected, DDoS label coverage stays incomplete, reserved external
data remains untouched, and the owner-deferred live pilot remains outside current offline work.
