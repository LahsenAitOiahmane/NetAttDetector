# Registered native capture diagnostic

Registered 1 November 2026, after native-cohort preparation and before inspecting any native
model scores. NAD-046 produced byte-identical repeated cohorts and validated accepted labels,
capture/time partitions, connection groups and exclusions. Their immutable logical row digest
is pinned in [native-study.json](../../configs/native-study.json).

This protocol explicitly authorizes **experimental fitting on that accepted-label subset**.
It does not promote the point-label rule, verify attack truth, authorize deployment/public model
release, or satisfy the broader device-held-out benchmark. Training-disabled preparation/global
plans stay unchanged. We do not repair excluded DDoS labels or drop conflicting feature rows.

## Fixed experiment

One prior dummy plus twelve candidates: logistic regression, histogram boosting and Extra Trees,
each with two feature policies and two training-weight policies. Seed 42, at most two native
threads, sequential fits, one-hour wall screen and 1.5 GiB sampled process-tree RSS screen.
No XGBoost, GPU, broad search or external dataset is needed for this baseline.

- `native11`: the eleven measured fields, with no metadata as inputs.
- `behavioral_derived`: exclude source/destination ports; keep the other nine measured fields
  and add reverse packet/byte fractions, overall and directional mean IP bytes/packet, and
  smoothed packets/bytes per second. The denominator for rates is duration + **1 ms**, a declared
  transform rather than a claim of precise instantaneous rates for single packets.
- `flow_natural`: retain natural training row weights.
- `connection_class_balanced`: each UID has equal total influence within its class, with equal
  total class weights. Compute weights from training only and normalize mean row weight to one.

Logistic regression uses train-fitted standard scaling after `log1p`, C=1, lbfgs and at most
300 iterations. Histogram boosting uses 80 iterations, learning rate .08, 15 leaves, minimum
30 leaf rows, L2=1 and **no internal random validation split/early stopping**. Extra Trees uses
64 trees, maximum depth 14, minimum five leaf rows, all input features, no bootstrap and two
threads. No preprocessing fits on calibration or validation observations.

Prediction uses one thread, including sequential forest-tree accumulation, to avoid introducing
parallel summation order into threshold decisions. Forest fitting can still use two workers.

## Threshold, selection and claims

For each candidate, aggregate segment scores by maximum within each accepted UID. This measures
whether **any matched segment** flags the connection, not a real incident/alert aggregation
policy. Choose a threshold using calibration connections only, maximizing recall subject to
precision >= .98, FPR <= .005 and at least 25 true-positive connections. Ties prefer precision,
then a higher threshold. If infeasible, explicitly use best calibration F1 as a diagnostic
fallback. Calibration and validation both need recall >= .8 for the full research screen.

Validation chooses the diagnostic candidate under the fixed ranking in the configuration;
it is therefore development data, not an untouched final test. If none passes, report failure
and retain the best diagnostic bundle only for offline research. Never call it a deployable
champion or retune thresholds against validation to obtain a passing result.

Report natural flow metrics, connection metrics, per-family connection support/recall,
conditional Wilson intervals, exact transformed-pattern overlap, sampled memory, fit/score time,
artifact size and batched inference timing. Conditional intervals do not cover dependence among
UIDs, single-capture sampling or model selection. There is only one capture in each partition,
so no capture-level uncertainty or cross-device generalization claim is possible.

Refit the selected specification twice as repeat/thread controls (two threads, then one),
compare validation scores with the original under absolute tolerance 1e-7 and require zero
threshold-decision differences to pass the repeatability screen. Compare saved/reloaded
predictions and decisions too. These are fifteen total fits; multiple seeds and independent
capture replication remain future work. Do not claim the old UQ/XGBoost discrepancy is resolved.

The bundle has a separate native format/schema, preprocessing, threshold, exact inputs and
research-only flags. It must not load through the UQ predictor. Keep models and feature/UID rows
private on the runner; retain aggregate reports/manifests only while terms remain unresolved.
CICIDS2017 stays reserved/unacquired and CSE stays observed. A fresh compatible test can be
planned after this diagnostic freeze without describing these three captures as independent
datasets. The small-office 50 reviewed alerts/day limit remains a separate, unmeasured pilot
requirement.
