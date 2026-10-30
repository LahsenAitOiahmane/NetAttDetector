# Evaluation protocol and improvement strategy

## Task and acceptable evidence

Phase 1 predicts benign versus attack for completed exported flows. Attack-family names support
error analysis; this is not yet a multiclass attack attribution system. Compare the new models
against each other under one frozen protocol, not against the old attack-only 100% reports.

Use a hierarchy of evidence:

1. Deterministic synthetic fixtures establish software behavior, not detection performance.
2. A deduplicated in-source random holdout establishes an exploratory benchmark.
3. Capture/device/day-held-out tests measure harder forms of generalization.
4. A separately sourced dataset tests distribution shift with a frozen model and threshold.
5. A time-bounded, authorized local shadow pilot establishes exporter parity and operational value.

Report each level separately. A model must not advance to production based on level 2 alone.

## Current sampling and split

The reader scans the complete input CSV, counts all valid binary labels and attack categories,
and retains up to `max_rows` observations by uniform random priorities. This avoids a first-N
slice which might contain only one time period or attack. No class resampling changes test data.
Rows with invalid binary labels are counted and excluded. Numeric invalid/nonfinite values are
treated as missing; individual missing values are imputed inside the training pipeline.

Within the retained sample, feature hashes exclude targets, attack names, addresses, and split
metadata. Exact repeated model inputs keep one row; feature-equivalent rows with conflicting
binary labels are all excluded and counted. These are **sample-level** checks. They do not
measure global duplicates, near-duplicates, conflicting multiclass names, or session similarity.
Sampling then deduplicating changes the empirical prevalence; reports retain both full-stream
counts and final cohort counts so the effect is visible. Train-only resampling is a future experiment.

The default random split is stratified 60% training, 20% validation, 20% test, seed 42. Disjoint
feature hashes are asserted. Models fit only training data. Median imputation, constant-feature
removal, and SGD scaling are pipeline steps, so validation/test statistics never fit them.
Histogram boosting may internally reserve part of its training partition for early stopping.

For grouping, copy a config and set `"split": "group"` with
`"group_column": "IPV4_SRC_ADDR"`, or another defensible capture/device identifier. The two
GroupShuffleSplit stages isolate that identifier; proportions may be uneven and are reported.
One source-IP split does not guarantee session or destination isolation. Use capture/session
identifiers when available. The command fails when a resulting partition lacks class support.

For chronological data, use `"split": "temporal"` with a parseable timestamp `group_column`.
Distinct timestamps are sorted; approximately 60/20/20 of unique time values define boundaries.
Equal timestamps stay together, so row proportions can differ substantially. The UQ v2 CSVs do
not inherently provide capture timestamps; do not invent dates from row order. Purged time gaps
and bidirectional-session isolation are still backlog items.

## Candidate selection and frozen threshold

Every run trains a prior-probability dummy, then configured candidates sequentially. The initial
models deliberately avoid an exhaustive search. Select hyperparameters and thresholds using
validation or future nested development folds only; never test scores.

On validation, choose the highest recall operating point satisfying attack precision ≥0.98 and
FPR ≤0.005. If no point meets those constraints, explicitly flag failure and use best validation
F1 as an exploratory fallback. Rank candidates first by feasibility, then recall (or fallback F1),
then average precision, then lower fit time as a tie breaker. The dummy is reported but cannot
be promoted. Evaluate the selected champion on the final test once per fixed experiment.
Changing a seed/hyperparameter after seeing test results creates a new development exercise;
it does not preserve that test set as an untouched publication holdout.

The external source uses the same stored pipeline and threshold. Remove feature-hash overlap
with the complete in-domain sampled cohort before scoring. Do not fit an external scaler,
choose a new threshold, or select a model on that external result. If used to improve training,
that source becomes development data and a new independent test source is necessary.

## Required report interpretation

| Metric | Interpretation |
|---|---|
| Attack precision | Fraction of attack alerts that are labeled attacks in this evaluated population |
| Attack recall | Fraction of labeled attacks that trigger an alert |
| Macro F1 / balanced accuracy | Give benign and attack behavior more equal influence |
| Average precision | Ranking quality across recall levels; baseline depends on attack prevalence |
| ROC AUC | Supplementary ranking measure; can look excellent despite an impractical alert rate |
| FPR | Benign flows mistakenly flagged divided by benign flows evaluated |
| False alarms / 10k benign flows | Analyst-facing scale; not automatically alerts/day |
| Attack-family support and recall | Identify failures hidden by common-class averages |
| Brier score | Score error; recorded scores are not yet externally calibrated |
| Inference timing / memory / size | Optimization evidence within the stated benchmark scope |

Prevalence matters: with 0.1% attacks, 90% recall and 0.5% FPR imply only about 15.3% precision
by Bayes' rule. The initial validation constraints are research targets, not a sufficient production
SLO. A site-specific daily alert budget must account for flow volume, event aggregation, and the
analyst's capacity. Accuracy alone must not decide promotion.

Warm batched timings exclude parsing, export, queueing, network I/O and model load. Seven
repetitions give a small timing sample, not a production latency guarantee. RSS is the Python
process sampled every 50 ms; short peaks and other OS processes are not measured. Measure
end-to-end throughput and dropped-flow rates in the actual sensor before deployment.

## Ordered accuracy and efficiency experiments

1. Audit labels, conflicting rows, benign diversity and source/capture overlap globally.
2. Establish capture/time/source-held-out baselines and inspect the worst attack slices.
3. Ablate ports, TTL, application IDs, and export-derived shortcuts on development partitions.
4. Add explicitly defined flow rates/ratios and log transforms; assess division-by-zero behavior.
5. Run a bounded randomized/Optuna search with a wall-time budget on development folds only.
6. Compare class weighting and carefully controlled train-only sampling; retain natural test prevalence.
7. Calibrate scores on a separate development partition; tune thresholds under confidence bounds
   and an operational alert budget, not a single desired accuracy number.
8. Try a two-stage binary detector then attack-family classifier. Use abstention for unknowns.
9. Compare benign-only Isolation Forest / incremental SGD as separate novelty/drift baselines.
10. Add local labeled benign and attack replay data with the selected exporter; retrain the contract.
11. Evaluate feature reduction, fewer trees, quantization/conversion, and batch sizes on a
    precision/recall/latency/memory Pareto frontier. Keep changes only when they have measured value.
12. Try deeper temporal/graph/neural models only if data and validated benefit justify their cost.

Future publication evidence must include confidence intervals, multiple preregistered development
seeds, a frozen independent holdout, threshold provenance, per-attack support, external failures,
resource measurements, data terms, and a model card. Do not claim zero-day protection from a
random split of known attack families.

Pipeline and threshold rationale follows the primary scikit-learn guidance on
[preprocessing and leakage](https://scikit-learn.org/stable/common_pitfalls.html) and
[decision thresholds](https://scikit-learn.org/stable/modules/classification_threshold.html).

The initial external failure also motivates reading Layeghy and Portmann's primary research,
[On Generalisability of Machine Learning-based Network Intrusion Detection Systems](https://arxiv.org/abs/2205.04112).
Their investigation links some transfer failures to feature-label relationships that change across
sources. That is a hypothesis for our next audit, not proof of the cause of this particular result.
