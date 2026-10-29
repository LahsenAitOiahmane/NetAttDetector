# Registered multi-source experiment protocol

Registered on 29 October 2026 before inspecting the new CSE-CIC holdout. Version 2 amends
cleaning/grouping after the development-only audit and before candidate fitting. The initial UNSW
and ToN transfer results are already known; both sources now become development data.
The exact candidate grid and budgets are committed in `configs/multisource.json`.

1. Scan all valid rows of UNSW and ToN. Audit numeric validity, label/category consistency,
   exact model-representation duplicates and conflicting binary labels globally using disk
   hash partitions, rather than an unbounded in-memory table. Report 64-bit fingerprint risk.
2. Retain a seeded uniform 100,000-row sample per source. Cache the numeric sample and
   profiles with input hashes to avoid rescanning during each model fit. Source and attack
   names are metadata, never features. No timestamp or capture metadata is fabricated.
3. Compare source/class-conditional feature distributions and feature-label correlations.
   Compare the embedded feature dictionaries; matching dictionaries establish intended
   definitions, not independently verified exporter parity or label ground truth.
4. Compare the 40-feature contract with a preregistered 34-feature alternative excluding
   source/destination ports, TTL extrema, IP protocol and application-protocol identifiers.
   Use this smallest feature set for grouping across development partitions, so removing
   features cannot create identical model inputs across training/calibration/validation.
5. Exclude globally conflicting **full 40-feature** fingerprints and repeated full-feature
   rows in the sample. Retain ambiguous reduced-feature groups: they measure information
   loss, not automatically incorrect labels. Use 20 stratified group folds over source/label
   strata: folds 0–11 train, 12–14 calibrate, 15–19 validate. This approximates 60/15/25;
   large groups can change row/class proportions. These are not host/session/time holdouts.
6. Train the nine declared candidates sequentially with two threads. Compare equal-source
   weights with equal-source/class weights. Fit preprocessing only on training. Fit sigmoid
   calibration only on the separate calibration set. Compare calibrated and uncalibrated
   variants on validation, with no additional estimator refits. Cap new fits at one hour;
   finish a fit already started, record overrun, and do not start another after the deadline.
7. Choose one global threshold using validation only. Require precision >=0.98, Wilson 95%
   precision lower bound >=0.95, Wilson FPR upper bound <=0.005, and >=25 predicted attacks
   in **each** source. Among feasible candidates maximize worst-source recall, then mean
   average precision, then smaller model size. If none is feasible, explicitly use best
   worst-source F1 as a research fallback. Never present a fallback as meeting the targets.
8. Freeze and hash the selected model, config, source hashes, partition fingerprints,
   validation leaderboard and threshold **before** reading new holdout flow records.
   Download/checksum/CRC verification before that freeze may inspect archive metadata only.
9. Evaluate NF-CSE-CIC-IDS2018-v2 once, with a uniform cap of 200,000 and seed 4242.
   Exclude full-feature fingerprints seen anywhere in the full development inputs,
   not merely their samples. Remove within-sample full-feature conflicts/duplicates.
   Report reduced-representation overlaps/ambiguity without deleting them: distinct full
   observations can legitimately share reduced measurements. Report exclusions/prevalence.
   Compare the frozen new model
   and the original saved UNSW champion on these identical holdout rows in one evaluation.
10. Report precision, recall, FPR, average precision, minority-family supports/recall,
    conditional Wilson 95% intervals, wall time, sampled process RSS, model size and warm
    inference timing. Intervals assume independent labeled rows; they do not quantify
    capture dependence, label errors, model-selection uncertainty or deployment drift.

No hyperparameter, feature, calibration or threshold change is permitted after the new holdout
result is observed. A failure remains a result. Improvements on ToN after including it in training
are development fit results, not independent transfer evidence. Final source generalization
is assessed on the new source; real exporter parity and local shadow testing remain future work.

Calibration follows the primary [scikit-learn guidance](https://scikit-learn.org/stable/modules/calibration.html)
on keeping classifier-fitting and calibration observations disjoint.

## Development-only amendment rationale

The first full audit found 7,429 rows in conflicting full-feature groups, but 10,090,679 rows
in conflicting reduced-feature groups. Dropping ports/protocol/TTL collapses many distinct
observations into the same feature state. Deleting those ambiguous states would remove hard
cases and change the task substantially. Version 2 retains them, isolates those groups during
development splitting, and restricts label-conflict exclusion to the full recorded representation.
The first audit/cache is retained as diagnostic evidence; it is not the training cohort. No model
candidate or new holdout result had been inspected when this amendment was registered.
