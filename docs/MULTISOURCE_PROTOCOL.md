# Registered multi-source experiment protocol

Registered on 29 October 2026 before inspecting the new CSE-CIC holdout. The initial UNSW
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
   Use this smallest feature set for duplicate/conflict grouping across every candidate and
   split, so removing features cannot create identical model inputs across partitions.
5. Exclude globally conflicting development fingerprints. Remove conflicting/duplicate
   smallest-representation rows in the sample, retaining source counts before/after. Split
   source/label strata 60% training, 15% calibration, 25% validation, with disjoint feature
   groups. These are not host-, session- or time-held-out splits; explicitly report that limit.
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
   Exclude smallest-feature fingerprints seen anywhere in the full development inputs,
   not merely their samples. Remove within-sample conflicts/duplicates using the same
   grouping. Report every exclusion and prevalence change. Compare the frozen new model
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
