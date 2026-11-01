# Registered statistical model comparison

Registered 1 November 2026 after the statistical measurement lab and before fitting models
on those extra fields. The separate [configuration](../../configs/statistical-study.json)
pins both the original core and statistical logical row digests. This is NAD-050, an offline
accepted-label diagnostic; no label promotion or production/public-model approval occurs.

## Fixed candidates and controls

One prior dummy and twelve candidates: logistic regression or histogram boosting, each using
core11-only control, all 27 fields, or 25 statistical fields without ports; each with natural
flow weights or connection/class-balanced training weights. Use the same train-fitted log/scaler
for logistic and the same bounded estimator parameters as the original native baseline. Do not
add Extra Trees or a search after their earlier failed transfer.

All training and prediction use **one thread**, fixed before these fits, following the measured
repeat evidence. Seed 42. Repeat the selected specification twice at the same setting, giving
fifteen total fits. Require scores within 1e-7, zero frozen-threshold decision differences and
exact saved/reloaded probe predictions. One hour / 1.5 GiB sampled RSS screens remain, without
claiming OS-enforced limits or cross-platform/cross-thread determinism.

The external input contract is always the 27 measured fields; the core control selects its
eleven fields within preprocessing. Metadata, labels, hashes, capture/source/device identifiers
and timestamps never become model inputs. All preprocessing/weights fit training only. Record
both float64 and float32 exact-pattern diagnostics so apparent information gains are not
confused with tiny arithmetic distinctions or estimator quantization.

## Unchanged evaluation

Use the same three chronological captures, source/capture/UID groups, unmodified exclusions,
natural calibration/validation distributions, UID-max connection scoring and calibration-only
threshold selection. Precision >= .98, FPR <= .005, recall >= .8 and at least 25 true-positive
connections remain research screens. Validation chooses the diagnostic under the same ranking
as the original study; it is not a fresh final test. An infeasible calibration frontier explicitly
falls back to best F1 without pretending the operating budget is satisfied.

Report every candidate, per-family connection support/recall, natural flow metrics, conditional
intervals, exact feature overlap, warnings, fit/inference resources, bundle size, serialization
and repeat controls. No cherry-picked replacement of the original 2.89%-FPR/1-of-114-DDoS
failure. Also report existing predefined aggregate error bins for the selected validation
diagnostic at its calibration-only threshold, without publishing UID/feature rows. No
validation threshold search or removal of ambiguous feature rows occurs. If all fail,
prioritize measured causal traffic context and representative development data instead of model
size. Reserved independent data remains untouched; the owner-deferred live pilot stays deferred.

Model bundles use `statistical-diagnostic-v1` and the exact measured schema/preprocessing/
threshold with research-only flags. Keep row databases and models private; retain aggregate
reports/configurations/freezes only. Reproduction rebuilds the same pinned inputs on Linux.
