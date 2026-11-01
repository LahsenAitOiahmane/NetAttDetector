# Context models: measured tradeoffs, no qualifying detector

Measured 1 November 2026. **None of twelve candidates passes the registered binary and
attack-family screens.** Added causal context reduces false positives in some configurations,
but does not produce a transferable decision rule on this development cohort. The selected
diagnostic is the statistics-only natural logistic control: precision98.89%, recall99.44%,
benign FPR4.04%, DDoS75/114. It remains rejected for deployment.

## Frozen experiment and complete results

[Protocol](../docs/research/CONTEXT_STUDY.md) registered at `ac54ffc` before any context fits.
[Run 41325831336](https://github.com/LahsenAitOiahmane/NetAttDetector/actions/runs/41325831336)
at `0631310` completed twelve candidates, dummy and two selected-model repeats. Core,
statistical and context logical digests match registration after complete read-only audits.
All original labels, warm-up rows and exclusions remain; only accepted rows fit models.

Thresholds use calibration UID-max scores only. Validation ranks diagnostics, preferring
candidates passing the binary precision98%/recall80%/FPR.5% screen **and** at least80% recall
for each required family with25+ accepted validation UIDs. No candidate meets even the FPR
screen: at most9/1,904 benign flags are permitted. No validation threshold adjustment or
external test occurred. Natural validation prevalence remains unchanged.

| Model / input / training weights | FP / 1,904 | Attack recall | DDoS / 114 | Scan / 103 |
|---|---:|---:|---:|---:|
| Logistic / statistics27 control / natural | 77 | 99.44% | 75 | 103 |
| Logistic / statistics27 control / balanced | 55 | 94.28% | 43 | 103 |
| Logistic / context36 / natural | 58 | 98.30% | 1 | 99 |
| Logistic / context36 / balanced | 26 | 83.53% | 1 | 101 |
| Logistic / context34 no ports / natural | 32 | 75.26% | 1 | 2 |
| Logistic / context34 no ports / balanced | 23 | 19.02% | 1 | 0 |
| Histogram / statistics27 control / natural | 53 | 73.13% | 0 | 0 |
| Histogram / statistics27 control / balanced | 56 | 72.34% | 0 | 0 |
| Histogram / context36 / natural | 56 | 79.84% | 0 | 0 |
| Histogram / context36 / balanced | 74 | 81.42% | 57 | 52 |
| Histogram / context34 no ports / natural | 49 | 63.76% | 0 | 0 |
| Histogram / context34 no ports / balanced | 56 | 79.84% | 0 | 0 |

Balanced means inverse accepted-UID segment and class weights computed on training only.
The dummy flags every validation UID:100% recall and100% benign FPR, illustrating why recall
alone is insufficient. Full calibration/validation metrics, per-family support and conditional
Wilson intervals, confusion matrices, pattern diagnostics and errors are in
[study.json](evidence/context-study/study.json). Conditional intervals do not account for
dependent UIDs, one capture per partition or repeated model selection.

For matched natural logistic comparisons, context changes FP77→58 but DDoS detection75→1.
Balanced context36 lowers FP55→26 relative to its statistics control, while attack recall
drops94.28%→83.53% and DDoS43→1. Reducing conflicting input vectors establishes additional
information, not successful learning or robustness. The prior no-port information bound applied
to statistics25, not the new context34 representation; it is not a ceiling on this new model.

## Coverage and repeatability

The training partition has **zero explicitly DDoS-labeled UIDs**, only8 C&C UIDs,5,742 generic
Attack UIDs and145,596 scan UIDs. Calibration contains8,222 C&C attack UIDs and no other attack
family. Validation is dominated by6,706 C&C UIDs with114 DDoS and103 scan UIDs. Binary class
support therefore hides severe family/source shift. The family guard exposes it; it cannot
repair missing development coverage. The training generic Attack label is not silently
reclassified as DDoS.

Only114 of14,394 raw DDoS labels link as accepted candidate flows; missing-counter and other
exclusions remain unresolved, with no repair or rebalancing. Device independence is unverified.
These results are conditional capture/time diagnostics, not a complete real-attack benchmark.
The next data work must improve independent benign/family coverage and reserve fresh tests;
another unrestricted parameter search is not justified by this result.

All thirteen serialized candidates reload with exact probe scores. Both selected one-thread
refits reproduce validation scores, decisions and calibration thresholds exactly. The selected
threshold is `.9999190519374116`; its tiny difference from the earlier separately fitted
statistics study does not alter the reported confusion counts. Do not substitute that older
model's bytes/hash for this study's separately frozen artifact.

Model work took29.93seconds /728.0MiB sampled process-tree RSS, excluding acquisition,
preparation, verification, imports and encryption. The selected bundle is2,471bytes, fit.64s;
one warmed5,000-row probe took2.13ms including feature validation/transforms. This excludes
I/O, packet metering and sensing and is not a live throughput claim. Linux focused checks
passed64 tests; Python3.11/3.13 CI each passed213 with7 platform/optional skips.

## Private retention

The selected research bundle is encrypted for the owner's local recipient. Only ciphertext,
public freeze/transfer metadata and synthetic portability probes leave private runner storage.
No plaintext model or real feature/UID/endpoint rows are publicly released. The private32-byte
key stays under ignored `.netatt-keys/` with a verified owner+SYSTEM Windows ACL.

The [trusted freeze](evidence/context-study/freeze.json) records model SHA256
`a504167cf9dbd384c17b192d2f1209714c8f4ef5c27e6b892b64fcbe04e19cc7`.
Recovery must check this independently trusted digest before joblib loading; anyone holding
the public key can encrypt arbitrary content. Follow the
[private-transfer protocol](../docs/research/PRIVATE_ARTIFACTS.md) and
[reproduction runbook](../docs/guides/CONTEXT_STUDY.md). Private retention does not resolve
dataset permissions for public weights or authorize a live pilot. The pilot remains deferred.
