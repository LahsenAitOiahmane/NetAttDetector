# Initial research baseline and model card

Evaluation date: 29 October 2026. This file is updated with verified experiment evidence below.

## Intended use and limits

The model is a binary research classifier for compatible UQ extended NetFlow v2 exports.
It is not validated for arbitrary packet captures, a different exporter, zero-day attack detection,
automated blocking, or commercial deployment. Do not use legacy 100% scores as a comparison.

The first local result uses a uniform capped sample, sample-level deduplication and a 60/20/20
stratified random split. Model and threshold were chosen only on validation. Random splitting
does not establish generalization across capture sessions, devices, time or deployment sites.
Any external failure is reported alongside the in-domain result. Detailed results and acquisition
records are appended once the corresponding runs finish.

## Local experiment

Command:

```powershell
uv run --no-sync netatt train --data data/raw/nf-unsw-nb15-v2.zip --config configs/benchmark.json --output artifacts/baseline-local
```

The full input scan verified 2,390,275 rows: 2,295,222 benign and 95,053 attacks, with no invalid
binary labels. Uniform sampling retained 200,000; duplicate removal excluded 7,109, leaving
192,891 flows. No conflicting binary feature rows occurred within this sample. Final prevalence
was 3.28% attacks, compared with 3.98% in the full input. That difference must be preserved in
interpretation. Splits were 115,734 training, 38,578 validation, and 38,579 test flows.

| Candidate | Validation precision | Validation recall | Average precision | Constraints met | Fit time (s) |
|---|---:|---:|---:|---|---:|
| Dummy | 3.28% | 100.00% | 0.03282 | No | 0.60 |
| SGD logistic | 76.24% | 97.31% | 0.69278 | No | 2.04 |
| Histogram boosting | 98.05% | 79.62% | 0.98570 | Yes | 16.39 |
| Extra Trees | 98.13% | 28.99% | 0.95737 | Yes | 4.90 |
| XGBoost | 98.04% | 79.07% | 0.98753 | Yes | 4.46 |

Histogram boosting won under the documented recall-with-constraints rule; XGBoost had a
slightly higher ranking score (average precision) but lower recall at its selected operating point.
The stored histogram threshold is `0.9957466410915312`. Scores are not calibrated probabilities.

| Final random holdout | Value |
|---|---:|
| Accuracy | 99.3079% |
| Attack precision | 97.1671% |
| Attack recall | 81.2796% |
| Attack F1 | 0.88516 |
| Macro F1 | 0.94080 |
| Average precision | 0.98012 |
| ROC AUC | 0.99946 |
| Benign FPR | 0.08040% |
| False alarms / 10,000 benign flows | 8.04 |
| True negatives / false positives | 37,283 / 30 |
| True positives / false negatives | 1,029 / 237 |

**The 98% precision target failed on test.** Accuracy above 99% hides 237 missed attacks.
Analysis recall was 50% (10 examples), Fuzzers 63.0% (357), Backdoor 73.7% (19), and DoS
74.7% (75). Worms recall was 100% on just one example, which does not establish reliable worm
detection. Minority support and the random split prevent strong production claims.

The complete local run took 60.13 seconds, with sampled peak process RSS 347.2 MiB.
The selected compressed bundle is 103,065 bytes. Warm 2048-row inference batches had median
11.32 ms and p95 13.87 ms across seven repetitions. These timings exclude CSV parsing, model
load and live export, and do not measure network capture capacity. They demonstrate that the
first benchmark is practical on the audited CPU, not an end-to-end production throughput claim.

Environment: Python 3.13.5, NumPy 2.5.4, pandas 2.3.3, scikit-learn 1.9.1, joblib 1.6.0,
psutil 7.2.2, XGBoost 3.4.1. The lockfile records the full environment. This run occurred during
development and correctly records a dirty worktree based on original revision `f1b2d59`.
The finalized source is reviewed in the modernization branch; the remote run supplies a clean
commit-bound reproduction. Model selection has not been changed in response to the test result.

![Initial source analysis, validation comparison and held-out errors](figures/baseline.png)

Machine-readable evidence: [metrics](evidence/local-metrics.json),
[data profile](evidence/local-data-profile.json),
[download manifest](evidence/nf-unsw-nb15-v2.manifest.json).
