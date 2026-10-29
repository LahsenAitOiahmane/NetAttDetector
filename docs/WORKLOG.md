# Modernization work log

## 29 October 2026 — audit and foundation

Started from a clean checkout of `main` at `f1b2d59`. Created branch
`codex/modernize-nids-foundation`. No subagents were used. No new cloud service or paid compute
was provisioned. Repository operations use the existing authenticated GitHub CLI account.

Reviewed source/configuration files, historical data/model inventory, log reports and Git history.
Measured the i5/16 GB hardware and limited free disk. Found missing original datasets,
unsaved scaler, unused test input, possible label proxy, attack-only evaluation excerpts, fabricated
flow statistics in live prediction, and uncontrolled parallel searches. Recorded evidence and
uncertainties in `AUDIT.md` rather than accepting legacy 100% accuracy.

Researched official UQ, UNSW, UNB/CIC and Stratosphere sources, including newer industrial
collections. Selected UQ v2 sources for initial common-schema experiments, retained license notes,
and documented the much larger collection and adapters still needed. UQ eSpace pages returned
403 in this environment; the publisher-linked public RDM records/API worked. Full-file S3 reads
timed out, while bounded range reads worked; the downloader now validates/retries bounded blocks.
Temporary signed URLs are not retained in project manifests.

Implemented a Python package/CLI, frozen uv environment, resource inventory, bounded download,
ZIP streaming, uniform sampling, sample-level deduplication/conflict removal, feature allowlist,
training-only preprocessing, validation threshold/model selection, test/external reports, model
serialization and strict flow CSV inference. Replaced executable prototype paths while retaining
historical model modules and artifacts as reference. Removed unused TensorFlow from supported
dependencies; XGBoost is optional. Current scikit-learn and XGBoost were resolved from package
metadata; exact versions are in the lockfile and experiment reports.

Added focused tests for sampling determinism, duplicates/conflicting binary labels, invalid
labels, ZIP reading, schema coercion, split isolation, threshold feasibility/fallback, confusion
metrics, train-only preprocessing, artifact compatibility, prediction reload, overwrite refusal,
external overlap rejection, configuration bounds, and download integrity/limits. Initial suite:
23 tests passed. Lint/format checks cover the supported package, tests and scripts; historical
modules are excluded intentionally because they are not the active implementation.

Added GitHub CI and real-data training workflows with read-only permissions, SHA-pinned actions,
thread/time limits and finite artifact retention. Documented operations, evaluation protocol,
architecture, ordered backlog, sprint process, release gates and dataset terms.

The first local benchmark scanned 2,390,275 UNSW flows, selected 200,000, removed 7,109 duplicate
feature rows, and selected histogram boosting on validation. Test accuracy was 99.3079%, precision
97.1671%, recall 81.2796%, FPR 0.0804%. The precision target failed on test. This was retained as
a valid research result, not tuned away. Full acquisition, external evaluation and remote workflow
outcomes are recorded in `reports/BASELINE.md` and appended below when verified.

Completed both real acquisitions: 19,330,771 verified flows across two compressed archives,
201,997,602 bytes in total. Download reuse passed the stored SHA-256 checks. Actual inference
through `app/app.py` successfully classified 1,000 exported real flow rows with the saved local
pipeline. Added an inference resource-budget regression check so a requested lower thread count
also controls saved forest/XGBoost estimators; added a clear optional-XGBoost installation error.
The supported test suite now has 24 passing tests, with lint and format checks passing locally.

The frozen local champion failed independent-source evaluation: ToN attack recall 0.3314%,
precision 29.94%, benign FPR 1.3503%, across 194,291 deduplicated sample rows. It scanned all
16,940,496 source rows, excluded two conflicting feature rows and 5,707 duplicates in the capped
sample, and found no overlap with the original cohort. No model or threshold was retuned.
Recorded exact counts, provenance, runtime/RSS and attack slices. Updated release assessment
to explicitly block deployment and prioritize generalization/feature audits. Root cause remains
unproven. This failed test is part of the deliverable, not discarded evidence.

The initial [GitHub CI run](https://github.com/LahsenAitOiahmane/NetAttDetector/actions/runs/40966748944)
passed on Python 3.11 and 3.13 (23 tests at that source revision, plus synthetic smoke training).
The initial real-data workflow runs against commit `e0a8352`; the temporary branch-push trigger
was removed after launch, leaving the final research workflow manual-only. Figure generation
used standard Matplotlib from JSON evidence; the optional analysis extra documents reproduction
without adding plotting dependencies to the prediction runtime.

The [real-data GitHub run](https://github.com/LahsenAitOiahmane/NetAttDetector/actions/runs/40966749037)
completed successfully. Downloaded its artifacts and compared them with the local experiment:
both data hashes, all split fingerprints, model selection, threshold, test metrics and external
metrics matched exactly. GitHub records a clean worktree at `e0a8352`, Python 3.13.17, the same
ML package versions, 78.44-second combined experiment runtime and 546.4 MiB sampled peak
process RSS. Preserved its metrics JSON in `reports/evidence/github-metrics.json`; large inputs
and model artifacts remain ignored. Initial remote artifact files are also available locally under
`artifacts/github-40966749037/`.

The [updated CI run](https://github.com/LahsenAitOiahmane/NetAttDetector/actions/runs/40967625888)
passed with 24 tests on Python 3.11 and 3.13 plus synthetic smoke training. Visually inspected both
generated figures. The second figure prominently compares transfer failure with the much better
in-domain scores. Prepared the modernization branch for a draft PR; no merge, release publication,
live network capture, traffic blocking, or production deployment was performed.

Created [draft PR #1](https://github.com/LahsenAitOiahmane/NetAttDetector/pull/1) with the final
problem statement, implementation summary, validation links, measured failures and remaining
release gates. All changes are on `codex/modernize-nids-foundation`; `main` is awaiting review.

## Outstanding work

This foundation does not complete the production roadmap. Global data audits, robust capture/time
holdouts, confidence intervals, domain adaptation, exporter parity, a live bounded sensor,
operational alert budgets, shadow validation, serving/monitoring/rollback, and commercial release
permissions remain explicit backlog items. No production accuracy or real-network alert quality
is claimed.
