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

## Outstanding work

This foundation does not complete the production roadmap. Global data audits, robust capture/time
holdouts, confidence intervals, domain adaptation, exporter parity, a live bounded sensor,
operational alert budgets, shadow validation, serving/monitoring/rollback, and commercial release
permissions remain explicit backlog items. No production accuracy or real-network alert quality
is claimed.
