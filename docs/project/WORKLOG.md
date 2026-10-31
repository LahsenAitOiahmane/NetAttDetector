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

## 29 October 2026 — multi-source development and registered fresh-source test

The user authorized the next experiment. PR #1 had been merged, so this phase starts from
`origin/main` at `0b090f2` on `codex/multisource-generalization`. Registered the initial protocol
in commit `899de37`, then amended it in `f94cbaa` using development-only audit evidence before
any candidate fitting or fresh holdout inspection. CSE-CIC-IDS2018-v2 is reserved for the final
frozen test. Its 612,059,380-byte archive exceeds the old 512 MiB budget; the explicit acquisition
budget is 768 MiB with ample disk headroom. Transfer uses two workers and bounded 4 MiB blocks.

Audited every one of 19,330,771 development rows with disk partitions. Full 40-feature
representations had 5,310,377 repeated rows and 2,542 conflicting groups containing 7,429 rows.
Only one full-feature group was shared across both sources. Both supplied feature dictionaries
have SHA-256 `f20e86a939f9eb53b06ace15de4cc977ed61c02e538c6465efe72b635448e939`.
No binary-label versus benign/attack-name disagreements or tested port/protocol/TTL range errors
were found. ToN has 216 nonfinite/coerced values in the two per-second byte fields (105/111).
These remain training-imputed missing values, not fabricated zeros.

The reduced representation produced ambiguous groups containing 10,090,679 raw rows. These
are not automatically label mistakes: losing measurements can collapse different observations.
The registered amendment retains reduced ambiguity, isolates its groups across development
partitions, and restricts conflict/duplicate deletion to the full representation. The initial
overcleaned 29,179-row cache is retained for diagnosis and is not used for fitting.

The corrected cache contains 195,894 rows, removing 55 sampled rows from globally conflicting
full groups and 4,051 sampled full duplicates. Source/label counts: UNSW benign 94,350 / attack
3,291; ToN benign 35,704 / attack 62,549. The corrected full preparation took 387.46 seconds
and sampled peak process RSS 365.7 MiB. Conditional benign-source KS distances reach 0.907
for source-to-destination average throughput, showing substantial measured distribution shift.
This supports investigating transfer failure but does not by itself prove a causal explanation.

Implemented globally checked reusable caches, group-stratified train/calibration/validation,
equal-source/class weighting, nine declared model/feature/weight variants, sigmoid calibration
on disjoint data through FrozenEstimator, per-source confidence/support constraints, immutable
model/index/protocol bindings, and exclusive holdout-access receipts. Version-2 bundles support
the selected ordered feature subset and calibrated inference while retaining version-1 support.
Exact selection, holdout and reproduction outcomes are recorded in `reports/MULTISOURCE.md`.

At clean source commit `07e25ff`, completed nine fits and eighteen calibration variants in
167.47 seconds with sampled peak process RSS 388.5 MiB. Only the uncalibrated/calibrated
XGBoost full-feature pair met all per-source constraints locally; the smaller uncalibrated
variant was selected at threshold 0.9936622381. Worst-source validation recall was only 9.40%.
No selected model or threshold was altered after freezing. Saved the original baseline hash,
all global-index hashes and partition membership fingerprints before opening CSE records.

The full CSE CSV scan verified 18,893,708 rows. All three archives now contain 38,224,479
verified flows, totaling 814,056,982 compressed bytes. CSE checksum
`ad6307995d08ff8825d0193d5a1cfa61cf3dc4c6f223962ba3762a7775b5dc24` matched the independent
GitHub download. Uniform holdout sampling removed 12 conflicting rows and 1,823 duplicates,
then excluded 151 full-feature matches present anywhere in development. The 198,014 remaining
rows included 23,764 attacks; 35,665 rows in ambiguous reduced groups and 13,648 reduced
development matches were retained as registered. The Windows frozen model detected one attack
and produced 67 false positives (precision 1.47%, recall 0.0042%, FPR 0.03845%). The old
baseline detected 146 attacks with 12,355 false positives. The candidate is not a deployable
improvement. Local holdout evaluation took 211.76 seconds and sampled peak RSS 317.4 MiB.

The [GitHub experiment](https://github.com/LahsenAitOiahmane/NetAttDetector/actions/runs/40972007019)
completed, but it did not reproduce the local fitted XGBoost/threshold. Input hashes, all global
indexes and partition membership matched; the Linux run found no feasible candidate and used
the declared F1 fallback at 0.7089409828. That independently frozen variant detected 39.03%
of attacks with 12.31% precision and 37.92% benign FPR on identical CSE rows. Its quality gate
also failed. Numerical/stochastic/ordering causes are not proven; preserve the discrepancy.
Both variants were frozen before CSE access and neither was selected using CSE results.

Baseline permutation diagnostics on capped development samples found MIN_TTL AP drops of
0.6963 in UNSW and -0.0199 in ToN. These samples may overlap old baseline fitting and are
diagnostic rather than independent performance evidence. The primary dataset paper also
describes excluding TTL in UNSW experiments. These observations support source-specific
dependence without establishing a complete causal explanation. No TTL or threshold change
was applied to the frozen artifacts after observing this evidence/test.

The cloud XGBoost memory snapshot failed to load on Windows with `input stream corrupted`.
Added stable-UBJSON export of existing fitted trees, preserving preprocessing/thresholds and
the original artifact. The [export-only workflow](https://github.com/LahsenAitOiahmane/NetAttDetector/actions/runs/40974339706)
converted the original Linux model without refitting. Its artifact loaded on Windows and passed
5,000 parity probes (maximum probability difference 7.45e-9, identical threshold decisions),
then classified 1,000 real development flow rows successfully. The local portable conversion
had zero probe difference. This fixes cross-platform inference export, not cross-platform
training determinism. Manifests and both failed holdout results are checked in as evidence.

Final validation: 35 local tests passed, lint/format checks passed, and local documentation links
resolved. GitHub CI on Python 3.11 and 3.13 passed with 33 core tests/two optional tests skipped;
the optional portable-export tests passed locally and the Linux export job exercised that path.
Visually verified both phase-2 figures. Created [draft PR #3](https://github.com/LahsenAitOiahmane/NetAttDetector/pull/3)
with the failed results, platform discrepancy, portability fix and remaining release gates.
All changes are on `codex/multisource-generalization`; no phase-2 merge/deployment was performed.

## 30 October 2026 — Development ablation and training stability

Delivered this follow-up on `codex/development-stability` in
[draft PR #4](https://github.com/LahsenAitOiahmane/NetAttDetector/pull/4), after phase 2 was
merged through PR #3. The model fits retain their original source revision and provenance.
Registered `configs/stability.json`/`STABILITY_PROTOCOL.md` in `b730df3`. Before fitting,
amended the grouping to the 30 fields shared by every policy, because dropping four rate
fields could otherwise create identical ablated observations in different partitions.
Committed the implementation/amendment as `8313635`; both real runs used that clean revision.

Reused the corrected 195,894-row UNSW/ToN cache. Preserved all observations in ambiguous
reduced groups: 29,361 common-input groups, largest group 20,302 rows, 48,138 observations
in mixed-label groups. Ran eight feature policies × two classifiers × three group partition
seeds (48 fits), eight same-payload XGBoost controls and four known-source transfer diagnostics.
Chose thresholds on calibration, then froze them for validation. No CSE file or new independent
source was read in this phase. No previous freeze/test artifact was changed.

Local completion: 714.2 seconds and sampled peak process RSS 424.5 MiB. The
[GitHub study](https://github.com/LahsenAitOiahmane/NetAttDetector/actions/runs/40979423634)
completed all 60 fits in 322.2 seconds with 551.7 MiB peak process RSS. Both used two-thread
study budgets; controls deliberately used one/two threads. These measurements exclude raw
archive preparation and cache loading; RSS includes the loaded cohort and is not whole-PC RAM.

All 48 pooled calibration operating points were infeasible on both platforms, so the declared
F1 fallbacks applied. None of the 16 model/policy combinations passed every source/seed.
Every candidate's exact post-preprocessing cross-partition overlap count was zero. TTL zeros
were source-dependent (84.77% of ToN benign versus 0.03% of UNSW benign); no zero was repaired
or relabeled. TTL removal did not establish useful source transfer. Results block promotion.

Canonical ordered feature/label/weight payloads and all partitions matched across platforms.
Control post-preprocessing train/validation digests matched. Repeated modes had exact scores
within each platform; Windows sampled one-versus-two-thread scores differed by up to 0.3438,
while full controls matched across thread counts. Cross-platform sampled-control differences
were up to 0.3907; full-control differences were at most 5.96e-8 with zero diagnostic-0.5
decision disagreements. Tree bytes still differed. Of 48 ablations, 44 had exactly matching
source metrics; three had minor score-metric differences and one XGBoost no-protocol-ID
seed-7 ablation changed its threshold/confusion matrices. Training determinism remains open;
these controls do not retroactively prove the old discrepancy's cause.

Added CLI study/comparison commands, a development-only workflow mode, strict comparison
payload/score integrity checks, saved JSON evidence, plot/report generators and reproduction
instructions. Updated README and the ordered Scrum backlog. [STABILITY.md](../../reports/STABILITY.md)
contains the measured trade-offs and two visually verified figures. The local suite passes
42 tests; lint/format checks pass. Source CI passed on Python 3.11/3.13 with 39 core tests
and three optional tests skipped in each core environment. The next sprint prioritizes capture/host/time provenance,
exporter/PCAP measurement parity and an operator alert budget before another model search.

## 30 October 2026 — Repository cleanup and capture/measurement sprint

Started `codex/repository-and-measurements` from current main after PR #4 was merged.
Merged the exact Dependabot PR #2 commits into this review branch, upgrading CPython pandas
to 3.1.0. The prior 42 tests passed in an isolated upgrade environment. All 195,894 cached
full/reduced fingerprints remained identical. Both existing portable model exports passed
5,000 probes with identical threshold decisions (maximum score deltas 0 and 7.45e-9).
Upgraded the default `.venv` after compatibility checks; old experiment artifacts/metrics retain
their original versions and were not refit. The temporary `.venv-pandas3` environment remains
ignored because automatic approval review blocked its recursive deletion with no further reason.

Removed obsolete 2024 models/utilities/data loader, redundant wrappers/requirements shim, old
editor settings/CSS sketch, tracked bytecode, five old model binaries, logs and eleven empty
captures. Kept raw archives, corrected caches, frozen runs, receipts and reviewed evidence;
Git history retains the removed originals. Grouped docs and scripts by purpose, rewrote links
and workflow commands, added navigation/repository guides, expanded `.gitignore`, and added
tracked-artifact/ignore/link checks. Added LF/binary attributes after observing Windows newline
conversion change an expected JSON digest. Canonical fixture hashes now match Linux/Windows.

Recorded the user's choices: small-office/home scope and up to 50 reviewed alerts/day. The
threat model separates reviewed incidents from per-flow predictions and leaves traffic volume,
aggregation, observation point, retention and pilot SLO open. No live capture or blocking occurred.

Inspected only headers of existing UNSW/ToN archives: both have endpoint addresses but lack
event timestamps, capture IDs and session IDs. Registered source roles/lineages, excluded merged
source contamination and reserved CICIDS2017 without acquiring or inspecting its flow data.
Added metadata inspection and declared capture-plan validation for checksums, host separation,
role/lineage isolation, timezone-aware chronology/purging and source-wise class support. The
actual plan remains empty/blocked; structural validation never authorizes training or proves labels.

Implemented an offline-only Linux NFStream 6.6.0 lab with 16 MiB inputs, one meter and 10,000-flow
cap. The separate 11-feature contract emits no UQ aliases/zero placeholders. Created a 948-byte
12-packet synthetic IPv4/IPv6 UDP/TCP fixture with zero-duration and idle-expiration cases.
The [Linux lab](https://github.com/LahsenAitOiahmane/NetAttDetector/actions/runs/41108865671)
at `2c28e22` passed all 85 expected fields on five flows; Windows verified the downloaded JSON
against the same expectations. Export loop: 0.093 seconds, sampled parent RSS 179.3 MiB;
imports and meter child memory are excluded. This proves selected native measurements only,
not nProbe parity, attack detection quality or production memory/throughput.

The integrated 59-test suite passed locally; 18 focused tests passed after the final source-wise
support check, including its added counterexample. Lint, formatting, documentation links, ignore
policy and deterministic fixture checks pass. Reports and evidence are in
[CAPTURE_MEASUREMENTS.md](../../reports/CAPTURE_MEASUREMENTS.md). Review:
[PR #5](https://github.com/LahsenAitOiahmane/NetAttDetector/pull/5), including the PR #2
dependency branch. PR #2 was merged to main separately during this work; incorporated its main
update and the remote branch merge without rewriting or overwriting concurrent changes.
The next dependency is trustworthy development captures/labels with metadata,
followed by re-export, label joins and capture-held-out training. No new real-data model was fitted.

## 31 October 2026 — Bounded development PCAPs and label-linkage audit

Fast-forwarded local main to merged PR #5 (`c38b80b`) and started
`codex/labelled-development-captures`. Checked original UNSW/ToN publisher downloads; their
SharePoint links redirected to sign-in. Registered IoT-23 as additional development data at
`a9534b4` before inspecting PCAP bytes/label logs. CICIDS2017 remains reserved and unacquired.

Added three-scenario bounded acquisition, observed input pins and a conservative TCP/UDP
five-tuple/interval matcher. Missing/unknown labels, durations, multiple candidates, partial
intervals and unsupported protocols remain unlabelled. Acquired all bundles on GitHub, then
retried successfully on Windows after transient DNS failures. All nine local file hashes match;
three bundles occupy 12,469,035 bytes under ignored `data/raw/iot23-development/`.

The first cloud run `41235768705` correctly stopped at the existing 10,000-flow export limit.
Introduced a separate one-pass aggregate audit with a 100,000-flow stop and unchanged 16 MiB
PCAP cap. Moved RSS sampling to a lightweight module, preserving the training import interface,
and sampled child RSS sums for the audit. Pinned replay `41235993410` at `a80dd43` passed:
45,588 flows, 41,613 candidate matches, 90.8 MiB peak sampled parent RSS and under 159 MiB
sampled parent/child sum. Original fixture checks also passed (85 fields). No feature rows or
raw traffic are uploaded as workflow artifacts.

Found a material data issue: 32,691 matched Torii attack segments in captures 20/21 map to only
16 distinct labelled connections. Total distinct matched attack UIDs across all three captures
is 21, with 8/8/5 per capture, below the planned support requirement. Capture 42-1's README
also disagrees with the PCAP's capture name/year. Kept training disabled and recorded the
license discrepancy between Zenodo metadata and individual mirror READMEs. No new model,
accuracy claim, live sensing or traffic blocking was introduced. Results, hashes, limitations,
run commands and ordered next work are in [DEVELOPMENT_CAPTURES.md](../../reports/DEVELOPMENT_CAPTURES.md).

All 83 tests then present passed locally in 67.23 seconds with two native threads. A subsequent
focused run passed 41 tests and skipped the added Linux-only exporter regression on Windows.
The Linux measurement workflow exercises both flow-cap settings against known fixture fields.
Lint, format, links, ignore/tracked-file hygiene and deterministic fixture bytes pass.

## Outstanding work

This work does not complete the production roadmap. Global audits cover the development inputs;
conditional confidence intervals and a fresh-source test are now implemented. Robust capture/time
holdouts, capture-level uncertainty, residual training stability, domain adaptation, exporter parity, a live bounded sensor,
operational alert budgets, shadow validation, serving/monitoring/rollback, and commercial release
permissions remain explicit backlog items. No production accuracy or real-network alert quality
is claimed.
