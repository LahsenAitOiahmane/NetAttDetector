# Product strategy and Scrum delivery plan

Planning baseline: 29 October 2026. Product goal: an efficient flow-based detector that produces
investigable alerts on a defined network, with credible detection evidence and a sustainable
false-alarm budget. The initial user is an operator of a small network, running CPU inference.
The first publishable release is a reproducible **research** release; a production service has
additional gates. No calendar deadline or guaranteed accuracy is implied by this plan.

## Working process

Use two-week sprints, initially 10–15 story points per solo-developer sprint, revised after two
sprints using observed capacity. Points below represent relative uncertainty, not hours. The
project owner acts as product owner and approves scope/operational alert budgets. The developer
implements stories; reviewers/analysts review data assumptions, experiments, and alert usefulness.
These are proposed responsibilities, not people silently assigned or notified.

Repository backlog states: Backlog → Ready → In progress → Review/Validation → Done. Limit
active implementation to one story and one experiment at a time on the laptop. A GitHub PR links
story IDs and validation artifacts. Update this file/work log at sprint review; a later GitHub
Project can mirror these stories after the owner chooses that workflow.

Each sprint has planning (goal and ready stories), a short daily progress/blocker note, a review
of runnable evidence, and a retrospective with one process improvement. Track accepted stories,
blocked dependencies, experiment reproducibility, metric changes with uncertainty, model size,
RSS, latency, and false alarms. Do not use increasing accuracy or lines of code as progress alone.

Definition of Ready: concrete user/problem, input provenance, known dependencies, measurable
acceptance, explicit resource budget, and small enough for a sprint. Definition of Done: reviewed
implementation, meaningful checks passing, documentation/model/data-card updates, reproducible
commands, known limitations, and linked experiment evidence if behavior changes. Passing a
synthetic smoke test alone does not complete an ML quality story.

## Ordered backlog

The order below is the default priority. `Implemented` describes present code, not production
acceptance. `Measured` requires a linked result. Remaining stories are not claimed complete.

The initial external test detected only 0.33% of ToN attacks at the frozen UNSW threshold.
That blocked deployment and motivated NAD-013–020. The registered multi-source phase now
also failed the fresh CSE source test; see `reports/MULTISOURCE.md`. Before further tuning,
reserve another independent source/capture. See the baseline report;
the roadmap must not be read as an endorsement of the initial champion for live protection.

| ID | Priority / points | Story and measurable acceptance | Dependencies | State |
|---|---|---|---|---|
| NAD-001 | P0 / 3 | Audit all original modules and artifact inventory; record missing data and misleading score limitations | — | Done: AUDIT.md |
| NAD-002 | P0 / 3 | Isolated package and frozen dependencies; clean install and CLI on supported Python versions | 001 | Implemented; CI validates |
| NAD-003 | P0 / 5 | Official dataset registry/downloader with byte limit, disk reserve, CRC and checksum manifests | 001 | Implemented; real acquisition recorded |
| NAD-004 | P0 / 5 | Bounded full-stream sampling, full label counts and sample EDA; deterministic seed and invalid-label accounting | 003 | Implemented; measured |
| NAD-005 | P0 / 5 | Target/identifier allowlist, duplicate/conflict removal and disjoint split checks | 004 | Implemented within sample |
| NAD-006 | P0 / 5 | Save fitted preprocessing, model, schema and threshold together; save/reload predictions agree | 005 | Implemented and tested |
| NAD-007 | P0 / 5 | Compare dummy, linear, histogram, Extra Trees and optional XGBoost with two-thread budgets | 006 | Implemented; measured baseline |
| NAD-008 | P0 / 5 | Validation-only selection and independent test report including precision, recall, FPR and slices | 007 | Implemented; targets may fail |
| NAD-009 | P0 / 5 | Freeze champion and evaluate a different source; report overlap exclusions and failures | 008 | Implemented; see BASELINE.md |
| NAD-010 | P0 / 3 | Strict flow-export inference; reject absent schema and retire fake live flow fields | 006 | Implemented and tested |
| NAD-011 | P0 / 5 | CI on Python 3.11/3.13 plus bounded GitHub real-data training and retained evidence | 002–009 | Implemented; record remote outcomes |
| NAD-012 | P0 / 3 | README, architecture, evaluation rules, operations, work log and honest model card | 001–011 | This documentation |
| NAD-013 | P1 / 5 | Global duplicate/conflict analysis using partitioned hashes/DuckDB; reconcile counts by source | 004 | Done for all 19.33M development rows; disk partitions and checked indexes |
| NAD-014 | P1 / 5 | Source/capture/session data cards and immutable split manifests; detect merged-source contamination | 013 | Source manifests, group fingerprints and contamination checks implemented; capture/session metadata unresolved |
| NAD-015 | P1 / 8 | Capture/time/device-held-out benchmark with purging; both labels and adequate attack support per fold | 014 | Backlog; basic split modes exist |
| NAD-016 | P1 / 5 | Port/TTL/protocol-ID ablations; retain only features robust across development sources | 015 | Eight targeted policies across three grouped development seeds measured in STABILITY.md; no subset passed; capture-aware acceptance remains |
| NAD-017 | P1 / 5 | Training-only rate/ratio/log features with unit tests and documented exporter availability | 016 | Backlog |
| NAD-018 | P1 / 5 | Bounded hyperparameter search (≤1 laptop-hour, 2 threads); select with development folds only | 015–017 | Registered nine-candidate comparison completed; broader/nested search remains |
| NAD-019 | P1 / 5 | Compare class weights/train-only sampling and calibrate on disjoint development observations | 018 | Equal-source/class weights and sigmoid calibration compared; alternative sampling remains |
| NAD-020 | P1 / 5 | Confidence intervals, predefined multiple development seeds, new frozen publication holdout | 015–019 | Conditional Wilson intervals, frozen CSE test and three registered development seeds measured; new untouched source and capture-level uncertainty remain |
| NAD-021 | P1 / 3 | Agree site threat model, flow volume, retained metadata and analyst alert/day budget | 009 | Backlog; owner/operator input |
| NAD-022 | P1 / 8 | Select exporter or retrain a reduced contract; demonstrate feature/units/direction/timeout parity on PCAP fixtures | 016,021 | Backlog |
| NAD-023 | P1 / 8 | Authorized passive sensor with bounded flow cache, IPv4/IPv6, TCP/UDP/ICMP and dropped-flow counters | 022 | Backlog |
| NAD-024 | P1 / 5 | Replay validation: same PCAP/export yields consistent online/offline predictions; measure end-to-end lag and loss | 023 | Backlog |
| NAD-025 | P1 / 5 | Event aggregation, suppression and rule correlation; alert includes evidence and schema/model versions | 024 | Backlog |
| NAD-026 | P1 / 8 | Shadow pilot over an agreed window with analyst labeling; compare alerts to rules and measured local benign FPR | 020,025 | Backlog |
| NAD-027 | P1 / 5 | Measure and reduce model/feature/batch cost without material loss under preregistered tolerances | 026 | Backlog |
| NAD-028 | P1 / 5 | Authentication, input bounds, backpressure, health/readiness and structured telemetry for optional inference API | 024 | Backlog |
| NAD-029 | P2 / 5 | Containerize pinned inference environment; smoke/replay behavior and process limits match local baseline | 027,028 | Backlog |
| NAD-030 | P2 / 5 | Model registry with immutable artifacts, hashes, review/promotion and tested rollback | 020,029 | Backlog |
| NAD-031 | P2 / 5 | Drift/missingness/alert-rate monitoring with measured trigger thresholds and a response playbook | 026,030 | Backlog |
| NAD-032 | P2 / 5 | Analyst feedback labeling and leakage-safe retraining; shadow challengers before promotion | 031 | Backlog |
| NAD-033 | P2 / 8 | Two-stage attack-family classifier; minimum per-class support and unknown/abstain behavior | 019,026 | Backlog |
| NAD-034 | P2 / 5 | Benign-only novelty baseline and incremental-learning comparison with correct binary task mapping | 026 | Backlog |
| NAD-035 | P2 / 8 | CICIoT2023 / DataSense / Zeek adapter chosen by deployment domain; preserve source metadata and terms | 014,021 | Backlog |
| NAD-036 | P2 / 5 | Minimal analyst dashboard for triage, timestamps, source evidence and suppression; no raw payload exposure | 025,028 | Backlog |
| NAD-037 | P1 / 3 | Resolve source-code license, commercial dataset permission, notices, privacy and retention before public product release | 021 | Backlog; owner decision |
| NAD-038 | P2 / 5 | Research release: versioned source, model/data cards, clean-environment reproduction and evaluation artifacts | 011,020,037 | Backlog |
| NAD-039 | P2 / 8 | Production candidate: pilot acceptance, reliability/load checks, failure handling, rollback drill and operator signoff | 026–032,037 | Backlog |
| NAD-040 | P1 / 5 | Compare ordered raw/transformed payloads, weights, repeated fits, threads and platform controls | 014 | Measured: 60 fits/platform, matching inputs, reduced control discrepancy with full sampling; one larger ablation retained decision differences |

## Next sprint after the development study

The [registered stability study](../reports/STABILITY.md) completed all 48 ablations, eight
controls and four transfer diagnostics on each platform. No calibration threshold met the
per-source precision/FPR/support constraints in any pooled-development ablation; all used
declared fallback thresholds. Removing TTL alone did not yield a robust cross-source detector.
The 30-field common grouping retained 48,138 observations in label-ambiguous groups. Do not
delete these observations or relax the quality gate to obtain an attractive score.

Sprint goal: make the next generalization test scientifically interpretable. Proposed capacity:
13 points over two weeks; these are backlog estimates, not a promise of delivery dates.

| Order / story | Concrete work | Evidence for acceptance |
|---|---|---|
| 1 / NAD-014,015 (5 points) | Inspect publisher metadata for capture/day/host/session provenance; register a source/capture split manifest before fitting. Keep CSE observed, and keep prospective BoT/other independent captures out of tuning. Do not use the merged UQ set as independent evidence. | Versioned provenance/data card, source overlap map, immutable split rules and adequate benign/attack support; explicitly record unavailable metadata. |
| 2 / NAD-022 (5 points) | Choose a domain-appropriate flow exporter and controlled benign PCAP fixtures; compare units, direction, duration quantization, TCP windows, TTL, protocol and derived rates with the published contract. | Reproducible export commands/version, fixture provenance and field comparison; unsupported fields fail explicitly. No production or live traffic capture is required for this task. |
| 3 / NAD-021 (3 points) | Define the intended environment, relevant attack types, expected benign flow volume, retention and tolerable alerts/day with the operator. Translate this into a registered evaluation budget. | Written threat model and operator-approved operating budget; 98% precision/0.5% FPR/80% recall remain research screens until then. |

Next, resolve the residual NAD-040 case with per-stage model/input instrumentation before
claiming reproducible training. Only after measurement/split requirements are ready should
NAD-017/018 explore training-only rate/ratio/log features, bounded search or domain adaptation.
Reserve a fresh source before those fits and evaluate it only after freezing the selected model.
Do not automatically run another broad search or acquire multi-gigabyte PCAP collections on
the laptop. A pilot, service API and dashboard do not address the current model-quality failure.

## Sprint sequence

The completed foundation is an initial sprint's deliverable bundle, not a claim it took the point
estimate in real time. Split later bundles to fit actual velocity; do not force 20+ points into a sprint.

| Sprint | Goal | Candidate stories | Demonstration / exit |
|---|---|---|---|
| Foundation | Replace unverifiable prototype behavior with reproducible evidence | 001–012 (delivered foundation bundle) | Fresh install, real data scan, trained bundle, report, checks and workflows |
| 1 | Trust the data and partitions | 013–015 | Global audit and source/capture-held-out baseline |
| 2 | Improve transferable features | 016–018 | Ablation/search comparison on development folds within budget |
| 3 | Make operating points defensible | 019–021 | Calibration/uncertainty report, frozen holdout, agreed alert budget |
| 4 | Match training and live measurements | 022–024 | Exporter parity and bounded replay sensor |
| 5 | Establish real alert value | 025–027 | Shadow-pilot errors, analyst feedback and efficiency frontier |
| 6 | Operate and recover reliably | 028–032 | Service/container/registry/monitoring/rollback demo |
| 7+ | Extend and publish | 033–039 as justified | Reviewed research release, then separate production-candidate review |

## Release gates and stop conditions

The multi-source candidate is rejected for deployment. Its Windows freeze detected 1 of
23,764 CSE attacks; the Linux fallback detected 39.03% with 37.92% benign FPR. Prioritize
measurement/TTL semantics, targeted single-feature-family ablations, repeated development
seeds and threshold stability, and capture-aware/local data. CSE is now observed and cannot
be reused as an untouched tuning holdout. Portable export fixes loading an existing cloud
model on Windows; it does not resolve training variability or generalization.

Research alpha: deterministic ingestion/splits, data terms/citations, train-only preprocessing,
dummy baseline, untouched independent holdout, external result, per-attack support, confidence
intervals, resource evidence, and reproduction from a clean checkout. Current foundation lacks
some of these gates, especially robust split evidence and intervals; label it accordingly.

Pilot candidate: exporter parity, local threat model and data handling agreed, bounded queues,
end-to-end replay evidence, useful alerts, and accepted local false-alarm budget. Run passively
in shadow mode first. A benchmark precision target is not an operational authorization to block.

Production candidate: sustained pilot performance, detection/false-alarm confidence bounds,
service SLOs chosen with the operator, recovery/rollback drills, monitoring, ownership and
commercial terms resolved. No promotion if minority recall collapses, data identity is uncertain,
external failures are unexplained, feature parity is absent, or the alert queue exceeds capacity.

If an experiment fails a gate, retain its result, explain the failure, and prioritize the cause.
Do not tune the final test repeatedly, discard inconvenient attacks, rebalance a test set to improve
precision, or add a large model merely to make the project appear more sophisticated.
