# NetAttDetector

CPU-conscious network intrusion detection research, rebuilt from the 2024 prototype with
bounded data processing, measured flow contracts and reproducible experiments.

**Current stage: offline research; no detector qualifies for deployment.** The latest
[statistical diagnostic](reports/STATISTICAL_STUDY.md) reaches 98.89% connection precision and
99.44% recall on its accepted validation subset, but its **4.04% benign FPR** fails the 0.5%
research screen. DDoS detection improves to 75/114 accepted UIDs, while label linkage covers
only 114/14,394 publisher DDoS UIDs. Validation selected that model; it is not a fresh test.

The intended environment is a [small office/home network](docs/project/THREAT_MODEL.md) with
up to 50 reviewed alerts/day. Traffic volume and incident aggregation remain unmeasured; the
owner deferred the live pilot. No live sensing, blocking or product promotion is enabled.
The [information-bound study](reports/INFORMATION_BOUND.md) now proves an observed no-port
ceiling of 60.71% recall at the validation FPR budget. Next work adds validated causal traffic
context before further fitting. See the [ordered backlog](docs/project/ROADMAP.md).

## Measured research stages

| Stage | Result and evidence |
|---|---|
| [Original UQ baseline](reports/BASELINE.md) | 99.31% random-split accuracy, but only 0.33% independent ToN attack recall; rejected |
| [Multi-source study](reports/MULTISOURCE.md) | Independent CSE test also fails; CSE is now observed evaluation, not a fresh tuning holdout |
| [Stability study](reports/STABILITY.md) | Sixty fits/platform; no policy passes the full source screen; residual differences retained |
| [Capture acquisition](reports/CAPTURE_EXPANSION.md) | Six bounded IoT-23 bundles available locally, with verified publisher inputs and disk-backed label audits |
| [Packet linkage](reports/PACKET_LABELS.md) | 248,997/250,819 flows link under an experimental candidate; not detector accuracy |
| [Native cohort](reports/NATIVE_COHORT.md) | All rows/exclusions retained, connection/time partitions verified, repeated builds byte-identical |
| [Native models](reports/NATIVE_STUDY.md) | Fifteen fits; best diagnostic FPR 2.89%, DDoS 1/114; no qualifying model |
| [Error/repeat diagnostics](reports/NATIVE_ERRORS.md) | Fixed-one-thread scores/coefficients repeat exactly; cross-thread differences remain |
| [Statistical measurement](reports/STATISTICAL_FEATURES.md) | 165 independent fixture values, 27 measured fields, unchanged core/label joins for every flow |
| [Statistical models](reports/STATISTICAL_STUDY.md) | Fifteen fixed-thread fits; natural logistic DDoS 1/114 to 75/114, but false positives 57 to 77 |
| [Input information bound](reports/INFORMATION_BOUND.md) | No-port validation ceiling 60.71% at 9 benign flags; even an optimistic relaxation needs 39 flags for 80% recall |

Native contracts (`nfstream-core-v1`, `nfstream-statistical-v1`) are separate from the historical
UQ models. They are not column aliases. Addresses, labels, capture IDs and timestamps stay out
of model inputs. Candidate labels, device independence, exclusions and data terms remain
explicit limitations. CI retains aggregate evidence; private native models/rows stay on the
ephemeral runner. Reproduction commands are indexed in [docs](docs/README.md).

Project code is [MIT](LICENSE), selected by the owner on 1 November 2026. Dataset permissions
and third-party dependency licenses remain separate; see [data sources](docs/research/DATASETS.md).

## What changed

- A versioned feature contract replaces hand-built packet placeholders.
- Chunked CSV/ZIP ingestion samples across the entire file without loading it all into RAM.
- Labels, addresses, dataset names, and DNS query identifiers cannot become model inputs.
- Historical UQ cleaning is documented; native cohorts retain exclusions and conflicting patterns.
- Training, validation, and test data are separate; preprocessing fits training data only.
- SGD logistic regression, histogram gradient boosting, Extra Trees, and optional XGBoost
  compete against a dummy baseline. Native thresholds use calibration only; validation chooses models.
- Reports include precision, recall, macro F1, average precision, false alarms, attack slices,
  input checksums, dependencies, model size, inference timing, and sampled peak process memory.
- A second dataset can test generalization with the model and threshold frozen.
- GitHub Actions tests Python 3.11/3.13 and runs a bounded real-data training workflow.

## Start on Windows / PowerShell

The audited machine has an Intel i5-1135G7, 4 cores / 8 threads, and 15.74 GiB RAM.
Native/statistical fits use one thread; historical UQ profiles allow two. Start with bounded
100k–200k samples and check available RAM with `doctor`. No GPU is required.
About 14 GiB disk was free at the initial audit; use compressed archives, not full PCAP downloads.

With [uv](https://docs.astral.sh/uv/):

```powershell
uv sync --frozen --extra dev
uv run --no-sync netatt doctor
uv run --no-sync netatt datasets
```

Current native research uses the verified Linux exporter. With an authenticated GitHub CLI
and the current branch pushed, run the bounded study:

```powershell
$studyBranch = git branch --show-current
gh workflow run measurements.yml --ref $studyBranch -f statistical_study=true
```

The workflow retains aggregate evidence, not real model/row binaries. Use the
[native cohort](docs/guides/NATIVE_COHORT.md), [statistical measurement](docs/guides/STATISTICAL_FEATURES.md)
and [model comparison](docs/guides/STATISTICAL_STUDY.md) runbooks for private local Linux
reproduction. Windows does not need a capture driver for setup, tests or report verification.

### Historical UQ experiments

These commands reproduce the original CSV experiments and their documented limitations:

```powershell
uv run --no-sync netatt download nf-unsw-nb15-v2 --max-mib 512
uv run --no-sync netatt profile --data data/raw/nf-unsw-nb15-v2.zip --output reports/local/unsw-profile.json
uv run --no-sync netatt train --data data/raw/nf-unsw-nb15-v2.zip --config configs/laptop.json --output artifacts/my-first-run
```

To include XGBoost and evaluate another network source:

```powershell
uv sync --frozen --extra dev --extra boosting
uv run --no-sync netatt download nf-ton-iot-v2 --max-mib 512
uv run --no-sync netatt train --data data/raw/nf-unsw-nb15-v2.zip --external data/raw/nf-ton-iot-v2.zip --config configs/benchmark.json --output artifacts/my-cross-dataset-run
```

Each output directory must be fresh. Experiments intentionally refuse to overwrite previous evidence.
The downloader reuses an existing archive only if its SHA-256 matches its local manifest.
`--max-mib` limits compressed download bytes; it does not limit decompressed bytes scanned or runtime.
The default 512 MiB budget can refuse larger catalog entries; inspect their record and available
disk before increasing it. No archive is automatically expanded on disk.

Without uv, create a virtual environment and run `python -m pip install -e ".[dev]"`;
this resolves compatible dependencies but is **not** the frozen lockfile environment.
`python -m netattdetector ...` invokes the same CLI after package installation.

## Use a trained UQ model

```powershell
uv run --no-sync netatt predict --model artifacts/my-first-run/model.joblib --data path/to/compatible-flows.csv --output reports/local/predictions.csv
```

Input must contain every named feature in [the contract](src/netattdetector/data.py), with matching
units and exporter semantics. Extra columns are ignored and columns are reordered by name.
Absent columns fail explicitly. Missing/nonfinite individual values use training-fitted imputation.
Output contains a row number, attack score, binary prediction, and the stored threshold.
Scores are uncalibrated model estimates, not established probabilities of a real attack.
Only load trusted joblib artifacts: deserializing untrusted pickle-based models can execute code.

The unused 2024 modules, old model binaries, bytecode, logs and empty capture files have been
removed from the active tree. They remain recoverable from Git history at `f1b2d59`.
Downloaded datasets and current experiment evidence are retained. See the
[repository guide](docs/project/REPOSITORY.md) and [documentation index](docs/README.md).

## Understand historical UQ results

An experiment writes `metrics.json`, `data-profile.json`, `config.json`, `summary.md`, and
`model.joblib` under its output directory. Download provenance lives in `data/raw/*.manifest.json`.
Read the checked-in [experiment report](reports/BASELINE.md) for measured results and limitations.

The initial operating-point targets are attack precision ≥ 0.98 and FPR ≤ 0.005 **on validation**.
The code maximizes recall under those constraints. If no threshold satisfies them, it chooses the
best validation F1 and records `validation_constraints_met: false`. Test constraint status is
reported separately. A successful command or green CI run is not a model-promotion decision.

The default split is a deduplicated, stratified 60/20/20 random split: useful for development,
insufficient for real-world claims. Group and temporal split modes are supported in configuration.
See [evaluation rules](docs/research/EVALUATION.md) before comparing experiments or publishing numbers.

## Documentation and ordered work

| Document | Purpose |
|---|---|
| [Audit](docs/project/AUDIT.md) | What the old code did, evidence, defects, and hardware limits |
| [Dataset research](docs/research/DATASETS.md) | Official sources, large datasets, terms, adapters and download policy |
| [Architecture](docs/architecture/ARCHITECTURE.md) | Current modules, feature contract, future sensor and service boundaries |
| [Evaluation](docs/research/EVALUATION.md) | Leakage controls, metrics, operating points, model-selection rules |
| [Scrum plan](docs/project/ROADMAP.md) | Prioritized stories, dependencies, estimates, sprints and release gates |
| [Operations](docs/guides/OPERATIONS.md) | Local and GitHub training, reproducibility, recovery and budgets |
| [Work log](docs/project/WORKLOG.md) | Actions taken, checks, decisions, results and outstanding work |
| [Baseline / model card](reports/BASELINE.md) | Measured results and limits of the first new model |
| [Multi-source report](reports/MULTISOURCE.md) | Global data audit, ablations, calibration, fresh-source results and limits |
| [Registered multi-source protocol](docs/research/MULTISOURCE_PROTOCOL.md) | Selection rules fixed before fresh-source scoring |
| [Multi-source runbook](docs/guides/MULTISOURCE_OPERATIONS.md) | Cache preparation, freezing, one-time evaluation and portable exports |
| [Documentation index](docs/README.md) | All maintained guides grouped by purpose |
| [Repository guide](docs/project/REPOSITORY.md) | Layout, cleanup and tracked/ignored file policy |
| [Provenance and capture splits](docs/research/PROVENANCE.md) | Source reservation, metadata evidence and structural split checks |
| [Offline measurement lab](docs/guides/MEASUREMENTS.md) | Synthetic PCAP checks and the separate NFStream contract |
| [Development capture audit](docs/guides/CAPTURE_LABELS.md) | Pinned bounded PCAP acquisition, conservative label matching and measured support limits |
| [Threat model / alert budget](docs/project/THREAT_MODEL.md) | User-selected environment, 50/day planning capacity and missing pilot inputs |

## Development

```powershell
uv sync --frozen --extra dev
uv run --no-sync ruff check src tests scripts
uv run --no-sync ruff format --check src tests scripts
uv run --no-sync pytest
uv run --no-sync python scripts/check_repository.py
```

CI smoke data is synthetic and tests software plumbing only. Real research data and new model
artifacts are ignored by Git. GitHub training uploads reports and the model with finite retention;
it does not publish a service or automatically deploy a model.

Dataset permissions are separate from source-code licensing. The UQ collection permits academic
research with citation and requires author agreement for commercial use. See the exact
[publisher terms](https://www.cyber.uq.edu.au/project/machine-learning-based-nids-datasets).
The repository has no newly assigned source-code license; the owner must choose one before an
open-source release. Do not assume dataset availability grants unrestricted commercial rights.
