# NetAttDetector

Resource-conscious network intrusion detection research, rebuilt from the 2024 prototype.

**Current stage: reproducible offline research baseline.** This repository trains on labeled
network **flows** and classifies compatible flow exports. A high benchmark score does not
establish that a model is ready to protect a real network. Live sensing, deployment validation,
alert management, and drift monitoring have explicit acceptance criteria in the roadmap.

The initial champion reached 99.31% accuracy on its random UNSW holdout but detected only
0.33% of attacks on the independent ToN-IoT sample. **It is unsuitable for deployment.**
The [model card](reports/BASELINE.md) retains this failure and explains the next work priorities.

The next experiment implements global development-data auditing, source-aware weighting,
grouped train/calibration/validation partitions, feature ablations and an untouched CSE-CIC
source test. Its preregistered rules are in [the multi-source protocol](docs/MULTISOURCE_PROTOCOL.md).
The [completed multi-source report](reports/MULTISOURCE.md) retains another failed independent
test and a cross-platform training discrepancy. Both new model variants remain research-only.

## What changed

- A versioned feature contract replaces hand-built packet placeholders.
- Chunked CSV/ZIP ingestion samples across the entire file without loading it all into RAM.
- Labels, addresses, dataset names, and DNS query identifiers cannot become model inputs.
- Duplicate feature rows and contradictory binary labels are removed from the sampled data.
- Training, validation, and test data are separate; preprocessing fits training data only.
- SGD logistic regression, histogram gradient boosting, Extra Trees, and optional XGBoost
  compete against a dummy baseline. Only validation chooses the model and attack threshold.
- Reports include precision, recall, macro F1, average precision, false alarms, attack slices,
  input checksums, dependencies, model size, inference timing, and sampled peak process memory.
- A second dataset can test generalization with the model and threshold frozen.
- GitHub Actions tests Python 3.11/3.13 and runs a bounded real-data training workflow.

## Start on Windows / PowerShell

The audited machine has an Intel i5-1135G7, 4 cores / 8 threads, and 15.74 GiB RAM.
Use two training threads and start with 100k–200k sampled flows. No GPU is required.
About 14 GiB disk was free at the initial audit; use compressed archives, not full PCAP downloads.

With [uv](https://docs.astral.sh/uv/):

```powershell
uv sync --frozen --extra dev
uv run --no-sync netatt doctor
uv run --no-sync netatt datasets
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
`python main.py ...` forwards to the same CLI after package installation.

## Use a trained model

```powershell
uv run --no-sync netatt predict --model artifacts/my-first-run/model.joblib --data path/to/compatible-flows.csv --output reports/local/predictions.csv
```

Input must contain every named feature in [the contract](src/netattdetector/data.py), with matching
units and exporter semantics. Extra columns are ignored and columns are reordered by name.
Absent columns fail explicitly. Missing/nonfinite individual values use training-fitted imputation.
Output contains a row number, attack score, binary prediction, and the stored threshold.
Scores are uncalibrated model estimates, not established probabilities of a real attack.
Only load trusted joblib artifacts: deserializing untrusted pickle-based models can execute code.

The old `app/app.py` packet sniffer has been retired. It now forwards to flow CSV prediction.
Existing `saved_models/`, `models/`, `data/data_loader.py`, logs, and captured CSVs are historical
material, outside the supported pipeline. Old artifacts are not compatible with the new contract.

## Understand the results

An experiment writes `metrics.json`, `data-profile.json`, `config.json`, `summary.md`, and
`model.joblib` under its output directory. Download provenance lives in `data/raw/*.manifest.json`.
Read the checked-in [experiment report](reports/BASELINE.md) for measured results and limitations.

The initial operating-point targets are attack precision ≥ 0.98 and FPR ≤ 0.005 **on validation**.
The code maximizes recall under those constraints. If no threshold satisfies them, it chooses the
best validation F1 and records `validation_constraints_met: false`. Test constraint status is
reported separately. A successful command or green CI run is not a model-promotion decision.

The default split is a deduplicated, stratified 60/20/20 random split: useful for development,
insufficient for real-world claims. Group and temporal split modes are supported in configuration.
See [evaluation rules](docs/EVALUATION.md) before comparing experiments or publishing numbers.

## Documentation and ordered work

| Document | Purpose |
|---|---|
| [Audit](docs/AUDIT.md) | What the old code did, evidence, defects, and hardware limits |
| [Dataset research](docs/DATASETS.md) | Official sources, large datasets, terms, adapters and download policy |
| [Architecture](docs/ARCHITECTURE.md) | Current modules, feature contract, future sensor and service boundaries |
| [Evaluation](docs/EVALUATION.md) | Leakage controls, metrics, operating points, model-selection rules |
| [Scrum plan](docs/ROADMAP.md) | Prioritized stories, dependencies, estimates, sprints and release gates |
| [Operations](docs/OPERATIONS.md) | Local and GitHub training, reproducibility, recovery and budgets |
| [Work log](docs/WORKLOG.md) | Actions taken, checks, decisions, results and outstanding work |
| [Baseline / model card](reports/BASELINE.md) | Measured results and limits of the first new model |
| [Multi-source report](reports/MULTISOURCE.md) | Global data audit, ablations, calibration, fresh-source results and limits |
| [Registered multi-source protocol](docs/MULTISOURCE_PROTOCOL.md) | Selection rules fixed before fresh-source scoring |
| [Multi-source runbook](docs/MULTISOURCE_OPERATIONS.md) | Cache preparation, freezing, one-time evaluation and portable exports |

## Development

```powershell
uv sync --frozen --extra dev
uv run --no-sync ruff check src tests scripts
uv run --no-sync ruff format --check src tests scripts
uv run --no-sync pytest
```

CI smoke data is synthetic and tests software plumbing only. Real research data and new model
artifacts are ignored by Git. GitHub training uploads reports and the model with finite retention;
it does not publish a service or automatically deploy a model.

Dataset permissions are separate from source-code licensing. The UQ collection permits academic
research with citation and requires author agreement for commercial use. See the exact
[publisher terms](https://www.cyber.uq.edu.au/project/machine-learning-based-nids-datasets).
The repository has no newly assigned source-code license; the owner must choose one before an
open-source release. Do not assume dataset availability grants unrestricted commercial rights.
