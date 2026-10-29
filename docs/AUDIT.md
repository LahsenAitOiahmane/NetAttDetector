# Repository and hardware audit

Audit date: 29 October 2026. Original reference: Git commit `f1b2d59` on `main`.

## Inventory and scope

Reviewed every tracked Python module, README, requirements, launch configuration, architecture
comment (`map.css`), CSV inventory/headers, log structure and training/evaluation excerpts, model
file sizes, Git history/remotes, and available hardware. Historical joblib files were inventoried
without deserializing them; they are not reproducible evidence without the missing inputs,
preprocessing state, and original dependency environment. Logs contain repeated runs; reports
below describe observed entries, not a claim that every historic experiment was valid.

The repository contained 7 model implementations: bagged Random Forest, XGBoost, PCA/SVM,
Extra Trees, scikit-learn MLP, Isolation Forest, and One-Class SVM. The latter two were disabled
in the final training script. Five joblib models were tracked. Eleven capture CSV files were
tracked: ten empty and one header-only. Neither `assets/train_net.csv` nor `assets/test_net.csv`
was present. There is no usable original training dataset in this checkout.

## Findings in priority order

| Severity | Evidence in original tree | Implication | Action |
|---|---|---|---|
| Critical | `app/app.py::extract_features` set flow duration/outbound counts to zero and used packet size as flow bytes | Training and serving observations have different meaning | Retire that inference path; implement exporter parity before live use |
| Critical | `data_loader.py` fitted a scaler, returned arrays, and never persisted the scaler | Saved estimators cannot reproduce training preprocessing | Save an entire fitted pipeline and named contract |
| Critical | Observed classification reports at log lines 285 onward contain DoS 299, Malware 2, Port Scanning 3005, no benign support | Reported 1.00 accuracy cannot establish benign/attack discrimination or false-positive rate | New binary target with explicit benign examples and confusion matrix |
| High | `ANOMALY` remained among inputs; provenance/meaning unavailable | Possible target proxy; cannot establish leakage conclusively without data | Exclude unverified labels/proxies by an allowlist |
| High | A test CSV was loaded but never used for reported evaluation | No untouched final holdout | Distinct training, validation and final test partitions |
| High | Random row split with no duplicate, host, session or time audit | Related observations can occur across splits | Deduplicate sampled model inputs; add group/time splits and external tests |
| High | Error handling could continue with undefined `train_df`, or return `None` | Failures may appear to be successful runs | Fail with explicit errors and nonzero CLI status |
| High | Feature ordering hard-coded independently in app and loader | Positional predictions can be silently wrong | One shared schema used in training and inference |
| Medium | Bagging wrapped 100-tree forests ten times | Up to 1000 trees without demonstrated value | Compare capped single estimators first |
| Medium | MLP/Extra Trees searches used `n_jobs=-1`, accuracy scoring and refitted winners | Resource contention and weak objective on imbalanced data | Two threads, fixed baselines, constrained validation selection |
| Medium | PCA fixed at 20 components; RBF SVM and One-Class SVM scale poorly | Fragile on changed feature counts and large datasets | Remove from default training; research only with evidence |
| Medium | Anomaly models emitted Normal/Anomaly against multiclass attack labels | Their zero scores were not a valid task-aligned comparison | Map tasks consistently; benign-only novelty models as separate experiments |
| Medium | TensorFlow was required but unused; PyShark not declared; old incompatible pins | Installation is heavy and not reliably reproducible on current Python | Minimal package, optional boosting, frozen uv lock |
| Medium | Binary models, bytecode, logs and capture files tracked | Repository noise and potential future traffic metadata disclosure | Ignore future outputs; retain original historical evidence for this audit |
| Medium | No tests, CI, data/model cards or release criteria | Changes cannot be evaluated consistently | Add focused tests, workflows, experiment records and Scrum backlog |

The log supports a 3306-row validation example and only two malware observations. It does not
prove the entire original dataset was that small. It also does not prove the `ANOMALY` field
caused the perfect scores; that is a leakage hypothesis which cannot now be verified.

## Hardware measured

| Resource | Observation | Working budget |
|---|---|---|
| CPU | Intel Core i5-1135G7 @ 2.40 GHz, 4 physical / 8 logical cores | 2 training/native threads |
| RAM | 16,905,965,568 bytes, 15.74 GiB | Start with 100k–200k flows; keep ample OS headroom |
| Available RAM | Variable; about 2.4 GiB during one development check | Avoid parallel experiments; reduce sample/chunk sizes if needed |
| Free disk | 15.3 billion bytes initially, about 14.25 GiB | Compressed CSV archives; keep ≥2 GiB reserve |
| Python | Global 3.13.5; a separate 3.12 interpreter also available | Isolated `.venv`, Python 3.13 for benchmark |
| GPU | Not required or relied on | CPU histogram boosting and bounded forests |

Working budgets are design choices, not measured throughput guarantees. `netatt doctor` reports
current memory/disk availability. Each real experiment records its own process RSS and timings.

## Compatibility

Legacy model modules remain for historical reading, outside the installed `src/` package.
The two executable entry points now use the supported CLI. Original code can be recovered from
the reference commit. Do not mix legacy artifacts with new models or silently populate missing
flow fields with zeros. The old multiclass models and new binary model solve different tasks;
their accuracy numbers are not directly comparable.
