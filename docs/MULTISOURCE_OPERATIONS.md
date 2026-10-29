# Multi-source experiment runbook

Use `configs/multisource.json` and the registered version-2 protocol. UNSW and ToN are now
development data. CSE-CIC-IDS2018-v2 is the designated independent source, never a tuning input.

```powershell
uv sync --frozen --extra dev --extra boosting
uv run --no-sync netatt download nf-cse-cic-ids2018-v2 --max-mib 768
uv run --no-sync netatt prepare-multisource --output data/processed/my-development
uv run --no-sync netatt train-multisource --prepared data/processed/my-development --baseline artifacts/baseline-local/model.joblib --output artifacts/my-multisource
uv run --no-sync netatt evaluate-frozen --run artifacts/my-multisource --prepared data/processed/my-development --data data/raw/nf-cse-cic-ids2018-v2.zip --baseline artifacts/baseline-local/model.joblib --output artifacts/my-multisource/holdout.json
```

The preparation and training directories must be fresh. Preparation reads all development rows
once, audits full/reduced fingerprint groups on disk, and caches a bounded numeric cohort. The
temporary raw hash partitions are deleted after sorting; sorted global indexes remain for overlap
checks. The cache uses NumPy archives with `allow_pickle=False`. No raw addresses are retained.
The original baseline path above is the locally trained phase-1 model; if absent, reproduce it
with the phase-1 benchmark command before any fresh holdout evaluation.

Training fits at most nine classifiers sequentially on two threads. Each compares no calibration
with sigmoid calibration fitted on a disjoint partition. The complete leaderboard reports each
source separately; pooled precision cannot establish constraint feasibility. Search has a one-hour
start-new-fit deadline. A fit already started can finish after that deadline, which is recorded.

`freeze.json` binds the selected model, protocol, prepared audit/cache, source hashes, global index
checksums, exact partition fingerprints, model-selection results, and baseline artifact hash. Keep
it with the model. `evaluate-frozen` checks those bindings and exclusively creates
`holdout-access.json` before parsing holdout records. It refuses a second access for the same
freeze even if a different output filename is supplied. If an evaluation fails, preserve the receipt
and investigate before intentionally creating another run. Creating another run does not restore
a scientifically untouched holdout once its outcomes have influenced development.

Full-feature development matches are excluded from the independent source. Reduced-feature
matches and ambiguous reduced groups are reported but retained: two distinct full observations
can legitimately have the same reduced measurements. Both frozen models score identical rows.
Do not fit, calibrate or select thresholds on the new-source data, and do not discard hard attack
families to improve the result. Report natural source counts and every cleaning exclusion.

Version-2 model bundles carry the selected ordered subset of the base feature contract and can
contain a frozen calibration wrapper. The normal `netatt predict` command supports both versions;
it restores preprocessing and enforces the requested thread budget through calibration wrappers.

Optional report/figures:

```powershell
uv run --frozen --extra analysis python scripts/report_multisource.py --prepared data/processed/my-development --run artifacts/my-multisource --holdout artifacts/my-multisource/holdout.json --output reports/local/MULTISOURCE.md
```

The existing GitHub training workflow also has a `multisource` experiment choice:

```powershell
gh workflow run train.yml --ref codex/multisource-generalization -f experiment=multisource
```

It regenerates development caches and the original fixed baseline, then follows the same
freeze-before-test sequence. Reproducing an already evaluated protocol is a reproducibility check,
not a second independent experiment. The job uses two threads, a 90-minute overall limit, and
retains compact evidence/model artifacts for 14 days. It does not upload the large raw archives
or global hash indexes. Preserve accepted evidence in the repository and regenerate indexes from
the checksummed inputs when needed. A completed workflow does not imply a deployment pass.
