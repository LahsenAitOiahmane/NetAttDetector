# Repository layout and maintenance

The active package lives only under `src/netattdetector/`. Run `netatt` or
`python -m netattdetector` after installing the project. Dependency declarations belong in
`pyproject.toml`; exact resolutions belong in `uv.lock`.

```text
.github/                 CI, bounded research workflows, dependency maintenance
configs/                 Experiment and validation settings
data/                    Local downloaded/processed data; only README is tracked
docs/
  architecture/          Contracts and system boundaries
  guides/                Reproduction and operations
  project/               Audit, decisions, Scrum roadmap and work log
  research/              Data provenance and registered evaluation protocols
reports/                 Reviewed reports, compact JSON evidence and figures
  local/                 Ignored generated/local reports
scripts/
  data/                  Synthetic fixtures and data preparation helpers
  reports/               Render saved results; never refit models
  research/              Additional experiment diagnostics
src/netattdetector/      Installed implementation
tests/                   Software checks and small reviewed synthetic fixtures
artifacts/               Ignored model/run outputs and retained workflow downloads
```

The cleanup removed 2024 model modules and utilities, the unused data loader, redundant entry
wrappers/dependency shim, generic editor launch settings, the obsolete CSS directory sketch,
tracked bytecode, five obsolete model binaries, logs and eleven empty/header-only captures.
The original inventory and its limitations remain in [the audit](AUDIT.md); historical files
remain available at Git revision `f1b2d59`. Git history was not rewritten. Existing raw archives,
corrected caches, frozen models and holdout receipts under ignored local directories were kept.

`.gitignore` protects environments, private/tabular captures, compressed downloads, trained
models, caches, credentials and local output folders. Curated synthetic capture/CSV fixtures
are permitted only under `tests/fixtures/`. Reviewed JSON evidence and report PNGs are tracked.
An ignore rule does not untrack a previously committed file; cleanup explicitly removed them.
Do not add a new output directory without updating this policy and the hygiene check.

`.gitattributes` fixes text checkouts to LF across platforms and treats capture/image/model
formats as binary. This prevents newline conversion from changing configuration/fixture hashes.
Historical run hashes describe the original bytes; do not rewrite them after a dependency upgrade.

CI checks repository hygiene and local documentation links in addition to lint, formatting,
tests and synthetic smoke training. Source and documentation may contain old filenames as
historical evidence; they must not refer to deleted files as current executable entry points.
