# Native error diagnostic operations

The [registered protocol](../research/NATIVE_ERRORS.md) permits three refits of the existing
diagnostic specification, not another search or a replacement validation threshold.

On working Linux, with fresh default directories:

```bash
uv sync --frozen --extra dev --extra measurement
uv run --no-sync python scripts/research/run_native_study.py --diagnostics
```

The acquisition/cohort/study/evidence path options and optional private `--captures` map are the
same as the [baseline runbook](NATIVE_STUDY.md). Existing default directories cannot be reused
as outputs; choose fresh paths. For GitHub:

```bash
gh workflow run measurements.yml --ref YOUR_BRANCH -f native_error_diagnostics=true
```

The workflow retains only aggregate `diagnostics.json`, its configuration, cohort manifest and
verification. A fitted fixed-one-thread model is written privately and is never uploaded. The
baseline freeze and its failed acceptance remain unchanged. Fixed-thread repeatability is
evaluated separately from two-versus-one-thread variation, FPR, DDoS support and data provenance.

Error tables count accepted UIDs by the highest-scoring segment's observed protocol, destination
service bucket, source-port range, packet-count range and duration. Ties keep the first stored
segment. They do not reveal raw endpoints/UIDs or prove a client/service identity, incident cause,
or an incorrect publisher label. These slices inform later measurements/data acquisition.

Public code is MIT; public model/data release and real-network pilot stages remain unavailable
until their separate terms/operating requirements are satisfied. The owner explicitly deferred
pilot traffic selection and requested continued offline research.
