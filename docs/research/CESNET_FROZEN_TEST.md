# Registered CESNET-CC25 frozen recall test

NAD-054, registered23November2026 before reading archive bytes or computing source features.
The [publisher record](https://zenodo.org/records/16752462) and its public API identify32 C&C
sample captures, six botnet families, a43,421,271-byte raw archive and CC-BY-4.0 terms.
Publisher metadata was inspected; raw captures and IPFIX/label rows were not. Keep
CICIDS2017 reserved/unacquired. CESNET is a new independent acquisition lineage, not verified
independence of every device, attack infrastructure or malware lineage from IoT-23.

## Freeze and no-fitting rule

Evaluate only the [already committed freeze](../../reports/evidence/context-study/freeze.json)
from NAD-053, model SHA256
`a504167cf9dbd384c17b192d2f1209714c8f4ef5c27e6b892b64fcbe04e19cc7`, threshold
`.9999190519374116`, statistics27 control inside the context36 input contract, one prediction
thread. Verify bytes before any joblib load. No fitting, recalibration, parameter/model search,
validation threshold adjustment or endpoint-based decision rule. The model remains rejected
on development criteria regardless of the external result. Record all outcomes and errors.

## Bounded complete acquisition and measurements

Pin URL/size/publisher MD5, metadata-derived sample inventory and limits in
`configs/data/cesnet-frozen-test.json` before acquisition. Verify archive size/MD5 and record
SHA256. Inspect a bounded tar inventory without extracting links, devices or paths; never
execute content. Select exactly one original complete capture per registered sample, excluding
the publisher's overlapping `-train`/`-test` copies. Freeze that exact member inventory before
export. Reject ambiguous/missing samples. Stream selected members to fresh private paths, with
128MiB per-capture and512MiB total expansion bounds and2GiB free-disk reserve.

Archive-header inspection after acquisition found AppleDouble `._` metadata forks and both
hyphenated and underscored split-copy suffixes. Exclude all those explicitly. Some complete
released captures use `-filtered.pcapng` filenames: retain the one unsplit file supplied by
the publisher, and describe it as the full released C&C capture rather than an unfiltered
network recording. This naming clarification precedes measurement/scoring and does not depend
on labels or predictions. The first conservative inventory attempt refused ambiguity; it did
not extract or score anything.

Export complete captures using pinned NFStream6.6.0/core/statistical/context settings already
fixture-validated, one meter/capture, original direction, idle2s/active1800s, 250k flow refusal
bound and100k context-event refusal bound. Preserve every supported exported flow and causal
history before applying label eligibility. Never reset state per scoring batch or omit flows
from history based on labels. Verify235 fixture values locally before real export. Use private
numeric SQLite rows, no endpoint addresses in retained features. Bound score batches to1,024.
Report wall time/50ms sampled tree RSS and exporter versions; no hard-memory or live-load claim.

## Labels, units and uncertainty

The publisher describes raw botnet C&C traffic only. Use its registered C&C IP/port in either
direction as a **label eligibility audit**, outside the model: flows touching that exact endpoint
are eligible positive C&C observations; other flows remain unknown, never benign. Preserve
asterisk annotations on public endpoint metadata and disclose their unresolved meaning.
Report coverage by capture/family and both total exported and eligible packet/flow counts.
If coverage is incomplete, the recall applies only to the endpoint-matched subset. Never use
these addresses/metadata as model features or to override a model decision.

Report positive-flow detection counts/recall per capture and malware family, pooled eligible
flow recall, and unweighted mean per-capture recall. These are NFStream segments, **not Zeek
UID connections or independent attacks**; do not equate publisher IPFIX flow counts to native
flows. The source has no compatible benign packet captures: precision, FPR, balanced accuracy,
ROC-AUC, PR-AUC and deployment acceptance remain unassessed. Do not publish misleading100%
precision/accuracy from positive-only labels. All unknown/unsupported coverage remains visible.

Use10,000 bootstrap resamples of the observed capture-level recall values with seed20261123
for the macro-recall percentile interval; explicitly condition on these32 captures and their
dependent families/infrastructure. Families with zero eligible flows are unassessed and do
not silently disappear from coverage. No per-flow binomial confidence claim. Publish aggregate
files only, not real numeric rows, endpoint traffic logs, model bytes or keys.

After scoring, mark this source observed evaluation. A later training expansion may use it only
after registering that role change and reserving another fresh test. Data from32 captures
increase diagnostic evidence, not approved production accuracy or the deferred pilot's scope.
The raw archive remains privately available for reproducible offline research with attribution.
