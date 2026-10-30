# Repository maintenance and capture/measurement sprint

Date: 30 October 2026. Scope: repository cleanup, dependency compatibility, source provenance,
offline flow measurement and initial small-office product assumptions. No new real-data model
was fitted, no fresh benchmark flow rows were inspected, and no live capture/deployment occurred.

## Repository and dependency maintenance

Removed the unused 2024 modules/utilities/data loader, redundant entry wrappers and dependency
shim, generic editor configuration, obsolete CSS tree sketch, tracked bytecode, five obsolete
model binaries, logs and eleven empty/header-only capture CSVs. Downloaded archives, corrected
cohorts, model freezes, holdout receipts and reviewed research evidence remain available.
Deleted files are recoverable from Git history; no history was rewritten.

Documentation now has project/research/guides/architecture sections, and helper scripts have
data/research/reports sections. Links and workflows use their new paths. The
[repository guide](../docs/project/REPOSITORY.md) explains the tree. `.gitignore` excludes private
tabular/capture data, model outputs, environments, caches, credentials and local reports; only
reviewed synthetic inputs under `tests/fixtures/` are exempted. CI checks tracked-file hygiene,
local Markdown links and representative ignore behavior. LF text checkout attributes avoid
cross-platform checksum changes from newline conversion; capture/image formats remain binary.

Integrated Dependabot's exact commits from [PR #2](https://github.com/LahsenAitOiahmane/NetAttDetector/pull/2)
into this branch. CPython uses pandas 3.1.0; the optional exporter's upstream PyPy constraint is
represented separately in the lock. The active Windows environment was upgraded after isolated
compatibility testing. [Compatibility evidence](evidence/dependencies/pandas3-compatibility.json)
records matching full/reduced hashes for all 195,894 cached observations, plus 5,000 saved probes
for each existing local/Linux portable export. Maximum score differences were 0 and 7.45e-9;
threshold decisions matched. This is input/inference compatibility, not new quality evidence or
a guarantee that retraining on another dependency version is identical. Historical metrics and
model artifacts retain their recorded environments and are not reinterpreted.

Primary compatibility references: [pandas 3.1 release notes](https://pandas.pydata.org/docs/whatsnew/v3.1.0.html)
and [pandas string migration guidance](https://pandas.pydata.org/docs/user_guide/migration-3-strings.html).

## Provenance and split preparation

The [source-role map](../configs/data/source_roles.json) keeps UNSW/ToN in development, CSE as
observed evaluation and the merged UQ collection excluded from independent testing. CICIDS2017
is registered as reserved/unacquired; only its publisher description was reviewed. No data was
downloaded from that reserved source. Its feature schema cannot be substituted for UQ fields.

The local UNSW/ToN headers contain endpoint addresses but no event timestamps, capture IDs or
session IDs. The [header evidence](evidence/provenance/unsw-header.json),
[ToN evidence](evidence/provenance/ton-header.json) and
[provenance guide](../docs/research/PROVENANCE.md) retain this limitation. Durations, row order and
archive names were not turned into invented timestamps/captures.

The declared capture-plan validator checks source roles/lineage, file hashes, duplicate capture
and input identities, explicit host groups, timezone-aware chronology, a 300-second within-source
purge and declared benign/attack support for each partition and source/partition. Even a valid
declaration does not authorize fitting: row-label linkage and metadata semantics need raw-data
audit. The [current plan](../configs/data/capture_plan.json) is empty and
[fails its gate](evidence/provenance/capture-plan.json). Thus **NAD-015's real capture-held-out
benchmark remains unfinished**; the preparation/validation tooling is implemented.

## Offline exporter measurements

The chosen candidate is NFStream 6.6.0, pinned in an optional Linux extra. It is configured for
one meter, IP-layer bytes, two-second idle/1,800-second active timeouts, and no application
dissection, tunnels or system inspection. The wrapper accepts local PCAP/PCAPNG files only,
caps input at 16 MiB and errors at its 10,000-flow limit. It cannot accept an interface name.

The [Linux workflow](https://github.com/LahsenAitOiahmane/NetAttDetector/actions/runs/41108865671)
at source commit `2c28e22` exported the 948-byte synthetic fixture and **passed all 85 expected
field checks on five flows**. The 12 packets cover bidirectional UDP, a no-reply/zero-duration
flow, TCP handshake/data/reset, IPv6 UDP and UDP tuple reuse after idle expiry. This is a
measurement regression, not attack-detection evidence or a throughput benchmark.

Measured export-loop wall time: 0.093 seconds; sampled parent-process RSS: 179.3 MiB. Imports and
the meter child's memory are outside this measurement, so it is not an end-to-end or total-process
memory bound. Evidence: [native export](evidence/measurements/linux-export.json),
[Linux comparison](evidence/measurements/linux-verification.json),
[Windows comparison of the same export](evidence/measurements/windows-verification.json), and
[contract matrix](evidence/measurements/contract.json). Both platforms verified the same canonical
fixture/expectation digests. NFStream itself was not run on Windows and no capture driver was installed.

`nfstream-core-v1` defines 11 measured fields and separate endpoint/time metadata. Ten UQ fields
have conceptual counterparts, but none has proven nProbe equivalence here; 30 have no adapter
mapping. **The existing UQ models cannot consume these records.** No values are fabricated and
no UQ aliases are emitted. A separately labelled re-export, preprocessing and model are required.
The [measurement guide](../docs/guides/MEASUREMENTS.md) records exact commands and missing cases,
including active-timeout boundaries, fragmentation, retransmissions, VLAN/tunnels, window scaling
and live overload/drop handling.

## Product scope and next work

The user selected a small office/home network and capacity for up to 50 reviewed alerts/day.
The [threat model](../docs/project/THREAT_MODEL.md) and
[planning record](../configs/deployment/small-office.json) distinguish reviewed incidents from
flagged flows. Traffic volume, aggregation/suppression, observation point, retention and pilot
quality SLO remain unspecified. For illustration, 0.5% FPR at 100,000 benign flows/day is 500
flagged benign flows/day before aggregation, exceeding the initial review capacity.

Next steps are bounded development-capture/label acquisition from observed source lineages,
preservation of source/capture/time/device metadata, validated label joins and wider measurement
fixtures. Keep the reserved source out of tuning. Only then build and evaluate a model for the
verified contract. Capture-aware detection quality, an operational alert budget, a live bounded
sensor, shadow validation and production release remain later gates.

Validation so far: the prior 42-test suite passed with pandas 3.1.0; the integrated 59-test suite
passed locally; 18 focused provenance/measurement tests passed after strengthening source support,
including the added pooled-support counterexample. Linux CI and the actual exporter lab passed.
Lint, formatting, fixture-byte checks, repository hygiene and local documentation links pass.
The review is [draft PR #5](https://github.com/LahsenAitOiahmane/NetAttDetector/pull/5), which
includes the dependency branch from PR #2. No main-branch merge or production promotion was performed.
