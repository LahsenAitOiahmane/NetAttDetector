# Development PCAP and label-linkage sprint

Measured 31 October 2026. **Acquisition and streaming label auditing work; these captures do not
meet the support requirements for a new detector.** No model was fitted and no detection accuracy
is claimed. The small-office/home scope and 50 reviewed alerts/day remain planning inputs.

## Inputs and reproducibility

PR #5 was merged; this sprint starts from `c38b80b`. Original UNSW/ToN publisher pages still
describe raw captures and ground truth, but their download links redirected this session to
Microsoft sign-in. Registered IoT-23 as an additional **development** lineage at `a9534b4`
before inspecting capture bytes or label logs. CICIDS2017 remains reserved and unacquired;
CSE remains observed evaluation and the merged UQ collection remains excluded.

Retrieved three complete PCAP/label/README bundles, **12,469,035 bytes** total, first on GitHub
and then on this Windows laptop under ignored `data/raw/iot23-development/`. All nine local
file digests match the pinned inputs. Downloaded data is present locally; NFStream processing
ran on Linux without installing Windows capture drivers. Local parser counts match the cloud.
See [local evidence](evidence/development-captures/local-acquisition.json) and the
[runbook](../docs/guides/CAPTURE_LABELS.md).

The first [run, 41235768705](https://github.com/LahsenAitOiahmane/NetAttDetector/actions/runs/41235768705)
at `65bb5c7` successfully acquired the bundles, then correctly failed at the materializing lab's
10,000-flow limit. Retained the observed hashes and introduced a separate streaming audit with
one meter and a 100,000-flow hard stop; the original lab limits remain unchanged. PCAP input
size remains capped at 16 MiB. This was a resource-limit adjustment before scoring a model,
not a relaxation of detection-quality requirements.

The [pinned replay, 41235993410](https://github.com/LahsenAitOiahmane/NetAttDetector/actions/runs/41235993410)
at `a80dd43` verified all input hashes and completed all captures. Its aggregate evidence,
receipts and frozen configuration are in [the evidence directory](evidence/development-captures/summary.json).
Original fixture regression also passed all 85 field comparisons on this revision.

## Measured coverage

| Scenario | Exported flows | Candidate matched flows | Distinct matched benign / attack Zeek connections | Audit seconds | Sampled tree RSS sum MiB |
|---|---:|---:|---:|---:|---:|
| 20-1 | 19,777 | 18,456 | 1,892 / 8 | 1.396 | 146.5 |
| 21-1 | 20,328 | 18,917 | 1,931 / 8 | 1.412 | 157.5 |
| 42-1 | 5,483 | 4,240 | 3,191 / 5 | 0.441 | 158.4 |
| Total | 45,588 | 41,613 | 7,014 / 21 | 3.249 | — |

These are **linkage counts, not prediction metrics**. The 10,921 publisher Zeek records include
36 explicitly malicious connections before matching. The two-second NFStream idle timeout
expands a small number of long C&C connections into thousands of segments: 16,343 matched
attack segments in 20-1 link to just eight distinct Zeek UIDs; 21-1 has 16,340 segments from
eight. Treating those segments as independent attacks would inflate support and permit leakage.
UIDs are capture-scoped, and even distinct UIDs are not proof of statistically independent events.

Excluded from candidate labels: 3,116 ambiguous flows, 806 with unknown label/duration, two
partial intervals and 51 unsupported-protocol flows. They remain unlabelled. No omitted row is
converted into a benign example. Whole-scenario malware names never become per-flow labels.

The audit joins an orientation-independent TCP/UDP five-tuple to exactly one overlapping,
containing labelled interval, with one millisecond of quantization tolerance. Decimal timestamps
are retained while comparing. Unknown duration can block subsequent same-tuple matching.
Duplicate candidates remain ambiguous even when labels agree. This conservative rule prevents
guessing; it does not independently validate publisher labels or packet-level ground truth.

The resource monitor is now a lightweight module, so export paths do not import model-training
libraries just to sample memory. The streaming audit retains identity keys, UID sets, label
indexes and counters, rather than all feature rows. Peak parent RSS was 90.8 MiB; sampled
process-tree RSS sums stayed below 159 MiB. Fifty-millisecond sampling includes label parsing,
exporter imports and matching, excludes download time, can miss brief peaks and double-count
shared child pages. These are small offline measurements, not production load/loss evidence.

## Metadata and release limitations

The pinned READMEs name origin devices RPi-01 (20-1), RPi-05 (21-1), and RPi-04 (42-1).
These publisher names do not verify persistent physical device identity. The 42-1 README also
names generic capture **41** and a **2018** start date, while its PCAP filename and exported
event times are in January **2019**. Keep this discrepancy visible; do not silently repair dates
or use it as a trusted split declaration. No capture manifest with fabricated host identities was
added. The global training plan remains empty and blocked.

Publisher sources: [IoT-23 description and citation](https://www.stratosphereips.org/datasets-iot23),
[Zenodo version record](https://doi.org/10.5281/zenodo.4743746),
[UNSW-NB15](https://research.unsw.edu.au/projects/unsw-nb15-dataset),
[ToN-IoT](https://research.unsw.edu.au/projects/toniot-datasets).
Zenodo's machine-readable metadata lists CC-BY-4.0, while individual mirror READMEs additionally
request authorization. This remains a release-terms discrepancy. The workflow retains only
aggregate provenance/counts, not raw PCAPs, connection logs, endpoint rows or joined feature data.
Resolve the terms before publishing data or a derived model; no commercial clearance is claimed.

## Acceptance and next work

Software acceptance: pinned bounded downloads, source reservation, conservative linkage,
streaming limits, original synthetic fixture parity and retained resource evidence. Detection
acceptance is **not met**: each capture has fewer than 25 distinct matched attack connections,
metadata needs reconciliation, and the selected scenarios cover only a narrow attack domain.

1. Resolve source access/terms and acquire development captures with enough distinct attacks
   and benign activity per planned partition. Keep CICIDS2017 out of fitting/threshold choice.
2. Reconcile capture/device metadata and label timing; investigate unmatched/ambiguous coverage
   and the segment multiplication before choosing a deployment timeout.
3. Extend fixture coverage for the chosen timeout, retransmissions, fragments and ICMP; freeze
   the final measurement/label contract. Do not alias native fields into existing UQ models.
4. Register capture/time/device partitions with UID groups and adequate connection-level
   support; fit a new bounded baseline only after those checks pass.
5. Measure site flow volume/aggregation to translate the 50/day review capacity into an
   operational false-alarm budget before a shadow pilot.

No live traffic, malware execution, traffic transmission, blocking, API service or model promotion
was introduced by this sprint.

Validation: all 83 tests present at the full local run passed in 67.23 seconds with two native
threads. After adding a Linux-only real-export regression, the focused provenance/measurement/
label suite passed 41 tests and skipped that one Linux-only check on Windows. Lint, formatting,
repository hygiene, documentation links and deterministic PCAP-byte checks passed. Linux CI
also exercises the 100,000-flow audit settings against the same five known flows/85 fields.
