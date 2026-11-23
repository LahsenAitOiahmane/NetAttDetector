# Kitsune development acquisition: new data with explicit coverage limits

Measured 23 November 2026. A complete **75.1 MB Mirai capture and 2.3 MB packet-label file**
are privately available and pinned. Full Windows and Linux audits agree on all
**764,137 packets and 764,137 binary labels**. This adds a separately sourced development
capture with benign traffic. It does not yet provide a labeled native-flow cohort or improve
the detector's measured accuracy: no model was fitted or scored in this phase.

## Registration and complete inputs

The [protocol](../docs/research/KITSUNE_DEVELOPMENT.md) and development source role were
committed at `033cf53` before raw packet or label access. Publisher metadata came from the
[UCI record](https://archive.ics.uci.edu/dataset/516/kitsune+network+attack+dataset), and the two
files came from the [author's Kaggle mirror](https://www.kaggle.com/datasets/ymirsky/network-attack-dataset-kitsune).
Public GET requests succeed; the earlier HEAD responses did not establish file availability.
No credentials, 1.37 GB derived-feature CSV, whole dataset archive or malware binary was used.

| Complete input | Bytes | Observed SHA256 |
|---|---:|---|
| Mirai capture | 75,075,844 | `3bebe40dbf24546be36880f89908e315f8f858c8256aa00de0b3bbab8c3c69e6` |
| Packet labels | 2,292,409 | `190d5cc70c35e11000f98f5e508d0d5e50eed82c0ea3acd1abefda8d18a95817` |

Input hashes and the [acquisition receipt](evidence/kitsune-development/acquisition.json) were
committed at `12c2aed` before the packet/label audit. They establish repeatability against
these observed originals, not publisher-authenticated checksums. Subsequent acquisition
and audits verify the independently committed pins. The downloader restricts HTTPS hosts,
response types, transfer/expanded bytes, CRC for any ZIP wrapper, deadlines and disk reserve;
it refuses overwrites and preserves pre-existing partial files.

## Packet labels and format

Despite its `.pcap` suffix, the capture is **PCAPNG with Ethernet enhanced-packet blocks**.
The classic-only reader first refused its magic. The existing bounded block reader was
extended at `6c529ed` with a streaming packet callback and validated per-interface clocks;
the classic reader's behavior remains separate. Decimal micro/nanosecond clocks and integer
timestamp offsets are supported; other packet blocks or clock representations fail explicitly.

The source label file is a one-column binary vector. Numeric initial values are consumed as
data, never guessed headers. Cardinality is exact under the publisher's one-label-per-packet
documentation, including unsupported and truncated packet records. This checks structural
alignment, not independent ground truth for each alleged attack.

| Structural packet status | Benign label | Malicious label |
|---|---:|---:|
| Fully decoded TCP/UDP | 84,088 | 68,583 |
| Non-IP Ethernet | 8,746 | 556,455 |
| Other IP transport, outside this structural decoder | 27,958 | 16,757 |
| Capture-truncated | 829 | 721 |
| **Total** | **121,621** | **642,516** |

**86.61% of malicious labels attach to non-IP Ethernet records.** The current IP-flow model
cannot represent those packets. Do not call this an equivalent 642,516-example native attack
cohort, discard the non-IP rows from the denominator without disclosure, or infer that every
such frame is an ARP attack. A separate layer-2 detector needs its own threat/measurement and
label evidence. The structural decoder's other-transport category is also distinct from
NFStream support: ICMP may be measurable even though this TCP/UDP decoder leaves it unparsed.

All 1,550 capture-truncated records retain their labels. Header sufficiency and native exporter
behavior on such records still require explicit auditing; missing packet bytes are not restored.
There are 173 timestamp regressions, so timestamp sorting must not replace original packet order.

Labels have exactly one transition: packets 1–121,621 are benign, and packets 121,622–764,137
are malicious. The observed benign maximum timestamp is `1540450841432895000` ns; the malicious
minimum is `1540450841473288000` ns, leaving a 40.393 ms gap. These aggregate ranges may support
a conservative callback-time labeling experiment, after registering and validating it. The
time/phase/ordinal itself must never become a model feature. A single phased capture cannot
by itself provide independent device/capture holdouts or a chronological both-class benchmark.

## Repeatability and resources

[Linux run 45241955600](https://github.com/LahsenAitOiahmane/NetAttDetector/actions/runs/45241955600)
at `e6d30f0` reacquired both complete files with identical hashes and reproduced every
non-resource audit field from Windows. Its suite passed 67 checks. A preceding run failed
in a new test because of a missing test import; the import was corrected before the successful
measurement run. No failed run is presented as completed data evidence.

The Windows audit took **5.42 seconds / 24.8 MiB sampled RSS**; Linux **6.37 seconds / 23.4 MiB**.
These include full packet/label traversal and aggregate counting, excluding input-hash checking,
download and imports. Windows acquisition took 124.19 seconds / 31.4 MiB. No entire packet/label
table was materialized. Sampled RSS is not a hard memory bound. Full local software checks
passed 277 with seven platform/optional skips; CI results are retained separately by commit.

Read the complete [local audit](evidence/kitsune-development/packet-audit.json) and
[cloud audit](evidence/kitsune-development/cloud/packet-audit.json), and follow the
[runbook](../docs/guides/KITSUNE_DEVELOPMENT.md). Kitsune is development data; CESNET remains
observed evaluation and CICIDS2017 reserved/unacquired. More native benign and missing-family
coverage, a defensible development split and a fresh frozen evaluation still remain.

## Attribution

Y. Mirsky, T. Doitshman, Y. Elovici and A. Shabtai, *Kitsune: An Ensemble of Autoencoders for
Online Network Intrusion Detection*, NDSS 2018. Cite the
[UCI dataset record / DOI 10.24432/C5D90Q](https://doi.org/10.24432/C5D90Q), licensed CC-BY-4.0,
and the author's mirror. This project's MIT source-code license is separate from data terms.
