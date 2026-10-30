# Dataset research and acquisition

Verified against publisher pages on 29 October 2026. Prioritize relevant environments, benign
coverage, credible labels, split metadata and reproducible feature definitions over file size.
Large synthetic benchmarks alone do not establish performance on the user's network.

## Priority catalog

Counts are publisher-reported flows, not verified counts until a complete scan finishes.

| Order | Dataset | Published scale | Use in this project | Acquisition |
|---|---|---:|---|---|
| 1 | NF-UNSW-NB15-v2 | 2,390,275 flows | Initial benign/attack model and attack-family slices | `netatt download nf-unsw-nb15-v2` |
| 2 | NF-ToN-IoT-v2 | 16,940,496 flows | Different-environment evaluation with the same NetFlow vocabulary | `netatt download nf-ton-iot-v2` |
| 3 | NF-CSE-CIC-IDS2018-v2 | 18,893,708 flows | Additional enterprise-like testbed | `netatt download nf-cse-cic-ids2018-v2` |
| 4 | NF-BoT-IoT-v2 | 37,763,497 flows | IoT DoS/botnet stress test; highly imbalanced | `netatt download nf-bot-iot-v2` |
| 5 | NF-UQ-NIDS-v2 | 75,987,976 flows | Multi-source experiments with source-held-out evaluation | `netatt download nf-uq-nids-v2` |

The [UQ publisher page](https://www.cyber.uq.edu.au/project/machine-learning-based-nids-datasets)
provides these counts, records and terms. These datasets share an extended NetFlow feature
vocabulary. The merged collection contains constituent datasets: **never train on the merged
collection and call a constituent an independent external test**. Shared schema also does not
guarantee identical distributions, independent sessions, clean labels, or current attacks.

The first downloaded UNSW archive is 33,877,423 compressed bytes and contains a 441,867,785-byte
data CSV plus a feature dictionary. These are measured archive sizes, not estimates. The
downloader records exact sizes and hashes for every completed acquisition; consult the checked-in
baseline report for the final acquisition ledger. No full PCAP collection is needed for this phase.

## Additional and more recent sources

The next phase also downloaded NF-CSE-CIC-IDS2018-v2: 612,059,380 compressed bytes, with
an 18,893,708-row CSV verified by a full scan. The three local archives now total 38,224,479
flows and 814,056,982 compressed bytes (about 776 MiB). CSE was used only after freezing the
registered multi-source model; it is now an observed benchmark, not a fresh holdout for tuning.
See the phase-2 model card and its manifest for exact source/checksum details.

| Source | Why evaluate it | Integration work before training |
|---|---|---|
| [Original UNSW-NB15](https://research.unsw.edu.au/projects/unsw-nb15-dataset) | 2,540,044 original records; documented 175,341/82,332 training/testing partitions | Separate Argus/Bro schema adapter; do not treat as independent from NF-UNSW |
| [CICIDS2017](https://www.unb.ca/cic/datasets/ids-2017.html) | Day/capture separation and widely used reference | CICFlowMeter adapter, duplicate/label audit, train on earlier days and test later days |
| [CSE-CIC-IDS2018](https://www.unb.ca/cic/datasets/ids-2018.html) | Public AWS collection of per-day flow data and raw captures | Select CSV objects first; do not sync the entire bucket on this laptop |
| [CICIoT2023](https://www.unb.ca/cic/datasets/iotdataset-2023.html) | 105-device topology, 33 attacks; distinct IoT behavior | Separate feature adapter and capture/device splits; check imbalance and acquisition size |
| [IoT-23](https://www.stratosphereips.org/datasets-iot23) | Labeled Zeek scenarios; real malware traffic and benign devices | Zeek schema adapter; split by capture/device. Publisher lists 8.8 GB flow-only and 21 GB full archives; defer on current disk |
| [DataSense / CIC IIoT 2025](https://www.unb.ca/cic/datasets/iiot-dataset-2025.html) | More recent industrial testbed with synchronized network/sensor observations | Domain-specific adapter and chronological evaluation; do not mix sensor labels into network features |
| [CIC current catalog](https://www.unb.ca/cic/datasets/index.html) | New releases include specialized 2024–2026 collections | Triage task relevance first. A newer malware-binary dataset is not automatically a better NIDS training source |

These sources are researched candidates, **not all downloaded or currently supported adapters**.
The supported loader accepts the UQ v2 binary `Label`/optional `Attack` schema. Feeding another
dataset to it fails when required fields are absent; renaming columns cannot make units or
feature definitions equivalent. We intentionally do not execute malware samples or download
large raw-binary/PCAP archives to improve a tabular model.

## Acquisition policy

1. Follow the permanent publisher record, inspect terms, feature definitions and expected size.
2. Resolve its public download link. A signed URL expires and is not committed to Git.
3. Check compressed size against `--max-mib` and free disk minus a 2 GiB reserve.
4. Download bounded byte ranges with retries. Partial blocks cannot duplicate appended data.
5. Verify total byte count, ZIP CRC and readable CSV members; atomically publish the archive.
6. Write download time, permanent record, SHA-256, sizes, license note and member list.
7. Scan all labels and create a seeded, bounded sample over the complete data CSV.

Local SHA-256 establishes repeatability against a previous local download. It is **not** a
publisher-authenticated checksum: manifests explicitly record `publisher_checksum_verified: false`.
The implementation streams ZIP members and never extracts archive paths. A feature dictionary
is excluded from automatic member selection; archives with multiple data CSVs require `--member`.
The byte cap covers transfer/storage, not uncompressed input volume or scan time. Automatic
resume across separate CLI invocations is not implemented; failed partial downloads are cleaned.

## Terms and citations

The UQ publisher grants academic research use with citation and asks for author agreement for
commercial use. This is material to any eventual product release. Keep terms with the experiment;
do not label the collection or derived model as commercially cleared without agreement.

For the v2 collection, cite Sarhan, Layeghy and Portmann, *Towards a Standard Feature Set for
Network Intrusion Detection System Datasets*, 2022,
[DOI](https://doi.org/10.1007/s11036-021-01843-0), and the appropriate original dataset paper.
Other sources have their own citation/license requirements; retain their exact publisher record
before redistribution. Source-code licensing and dataset licensing are separate decisions.

## Data-quality backlog

Preserve source/capture/day metadata outside the feature matrix. Quantify duplicate and
conflicting rows globally, not only within the bounded sample. Record minority attack support,
missingness by class/source, timestamp ordering, protocol coverage, invalid units, label timing,
and feature drift. Compare port/TTL/protocol ablations to detect testbed shortcuts. A future
training mixture must hold out one entire source and retain the naturally distributed test set.
