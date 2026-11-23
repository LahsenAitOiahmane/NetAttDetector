# Registered Kitsune development acquisition

NAD-055 development expansion, registered 23 November 2026 before reading raw packet/label
bytes. The [UCI publisher record](https://archive.ics.uci.edu/dataset/516/kitsune+network+attack+dataset)
describes nine captures with original packets, one binary label per packet and 115 derived
packet/context statistics. Only publisher metadata and file-list sizes have been inspected.
The [author's Kaggle mirror](https://www.kaggle.com/datasets/ymirsky/network-attack-dataset-kitsune)
lists a 75,075,844-byte Mirai PCAP and 2,292,409-byte label CSV. Register the entire Kitsune
lineage as development now; it cannot become a fresh publication test after inspection.
CICIDS2017 remains reserved and unacquired; CESNET remains observed evaluation.

## Bounded acquisition, not fitting

Acquire only the two complete Mirai files named in `configs/data/kitsune-development.json`,
through the public author-mirror file API. No credentials are required by the public file list;
if file retrieval requires authentication, report that limitation without bypassing it. Permit
HTTPS redirects only to the documented service's public download storage hosts. Never print
temporary signed URLs, tokens or file contents. Refuse HTML/login responses and noncomplete
HTTP statuses. A HEAD failure alone does not prove the GET file route is unavailable.

Cap transferred bytes at 96 MiB for PCAP and 4 MiB for labels, expanded input at 96 MiB total,
and leave a 2 GiB disk reserve. The service may wrap a requested file in ZIP: require exactly
the intended single member, valid CRC and the metadata-pinned expanded byte count. Do not
extract paths, links or additional members. Use fresh private paths, streaming checksums and
byte/deadline bounds. Preserve any error; do not overwrite previous acquisitions or partials.
Observed SHA256 pins are repeatability evidence, not publisher-authenticated checksums.

Commit an aggregate receipt and input hashes before measuring labels or native features.
Do not acquire the 1.37 GB packet-feature CSV or whole multi-gigabyte dataset archive. This
phase adds compatible source material; the large dataset's entire volume is not needed to
answer the current feature/label question on the user's small PC.

## Label/packet audit before native preparation

Independently parse the complete classic-PCAP record structure and the complete label CSV.
Verify exact packet/label cardinality, header/index semantics, binary values, timestamp order,
captured versus original lengths, supported protocols and label distribution. Validate a
constant-memory or disk-backed audit against controlled synthetic fixtures. Preserve labels
for unsupported, truncated and ambiguous packets rather than treating them as benign.
The author's demo statement about an initial benign prefix is context, not a replacement
for per-packet labels. Do not infer a connection label from the capture's Mirai name.

Packet labels cannot be assigned to NFStream flow segments by row number. First register a
packet-to-exporter linkage rule that handles omitted protocols, timestamp precision, direction,
fragments, duplicate packets, truncation and mixed-label flows. Require controlled callback
parity and full coverage accounting before deriving any labeled native cohort. All traffic
contributes causal history irrespective of labels. No source identifiers, endpoint identities,
packet labels, sequence numbers or time indices may enter model features.

The 115 published packet features have different units and aggregation from the native
27/36-field contracts. Do not rename or zero-fill them into those models. A later model study
must register development capture roles, leakage-safe grouping/splits, sufficient benign and
attack-family support and a new frozen test plan. This acquisition alone authorizes no fit,
threshold change, deployment or live pilot.

## Terms and attribution

The UCI record lists CC-BY-4.0. Attribute Yisroel Mirsky, Tomer Doitshman, Yuval Elovici and
Asaf Shabtai, *Kitsune: An Ensemble of Autoencoders for Online Network Intrusion Detection*,
NDSS 2018, and [the dataset DOI](https://doi.org/10.24432/C5D90Q). The
[author implementation](https://github.com/ymirsky/Kitsune-py) is a separate MIT code release.
Retain original data privately; publish aggregate evidence, not real packet/feature rows.
