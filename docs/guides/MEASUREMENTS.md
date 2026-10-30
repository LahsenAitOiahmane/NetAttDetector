# Offline flow measurement lab

The chosen measurement candidate is [NFStream](https://www.nfstream.org/docs/api), pinned to
6.6.0 in the optional Linux `measurement` extra. Its documentation exposes directional counters,
timestamps/durations and byte-accounting choices. We verify an explicit subset on generated
PCAPs rather than assuming equivalence to the old benchmark exporter.

The adapter accepts only existing local PCAP/PCAPNG files up to 16 MiB. It never accepts an
interface name. It uses one meter process, IP-layer byte accounting, a two-second idle timeout,
1,800-second active timeout, no tunnels, no application dissection, no system visibility and no
post-mortem statistical analysis. A 10,000-flow export limit causes an error rather than reporting
a potentially partial file as complete. These are lab limits, not a live sensor design.

## Reproduce on Linux

```bash
uv sync --frozen --extra dev --extra measurement
uv run --no-sync python scripts/data/make_pcap_fixtures.py --check
uv run --no-sync netatt export-pcap --pcap tests/fixtures/pcap/core.pcap --output artifacts/measurements/export.json
uv run --no-sync netatt verify-measurements --export artifacts/measurements/export.json --expected tests/fixtures/pcap/expected.json --output artifacts/measurements/verification.json
uv run --no-sync netatt measurement-contract --output artifacts/measurements/contract.json
```

Use a fresh output directory for a repeat. The GitHub **Offline flow measurement lab** workflow
runs the same commands and retains evidence for 14 days. This project does not install Windows
capture drivers. NFStream's [Windows installation requirements](https://github.com/nfstream/nfstream#how-to-get-it)
are separate; the currently verified adapter is Linux-only. Windows can regenerate/check fixture
bytes, inspect provenance, validate plans and compare downloaded measurement JSON without drivers.

## Contract and fixtures

`nfstream-core-v1` has 11 numeric fields: ports, IP protocol/version, directional packets/bytes,
and bidirectional/directional durations. IPs and event timestamps remain metadata. The native
counter totals and timestamps are included as extra measurement-check fields, not model inputs.

The 948-byte fixture contains 12 synthetic packets and five expected flows: bidirectional UDP,
a single packet with zero duration/no reply, a TCP handshake/data/reset sequence, bidirectional
IPv6 UDP, and reuse of a UDP tuple after idle expiry. Known IP lengths include headers but not
Ethernet padding. Expected values are explicit and independent of NFStream's export. The generator
uses only Python's standard library and sends no packets. The comparison checks all 85 expected
fields, input digest, exporter version and exact settings. A wrong byte layer, unit, flow identity,
missing flow, duplicate or counter value fails.

The fixture does not validate active-timeout boundary behavior, fragmentation, retransmission
accounting, VLAN/tunnel traffic, TCP window scaling, application IDs, live packet loss or capture
queue overload. Extend the lab with independent known expectations before supporting those cases.

**Passing this lab does not make an NFStream export compatible with an existing UQ model.**
Ten UQ fields have conceptual counterparts; their benchmark semantics have not been proved equal.
Thirty other fields have no mapping in this adapter, including TTL/application/statistical fields.
No missing values are filled and no UQ-named prediction CSV is emitted. The existing feature
validator rejects these records. A compatible labelled re-export and a newly trained contract
are required before this measurement path can produce attack predictions.

NFStream's own LGPL license, the project's source-code license decision, and dataset use terms
are separate matters. This sprint adds a dependency and an offline lab, not a licensed product
release, live capture service or network-protection deployment.
