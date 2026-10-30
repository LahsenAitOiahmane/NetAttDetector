# Synthetic measurement fixtures

`core.pcap` is a 948-byte, 12-packet deterministic Ethernet capture generated offline by
[make_pcap_fixtures.py](../../../scripts/data/make_pcap_fixtures.py). It contains no captured
user traffic, attack tools or malware. IPv4/IPv6 documentation addresses and fixed payloads
make regeneration independent of a network connection. `expected.json` lists known flow values.

Check bytes with `python scripts/data/make_pcap_fixtures.py --check` from the repository root.
To intentionally regenerate, choose a fresh directory, review the capture and expectations,
then replace the curated fixture explicitly. See the [measurement guide](../../../docs/guides/MEASUREMENTS.md).
