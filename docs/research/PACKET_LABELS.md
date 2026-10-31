# Registered packet-backed label validation

Registered 31 October 2026 after the expanded support study, before independent packet-level
inspection. Uses only the pinned IoT-23 development captures 8/34/3. CICIDS2017 remains reserved.
The [configuration](../../configs/data/packet-evidence.json) defines a candidate rule; the original
`iot23-interval-v1` rule and frozen experiments are retained.

The label-log inspection found 14,260 missing-duration DDoS records with zero declared packets,
and 121 with one declared packet. Other excluded labels often declare one packet. Do not equate
missing duration with zero duration. Establish an exact packet witness first.

An eligible point witness requires all of the following:

- The label declares exactly one directional packet and consistent IP-byte counters.
- Exactly one structurally decoded TCP/UDP packet has the same tuple and exact nanosecond
  timestamp. There is no nearest-time search or extra timestamp tolerance.
- Direction and IP bytes agree, and no other eligible label UID claims that packet. Include
  known-duration one-packet claims in collision checks.
- A newly point-linked NFStream flow itself contains one packet, zero duration, the same
  first-observed direction/IP bytes, and the packet's timestamp quantized to milliseconds.

Only those witnessed missing ends become points in the candidate index. Unknown/unproven ends
remain blockers. Known intervals retain their original semantics; no packet-tail extension is
inferred. Compare baseline and candidate on the same flow stream, record every gain/loss/change,
and preserve all prior accepted matches. Any loss/change or saved-baseline discrepancy fails
the acceptance screen. Report unique UIDs, segment counts and detailed attack coverage.

The independent reader supports bounded classic PCAP/Ethernet with micro/nanosecond timestamps,
IPv4/IPv6 base headers, up to two VLAN tags and TCP/UDP structure. Other formats, fragments,
IPv6 extension headers and transports are explicit unsupported cases. Invalid structures are
counted, never silently coerced into valid packets. No payload bytes enter an index or report.
This is structural decoding; checksums, reassembly, payload/attack semantics and capture loss
are not validated. Tests use independent known packet lengths/timestamps and failure cases.

The private packet/claim index uses an 8 MiB cache target and 512 MiB main-file limit. PCAPs
remain limited to 128 MiB and two million records. Process one capture at a time. Resource
screen: sampled process-tree RSS sum below 1 GiB and audit time below ten minutes per capture,
excluding downloads; sampling is not an OS-enforced bound. Artifacts contain hashes and aggregate
counts only. Raw packets, endpoint rows, claims, label UIDs and indexes stay local/ignored.

This phase does not authorize model fitting, change source roles, prove device independence,
or certify a label as a true attack. Those gates still apply after a successful coverage audit.

The [proposed native dataset plan](../../configs/data/native-dataset-plan.json) assigns capture
3 to training, 8 to calibration and 34 to validation, in verified event-time order. The experiment
checks connection support and a 300-second inter-capture purge. Device isolation stays false,
the independent source is unacquired, and no row-level training cohort is materialized by this
phase. The plan is explicit preparation and does not replace the blocked global capture plan.

Primary format/semantics references: [PCAP format draft](https://www.ietf.org/ietf-ftp/internet-drafts/draft-ietf-opsawg-pcap-09.html),
[Zeek connection fields](https://docs.zeek.org/en/current/scripts/base/protocols/conn/main.zeek.html),
[NFStream API](https://www.nfstream.org/docs/api). Current exporter documentation is context;
the old dataset's semantics require the actual pinned packet/log evidence.
