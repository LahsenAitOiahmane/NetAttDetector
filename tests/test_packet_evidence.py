import importlib.util
import struct

import pytest

from netattdetector.capture_labels import match_flow, read_zeek_labels
from netattdetector.label_index import SQLiteLabelIndex
from netattdetector.packet_evidence import PacketIndex, PointLabels, candidate_match
from netattdetector.packet_reader import decode_frame, packet_records

spec = importlib.util.spec_from_file_location(
    "packet_fixture_builder", "scripts/data/make_pcap_fixtures.py"
)
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)


def write_pcap(path, frames, *, endian="<", nano=False):
    magic = 0xA1B23C4D if nano else 0xA1B2C3D4
    content = struct.pack(endian + "IHHIIII", magic, 2, 4, 0, 0, 65535, 1)
    for frame in frames:
        fraction = 123456000 if nano else 123456
        content += (
            struct.pack(endian + "IIII", 1_600_000_001, fraction, len(frame), len(frame)) + frame
        )
    path.write_bytes(content)
    return path


def packet():
    return builder.packet("192.0.2.1", "198.51.100.1", 1234, 53, b"abcd")


@pytest.mark.parametrize("endian,nano", [("<", False), (">", False), ("<", True), (">", True)])
def test_timestamp_encodings_and_ip_byte_layer(tmp_path, endian, nano):
    rows = list(
        packet_records(write_pcap(tmp_path / "known.pcap", [packet()], endian=endian, nano=nano))
    )
    assert len(rows) == 1 and rows[0]["time_ns"] == 1_600_000_001_123_456_000
    assert rows[0]["status"] == "decoded_tcp_udp" and rows[0]["ip_bytes"] == 32
    assert rows[0]["src_port"] == 1234 and rows[0]["dst_port"] == 53


def test_existing_independent_fixture_census():
    rows = list(packet_records("tests/fixtures/pcap/core.pcap"))
    assert len(rows) == 12 and {row["status"] for row in rows} == {"decoded_tcp_udp"}
    # 68 + 28 + 252 + 103 + 32 IP bytes across the five documented flows.
    assert sum(row["ip_bytes"] for row in rows) == 483
    assert sum(row["ip_version"] == 6 for row in rows) == 2


def test_vlan_and_unsupported_structure_are_explicit():
    raw = packet()
    vlan = raw[:12] + bytes.fromhex("81000001") + raw[12:]
    assert decode_frame(vlan)["ip_bytes"] == 32
    broken = bytearray(builder.packet("192.0.2.1", "198.51.100.1", 1234, 443, tcp=True))
    broken[14 + 20 + 12] = 0  # An invalid TCP data offset must never prove a valid point.
    assert decode_frame(broken)["status"] == "invalid_tcp_header_length"
    fragmented = bytearray(raw)
    fragmented[14 + 6 : 14 + 8] = bytes.fromhex("2000")
    assert decode_frame(fragmented)["status"] == "unsupported_ipv4_fragment"
    short_udp = bytearray(raw)
    short_udp[14 + 20 + 4 : 14 + 20 + 6] = bytes.fromhex("0001")
    assert decode_frame(short_udp)["status"] == "invalid_udp_length"


@pytest.mark.parametrize("case", ["header", "body", "magic", "length", "limit"])
def test_incomplete_or_unregistered_captures_fail(tmp_path, case):
    path = write_pcap(tmp_path / "known.pcap", [packet()])
    content = path.read_bytes()
    if case == "header":
        content = content[:29]
    elif case == "body":
        content = content[:-1]
    elif case == "magic":
        content = bytes.fromhex("0a0d0d0a") + content[4:]
    elif case == "length":
        content = content[:36] + struct.pack("<I", 1) + content[40:]
    path.write_bytes(content)
    with pytest.raises(ValueError):
        list(packet_records(path, max_packets=0 if case == "limit" else 2_000_000))


def log(path, rows):
    columns = "ts uid id.orig_h id.orig_p id.resp_h id.resp_p proto duration orig_pkts resp_pkts orig_ip_bytes resp_ip_bytes label detailed-label".split()
    lines = []
    for changes in rows:
        data = {
            "ts": "1600000001.123456",
            "uid": "u1",
            "id.orig_h": "192.0.2.1",
            "id.orig_p": "1234",
            "id.resp_h": "198.51.100.1",
            "id.resp_p": "53",
            "proto": "udp",
            "duration": "-",
            "orig_pkts": "1",
            "resp_pkts": "0",
            "orig_ip_bytes": "32",
            "resp_ip_bytes": "0",
            "label": "Malicious",
            "detailed-label": "DDoS",
            **changes,
        }
        lines.append("\t".join(data[col] for col in columns))
    path.write_text("#fields\t" + "\t".join(columns) + "\n" + "\n".join(lines) + "\n")
    return path


def flow():
    return {
        "src_ip": "192.0.2.1",
        "dst_ip": "198.51.100.1",
        "src_port": 1234,
        "dst_port": 53,
        "protocol": 17,
        "src2dst_packets": 1,
        "dst2src_packets": 0,
        "bidirectional_packets": 1,
        "src2dst_bytes": 32,
        "dst2src_bytes": 0,
        "bidirectional_first_seen_ms": 1600000001123,
        "bidirectional_last_seen_ms": 1600000001123,
        "bidirectional_duration_ms": 0,
    }


def test_packet_backed_point_is_not_blind_zero_fill(tmp_path):
    pcap = write_pcap(tmp_path / "known.pcap", [packet()])
    labels = log(tmp_path / "labels.log", [{}])
    packets = PacketIndex(tmp_path / "packets.sqlite")
    base = SQLiteLabelIndex(tmp_path / "labels.sqlite")
    try:
        packets.build(pcap)
        evidence = packets.establish_points(labels)
        assert evidence["proven_points_by_detail"] == {"DDoS": 1}
        read_zeek_labels(labels, index=base)
        candidate = PointLabels(base, packets)
        assert match_flow(flow(), base) == ("unknown_label_or_duration", None)
        status, row = candidate_match(flow(), candidate)
        assert status == "matched" and row["uid"] == "u1" and row["label"] == 1
        wrong = flow()
        wrong["bidirectional_packets"] = 2
        assert candidate_match(wrong, candidate) == ("point_flow_counter_mismatch", None)
        wrong = flow()
        wrong["src2dst_bytes"] = 46
        assert candidate_match(wrong, candidate) == ("point_flow_counter_mismatch", None)
    finally:
        base.close()
        packets.close()


@pytest.mark.parametrize(
    "case,reason",
    [
        ("zero", "zero_packet_claim"),
        ("multi", "multi_packet_claim"),
        ("unknown_bytes", "unknown_packet_or_byte_counts"),
        ("bytes", "direction_or_ip_bytes_mismatch"),
        ("timestamp", "no_exact_packet"),
        ("duplicate_packet", "multiple_exact_packets"),
        ("shared_uid_claim", "shared_packet_claim"),
        ("known_uid_collision", "shared_packet_claim"),
    ],
)
def test_uncertain_packet_claims_stay_unproven(tmp_path, case, reason):
    frames = [packet(), packet()] if case == "duplicate_packet" else [packet()]
    pcap = write_pcap(tmp_path / "known.pcap", frames)
    changes = (
        {"orig_pkts": "0", "orig_ip_bytes": "0"}
        if case == "zero"
        else {"orig_pkts": "2", "orig_ip_bytes": "64"}
        if case == "multi"
        else {"orig_ip_bytes": "-"}
        if case == "unknown_bytes"
        else {"orig_ip_bytes": "46"}
        if case == "bytes"
        else {"ts": "1600000001.123457"}
        if case == "timestamp"
        else {}
    )
    rows = [changes]
    if case in ("shared_uid_claim", "known_uid_collision"):
        rows.append({"uid": "u2", "duration": "0" if case == "known_uid_collision" else "-"})
    labels = log(tmp_path / "labels.log", rows)
    index = PacketIndex(tmp_path / "packets.sqlite")
    try:
        index.build(pcap)
        report = index.establish_points(labels)
        assert report["proven_points_by_detail"] == {}
        assert report["excluded_missing_records_by_detail"]["DDoS"][reason] >= 1
    finally:
        index.close()


def test_response_direction_and_unknown_binary_label_remain_explicit(tmp_path):
    pcap = write_pcap(tmp_path / "known.pcap", [packet()])
    labels = log(
        tmp_path / "labels.log",
        [
            {
                "id.orig_h": "198.51.100.1",
                "id.orig_p": "53",
                "id.resp_h": "192.0.2.1",
                "id.resp_p": "1234",
                "orig_pkts": "0",
                "resp_pkts": "1",
                "orig_ip_bytes": "0",
                "resp_ip_bytes": "32",
                "label": "Unknown",
            }
        ],
    )
    packets = PacketIndex(tmp_path / "packets.sqlite")
    base = SQLiteLabelIndex(tmp_path / "labels.sqlite")
    try:
        packets.build(pcap)
        assert packets.establish_points(labels)["proven_points_by_detail"] == {"DDoS": 1}
        read_zeek_labels(labels, index=base)
        status, row = candidate_match(flow(), PointLabels(base, packets))
        assert status == "unknown_label_or_duration" and row is None
    finally:
        base.close()
        packets.close()
