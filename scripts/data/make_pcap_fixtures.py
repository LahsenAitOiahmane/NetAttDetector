"""Generate tiny deterministic synthetic Ethernet/IP PCAPs; sends no network traffic."""

import argparse
import hashlib
import ipaddress
import json
import struct
from pathlib import Path


def checksum(data):
    data += b"\0" * (len(data) % 2)
    value = sum(struct.unpack(f"!{len(data) // 2}H", data))
    while value >> 16:
        value = (value & 65535) + (value >> 16)
    return (~value) & 65535


def packet(src, dst, sport, dport, payload=b"", tcp=False, flags=0x10, seq=1, ack=1):
    source, target = ipaddress.ip_address(src), ipaddress.ip_address(dst)
    proto = 6 if tcp else 17
    if tcp:
        transport = (
            struct.pack("!HHIIBBHHH", sport, dport, seq, ack, 5 << 4, flags, 8192, 0, 0) + payload
        )
    else:
        transport = struct.pack("!HHHH", sport, dport, 8 + len(payload), 0) + payload
    if source.version == 4:
        pseudo = source.packed + target.packed + struct.pack("!BBH", 0, proto, len(transport))
        header = struct.pack(
            "!BBHHHBBH4s4s",
            0x45,
            0,
            20 + len(transport),
            1,
            0x4000,
            64,
            proto,
            0,
            source.packed,
            target.packed,
        )
        header = header[:10] + struct.pack("!H", checksum(header)) + header[12:]
        ethernet_type = 0x0800
    else:
        pseudo = source.packed + target.packed + struct.pack("!I3xB", len(transport), proto)
        header = struct.pack(
            "!IHBB16s16s", 6 << 28, len(transport), proto, 64, source.packed, target.packed
        )
        ethernet_type = 0x86DD
    offset = 16 if tcp else 6
    transport = (
        transport[:offset]
        + struct.pack("!H", checksum(pseudo + transport) or 65535)
        + transport[offset + 2 :]
    )
    ethernet = bytes.fromhex("020000000002020000000001") + struct.pack("!H", ethernet_type)
    frame = ethernet + header + transport
    return frame + b"\0" * max(0, 60 - len(frame))


def expected(
    src,
    dst,
    sport,
    dport,
    proto,
    version,
    start,
    end,
    forward_packets,
    reverse_packets,
    forward_bytes,
    reverse_bytes,
    forward_duration,
    reverse_duration,
):
    return {
        "src_ip": src,
        "dst_ip": dst,
        "src_port": sport,
        "dst_port": dport,
        "protocol": proto,
        "ip_version": version,
        "bidirectional_first_seen_ms": 1_600_000_000_000 + start,
        "bidirectional_last_seen_ms": 1_600_000_000_000 + end,
        "bidirectional_duration_ms": end - start,
        "src2dst_duration_ms": forward_duration,
        "dst2src_duration_ms": reverse_duration,
        "src2dst_packets": forward_packets,
        "dst2src_packets": reverse_packets,
        "src2dst_bytes": forward_bytes,
        "dst2src_bytes": reverse_bytes,
        "bidirectional_packets": forward_packets + reverse_packets,
        "bidirectional_bytes": forward_bytes + reverse_bytes,
    }


def fixtures():
    udp = ("192.0.2.10", "198.51.100.20", 53000, 53)
    tcp = ("192.0.2.12", "198.51.100.22", 55000, 443)
    ipv6 = ("2001:db8::1", "2001:db8::2", 56000, 9999)
    packets = [
        (1000, packet(*udp, b"abcd")),
        (1250, packet(udp[1], udp[0], udp[3], udp[2], b"abcdefgh")),
        (2000, packet("192.0.2.11", "198.51.100.21", 54000, 9999)),
        (3000, packet(*tcp, tcp=True, flags=2, seq=100, ack=0)),
        (3100, packet(tcp[1], tcp[0], tcp[3], tcp[2], tcp=True, flags=18, seq=200, ack=101)),
        (3200, packet(*tcp, tcp=True, seq=101, ack=201)),
        (3320, packet(*tcp, b"hello", tcp=True, flags=24, seq=101, ack=201)),
        (
            3450,
            packet(
                tcp[1], tcp[0], tcp[3], tcp[2], b"welcome", tcp=True, flags=24, seq=201, ack=106
            ),
        ),
        (3500, packet(*tcp, tcp=True, flags=20, seq=106, ack=208)),
        (4000, packet(*ipv6, b"abc")),
        (4125, packet(ipv6[1], ipv6[0], ipv6[3], ipv6[2], b"abcd")),
        (10000, packet(*udp, b"idle")),
    ]
    rows = [
        expected(*udp, 17, 4, 1000, 1250, 1, 1, 32, 36, 0, 0),
        expected("192.0.2.11", "198.51.100.21", 54000, 9999, 17, 4, 2000, 2000, 1, 0, 28, 0, 0, 0),
        expected(*tcp, 6, 4, 3000, 3500, 4, 2, 165, 87, 500, 350),
        expected(*ipv6, 17, 6, 4000, 4125, 1, 1, 51, 52, 0, 0),
        expected(*udp, 17, 4, 10000, 10000, 1, 0, 32, 0, 0, 0),
    ]
    content = struct.pack("<IHHIIII", 0xA1B2C3D4, 2, 4, 0, 0, 65535, 1)
    for ms, frame in packets:
        content += (
            struct.pack(
                "<IIII", 1_600_000_000 + ms // 1000, (ms % 1000) * 1000, len(frame), len(frame)
            )
            + frame
        )
    return content, {
        "scope": "synthetic protocol/counter fixtures; not attack labels or captured user traffic",
        "pcap_sha256": hashlib.sha256(content).hexdigest(),
        "packets": len(packets),
        "rows": rows,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="tests/fixtures/pcap")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    root = Path(args.output)
    content, expectations = fixtures()
    serialized = json.dumps(expectations, indent=2) + "\n"
    if args.check:
        if (root / "core.pcap").read_bytes() != content or (root / "expected.json").read_text(
            encoding="utf-8"
        ) != serialized:
            raise SystemExit(
                "Synthetic fixture bytes/expectations changed; review and regenerate explicitly."
            )
    else:
        if (root / "core.pcap").exists() or (root / "expected.json").exists():
            raise SystemExit("Refusing to overwrite existing fixtures.")
        root.mkdir(parents=True, exist_ok=True)
        (root / "core.pcap").write_bytes(content)
        (root / "expected.json").write_text(serialized, encoding="utf-8")
    print(
        f"Synthetic fixtures {'verified' if args.check else 'created'}: {len(content)} bytes, {expectations['packets']} packets."
    )
