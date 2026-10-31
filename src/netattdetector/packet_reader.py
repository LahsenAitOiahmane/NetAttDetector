"""Bounded structural PCAP decoding. No interfaces, reassembly or payload inspection."""

import ipaddress
import struct
from pathlib import Path

MAGIC = {
    b"\xd4\xc3\xb2\xa1": ("<", 1000),
    b"\xa1\xb2\xc3\xd4": (">", 1000),
    b"\x4d\x3c\xb2\xa1": ("<", 1),
    b"\xa1\xb2\x3c\x4d": (">", 1),
}


def decode_frame(frame):
    frame = bytes(frame)
    if len(frame) < 14:
        return {"status": "short_ethernet"}
    offset, ethertype = 14, int.from_bytes(frame[12:14], "big")
    for _ in range(2):
        if ethertype not in (0x8100, 0x88A8):
            break
        if len(frame) < offset + 4:
            return {"status": "short_vlan"}
        ethertype = int.from_bytes(frame[offset + 2 : offset + 4], "big")
        offset += 4
    if ethertype in (0x8100, 0x88A8):
        return {"status": "unsupported_vlan_depth"}
    ip = frame[offset:]
    if ethertype == 0x0800:
        if len(ip) < 20 or ip[0] >> 4 != 4:
            return {"status": "invalid_ipv4_header"}
        header = (ip[0] & 15) * 4
        length = int.from_bytes(ip[2:4], "big")
        if header < 20 or header > len(ip) or length < header or length > len(ip):
            return {"status": "invalid_or_truncated_ipv4_length"}
        if int.from_bytes(ip[6:8], "big") & 0x3FFF:
            return {"status": "unsupported_ipv4_fragment"}
        version, protocol = 4, ip[9]
        src, dst = str(ipaddress.ip_address(ip[12:16])), str(ipaddress.ip_address(ip[16:20]))
    elif ethertype == 0x86DD:
        if len(ip) < 40 or ip[0] >> 4 != 6:
            return {"status": "invalid_ipv6_header"}
        header, length, version, protocol = 40, 40 + int.from_bytes(ip[4:6], "big"), 6, ip[6]
        if length > len(ip):
            return {"status": "truncated_ipv6"}
        src, dst = str(ipaddress.ip_address(ip[8:24])), str(ipaddress.ip_address(ip[24:40]))
        if protocol in (0, 43, 44, 50, 51, 60):
            return {"status": "unsupported_ipv6_extension"}
    else:
        return {"status": "non_ip_ethernet"}
    if protocol not in (6, 17):
        return {"status": "unsupported_transport"}
    transport = ip[header:length]
    if len(transport) < 4:
        return {"status": "short_transport_tuple"}
    result = {
        "src_ip": src,
        "dst_ip": dst,
        "src_port": int.from_bytes(transport[:2], "big"),
        "dst_port": int.from_bytes(transport[2:4], "big"),
        "protocol": protocol,
        "ip_version": version,
        "ip_bytes": length,
    }
    if protocol == 6:
        result["tcp_flags"] = transport[13] if len(transport) >= 14 else None
        if len(transport) < 20 or not 20 <= (transport[12] >> 4) * 4 <= len(transport):
            result["status"] = "invalid_tcp_header_length"
        else:
            result["status"] = "decoded_tcp_udp"
            result["transport_payload_bytes"] = len(transport) - (transport[12] >> 4) * 4
    elif len(transport) < 8 or not 8 <= int.from_bytes(transport[4:6], "big") <= len(transport):
        result["status"] = "invalid_udp_length"
    else:
        result["status"] = "decoded_tcp_udp"
        result["transport_payload_bytes"] = int.from_bytes(transport[4:6], "big") - 8
    return result


def packet_records(path, *, max_packets=2_000_000):
    path = Path(path)
    if (
        not path.is_file()
        or path.suffix.lower() != ".pcap"
        or not 24 <= path.stat().st_size <= 128 * 2**20
    ):
        raise ValueError("Provide a bounded existing classic .pcap file (at most 128 MiB).")
    with path.open("rb") as stream:
        header = stream.read(24)
        if header[:4] not in MAGIC:
            raise ValueError("Unsupported PCAP magic; PCAPNG is not supported by this reader.")
        endian, multiplier = MAGIC[header[:4]]
        major, minor, _, _, snaplen, network = struct.unpack(endian + "HHIIII", header[4:])
        if (major, minor) != (2, 4) or network != 1 or not 0 < snaplen <= 2**20:
            raise ValueError("Use classic PCAP 2.4 Ethernet without additional link flags.")
        count = 0
        while record := stream.read(16):
            count += 1
            if count > max_packets or len(record) != 16:
                raise ValueError("Packet limit or incomplete PCAP record header.")
            seconds, fraction, captured, original = struct.unpack(endian + "IIII", record)
            if (
                fraction >= 1_000_000_000 // multiplier
                or not 0 < captured <= min(snaplen, 2**20)
                or original < captured
            ):
                raise ValueError("Invalid PCAP timestamp or packet lengths.")
            frame = stream.read(captured)
            if len(frame) != captured:
                raise ValueError("Incomplete PCAP packet body.")
            decoded = decode_frame(frame)
            if captured < original:
                decoded["status"] = "capture_truncated"
            yield {
                "packet_id": count,
                "time_ns": seconds * 1_000_000_000 + fraction * multiplier,
                **decoded,
            }
