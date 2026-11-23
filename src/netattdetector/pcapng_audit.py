"""Bounded block/packet accounting, without a flow engine or packet-label inference.

Supports section/interface/enhanced-packet blocks. Other packet block types fail
explicitly. Counts and lengths do not validate checksums, payloads or attack truth.
"""

import hashlib
import struct
from collections import Counter


def interface_clock(options, endian):
    """Callback mode supports verified decimal micro/nanoseconds and integer offsets."""
    multiplier, offset, seen, cursor = 1000, 0, set(), 0
    while cursor < len(options):
        if cursor + 4 > len(options):
            raise ValueError("Truncated PCAPNG interface option.")
        code, length = struct.unpack(endian + "HH", options[cursor : cursor + 4])
        cursor += 4
        padded = (length + 3) // 4 * 4
        if cursor + padded > len(options):
            raise ValueError("PCAPNG interface option exceeds its block.")
        value = options[cursor : cursor + length]
        cursor += padded
        if code == 0:
            if length or any(options[cursor:]):
                raise ValueError("Invalid PCAPNG end-of-options marker.")
            break
        if code in (9, 14):
            if code in seen:
                raise ValueError("Ambiguous duplicate PCAPNG clock option.")
            seen.add(code)
        if code == 9:
            if length != 1 or value[0] not in (6, 9):
                raise ValueError("Packet callbacks require decimal micro/nanosecond timestamps.")
            multiplier = 1000 if value[0] == 6 else 1
        if code == 14:
            if length != 8:
                raise ValueError("Invalid PCAPNG timestamp offset.")
            offset = struct.unpack(endian + "q", value)[0] * 1_000_000_000
    return multiplier, offset


def audit_stream(stream, *, max_bytes=128 * 2**20, max_packets=2_000_000, on_packet=None):
    count = packets = truncated = captured_bytes = wire_bytes = 0
    types = Counter()
    links, interfaces = [], []
    endian = None
    digest = hashlib.sha256()

    def read(size):
        nonlocal count
        if count + size > max_bytes:
            raise ValueError("PCAPNG byte bound exceeded.")
        raw = stream.read(size)
        count += len(raw)
        digest.update(raw)
        if len(raw) != size:
            raise ValueError("Truncated PCAPNG block.")
        return raw

    while True:
        header = stream.read(8)
        if not header:
            break
        count += len(header)
        digest.update(header)
        if len(header) != 8 or count > max_bytes:
            raise ValueError("Truncated or oversized PCAPNG header.")
        prefix = b""
        if header[:4] == b"\x0a\x0d\x0d\x0a":
            prefix = read(4)
            endian = {b"\x4d\x3c\x2b\x1a": "<", b"\x1a\x2b\x3c\x4d": ">"}.get(prefix)
            if endian is None:
                raise ValueError("Unknown PCAPNG byte-order magic.")
            interfaces = []
        if endian is None:
            raise ValueError("PCAPNG must begin with a section header.")
        kind, length = struct.unpack(endian + "II", header)
        if not 12 <= length <= 2**20 or length % 4 or length < 8 + len(prefix) + 4:
            raise ValueError("Invalid or over-bound PCAPNG block length.")
        body = prefix + read(length - 8 - len(prefix))
        if struct.unpack(endian + "I", body[-4:])[0] != length:
            raise ValueError("PCAPNG trailing block length mismatch.")
        types[kind] += 1
        if kind == 0x0A0D0D0A:
            if length < 28 or struct.unpack(endian + "HH", body[4:8]) != (1, 0):
                raise ValueError("Unsupported PCAPNG section version/header.")
        elif kind == 1:
            if length < 20:
                raise ValueError("Truncated PCAPNG interface description.")
            link, reserved, snapshot = struct.unpack(endian + "HHI", body[:8])
            if reserved != 0:
                raise ValueError("Invalid PCAPNG reserved interface field.")
            interfaces.append((link, snapshot))
            if on_packet is not None:
                interfaces[-1] = (*interfaces[-1], *interface_clock(body[8:-4], endian))
            links.append(link)
        elif kind == 6:
            if length < 32:
                raise ValueError("Truncated enhanced packet block.")
            interface, high, low, caplen, wirelen = struct.unpack(endian + "IIIII", body[:20])
            if (
                interface >= len(interfaces)
                or caplen > wirelen
                or 32 + ((caplen + 3) // 4) * 4 > length
            ):
                raise ValueError("Invalid enhanced packet interface or captured length.")
            snapshot = interfaces[interface][1]
            if snapshot and caplen > snapshot:
                raise ValueError("Captured packet exceeds its interface snapshot length.")
            packets += 1
            if packets > max_packets:
                raise ValueError("PCAPNG packet count bound exceeded.")
            truncated += caplen < wirelen
            captured_bytes += caplen
            wire_bytes += wirelen
            if on_packet is not None:
                link, _, multiplier, offset = interfaces[interface]
                on_packet(
                    {
                        "packet_id": packets,
                        "time_ns": ((high << 32) + low) * multiplier + offset,
                        "linktype": link,
                        "captured_bytes": caplen,
                        "original_bytes": wirelen,
                        "frame": body[20 : 20 + caplen],
                    }
                )
        elif kind in (2, 3):
            raise ValueError(
                "This audit requires enhanced packets; other packet blocks are unassessed."
            )
    if endian is None:
        raise ValueError("Empty PCAPNG input.")
    return {
        "bytes": count,
        "sha256": digest.hexdigest(),
        "packet_records": packets,
        "truncated_packet_records": truncated,
        "captured_packet_bytes": captured_bytes,
        "original_packet_bytes": wire_bytes,
        "linktypes": links,
        "block_types": {str(key): value for key, value in sorted(types.items())},
    }
