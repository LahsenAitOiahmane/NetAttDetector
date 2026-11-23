import io
import struct

import pytest

from netattdetector.pcapng_audit import audit_stream


def block(kind, body, endian="<"):
    size = len(body) + 12
    return struct.pack(endian + "II", kind, size) + body + struct.pack(endian + "I", size)


def section(endian="<"):
    return block(0x0A0D0D0A, struct.pack(endian + "IHHq", 0x1A2B3C4D, 1, 0, -1), endian)


def interface(endian="<"):
    return block(1, struct.pack(endian + "HHI", 1, 0, 65535), endian)


def packet(endian="<", interface_id=0, caplen=3, wirelen=5):
    return block(
        6, struct.pack(endian + "IIIII", interface_id, 0, 1, caplen, wirelen) + b"abc\0", endian
    )


def test_known_counts_truncation_and_section_byte_order_reset():
    raw = (
        section()
        + interface()
        + packet()
        + section(">")
        + interface(">")
        + packet(">", caplen=3, wirelen=3)
    )
    result = audit_stream(io.BytesIO(raw))
    assert result["bytes"] == len(raw)
    assert result["packet_records"] == 2
    assert result["truncated_packet_records"] == 1
    assert result["captured_packet_bytes"] == 6
    assert result["original_packet_bytes"] == 8
    assert result["linktypes"] == [1, 1]


@pytest.mark.parametrize(
    "kind",
    [
        "trailing",
        "truncated",
        "bad_interface",
        "bad_caplen",
        "simple_packet",
        "missing_section",
        "overflow",
        "length_alignment",
    ],
)
def test_invalid_unbounded_or_unsupported_packet_records_are_rejected(kind):
    raw = section() + interface() + packet()
    bound = 128 * 2**20
    if kind == "trailing":
        raw = raw[:-4] + b"\0" * 4
    elif kind == "truncated":
        raw = raw[:-1]
    elif kind == "bad_interface":
        raw = section() + interface() + packet(interface_id=2)
    elif kind == "bad_caplen":
        raw = section() + interface() + packet(caplen=6)
    elif kind == "simple_packet":
        raw = section() + interface() + block(3, struct.pack("<I", 0))
    elif kind == "missing_section":
        raw = interface() + packet()
    elif kind == "overflow":
        bound = len(raw) - 1
    else:
        raw = raw[:-36] + struct.pack("<II", 6, 35) + bytes(27)
    with pytest.raises(ValueError):
        audit_stream(io.BytesIO(raw), max_bytes=bound)
