import hashlib
import io
from collections import defaultdict
from decimal import Decimal

import pytest

from netattdetector.capture_labels import bounded_fetch, connection_key, match_flow
from netattdetector.label_index import SQLiteLabelIndex, nanoseconds


def test_disk_and_memory_matchers_agree_at_exact_boundaries(tmp_path):
    key = connection_key("192.0.2.1", 1234, "198.51.100.1", 443, 6)
    memory = defaultdict(list)
    disk = SQLiteLabelIndex(tmp_path / "labels.sqlite")
    try:
        for i in range(100):
            start = Decimal("1538478769000.123456") + i * 100
            row = {
                "uid": str(i),
                "start": start,
                "end": start + 20 if i % 7 else None,
                "label": i % 2 if i % 5 else None,
                "detail": "test",
            }
            memory[key].append(row)
            disk.add(key, row)
        disk.finish()
        for offset in range(-2, 10100, 7):
            first = float(Decimal("1538478769000") + offset)
            flow = {
                "src_ip": "198.51.100.1",
                "src_port": 443,
                "dst_ip": "192.0.2.1",
                "dst_port": 1234,
                "protocol": 6,
                "bidirectional_first_seen_ms": first,
                "bidirectional_last_seen_ms": first + 1,
            }
            assert match_flow(flow, disk) == match_flow(flow, memory)
        assert disk.connection.execute("PRAGMA cache_size").fetchone()[0] == -8192
    finally:
        disk.close()


def test_disk_index_does_not_round_away_one_nanosecond(tmp_path):
    disk = SQLiteLabelIndex(tmp_path / "labels.sqlite")
    key = connection_key("192.0.2.1", 1234, "198.51.100.1", 443, 6)
    start = Decimal("1538478769000.000001")
    try:
        disk.add(key, {"uid": "u1", "start": start, "end": start + 1, "label": 1, "detail": "C&C"})
        disk.finish()
        assert disk.overlapping(key, start, start, Decimal(0))[0]["start"] == start
        assert (
            disk.overlapping(
                key, start - Decimal("0.000001"), start - Decimal("0.000001"), Decimal(0)
            )
            == []
        )
    finally:
        disk.close()
    with pytest.raises(ValueError, match="exact"):
        nanoseconds(Decimal("1538478769000.0000001"))


def test_disk_enforces_uid_uniqueness_for_unsupported_protocols(tmp_path):
    disk = SQLiteLabelIndex(tmp_path / "labels.sqlite")
    row = {"uid": "u1", "start": Decimal(1000), "end": None, "label": 0, "detail": "-"}
    try:
        disk.add(None, row)
        with pytest.raises(ValueError, match="Duplicate Zeek UID"):
            disk.add(None, row)
    finally:
        disk.close()
    with pytest.raises(FileExistsError):
        SQLiteLabelIndex(tmp_path / "labels.sqlite")


def test_large_download_is_chunked_and_hash_verified(tmp_path, monkeypatch):
    content = b"abcde" * 40_000
    read_sizes = []

    class Response(io.BytesIO):
        status = 200
        headers = {"Content-Length": str(len(content))}

        def read(self, size):
            assert 0 < size <= 65536
            read_sizes.append(size)
            return super().read(size)

        def read1(self, size):
            return self.read(size)

    class Opener:
        def open(self, request, timeout):
            return Response(content)

    monkeypatch.setattr("urllib.request.build_opener", lambda *args: Opener())
    target = tmp_path / "capture.pcap"
    digest = hashlib.sha256(content).hexdigest()
    result = bounded_fetch("https://mcfp.felk.cvut.cz/test.pcap", target, len(content), digest)
    assert target.read_bytes() == content and result["sha256"] == digest
    assert len(read_sizes) >= 4 and not target.with_name(target.name + ".partial").exists()


def test_download_deadline_removes_only_its_own_partial(tmp_path, monkeypatch):
    class Response(io.BytesIO):
        status = 200
        headers = {"Content-Length": "3"}

    class Opener:
        def open(self, request, timeout):
            return Response(b"abc")

    ticks = iter([0, 0, 301])
    monkeypatch.setattr("urllib.request.build_opener", lambda *args: Opener())
    monkeypatch.setattr("netattdetector.capture_labels.time.monotonic", lambda: next(ticks))
    target = tmp_path / "capture.pcap"
    unrelated = tmp_path / "another.partial"
    unrelated.write_text("retained")
    with pytest.raises(TimeoutError, match="300-second"):
        bounded_fetch("https://mcfp.felk.cvut.cz/test.pcap", target, 3)
    assert not target.exists() and not target.with_name(target.name + ".partial").exists()
    assert unrelated.read_text() == "retained"
