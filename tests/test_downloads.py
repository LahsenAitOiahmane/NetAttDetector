import io
import threading
import zipfile

import pytest

from netattdetector import datasets


class Response(io.BytesIO):
    status = 206

    def __init__(self, content, headers):
        super().__init__(content)
        self.headers = headers


def test_download_budget_crc_provenance_and_reuse(tmp_path, monkeypatch):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("flows.csv", "Label,Attack\n0,Benign\n")
    payload = buffer.getvalue()
    monkeypatch.setattr(datasets, "resolve_download", lambda _: "https://example.invalid/archive")

    def fake_open(url, headers=None):
        first, last = map(int, headers["Range"].removeprefix("bytes=").split("-"))
        return Response(
            payload[first : last + 1], {"Content-Range": f"bytes {first}-{last}/{len(payload)}"}
        )

    monkeypatch.setattr(datasets, "open_url", fake_open)
    manifest = datasets.download("nf-unsw-nb15-v2", tmp_path, max_mib=1)
    assert manifest["publisher_checksum_verified"] is False
    assert manifest["csv_members"][0]["name"] == "flows.csv"
    assert "example.invalid" not in str(manifest)
    assert datasets.download("nf-unsw-nb15-v2", tmp_path) == manifest
    (tmp_path / "nf-unsw-nb15-v2.zip").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="Checksum mismatch"):
        datasets.download("nf-unsw-nb15-v2", tmp_path)


def test_budget_refuses_before_full_transfer(tmp_path, monkeypatch):
    monkeypatch.setattr(datasets, "resolve_download", lambda _: "https://example.invalid/archive")
    monkeypatch.setattr(
        datasets, "open_url", lambda *args: Response(b"P", {"Content-Range": "bytes 0-0/999999999"})
    )
    with pytest.raises(ValueError, match="exceeds limit"):
        datasets.download("nf-unsw-nb15-v2", tmp_path, max_mib=1)
    assert not list(tmp_path.glob("*.zip"))


def test_parallel_ranges_are_written_in_order(tmp_path, monkeypatch):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("flows.csv", "Label,Attack\n" + "0,Benign\n" * 250000)
    payload = buffer.getvalue()
    second_started = threading.Event()
    monkeypatch.setattr(datasets, "resolve_download", lambda _: "https://example.invalid/archive")

    def fake_open(url, headers=None):
        first, last = map(int, headers["Range"].removeprefix("bytes=").split("-"))
        if first == 2**20:
            second_started.set()
        elif first == 0 and last > 0:
            assert second_started.wait(2)
        return Response(
            payload[first : last + 1], {"Content-Range": f"bytes {first}-{last}/{len(payload)}"}
        )

    monkeypatch.setattr(datasets, "open_url", fake_open)
    datasets.download("nf-unsw-nb15-v2", tmp_path, max_mib=4, block_mib=1, workers=2)
    assert (tmp_path / "nf-unsw-nb15-v2.zip").read_bytes() == payload
