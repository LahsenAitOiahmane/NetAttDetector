import copy
import io
import json
import struct
import zipfile

import pytest

from netattdetector import kitsune


def labels(path, values):
    path.write_text("\n".join(map(str, values)) + "\n", encoding="utf-8")
    return path


def test_literal_binary_vector_header_and_no_guessed_numeric_header(tmp_path):
    path = labels(tmp_path / "labels.csv", ["Label", 0, 0, 1])
    assert list(kitsune.binary_labels(path)) == [0, 0, 1]
    path = labels(path, [0, 0, 1])
    assert list(kitsune.binary_labels(path)) == [0, 0, 1]
    with pytest.raises(ValueError, match="row bound"):
        list(kitsune.binary_labels(path, max_rows=2))


@pytest.mark.parametrize(
    "values",
    [[], [""], ["2"], ["0.0"], ["0,1"], ["index,label"], ["0", "label"], ["0", " "], ["0", "NaN"]],
)
def test_invalid_label_formats_fail_without_coercion(tmp_path, values):
    path = labels(tmp_path / "labels.csv", values)
    with pytest.raises(ValueError):
        list(kitsune.binary_labels(path))


def test_independent_known_packet_cardinality_and_prefix(tmp_path):
    path = labels(tmp_path / "labels.csv", [0] * 7 + [1] * 5)
    result = kitsune.audit_pairs("tests/fixtures/pcap/core.pcap", path)
    assert result["packets"] == result["labels"] == 12
    assert result["binary_class_counts"] == {"0": 7, "1": 5}
    assert result["class_counts_by_packet_status"] == {"decoded_tcp_udp": {"0": 7, "1": 5}}
    assert result["first_attack_packet_ordinal"] == 8
    assert result["last_benign_packet_ordinal"] == 7
    assert result["single_benign_prefix_then_attack_suffix"]
    assert result["label_transitions"] == 1
    assert not result["native_flow_labels_authorized"]
    labels(path, [0, 1] * 6)
    result = kitsune.audit_pairs("tests/fixtures/pcap/core.pcap", path)
    assert not result["single_benign_prefix_then_attack_suffix"]
    assert result["label_transitions"] == 11


@pytest.mark.parametrize("size", [11, 13])
def test_missing_or_extra_labels_do_not_shift_packet_alignment(tmp_path, size):
    path = labels(tmp_path / "labels.csv", [0] * size)
    with pytest.raises(ValueError, match="cardinality"):
        kitsune.audit_pairs("tests/fixtures/pcap/core.pcap", path)


def test_truncated_and_unsupported_packet_labels_are_retained(tmp_path):
    # Two Ethernet records: non-IP and capture-truncated. Neither becomes benign by default.
    header = struct.pack("<IHHIIII", 0xA1B2C3D4, 2, 4, 0, 0, 65535, 1)
    frame = bytes(12) + bytes.fromhex("0806")
    raw = header
    raw += struct.pack("<IIII", 2, 0, len(frame), len(frame)) + frame
    raw += struct.pack("<IIII", 1, 0, len(frame), len(frame) + 10) + frame
    path = tmp_path / "known.pcap"
    path.write_bytes(raw)
    result = kitsune.audit_pairs(path, labels(tmp_path / "labels.csv", [1, 1]))
    assert result["binary_class_counts"] == {"1": 2}
    assert result["class_counts_by_packet_status"] == {
        "capture_truncated": {"1": 1},
        "non_ip_ethernet": {"1": 1},
    }
    assert result["timestamp_regressions"] == 1


def specification():
    return {
        "id": "pcap",
        "name": "Mirai Botnet/Mirai_pcap.pcap",
        "expected_uncompressed_bytes": 9,
        "maximum_transfer_bytes": 128,
    }


def test_raw_and_single_member_zip_publication_are_exact_and_exclusive(tmp_path):
    spec = specification()
    source = tmp_path / "raw.download"
    source.write_bytes(b"synthetic")
    raw = kitsune.publish_download(source, tmp_path / "raw.pcap", spec)
    assert raw["bytes"] == 9 and raw["container"] == "raw"
    assert not raw["publisher_checksum_verified"]
    with pytest.raises(FileExistsError):
        kitsune.publish_download(source, tmp_path / "raw.pcap", spec)
    zipped = tmp_path / "wrapped.zip"
    with zipfile.ZipFile(zipped, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(spec["name"], b"synthetic")
    result = kitsune.publish_download(zipped, tmp_path / "zip.pcap", spec)
    assert result["container"] == "single-member-zip-crc-checked"
    assert result["sha256"] == raw["sha256"]


@pytest.mark.parametrize("case", ["extra", "path", "size", "symlink"])
def test_unexpected_zip_paths_sizes_types_and_members_are_rejected(tmp_path, case):
    source = tmp_path / "bad.zip"
    spec = specification()
    with zipfile.ZipFile(source, "w") as archive:
        name = "../Mirai_pcap.pcap" if case == "path" else "Mirai_pcap.pcap"
        member = zipfile.ZipInfo(name)
        if case == "symlink":
            member.create_system = 3
            member.external_attr = 0o120777 << 16
        archive.writestr(member, b"wrong" if case == "size" else b"synthetic")
        if case == "extra":
            archive.writestr("unrequested.txt", b"extra")
    with pytest.raises(ValueError):
        kitsune.publish_download(source, tmp_path / "target.pcap", spec)
    assert not (tmp_path / "target.pcap").exists()


class Response(io.BytesIO):
    status = 200

    def __init__(self, raw, headers):
        super().__init__(raw)
        self.headers = headers


def fake_response(monkeypatch, response):
    class Opener:
        def open(self, *_args, **_kwargs):
            return response

    monkeypatch.setattr(kitsune.urllib.request, "build_opener", lambda *_: Opener())


def test_complete_bounded_transfer_and_owned_partial_cleanup(tmp_path, monkeypatch):
    spec = specification()
    fake_response(
        monkeypatch,
        Response(b"synthetic", {"Content-Type": "application/octet-stream", "Content-Length": "9"}),
    )
    result = kitsune.fetch_file(spec, tmp_path)
    assert result["transfer_bytes"] == 9
    assert (tmp_path / "Mirai_pcap.pcap").read_bytes() == b"synthetic"
    assert not list(tmp_path.glob("*.partial"))


@pytest.mark.parametrize("case", ["html", "over_limit", "truncated", "noncomplete"])
def test_login_partial_or_overbound_responses_never_publish(tmp_path, monkeypatch, case):
    spec = specification()
    headers = {"Content-Type": "application/octet-stream"}
    raw = b"synthetic"
    if case == "html":
        headers["Content-Type"] = "text/html"
    if case == "over_limit":
        spec["maximum_transfer_bytes"] = 8
    if case == "truncated":
        headers["Content-Length"] = "10"
    response = Response(raw, headers)
    if case == "noncomplete":
        response.status = 206
    fake_response(monkeypatch, response)
    with pytest.raises(ValueError):
        kitsune.fetch_file(spec, tmp_path)
    assert not list(tmp_path.iterdir())


def test_preexisting_partial_is_never_removed_or_overwritten(tmp_path, monkeypatch):
    partial = tmp_path / "Mirai_pcap.pcap.download.partial"
    partial.write_bytes(b"preserve")
    fake_response(monkeypatch, Response(b"synthetic", {"Content-Type": "application/octet-stream"}))
    with pytest.raises(FileExistsError):
        kitsune.fetch_file(specification(), tmp_path)
    assert partial.read_bytes() == b"preserve"


def test_modified_acquisition_policy_is_rejected(tmp_path):
    policy = copy.deepcopy(kitsune.configuration())
    policy["training_authorized"] = True
    path = tmp_path / "policy.json"
    path.write_text(json.dumps(policy))
    with pytest.raises(ValueError, match="registered"):
        kitsune.configuration(path)


def test_pcapng_magic_dispatch_consumes_labels_for_unsupported_and_truncated_packets(tmp_path):
    def block(kind, body):
        size = len(body) + 12
        return struct.pack("<II", kind, size) + body + struct.pack("<I", size)

    raw = block(0x0A0D0D0A, struct.pack("<IHHq", 0x1A2B3C4D, 1, 0, -1))
    raw += block(1, struct.pack("<HHI", 1, 0, 65535))
    frame = bytes(12) + bytes.fromhex("0806")
    for timestamp, original in [(100, len(frame)), (50, len(frame) + 5)]:
        raw += block(
            6, struct.pack("<IIIII", 0, 0, timestamp, len(frame), original) + frame + bytes(2)
        )
    path = tmp_path / "misleading-name.pcap"
    path.write_bytes(raw)
    result = kitsune.audit_pairs(path, labels(tmp_path / "labels.csv", [0, 1]))
    assert result["capture_format"] == "pcapng-enhanced"
    assert result["pcapng_structure"]["packet_records"] == 2
    assert result["binary_class_counts"] == {"0": 1, "1": 1}
    assert result["class_counts_by_packet_status"] == {
        "capture_truncated": {"1": 1},
        "non_ip_ethernet": {"0": 1},
    }
    assert result["timestamp_regressions"] == 1
    labels(tmp_path / "labels.csv", [0, 1, 0])
    with pytest.raises(ValueError, match="cardinality"):
        kitsune.audit_pairs(path, tmp_path / "labels.csv")


def test_repeatability_pins_reject_changes_while_allowing_new_transfer_timings(tmp_path):
    receipt = json.loads(Path(kitsune.PINS).read_text())
    pins = tmp_path / "pins.json"
    pins.write_text(json.dumps(receipt))
    repeat = copy.deepcopy(receipt)
    repeat["resources"] = {"seconds": 0.1, "sampled_rss_mib": 30}
    kitsune.verify_pins(repeat, pins)
    repeat["files"][0]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="committed"):
        kitsune.verify_pins(repeat, pins)
