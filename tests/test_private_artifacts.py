import hashlib
import json
import struct

import pytest

pytest.importorskip("cryptography")

from cryptography.exceptions import InvalidTag

from netattdetector import private_artifacts as transfer


@pytest.fixture
def sealed(tmp_path):
    recipient = tmp_path / "recipient.json"
    transfer.generate_recipient(tmp_path / "keys", recipient)
    model = tmp_path / "model.joblib"
    model.write_bytes(b"test-only opaque model bytes; never deserialize")
    freeze = {"artifact_sha256": hashlib.sha256(model.read_bytes()).hexdigest()}
    envelope = tmp_path / "model.nadenc"
    probes = {"scope": "synthetic", "rows": [[1, 2]], "scores": [0.4]}
    transfer.seal_model(model, freeze, probes, recipient, envelope)
    return tmp_path, recipient, model, freeze, envelope, probes


def test_round_trip_randomized_ciphertext_and_no_overwrite(sealed):
    directory, recipient, model, freeze, envelope, probes = sealed
    recovered, actual = transfer.recover_model(
        envelope, directory / "keys/recipient-v1.key", freeze
    )
    assert recovered == model.read_bytes() and actual == probes
    second = directory / "second.nadenc"
    transfer.seal_model(model, freeze, probes, recipient, second)
    assert envelope.read_bytes() != second.read_bytes()
    assert model.read_bytes() not in envelope.read_bytes()
    with pytest.raises(FileExistsError):
        transfer.seal_model(model, freeze, probes, recipient, envelope)
    with pytest.raises(FileExistsError):
        transfer.generate_recipient(directory / "keys", recipient)


@pytest.mark.parametrize(
    "corruption", ["tag", "header", "truncate", "version", "length", "oversize"]
)
def test_corruption_is_rejected_without_model_deserialization(sealed, corruption):
    directory, _, _, freeze, envelope, _ = sealed
    raw = bytearray(envelope.read_bytes())
    if corruption == "tag":
        raw[-1] ^= 1
    elif corruption == "header":
        offset = len(transfer.MAGIC) + 4
        length = struct.unpack(">I", raw[len(transfer.MAGIC) : offset])[0]
        header = json.loads(raw[offset : offset + length])
        header["salt"] = transfer.encode(bytes(32))
        encoded = transfer.canonical(header)
        assert len(encoded) == length
        raw[offset : offset + length] = encoded
    elif corruption == "truncate":
        raw = raw[:-1]
    elif corruption == "version":
        raw[6] = ord("9")
    elif corruption == "length":
        raw[len(transfer.MAGIC) : len(transfer.MAGIC) + 4] = struct.pack(
            ">I", transfer.MAX_HEADER + 1
        )
    else:
        raw = b"x" * (transfer.MAX_PAYLOAD + transfer.MAX_HEADER + len(transfer.MAGIC) + 21)
    altered = directory / "corrupt.nadenc"
    altered.write_bytes(raw)
    with pytest.raises((ValueError, InvalidTag)):
        transfer.recover_model(altered, directory / "keys/recipient-v1.key", freeze)


def test_wrong_recipient_and_forged_freeze_cannot_release_model(sealed):
    directory, _, _, freeze, envelope, _ = sealed
    transfer.generate_recipient(directory / "other-keys", directory / "other.json")
    with pytest.raises(ValueError, match="different recipient"):
        transfer.recover_model(envelope, directory / "other-keys/recipient-v1.key", freeze)
    with pytest.raises(ValueError, match="trusted freeze"):
        transfer.recover_model(
            envelope, directory / "keys/recipient-v1.key", {"artifact_sha256": "0" * 64}
        )


def test_payload_model_hash_is_independently_checked(sealed):
    directory, _, _, freeze, envelope, _ = sealed
    # A matching arbitrary inner freeze cannot authorize loading another model.
    from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

    raw = envelope.read_bytes()
    offset = len(transfer.MAGIC) + 4
    length = struct.unpack(">I", raw[len(transfer.MAGIC) : offset])[0]
    header = json.loads(raw[offset : offset + length])
    private = X25519PrivateKey.from_private_bytes(
        (directory / "keys/recipient-v1.key").read_bytes()
    )
    public = private.public_key().public_bytes_raw()
    key = transfer.derive(
        private,
        transfer.decode(header["ephemeral"]),
        transfer.decode(header["salt"]),
        public,
        transfer.decode(header["ephemeral"]),
    )
    aad = raw[: offset + length]
    payload = json.loads(
        AESGCM(key).decrypt(transfer.decode(header["nonce"]), raw[offset + length :], aad)
    )
    payload["model"] = transfer.encode(b"different malicious pickle bytes")
    encoded = transfer.canonical(payload)
    header["payload_bytes"] = len(encoded)
    encoded_header = transfer.canonical(header)
    aad = transfer.MAGIC + struct.pack(">I", len(encoded_header)) + encoded_header
    altered = directory / "forged.nadenc"
    altered.write_bytes(aad + AESGCM(key).encrypt(transfer.decode(header["nonce"]), encoded, aad))
    with pytest.raises(ValueError, match="trusted model digest"):
        transfer.recover_model(altered, directory / "keys/recipient-v1.key", freeze)
