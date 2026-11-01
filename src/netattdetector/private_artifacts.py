"""Bounded encrypted research transfer; never deserialize a model here.

Confidentiality does not identify the sender. Recovery requires a separately trusted
model digest, checked before returning any model bytes to the caller.
"""

import base64
import hashlib
import json
import os
import struct
import subprocess
from pathlib import Path

MAGIC = b"NADENC1\n"
ALGORITHM = "X25519-HKDF-SHA256-AES256GCM-v1"
MAX_MODEL = 16 * 2**20
MAX_PAYLOAD = 24 * 2**20
MAX_HEADER = 4096


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def encode(value):
    return base64.b64encode(value).decode("ascii")


def decode(value, length=None):
    if not isinstance(value, str):
        raise ValueError("Invalid encoded transfer field.")
    try:
        result = base64.b64decode(value, validate=True)
    except ValueError as error:
        raise ValueError("Invalid encoded transfer field.") from error
    if length is not None and len(result) != length:
        raise ValueError("Invalid transfer field length.")
    return result


def bounded_read(path, limit):
    with Path(path).open("rb") as handle:
        result = handle.read(limit + 1)
    if len(result) > limit:
        raise ValueError("Research transfer exceeds its byte bound.")
    return result


def protect_directory(path):
    path = Path(path)
    path.mkdir(parents=True, mode=0o700, exist_ok=False)
    restrict_directory(path)
    return path


def restrict_directory(path):
    """Apply an exact ACL, rather than preserving pre-existing explicit grants."""
    if os.name == "nt":
        script = """
        $ErrorActionPreference = 'Stop'
        $taskOwner = [System.Security.Principal.WindowsIdentity]::GetCurrent().User
        $taskSystem = [System.Security.Principal.SecurityIdentifier]::new('S-1-5-18')
        $taskAcl = [System.Security.AccessControl.DirectorySecurity]::new()
        $taskAcl.SetOwner($taskOwner)
        $taskAcl.SetAccessRuleProtection($true, $false)
        foreach ($taskPrincipal in @($taskOwner, $taskSystem)) {
            $taskRule = [System.Security.AccessControl.FileSystemAccessRule]::new(
                $taskPrincipal, 'FullControl', 'ContainerInherit,ObjectInherit', 'None', 'Allow')
            $taskAcl.AddAccessRule($taskRule)
        }
        [System.IO.Directory]::SetAccessControl($env:NETATT_PRIVATE_DIRECTORY, $taskAcl)
        """
        subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", script],
            env={**os.environ, "NETATT_PRIVATE_DIRECTORY": str(Path(path).resolve())},
            check=True,
            capture_output=True,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
    else:
        os.chmod(path, 0o700)


def generate_recipient(directory, public_path):
    from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey

    public_path = Path(public_path)
    if public_path.exists():
        raise FileExistsError(public_path)
    directory = protect_directory(directory)
    private = X25519PrivateKey.generate()
    key_path = directory / "recipient-v1.key"
    fd = os.open(key_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as handle:
        handle.write(private.private_bytes_raw())
    public = private.public_key().public_bytes_raw()
    recipient = {
        "algorithm": ALGORITHM,
        "public_key": encode(public),
        "fingerprint_sha256": hashlib.sha256(public).hexdigest(),
        "purpose": "Private research artifact recipient; does not authenticate senders",
    }
    public_path.parent.mkdir(parents=True, exist_ok=True)
    with public_path.open("xb") as handle:
        handle.write(canonical(recipient) + b"\n")
    return recipient


def derive(private, public, salt, recipient, ephemeral):
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PublicKey
    from cryptography.hazmat.primitives.kdf.hkdf import HKDF

    shared = private.exchange(X25519PublicKey.from_public_bytes(public))
    return HKDF(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        info=MAGIC + recipient + ephemeral,
    ).derive(shared)


def seal_model(model_path, freeze, probes, recipient_path, output):
    from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

    model = bounded_read(model_path, MAX_MODEL)
    if hashlib.sha256(model).hexdigest() != freeze["artifact_sha256"]:
        raise ValueError("Model does not match its diagnostic freeze.")
    recipient = json.loads(bounded_read(recipient_path, MAX_HEADER))
    public = decode(recipient["public_key"], 32)
    fingerprint = hashlib.sha256(public).hexdigest()
    if recipient["algorithm"] != ALGORITHM or recipient["fingerprint_sha256"] != fingerprint:
        raise ValueError("Invalid recipient contract.")
    payload = canonical({"model": encode(model), "freeze": freeze, "synthetic_probes": probes})
    if len(payload) > MAX_PAYLOAD:
        raise ValueError("Research payload exceeds its byte bound.")
    private = X25519PrivateKey.generate()
    ephemeral = private.public_key().public_bytes_raw()
    salt, nonce = os.urandom(32), os.urandom(12)
    header = canonical(
        {
            "algorithm": ALGORITHM,
            "recipient": fingerprint,
            "ephemeral": encode(ephemeral),
            "salt": encode(salt),
            "nonce": encode(nonce),
            "payload_bytes": len(payload),
        }
    )
    prefix = MAGIC + struct.pack(">I", len(header)) + header
    ciphertext = AESGCM(derive(private, public, salt, public, ephemeral)).encrypt(
        nonce, payload, prefix
    )
    with Path(output).open("xb") as handle:
        handle.write(prefix + ciphertext)
    return {
        "algorithm": ALGORITHM,
        "recipient": fingerprint,
        "artifact_sha256": freeze["artifact_sha256"],
        "ciphertext_bytes": len(prefix) + len(ciphertext),
    }


def recover_model(envelope, private_path, trusted_freeze):
    from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM

    raw = bounded_read(envelope, MAX_PAYLOAD + MAX_HEADER + len(MAGIC) + 20)
    offset = len(MAGIC) + 4
    if len(raw) < offset or raw[: len(MAGIC)] != MAGIC:
        raise ValueError("Unsupported research envelope.")
    length = struct.unpack(">I", raw[len(MAGIC) : offset])[0]
    if not 0 < length <= MAX_HEADER or len(raw) < offset + length + 16:
        raise ValueError("Invalid/truncated research header.")
    header = json.loads(raw[offset : offset + length])
    if (
        set(header) != {"algorithm", "recipient", "ephemeral", "salt", "nonce", "payload_bytes"}
        or header["algorithm"] != ALGORITHM
    ):
        raise ValueError("Unsupported research header.")
    size = header["payload_bytes"]
    if (
        type(size) is not int
        or not 0 < size <= MAX_PAYLOAD
        or len(raw) != offset + length + size + 16
    ):
        raise ValueError("Invalid/truncated research payload.")
    private = X25519PrivateKey.from_private_bytes(bounded_read(private_path, 32))
    public = private.public_key().public_bytes_raw()
    if header["recipient"] != hashlib.sha256(public).hexdigest():
        raise ValueError("Research artifact targets a different recipient.")
    ephemeral, salt, nonce = (
        decode(header["ephemeral"], 32),
        decode(header["salt"], 32),
        decode(header["nonce"], 12),
    )
    key = derive(private, ephemeral, salt, public, ephemeral)
    payload = json.loads(AESGCM(key).decrypt(nonce, raw[offset + length :], raw[: offset + length]))
    if (
        set(payload) != {"model", "freeze", "synthetic_probes"}
        or payload["freeze"] != trusted_freeze
    ):
        raise ValueError("Payload differs from the separately trusted freeze.")
    model = decode(payload["model"])
    if (
        len(model) > MAX_MODEL
        or hashlib.sha256(model).hexdigest() != trusted_freeze["artifact_sha256"]
    ):
        raise ValueError("Recovered model differs from the trusted model digest.")
    return model, payload["synthetic_probes"]
