"""Download public UQ archives with size limits, checksums and provenance.

Temporary download URLs are never persisted. Archives stay compressed;
pandas streams CSV members directly. No archive paths are extracted.
"""

import json
import shutil
import time
import urllib.error
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

from netattdetector.io import sha256_file, write_json

SOURCE = "https://www.cyber.uq.edu.au/project/machine-learning-based-nids-datasets"
CATALOG = {
    "nf-unsw-nb15-v2": {"uuid": "8c6e2a00-ef9c-11ed-827d-e762de186848", "flows": 2_390_275},
    "nf-ton-iot-v2": {"uuid": "a4ad7080-ef9c-11ed-a964-b70596e96ad5", "flows": 16_940_496},
    "nf-cse-cic-ids2018-v2": {"uuid": "ce5161d0-ef9c-11ed-827d-e762de186848", "flows": 18_893_708},
    "nf-bot-iot-v2": {"uuid": "b97fa3c0-ef9c-11ed-8007-832d3bc76ba8", "flows": 37_763_497},
    "nf-uq-nids-v2": {"uuid": "e2412450-ef9c-11ed-827d-e762de186848", "flows": 75_987_976},
}
LICENSE_NOTE = "Academic research with citation; commercial use requires author agreement."


def open_url(url, headers=None):
    request = urllib.request.Request(
        url, headers={"User-Agent": "NetAttDetector/0.2", **(headers or {})}
    )
    for attempt in range(3):
        try:
            return urllib.request.urlopen(request, timeout=60)
        except (urllib.error.URLError, TimeoutError):
            if attempt == 2:
                raise
            time.sleep(2**attempt)


def resolve_download(dataset):
    uid = CATALOG[dataset]["uuid"]
    with open_url(f"https://api.rdm.uq.edu.au/datasets/{uid}/download") as response:
        details = json.load(response)
    if details.get("accessType") != "OPEN" or details.get("packagingStatus") != 0:
        raise ValueError("Dataset is not ready for public download; check the official record.")
    return details["url"]


def archive_members(path):
    with zipfile.ZipFile(path) as archive:
        bad = archive.testzip()
        if bad:
            raise ValueError(f"Archive CRC check failed: {bad}")
        members = [
            {"name": x.filename, "bytes": x.file_size}
            for x in archive.infolist()
            if x.filename.lower().endswith(".csv") and not x.filename.startswith("__MACOSX/")
        ]
    if not members:
        raise ValueError("Downloaded archive has no CSV members.")
    return members


def download(dataset, output="data/raw", max_mib=512, workers=2, block_mib=4):
    if max_mib <= 0:
        raise ValueError("max-mib must be positive.")
    if not 1 <= workers <= 4 or not 1 <= block_mib <= 8:
        raise ValueError("Use 1..4 download workers and 1..8 MiB blocks.")
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    target = root / f"{dataset}.zip"
    manifest_path = target.with_suffix(".manifest.json")
    if target.exists():
        if not manifest_path.exists():
            raise ValueError(
                f"Existing archive lacks provenance: {target}; move it before downloading."
            )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["sha256"] != sha256_file(target):
            raise ValueError(f"Checksum mismatch: {target}")
        return manifest

    url = resolve_download(dataset)
    # Signed URLs authorize GET, not HEAD. A one-byte GET exposes total size.
    with open_url(url, {"Range": "bytes=0-0"}) as response:
        content_range = response.headers.get("Content-Range", "")
        size = (
            int(content_range.rsplit("/", 1)[-1])
            if "/" in content_range
            else int(response.headers.get("Content-Length", 0))
        )
    limit = int(max_mib * 2**20)
    if size <= 0 or size > limit:
        raise ValueError(f"Archive size {size:,} bytes exceeds limit {limit:,} or is unknown.")
    if shutil.disk_usage(root).free < size + 2 * 2**30:
        raise ValueError("Insufficient disk: keep at least 2 GiB free after downloading.")

    partial = target.with_suffix(".zip.part")
    try:
        block_bytes = block_mib * 2**20

        def fetch(start):
            end = min(start + block_bytes, size) - 1
            for attempt in range(3):
                try:
                    with open_url(url, {"Range": f"bytes={start}-{end}"}) as response:
                        if (
                            response.status != 206
                            or response.headers.get("Content-Range")
                            != f"bytes {start}-{end}/{size}"
                        ):
                            raise ValueError("Server did not honor the bounded range request.")
                        chunk = response.read(end - start + 2)
                    if len(chunk) != end - start + 1:
                        raise ValueError("Incomplete range response.")
                    return chunk
                except (OSError, ValueError):
                    if attempt == 2:
                        raise
                    time.sleep(2**attempt)

        # A fixed window bounds buffered data; executor.map would eagerly queue all blocks.
        with partial.open("wb") as stream, ThreadPoolExecutor(max_workers=workers) as pool:
            starts = iter(range(0, size, block_bytes))
            pending = [
                (start, pool.submit(fetch, start))
                for start in list(range(0, min(size, workers * block_bytes), block_bytes))
            ]
            for _ in pending:
                next(starts)
            total = 0
            while pending:
                _, future = pending.pop(0)
                chunk = future.result()
                stream.write(chunk)
                total += len(chunk)
                start = next(starts, None)
                if start is not None:
                    pending.append((start, pool.submit(fetch, start)))
        if total != size:
            raise ValueError(f"Incomplete download: {total} of {size} bytes.")
        members = archive_members(partial)
        manifest = {
            "dataset": dataset,
            "downloaded_at_utc": datetime.now(UTC).isoformat(),
            "source": SOURCE,
            "record": f"https://rdm.uq.edu.au/files/{CATALOG[dataset]['uuid']}",
            "license_note": LICENSE_NOTE,
            "publisher_reported_flows": CATALOG[dataset]["flows"],
            "bytes": size,
            "sha256": sha256_file(partial),
            "publisher_checksum_verified": False,
            "csv_members": members,
        }
        partial.replace(target)
        write_json(manifest_path, manifest)
        return manifest
    finally:
        partial.unlink(missing_ok=True)
