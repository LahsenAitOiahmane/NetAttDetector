"""Hardware inventory without identifiers or privileged commands."""

import platform
import shutil
from pathlib import Path

import psutil


def inventory(path="."):
    memory = psutil.virtual_memory()
    disk = shutil.disk_usage(Path(path).resolve())
    return {
        "platform": platform.system(),
        "python": platform.python_version(),
        "cpu": platform.processor(),
        "physical_cores": psutil.cpu_count(logical=False),
        "logical_cores": psutil.cpu_count(),
        "ram_gib": round(memory.total / 2**30, 2),
        "available_ram_gib": round(memory.available / 2**30, 2),
        "free_disk_gib": round(disk.free / 2**30, 2),
        "recommended_threads": min(2, psutil.cpu_count() or 1),
        "recommended_max_rows": 200_000,
        "recommended_chunk_rows": 50_000,
    }
