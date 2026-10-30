"""Reject tracked generated/private artifacts and broken local Markdown links."""

import re
import subprocess
from pathlib import Path
from urllib.parse import unquote


def check(root=Path(".")):
    root = root.resolve()
    names = subprocess.check_output(["git", "ls-files"], cwd=root, text=True).splitlines()
    errors = []
    generated = {
        ".pyc",
        ".pyo",
        ".joblib",
        ".pickle",
        ".pkl",
        ".onnx",
        ".npz",
        ".npy",
        ".parquet",
        ".csv",
        ".tsv",
        ".pcap",
        ".pcapng",
        ".zip",
        ".7z",
        ".gz",
    }
    for name in names:
        path = root / name
        if not path.exists():
            errors.append(f"Tracked file is missing: {name}")
            continue
        fixture = name.startswith("tests/fixtures/") and path.suffix in {
            ".csv",
            ".tsv",
            ".pcap",
            ".pcapng",
        }
        private_root = name.startswith(
            ("artifacts/", "reports/local/", "logs/", "saved_models/", ".venv", ".vscode/")
        )
        secret = path.name == ".env" or (
            path.name.startswith(".env.") and path.name != ".env.example"
        )
        if (
            private_root
            or secret
            or "__pycache__" in path.parts
            or (path.suffix in generated and not fixture)
            or (name.startswith("data/") and name != "data/README.md")
        ):
            errors.append(f"Generated/private file is tracked: {name}")
        if path.suffix != ".md":
            continue
        for target in re.findall(r"\[[^\]]*\]\(([^)]+)\)", path.read_text(encoding="utf-8")):
            if target.startswith(("https:", "http:", "mailto:", "#")):
                continue
            target = unquote(target.split("#", 1)[0].strip("<>"))
            if not (path.parent / target).exists():
                errors.append(f"Broken local link in {name}: {target}")
    protected = [
        ".venv-pandas3/pyvenv.cfg",
        "data/raw/private.zip",
        "data/processed/cache.npz",
        "artifacts/run/model.joblib",
        "reports/local/private.json",
        "private.pcap",
        "traffic.csv",
        ".env.local",
    ]
    public = [
        "uv.lock",
        "src/netattdetector/cli.py",
        "reports/evidence/example.json",
        "reports/figures/example.png",
        "data/README.md",
        "tests/fixtures/pcap/synthetic.pcap",
    ]
    for candidates, expected in ((protected, True), (public, False)):
        result = subprocess.run(
            ["git", "check-ignore", "--no-index", "--stdin", "-z"],
            input=("\0".join(candidates) + "\0").encode(),
            cwd=root,
            capture_output=True,
            check=False,
        )
        if result.returncode not in (0, 1):
            raise RuntimeError(result.stderr.decode())
        ignored = set(result.stdout.decode().strip("\0").split("\0"))
        for name in candidates:
            if (name in ignored) != expected:
                errors.append(f"Unexpected ignore policy for {name}: ignored={name in ignored}")
    if errors:
        raise ValueError("\n".join(errors))
    print(f"Repository hygiene and documentation links passed ({len(names)} tracked files).")


if __name__ == "__main__":
    check()
