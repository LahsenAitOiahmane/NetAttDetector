"""Verify independent statistics, then preserve every core row in a private sidecar."""

import argparse
import json
from pathlib import Path

from netattdetector.capture_labels import acquire_capture
from netattdetector.io import write_json
from netattdetector.native_cohort import load_config, prepare_native_cohort
from netattdetector.statistical import export_statistical_pcap, verify_statistical_measurements
from netattdetector.statistical_cohort import prepare_statistical_cohort


def run(data_root, cohort, statistics, evidence, captures=None):
    data_root, cohort, statistics, evidence = map(Path, (data_root, cohort, statistics, evidence))
    if any(path.exists() for path in (cohort, statistics, evidence)) or (
        captures is None and data_root.exists()
    ):
        raise FileExistsError("Use fresh data/cohort/statistics/evidence directories.")
    evidence.mkdir(parents=True)
    export_statistical_pcap("tests/fixtures/pcap/core.pcap", evidence / "synthetic-export.json")
    fixture = verify_statistical_measurements(
        evidence / "synthetic-export.json",
        "tests/fixtures/pcap/statistical-expected.json",
        evidence / "fixture-verification.json",
    )
    if not fixture["passed"]:
        raise ValueError("Independent statistical fixture failed before real-data acquisition.")
    source, _, plan = load_config("configs/data/native-cohort.json")
    if captures is None:
        captures = {}
        for capture_id in plan["partitions"].values():
            directory = data_root / capture_id
            acquire_capture(capture_id, directory, source["source_config"])
            captures[capture_id] = str(directory)
    core_manifest = prepare_native_cohort(captures, cohort)
    write_json(evidence / "core-manifest.json", core_manifest)
    try:
        manifest = prepare_statistical_cohort(captures, cohort, statistics)
    finally:
        for name in ("manifest.json", "progress.json", "core-verification.json"):
            path = statistics / name
            if path.exists():
                write_json(evidence / name, json.loads(path.read_text(encoding="utf-8")))
    print(
        json.dumps(
            {
                "schema": manifest["schema"],
                "rows": manifest["audit"]["rows"],
                "patterns": manifest["audit"]["patterns"],
                "training_authorized": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="data/raw/statistical-lab")
    parser.add_argument("--cohort", default="data/processed/statistical-core")
    parser.add_argument("--statistics", default="data/processed/statistical-sidecar")
    parser.add_argument("--evidence", default="artifacts/statistical-lab")
    parser.add_argument("--captures", help="Private capture-ID/path JSON to reuse pinned inputs")
    args = parser.parse_args()
    run(
        args.data_root,
        args.cohort,
        args.statistics,
        args.evidence,
        json.loads(Path(args.captures).read_text()) if args.captures else None,
    )
