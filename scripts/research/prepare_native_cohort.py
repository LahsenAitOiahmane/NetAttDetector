"""Prepare pinned native rows privately and publish only an explicit aggregate evidence allowlist."""

import argparse
import json
from pathlib import Path

from netattdetector.capture_labels import acquire_capture
from netattdetector.io import write_json
from netattdetector.native_cohort import load_config, prepare_native_cohort, verify_native_cohort


def run(data_root, cohort, evidence, captures=None, repeat_check=False):
    data_root, cohort, evidence = Path(data_root), Path(cohort), Path(evidence)
    if cohort.exists() or evidence.exists() or (captures is None and data_root.exists()):
        raise FileExistsError("Use fresh cohort/evidence and acquisition directories.")
    config, _, plan = load_config("configs/data/native-cohort.json")
    if captures is None:
        captures = {}
        for capture_id in plan["partitions"].values():
            directory = data_root / capture_id
            acquire_capture(capture_id, directory, config["source_config"])
            captures[capture_id] = str(directory)
    manifest = prepare_native_cohort(captures, cohort)
    evidence.mkdir(parents=True)
    verification = verify_native_cohort(cohort, cohort / "verification.json")
    # No glob/copytree: rows, packet databases and failure outputs cannot enter CI artifacts.
    for name, value in (("manifest.json", manifest), ("verification.json", verification)):
        write_json(evidence / name, value)
    if repeat_check:
        repeated = prepare_native_cohort(captures, cohort.with_name(cohort.name + "-repeat"))
        comparison = {
            "same_inputs_and_environment": True,
            "ordered_logical_rows_identical": manifest["audit"] == repeated["audit"],
            "database_bytes_identical": manifest["database"] == repeated["database"],
            "first_database": manifest["database"],
            "second_database": repeated["database"],
            "first_logical_sha256": manifest["audit"]["rows_logical_sha256"],
            "second_logical_sha256": repeated["audit"]["rows_logical_sha256"],
            "repeat_records": repeated["records"],
            "training_authorized": False,
            "scope": "Two complete native builds on one Linux runner/version, with existing pinned captures. Does not establish cross-version/platform or model-fit determinism.",
        }
        write_json(evidence / "reproducibility.json", comparison)
        if not comparison["ordered_logical_rows_identical"]:
            raise ValueError("Repeated native export changed stored rows or partition diagnostics.")
    print(
        json.dumps(
            {
                "rows": manifest["audit"]["rows"],
                "audit": manifest["audit"],
                "verified": verification["verified"],
                "training_authorized": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="data/raw/native-cohort")
    parser.add_argument("--cohort", required=True)
    parser.add_argument("--evidence", required=True)
    parser.add_argument(
        "--captures", help="Optional existing local capture-ID/path JSON; no acquisition"
    )
    parser.add_argument(
        "--repeat-check",
        action="store_true",
        help="Rebuild using the same pinned inputs and compare every logical row",
    )
    args = parser.parse_args()
    run(
        args.data_root,
        args.cohort,
        args.evidence,
        json.loads(Path(args.captures).read_text()) if args.captures else None,
        args.repeat_check,
    )
