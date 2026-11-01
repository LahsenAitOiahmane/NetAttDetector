"""Recreate private native rows, fit the registered diagnostic and retain aggregate evidence."""

import argparse
import json
from pathlib import Path

from netattdetector.capture_labels import acquire_capture
from netattdetector.io import write_json
from netattdetector.native_cohort import load_config, prepare_native_cohort


def run(data_root, cohort, study, evidence, captures=None, diagnostics=False):
    data_root, cohort, study, evidence = map(Path, (data_root, cohort, study, evidence))
    if any(path.exists() for path in (cohort, study, evidence)) or (
        captures is None and data_root.exists()
    ):
        raise FileExistsError("Use fresh acquisition/cohort/study/evidence directories.")
    config, _, plan = load_config("configs/data/native-cohort.json")
    if captures is None:
        captures = {}
        for capture_id in plan["partitions"].values():
            directory = data_root / capture_id
            acquire_capture(capture_id, directory, config["source_config"])
            captures[capture_id] = str(directory)
    manifest = prepare_native_cohort(captures, cohort)
    # Load numerical/model libraries only after bounded packet/export preparation completes.
    if diagnostics:
        from netattdetector.native_errors import run_native_errors as run_study
    else:
        from netattdetector.native_study import run_native_study as run_study

    evidence.mkdir(parents=True)
    write_json(evidence / "cohort-manifest.json", manifest)
    try:
        result = run_study(cohort, study)
    finally:
        # Explicit allowlist; model binaries and per-flow/group/feature rows never leave data/.
        for name in (
            "study.json",
            "freeze.json",
            "cohort-verification.json",
            "study-config.json",
            "progress.json",
            "diagnostics.json",
            "diagnostics-config.json",
        ):
            source = study / name
            if source.exists():
                write_json(evidence / name, json.loads(source.read_text(encoding="utf-8")))
    print(
        json.dumps(
            {
                "completed_fits": result["completed_fits"],
                "selected": result.get("selected_diagnostic_id"),
                "qualifying_candidates": result.get("qualifying_non_dummy_candidates"),
                "fixed_thread_repeat": result.get("fixed_one_thread_repeat"),
                "resources": result["resources"],
                "deployment_authorized": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="data/raw/native-study")
    parser.add_argument("--cohort", default="data/processed/native-study-cohort")
    parser.add_argument("--study", default="data/processed/native-study-models")
    parser.add_argument("--evidence", default="artifacts/native-study")
    parser.add_argument(
        "--captures", help="Optional private capture-ID/path JSON map, skipping acquisition"
    )
    parser.add_argument(
        "--diagnostics",
        action="store_true",
        help="Run the registered three-fit error study instead of baseline selection",
    )
    args = parser.parse_args()
    run(
        args.data_root,
        args.cohort,
        args.study,
        args.evidence,
        json.loads(Path(args.captures).read_text()) if args.captures else None,
        args.diagnostics,
    )
