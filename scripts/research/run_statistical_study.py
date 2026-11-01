"""Prepare private statistical measurements and run only the registered bounded comparison."""

import argparse
import json
from pathlib import Path

from audit_statistical_features import run as prepare_lab

from netattdetector.io import write_json


def run(data_root, cohort, statistics, study, evidence, captures=None):
    evidence, study = Path(evidence), Path(study)
    if evidence.exists() or study.exists():
        raise FileExistsError("Use fresh statistical model/evidence directories.")
    evidence.mkdir(parents=True)
    prepare_lab(data_root, cohort, statistics, evidence / "measurements", captures)
    from netattdetector.statistical_study import run_statistical_study

    try:
        result = run_statistical_study(cohort, statistics, study)
    finally:
        for name in (
            "study.json",
            "freeze.json",
            "study-config.json",
            "core-verification.json",
            "statistical-verification.json",
            "progress.json",
        ):
            path = study / name
            if path.exists():
                write_json(evidence / name, json.loads(path.read_text(encoding="utf-8")))
    print(
        json.dumps(
            {
                "completed_fits": result["completed_fits"],
                "selected": result["selected_diagnostic_id"],
                "qualifying_candidates": result["qualifying_non_dummy_candidates"],
                "resources": result["resources"],
                "deployment_authorized": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="data/raw/statistical-study")
    parser.add_argument("--cohort", default="data/processed/statistical-study-core")
    parser.add_argument("--statistics", default="data/processed/statistical-study-sidecar")
    parser.add_argument("--study", default="data/processed/statistical-study-models")
    parser.add_argument("--evidence", default="artifacts/statistical-study")
    parser.add_argument(
        "--captures", help="Optional private capture-ID/path map to reuse verified inputs"
    )
    args = parser.parse_args()
    run(
        args.data_root,
        args.cohort,
        args.statistics,
        args.study,
        args.evidence,
        json.loads(Path(args.captures).read_text()) if args.captures else None,
    )
