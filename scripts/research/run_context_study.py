"""Recreate private pinned measurements and compare the registered context models."""

import argparse
import json
import shutil
from pathlib import Path

from audit_causal_context import run as prepare_lab

from netattdetector.io import write_json


def run(data_root, cohort, statistics, context, study, evidence, captures=None):
    evidence, study = Path(evidence), Path(study)
    evidence.mkdir(parents=True, exist_ok=False)
    if study.exists():
        raise FileExistsError(study)
    prepare_lab(data_root, cohort, statistics, context, evidence / "measurements", captures)
    from netattdetector.context_study import run_context_study

    try:
        result = run_context_study(cohort, statistics, context, study)
    finally:
        for name in (
            "study.json",
            "freeze.json",
            "study-config.json",
            "core-verification.json",
            "statistical-verification.json",
            "context-verification.json",
            "progress.json",
            "transfer.json",
        ):
            path = study / name
            if path.exists():
                write_json(evidence / name, json.loads(path.read_text(encoding="utf-8")))
        encrypted = study / "selected.nadenc"
        if encrypted.exists():
            shutil.copyfile(encrypted, evidence / "selected.nadenc")
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
    parser.add_argument("--data-root", default="data/raw/context-study")
    parser.add_argument("--cohort", default="data/processed/context-study-core")
    parser.add_argument("--statistics", default="data/processed/context-study-statistics")
    parser.add_argument("--context", default="data/processed/context-study-context")
    parser.add_argument("--study", default="data/processed/context-study-models")
    parser.add_argument("--evidence", default="artifacts/context-study")
    parser.add_argument("--captures")
    args = parser.parse_args()
    run(
        args.data_root,
        args.cohort,
        args.statistics,
        args.context,
        args.study,
        args.evidence,
        json.loads(Path(args.captures).read_text()) if args.captures else None,
    )
