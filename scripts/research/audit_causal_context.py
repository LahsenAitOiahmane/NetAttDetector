"""Validate bounded callback-time context and independent replay without fitting a model."""

import argparse
import json
from pathlib import Path

from audit_statistical_features import run as prepare_lab

from netattdetector.context_cohort import prepare_context_cohort
from netattdetector.context_meter import verify_context_fixture
from netattdetector.io import write_json
from netattdetector.native_cohort import load_config


def run(data_root, cohort, statistics, context, evidence, captures=None):
    data_root, evidence, context = Path(data_root), Path(evidence), Path(context)
    if evidence.exists() or context.exists():
        raise FileExistsError("Use fresh causal context/evidence directories.")
    evidence.mkdir(parents=True)
    verify_context_fixture(evidence / "fixture-verification.json")
    prepare_lab(data_root, cohort, statistics, evidence / "base-measurements", captures)
    if captures is None:
        _, _, plan = load_config("configs/data/native-cohort.json")
        captures = {name: str(data_root / name) for name in plan["partitions"].values()}
    try:
        manifest = prepare_context_cohort(captures, cohort, statistics, context)
    finally:
        for name in (
            "manifest.json",
            "progress.json",
            "core-verification.json",
            "statistical-verification.json",
        ):
            path = context / name
            if path.exists():
                write_json(evidence / name, json.loads(path.read_text(encoding="utf-8")))
    print(
        json.dumps(
            {
                "schema": manifest["schema"],
                "rows": manifest["audit"]["rows"],
                "patterns": manifest["audit"]["patterns"],
                "records": manifest["records"],
                "training_authorized": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="data/raw/causal-context")
    parser.add_argument("--cohort", default="data/processed/context-core")
    parser.add_argument("--statistics", default="data/processed/context-statistics")
    parser.add_argument("--context", default="data/processed/context-candidate")
    parser.add_argument("--evidence", default="artifacts/causal-context")
    parser.add_argument(
        "--captures", help="Optional private capture-ID/path map to reuse pinned inputs"
    )
    args = parser.parse_args()
    run(
        args.data_root,
        args.cohort,
        args.statistics,
        args.context,
        args.evidence,
        json.loads(Path(args.captures).read_text()) if args.captures else None,
    )
