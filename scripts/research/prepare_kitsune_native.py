"""Measure registered Kitsune phase labels and unchanged native context features."""

import argparse
import json

from netattdetector.kitsune_native import prepare

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--source", default="data/raw/kitsune-mirai")
parser.add_argument("--output", default="data/processed/kitsune-native")
args = parser.parse_args()
result = prepare(args.source, args.output)
print(
    json.dumps(
        {
            "study_id": result["study_id"],
            "audit": result["audit"],
            "context_control": result["context_control"],
            "training_authorized": False,
        },
        indent=2,
    )
)
