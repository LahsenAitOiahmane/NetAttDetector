"""Acquire, measure or score the registered CESNET frozen positive-only test."""

import argparse
import json

from netattdetector.cesnet import acquire, evaluate, prepare

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("stage", choices=("acquire", "prepare", "evaluate"))
parser.add_argument("--source", default="data/raw/cesnet-cc25")
parser.add_argument("--measurements", default="data/processed/cesnet-cc25")
parser.add_argument("--model", default="artifacts/private-context-model/selected.joblib")
parser.add_argument("--output", default="artifacts/cesnet-frozen-test")
args = parser.parse_args()
if args.stage == "acquire":
    result = acquire(args.source)
elif args.stage == "prepare":
    result = prepare(args.source, args.measurements)
else:
    result = evaluate(args.measurements, args.model, args.output)
print(
    json.dumps(
        {
            key: result[key]
            for key in ("study_id", "rows", "summary", "model_fits", "deployment_authorized")
            if key in result
        },
        indent=2,
    )
)
