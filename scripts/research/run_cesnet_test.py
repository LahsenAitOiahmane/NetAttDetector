"""Acquire, measure or score the registered CESNET frozen positive-only test."""

import argparse
import json
from pathlib import Path

from netattdetector.cesnet import (
    acquire,
    evaluate,
    prepare,
    recover_measurements,
    seal_measurements,
)

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("stage", choices=("acquire", "prepare", "seal", "recover", "evaluate"))
parser.add_argument("--source", default="data/raw/cesnet-cc25")
parser.add_argument("--measurements", default="data/processed/cesnet-cc25")
parser.add_argument("--model", default="artifacts/private-context-model/selected.joblib")
parser.add_argument("--output", default="artifacts/cesnet-frozen-test")
parser.add_argument("--envelope")
parser.add_argument("--private-key", default=".netatt-keys/recipient-v1.key")
parser.add_argument(
    "--trusted-manifest", default="reports/evidence/cesnet-frozen-test/manifest.json"
)
args = parser.parse_args()
if args.stage == "acquire":
    result = acquire(args.source)
elif args.stage == "prepare":
    result = prepare(args.source, args.measurements)
elif args.stage == "seal":
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    result = seal_measurements(args.measurements, args.output)
elif args.stage == "recover":
    if not args.envelope:
        parser.error("recover requires --envelope")
    result = recover_measurements(
        args.envelope, args.private_key, args.trusted_manifest, args.measurements
    )
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
