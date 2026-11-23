"""Acquire or audit the bounded, registered Kitsune development capture."""

import argparse
import json

from netattdetector.kitsune import acquire, audit

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("stage", choices=("acquire", "audit"))
parser.add_argument("--source", default="data/raw/kitsune-mirai")
parser.add_argument("--output", default="artifacts/kitsune-packet-audit.json")
args = parser.parse_args()
result = acquire(args.source) if args.stage == "acquire" else audit(args.source, args.output)
print(
    json.dumps(
        {
            key: result[key]
            for key in (
                "profile_id",
                "packets",
                "binary_class_counts",
                "class_counts_by_packet_status",
                "packet_label_cardinality_verified",
                "training_authorized",
            )
            if key in result
        },
        indent=2,
    )
)
