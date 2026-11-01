"""Generate independent size/time/flag expectations; never import or query NFStream."""

import argparse
import json
import statistics
from pathlib import Path


def expected_values():
    core = json.loads(Path("tests/fixtures/pcap/expected.json").read_text(encoding="utf-8"))
    # These lists are specified from the generated packet construction, not meter output.
    samples = [
        ([32, 36], [250], {}),
        ([28], [], {}),
        (
            [40, 40, 40, 45, 47, 40],
            [100, 100, 120, 130, 50],
            {"syn": 2, "ack": 5, "psh": 2, "rst": 1},
        ),
        ([51, 52], [125], {}),
        ([32], [], {}),
    ]
    for row, (sizes, intervals, flags) in zip(core["rows"], samples, strict=True):
        for suffix, values in (("ps", sizes), ("piat_ms", intervals)):
            row.update(
                {
                    f"bidirectional_min_{suffix}": min(values) if values else 0,
                    f"bidirectional_mean_{suffix}": statistics.mean(values) if values else 0.0,
                    f"bidirectional_stddev_{suffix}": statistics.stdev(values)
                    if len(values) > 1
                    else 0.0,
                    f"bidirectional_max_{suffix}": max(values) if values else 0,
                }
            )
        for flag in ("syn", "cwr", "ece", "urg", "ack", "psh", "rst", "fin"):
            row[f"bidirectional_{flag}_packets"] = flags.get(flag, 0)
    core["scope"] = (
        "Independent explicit packet-size/interval/flag lists on the synthetic twelve-packet fixture; not exporter-derived or attack labels."
    )
    return core


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="tests/fixtures/pcap/statistical-expected.json")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    path = Path(args.output)
    content = json.dumps(expected_values(), indent=2) + "\n"
    if args.check:
        if path.read_text(encoding="utf-8") != content:
            raise ValueError("Independent statistical fixture changed; review before regenerating.")
    else:
        with path.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(content)
    print(
        "Independent statistical expectations verified."
        if args.check
        else "Independent statistical expectations created."
    )
