"""Public command-line entry points. Heavy libraries load only when needed."""

import argparse
import json
import sys
from pathlib import Path

from netattdetector import __version__
from netattdetector.io import write_json


def parser():
    root = argparse.ArgumentParser(description="NetAttDetector flow classification research tools")
    root.add_argument("--version", action="version", version=__version__)
    commands = root.add_subparsers(dest="command", required=True)
    doctor = commands.add_parser(
        "doctor", help="Inventory CPU, RAM, disk and conservative defaults"
    )
    doctor.add_argument("--output")
    commands.add_parser("datasets", help="List official dataset records and terms")
    download = commands.add_parser(
        "download", help="Download a public UQ archive with a byte budget"
    )
    download.add_argument("dataset")
    download.add_argument("--output", default="data/raw")
    download.add_argument("--max-mib", type=int, default=512)
    profile = commands.add_parser(
        "profile", help="Scan labels and profile a bounded uniform sample"
    )
    profile.add_argument("--data", required=True)
    profile.add_argument("--member")
    profile.add_argument("--max-rows", type=int, default=200_000)
    profile.add_argument("--chunk-rows", type=int, default=50_000)
    profile.add_argument("--seed", type=int, default=42)
    profile.add_argument("--output", required=True)
    train = commands.add_parser(
        "train", help="Compare models on validation; evaluate champion on holdout"
    )
    train.add_argument("--data", required=True)
    train.add_argument("--external")
    train.add_argument("--external-member")
    train.add_argument("--config", default="configs/laptop.json")
    train.add_argument("--output", required=True)
    predict = commands.add_parser("predict", help="Classify compatible exported flow CSVs")
    predict.add_argument("--model", required=True)
    predict.add_argument("--data", required=True)
    predict.add_argument("--output", required=True)
    predict.add_argument("--chunk-rows", type=int, default=10_000)
    predict.add_argument("--threads", type=int, default=2)
    return root


def execute(args):
    if args.command == "doctor":
        from netattdetector.hardware import inventory

        result = inventory()
        if args.output:
            write_json(args.output, result)
        return result
    if args.command in {"datasets", "download"}:
        from netattdetector.datasets import CATALOG, LICENSE_NOTE, SOURCE, download

        if args.command == "datasets":
            return {"source": SOURCE, "license_note": LICENSE_NOTE, "datasets": CATALOG}
        if args.dataset not in CATALOG:
            raise ValueError(f"Unknown dataset. Choices: {', '.join(CATALOG)}")
        return download(args.dataset, args.output, args.max_mib)
    if args.command == "profile":
        from netattdetector.data import describe_sample, load_sample

        frame, result = load_sample(
            args.data, args.max_rows, args.chunk_rows, args.seed, args.member
        )
        result.update(describe_sample(frame))
        write_json(args.output, result)
        return {
            "report": args.output,
            "rows_scanned": result["rows_scanned"],
            "sample_rows": len(frame),
        }
    if args.command == "train":
        from netattdetector.training import TrainConfig, run_experiment

        config = TrainConfig(**json.loads(Path(args.config).read_text(encoding="utf-8")))
        result = run_experiment(args.data, args.output, config, args.external, args.external_member)
        return {
            "report": str(Path(args.output) / "metrics.json"),
            "champion": result["champion"],
            "test": result["test"],
            "external": result["external"]["metrics"] if result["external"] else None,
        }
    if args.command == "predict":
        from netattdetector.inference import predict_csv

        return predict_csv(args.model, args.data, args.output, args.chunk_rows, args.threads)
    raise ValueError("Unknown command.")


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        print(json.dumps(execute(args), indent=2, allow_nan=False))
    except (ValueError, OSError, KeyError, TypeError) as error:
        print(f"netatt: {error}", file=sys.stderr)
        sys.exit(2)
