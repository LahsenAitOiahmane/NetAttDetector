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
    prepare = commands.add_parser(
        "prepare-multisource", help="Audit all development rows and cache a bounded cohort"
    )
    prepare.add_argument("--config", default="configs/multisource.json")
    prepare.add_argument("--data-root", default="data/raw")
    prepare.add_argument("--output", required=True)
    multi = commands.add_parser(
        "train-multisource", help="Select and freeze a model using development data only"
    )
    multi.add_argument("--config", default="configs/multisource.json")
    multi.add_argument("--prepared", required=True)
    multi.add_argument("--baseline")
    multi.add_argument("--output", required=True)
    frozen = commands.add_parser(
        "evaluate-frozen", help="Unseal the registered independent holdout once"
    )
    frozen.add_argument("--run", required=True)
    frozen.add_argument("--prepared", required=True)
    frozen.add_argument("--data", required=True)
    frozen.add_argument("--baseline")
    frozen.add_argument("--output", required=True)
    export = commands.add_parser(
        "export-portable", help="Convert XGBoost serialization without refitting"
    )
    export.add_argument("--model", required=True)
    export.add_argument("--output", required=True)
    verify = commands.add_parser(
        "verify-portable", help="Check a portable export's prediction probes"
    )
    verify.add_argument("path")
    study = commands.add_parser(
        "study-stability", help="Run bounded development-only ablations and repeat controls"
    )
    study.add_argument("--prepared", required=True)
    study.add_argument("--config", default="configs/stability.json")
    study.add_argument("--output", required=True)
    compare = commands.add_parser(
        "compare-stability", help="Compare identical canonical control payloads across runs"
    )
    compare.add_argument("first")
    compare.add_argument("second")
    compare.add_argument("--output", required=True)
    provenance = commands.add_parser(
        "inspect-provenance", help="Inspect source metadata headers without training"
    )
    provenance.add_argument("--data", required=True)
    provenance.add_argument("--source", required=True)
    provenance.add_argument("--roles", default="configs/data/source_roles.json")
    provenance.add_argument("--output", required=True)
    plan = commands.add_parser(
        "check-capture-plan", help="Validate declared capture/time/host split isolation"
    )
    plan.add_argument("--plan", default="configs/data/capture_plan.json")
    plan.add_argument("--output", required=True)
    export_flow = commands.add_parser(
        "export-pcap", help="Export bounded offline NFStream lab measurements on Linux"
    )
    export_flow.add_argument("--pcap", required=True)
    export_flow.add_argument("--output", required=True)
    verify_flow = commands.add_parser(
        "verify-measurements", help="Compare offline export with known synthetic fixture values"
    )
    verify_flow.add_argument("--export", required=True)
    verify_flow.add_argument("--expected", required=True)
    verify_flow.add_argument("--output", required=True)
    contract = commands.add_parser(
        "measurement-contract",
        help="Describe the separate NFStream lab contract and unsupported UQ fields",
    )
    contract.add_argument("--output", required=True)
    capture = commands.add_parser(
        "download-capture", help="Acquire a bounded registered IoT-23 development capture"
    )
    capture.add_argument("capture_id")
    capture.add_argument("--output", required=True)
    capture.add_argument("--config", default="configs/data/iot23-development.json")
    capture.add_argument(
        "--record-first",
        action="store_true",
        help="Record first acquisition before pinning its observed digests",
    )
    labels = commands.add_parser(
        "audit-capture-labels", help="Audit native flows against pinned IoT-23 connection labels"
    )
    labels.add_argument("--export", required=True)
    labels.add_argument("--capture-dir", required=True)
    labels.add_argument("--config", default="configs/data/iot23-development.json")
    labels.add_argument("--output", required=True)
    return root


def execute(args):
    if args.command == "download-capture":
        from netattdetector.capture_labels import acquire_capture

        return acquire_capture(args.capture_id, args.output, args.config, not args.record_first)
    if args.command == "audit-capture-labels":
        from netattdetector.capture_labels import audit_capture_labels

        return audit_capture_labels(args.export, args.capture_dir, args.output, args.config)
    if args.command == "measurement-contract":
        from netattdetector.measurement import contract_report

        result = contract_report()
        write_json(args.output, result)
        return {"contract": args.output, "schema": result["schema"], "uq_model_compatible": False}
    if args.command == "export-pcap":
        from netattdetector.measurement import export_pcap

        result = export_pcap(args.pcap, args.output)
        return {
            "export": args.output,
            "schema": result["schema"],
            "flows": len(result["rows"]),
            "resources": result["resources"],
        }
    if args.command == "verify-measurements":
        from netattdetector.measurement import verify_measurements

        result = verify_measurements(args.export, args.expected, args.output)
        if not result["passed"]:
            raise ValueError(f"Measurement comparison failed; see {args.output}.")
        return result
    if args.command == "inspect-provenance":
        from netattdetector.provenance import inspect_provenance

        return inspect_provenance(args.data, args.source, args.roles, args.output)
    if args.command == "check-capture-plan":
        from netattdetector.provenance import validate_capture_plan

        result = validate_capture_plan(args.plan, args.output)
        if not result["structure_valid"]:
            raise ValueError(
                f"Capture plan blocked; see {args.output}: " + "; ".join(result["blockers"])
            )
        return result
    if args.command == "study-stability":
        from netattdetector.stability import run_stability_study

        result = run_stability_study(args.prepared, args.config, args.output)
        return {
            "study": str(Path(args.output) / "study.json"),
            "completed_ablation_fits": len(result["records"]),
            "passes_all_seeds": sum(x["passes_all_seeds"] for x in result["summary"]),
            "resources": result["resources"],
        }
    if args.command == "compare-stability":
        from netattdetector.stability import compare_studies

        result = compare_studies(args.first, args.second)
        write_json(args.output, result)
        return result
    if args.command == "export-portable":
        from netattdetector.portable import export_portable

        return export_portable(args.model, args.output)
    if args.command == "verify-portable":
        from netattdetector.portable import verify_portable

        return verify_portable(args.path)
    if args.command == "prepare-multisource":
        from netattdetector.audit import prepare_sources
        from netattdetector.multisource import load_protocol
        from netattdetector.training import ResourceMonitor

        config = load_protocol(args.config)
        paths = {
            name: str(Path(args.data_root) / f"{name}.zip")
            for name in config["development_sources"]
        }
        with ResourceMonitor() as monitor:
            result = prepare_sources(
                paths,
                args.output,
                config["rows_per_source"],
                config["chunk_rows"],
                config["seed"],
                config["audit_buckets"],
            )
        write_json(
            Path(args.output) / "resources.json",
            {"wall_seconds": monitor.elapsed, "sampled_peak_process_rss_mib": monitor.peak / 2**20},
        )
        return {
            "audit": str(Path(args.output) / "audit.json"),
            "cohort_rows": result["cohort_rows"],
        }
    if args.command == "train-multisource":
        from netattdetector.multisource import train_multisource

        result = train_multisource(args.prepared, args.config, args.output, args.baseline)
        return {
            "freeze": str(Path(args.output) / "freeze.json"),
            "selected": result["selected"]["id"],
            "constraints_met_each_source": result["selected"]["constraints_met_each_source"],
        }
    if args.command == "evaluate-frozen":
        from netattdetector.multisource import evaluate_frozen

        result = evaluate_frozen(args.run, args.prepared, args.data, args.output, args.baseline)
        return {name: item["metrics"] for name, item in result["models"].items()}
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
