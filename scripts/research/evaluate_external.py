"""Evaluate a saved champion on a new source without refitting or retuning it."""

import argparse
import json
from pathlib import Path

from threadpoolctl import threadpool_limits

from netattdetector.data import load_sample
from netattdetector.evaluation import attack_slices, metrics
from netattdetector.inference import load_bundle
from netattdetector.io import sha256_file, write_json
from netattdetector.training import ResourceMonitor, TrainConfig, score


def evaluate(model_path, reference_data, external_data, output):
    output = Path(output)
    if output.exists():
        raise ValueError("External report already exists; use a fresh path.")
    bundle = load_bundle(model_path)
    config = TrainConfig(**bundle["config"])
    original = json.loads(Path(model_path).with_name("metrics.json").read_text(encoding="utf-8"))
    if sha256_file(reference_data) != original["input_profile"]["sha256"]:
        raise ValueError("Reference data does not match the original experiment checksum.")
    with ResourceMonitor() as resources, threadpool_limits(limits=config.threads):
        reference, _ = load_sample(
            reference_data,
            config.max_rows,
            config.chunk_rows,
            config.seed,
            config.member,
            config.group_column,
        )
        external, profile = load_sample(
            external_data, config.max_rows, config.chunk_rows, config.seed
        )
        overlap = external["__hash"].isin(reference["__hash"])
        external = external.loc[~overlap].reset_index(drop=True)
        if external["__label"].nunique() != 2:
            raise ValueError("External set needs both classes after overlap exclusion.")
        scores = score(bundle["model"], external)
        report = {
            "model_sha256": sha256_file(model_path),
            "reference_data_sha256": original["input_profile"]["sha256"],
            "profile": profile,
            "overlapping_feature_rows_excluded": int(overlap.sum()),
            "threshold_policy": "saved in-domain validation threshold; no fitting or tuning",
            "metrics": metrics(external["__label"], scores, bundle["threshold"]),
            "attack_slices": attack_slices(external, scores, bundle["threshold"]),
        }
    report["resources"] = {
        "wall_seconds": resources.elapsed,
        "sampled_peak_process_rss_mib": resources.peak / 2**20,
    }
    write_json(output, report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--reference-data", required=True)
    parser.add_argument("--external", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = evaluate(args.model, args.reference_data, args.external, args.output)
    print(json.dumps(report["metrics"], indent=2))
