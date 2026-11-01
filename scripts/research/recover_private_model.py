"""Recover only a model matching a separately trusted freeze; test synthetic portability."""

import argparse
import importlib.metadata
import io
import json
import platform

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from netattdetector.io import write_json
from netattdetector.native_errors import compare_predictions
from netattdetector.private_artifacts import bounded_read, protect_directory, recover_model


def run(envelope, private_key, freeze_path, output):
    freeze = json.loads(bounded_read(freeze_path, 128 * 1024))
    model_bytes, probes = recover_model(envelope, private_key, freeze)
    # Only reached after matching the independent trusted digest, before pickle loading.
    bundle = joblib.load(io.BytesIO(model_bytes))
    if bundle["threshold"] != freeze["threshold"] or bundle["deployment_authorized"] is not False:
        raise ValueError("Recovered bundle differs from its diagnostic freeze.")
    if (
        not isinstance(probes["rows"], list)
        or not 1 <= len(probes["rows"]) <= 1000
        or probes["features"] != bundle["features"]
    ):
        raise ValueError("Invalid synthetic portability probes.")
    frame = pd.DataFrame(probes["rows"], columns=probes["features"])
    with threadpool_limits(limits=1):
        actual = bundle["model"].predict_proba(frame)[:, 1]
    expected = np.asarray(probes["scores"], dtype=float)
    if expected.shape != actual.shape or not np.isfinite(expected).all():
        raise ValueError("Invalid expected synthetic scores.")
    comparison = compare_predictions(expected, actual, freeze["threshold"])
    passed = (
        comparison["max_absolute_score_difference"] <= 1e-7
        and comparison["frozen_threshold_decision_differences"] == 0
    )
    result = {
        "artifact_sha256": freeze["artifact_sha256"],
        "rows": len(frame),
        "comparison": comparison,
        "passed": passed,
        "platform": platform.system(),
        "python": platform.python_version(),
        "versions": {
            name: importlib.metadata.version(name)
            for name in ("numpy", "pandas", "scikit-learn", "joblib", "cryptography")
        },
        "scope": "Synthetic numeric portability only; no accuracy/generalization/live-network claim.",
        "deployment_authorized": False,
    }
    output = protect_directory(output)
    with (output / "selected.joblib").open("xb") as handle:
        handle.write(model_bytes)
    write_json(output / "portability.json", result)
    if not passed:
        raise ValueError("Synthetic cross-platform portability screen failed; evidence retained.")
    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--envelope", required=True)
    parser.add_argument("--private-key", default=".netatt-keys/recipient-v1.key")
    parser.add_argument(
        "--freeze",
        required=True,
        help="Separately trusted committed freeze, never an arbitrary envelope's claim",
    )
    parser.add_argument("--output", required=True, help="Fresh private ignored directory")
    args = parser.parse_args()
    run(args.envelope, args.private_key, args.freeze, args.output)
