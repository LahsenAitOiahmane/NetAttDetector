"""Generate deterministic synthetic flow fixtures for CI, never accuracy evidence."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from netattdetector.data import FEATURES


def make_data(rows=2000, seed=42):
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame(rng.lognormal(2, 1, (rows, len(FEATURES))), columns=FEATURES)
    target = rng.binomial(1, 0.25, rows)
    frame["IN_BYTES"] += target * 50
    frame["Label"] = target
    frame["Attack"] = np.where(target, "synthetic_attack", "Benign")
    frame["IPV4_SRC_ADDR"] = [f"10.0.0.{i % 20}" for i in range(rows)]
    frame["IPV4_DST_ADDR"] = "10.0.1.1"
    frame["Timestamp"] = pd.date_range("2024-01-01", periods=rows, freq="s").astype(str)
    return frame


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="data/processed/smoke.csv")
    args = parser.parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    make_data().to_csv(output, index=False)
