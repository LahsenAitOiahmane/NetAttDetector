"""Plot saved baseline/candidate label coverage; never run an exporter or fit a model."""

import argparse
import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot(summary_path, output):
    summary = json.loads(Path(summary_path).read_text(encoding="utf-8"))
    original, candidate, publisher = Counter(), Counter(), Counter()
    for record in summary["records"]:
        original.update(record["baseline"]["matched_connections_by_detail"])
        candidate.update(record["candidate"]["matched_connections_by_detail"])
        source = (
            Path("reports/evidence/capture-expansion/expanded")
            / f"{record['capture_id']}-audit.json"
        )
        counts = json.loads(source.read_text(encoding="utf-8"))["label_log_counts"]
        publisher.update(
            {
                key.removeprefix("detail:"): value
                for key, value in counts.items()
                if key.startswith("detail:")
            }
        )
    names = ["C&C", "PartOfAHorizontalPortScan", "Attack", "DDoS"]
    before = [100 * original[name] / publisher[name] for name in names]
    after = [100 * candidate[name] / publisher[name] for name in names]
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(10.5, 4.7), layout="constrained")
    y = np.arange(len(names))
    ax.barh(y - 0.18, before, 0.34, color="#c7cfd7", label="Original interval rule")
    ax.barh(y + 0.18, after, 0.34, color="#176b87", label="Packet-backed candidate")
    ax.set(
        yticks=y,
        yticklabels=["C&C", "Horizontal port scan", "Attack", "DDoS"],
        xlim=(0, 121),
        xlabel="Matched distinct connections / publisher connections (%)",
        title="Candidate label linkage with exact packet witnesses",
    )
    for i, (old, new) in enumerate(zip(before, after, strict=True)):
        for offset, value in ((-0.18, old), (0.18, new)):
            label = (
                f"{value:.2f}%"
                if 99 < value < 100
                else f"{value:.1f}%"
                if value >= 1
                else f"{value:.3f}%"
            )
            ax.text(value + 1.5, i + offset, label, va="center", fontsize=9)
    ax.legend(loc="upper right", fontsize=9)
    fig.text(
        0.5,
        -0.035,
        "Coverage is a data-preparation measure. No detector was fitted.",
        ha="center",
        fontsize=10,
    )
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    plot(args.summary, args.output)
