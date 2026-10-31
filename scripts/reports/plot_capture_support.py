"""Plot saved connection-level linkage coverage; never fit or acquire data."""

import argparse
import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot(evidence, output):
    reports = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(Path(evidence).glob("*-audit.json"))
    ]
    if len(reports) != 3:
        raise ValueError("Expected three reviewed expansion capture audits.")
    labels = [item["capture_id"].removeprefix("CTU-IoT-Malware-Capture-") for item in reports]
    publisher = [item["label_log_counts"]["coarse:Malicious"] for item in reports]
    matched = [item["matched_zeek_connection_class_counts"]["1"] for item in reports]
    totals, candidates = Counter(), Counter()
    for item in reports:
        for name, counts in item["connection_coverage_by_detail"].items():
            if name != "-":
                totals[name] += counts["publisher_connections"]
                candidates[name] += counts["candidate_matched_connections"]
    families = ["C&C", "PartOfAHorizontalPortScan", "Attack", "DDoS"]
    percentages = [100 * candidates[name] / totals[name] for name in families]
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, (left, right) = plt.subplots(1, 2, figsize=(11.4, 4.4), layout="constrained")
    y = np.arange(3)
    left.barh(y - 0.18, publisher, 0.34, label="Publisher attack connections", color="#c7cfd7")
    left.barh(
        y + 0.18, matched, 0.34, label="Candidate matched attack connections", color="#176b87"
    )
    left.set(
        yticks=y,
        yticklabels=labels,
        xscale="log",
        xlabel="Distinct Zeek connections (log scale)",
        ylabel="IoT-23 scenario",
        title="Attack support retained per capture",
    )
    left.legend(loc="lower right", fontsize=8)
    for i, value in enumerate(matched):
        left.text(value * 1.12, i + 0.18, f"{value:,}", va="center", fontsize=9)
    right.barh(families, percentages, color=["#176b87"] * 3 + ["#bd663c"])
    right.set(
        xlim=(0, 116),
        xlabel="Candidate matched / publisher connections (%)",
        title="Coverage varies sharply by attack label",
    )
    right.set_yticks(range(4), ["C&C", "Horizontal port scan", "Attack", "DDoS"])
    for i, value in enumerate(percentages):
        text = f"{value:.1f}%" if value >= 0.1 else f"{value:.3f}%"
        right.text(value + 2, i, text, va="center", fontsize=9)
    fig.suptitle("Development label linkage — connection counts", fontsize=14)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    plot(args.evidence, args.output)
