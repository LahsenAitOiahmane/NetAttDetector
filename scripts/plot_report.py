"""Render compact benchmark figures from JSON evidence (requires the analysis extra)."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot(metrics_path, profile_path, output):
    report = json.loads(Path(metrics_path).read_text(encoding="utf-8"))
    profile = json.loads(Path(profile_path).read_text(encoding="utf-8"))
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), layout="constrained")
    fig.suptitle("NetAttDetector — initial UNSW research baseline", fontsize=19, weight="bold")
    counts = profile["full_attack_counts"]
    names = sorted(counts, key=counts.get)
    ax = axes[0, 0]
    ax.barh(
        names,
        [counts[x] for x in names],
        color=["#147d92" if x == "Benign" else "#df813e" for x in names],
    )
    ax.set_xscale("log")
    ax.set_title("Complete source: class imbalance")
    ax.set_xlabel("Flows (log scale)")
    ax = axes[0, 1]
    entries = report["validation_leaderboard"]
    names = [x["model"] for x in entries]
    positions = np.arange(len(names))
    ax.bar(
        positions - 0.18,
        [x["validation"]["average_precision"] for x in entries],
        0.36,
        label="Average precision",
        color="#147d92",
    )
    ax.bar(
        positions + 0.18,
        [x["validation"]["attack_recall"] for x in entries],
        0.36,
        label="Recall at selected threshold",
        color="#df813e",
    )
    ax.set_xticks(positions, names, rotation=15)
    ax.set_ylim(0, 1.1)
    ax.set_title("Validation comparison — only this split selects the model")
    ax.legend(loc="upper left", fontsize=8)
    ax = axes[1, 0]
    c = report["test"]["confusion_matrix"]
    matrix = np.array([[c["tn"], c["fp"]], [c["fn"], c["tp"]]])
    ax.imshow(np.log1p(matrix), cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(
                j,
                i,
                f"{matrix[i, j]:,}",
                ha="center",
                va="center",
                fontsize=18,
                color="white" if np.log1p(matrix[i, j]) > 7 else "#163848",
            )
    ax.set_xticks([0, 1], ["Benign", "Attack"])
    ax.set_yticks([0, 1], ["Benign", "Attack"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Untouched random holdout: confusion counts")
    ax = axes[1, 1]
    slices = {k: v for k, v in report["test_attack_slices"].items() if v["attack_rows"]}
    names = sorted(slices, key=lambda x: slices[x]["attack_recall"])
    ax.barh(
        [f"{x} (n={slices[x]['support']})" for x in names],
        [slices[x]["attack_recall"] for x in names],
        color="#147d92",
    )
    ax.set_xlim(0, 1.1)
    ax.set_xlabel("Binary attack recall; tiny supports are not reliable estimates")
    ax.set_title("Attack-family errors hidden by 99.31% accuracy")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    plot(args.metrics, args.profile, args.output)
