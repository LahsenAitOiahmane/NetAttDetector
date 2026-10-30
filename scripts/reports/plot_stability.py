"""Plot saved development metrics and controlled score differences without fitting."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def draw(study, comparison, output):
    report = json.loads((Path(study) / "study.json").read_text())
    parity = json.loads(Path(comparison).read_text())
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    policies = report["config"]["feature_policies"]
    y = np.arange(len(policies))
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "savefig.facecolor": "white",
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.8), sharey=True, layout="constrained")
    for name, offset, color in (("hist", -0.12, "#007f86"), ("xgboost", 0.12, "#c36b29")):
        rows = [
            next(r for r in report["summary"] if r["model"] == name and r["policy"] == p)
            for p in policies
        ]
        for ax, metric in zip(
            axes, ("worst_source_seed_precision", "max_source_seed_fpr"), strict=True
        ):
            ax.scatter(
                [100 * r[metric] for r in rows], y + offset, color=color, s=48, label=name, zorder=3
            )
    axes[0].axvline(98, color="#9b3240", linestyle="--", label="98% precision target")
    axes[1].axvline(0.5, color="#9b3240", linestyle="--", label="0.5% point-FPR reference")
    axes[0].set(xlabel="Minimum precision across sources and seeds (%)", xlim=(78, 100))
    axes[1].set(xlabel="Maximum benign FPR across sources and seeds (%)", xlim=(0, 20))
    axes[0].set_yticks(y, [p.replace("_", " ") for p in policies])
    axes[0].invert_yaxis()
    for ax in axes:
        ax.grid(axis="x", alpha=0.2)
        ax.legend(loc="upper right", fontsize=8)
    fig.suptitle("No tested feature policy met the development quality screen", fontsize=15)
    fig.supxlabel(
        "UNSW + ToN development only • 48 ablations • thresholds frozen on calibration\nAll calibration constraints were infeasible; points use declared F1 fallbacks. FPR confidence gates also apply.",
        fontsize=10,
    )
    fig.savefig(output / "stability-ablation.png", dpi=180)
    plt.close(fig)

    modes = ["stochastic-1t:0", "stochastic-2t:0", "full-1t:0", "full-2t:0"]
    rows = [next(r for r in parity["controls"] if r["control"] == mode) for mode in modes]
    values = [r["max_probability_difference"] for r in rows]
    fig, ax = plt.subplots(figsize=(10.2, 4.3), layout="constrained")
    bars = ax.barh(
        np.arange(4),
        np.maximum(np.asarray(values) - 1e-9, 0),
        color=["#c36b29"] * 2 + ["#007f86"] * 2,
        height=0.55,
        left=1e-9,
    )
    ax.set_xscale("log")
    ax.set_xlim(1e-9, 1)
    ax.set_yticks(
        np.arange(4),
        [
            "0.8 sampling / 1 thread",
            "0.8 sampling / 2 threads",
            "full sampling / 1 thread",
            "full sampling / 2 threads",
        ],
    )
    ax.invert_yaxis()
    ax.set_xlabel("Maximum Windows/Linux probability difference (log scale)")
    ax.grid(axis="x", alpha=0.2)
    for bar, value in zip(bars, values, strict=True):
        ax.text(
            max(2e-9, value * 1.35),
            bar.get_y() + bar.get_height() / 2,
            f"{value:.3g}",
            va="center",
            fontsize=10,
        )
    fig.suptitle("Full sampling reduced the observed cross-platform score difference", fontsize=14)
    fig.supxlabel(
        "Same ordered raw and transformed control inputs • 100 trees • repeated modes matched within each platform\nThese controls do not establish complete training determinism; one 150-tree ablation retained decision differences.",
        fontsize=9,
    )
    fig.savefig(output / "stability-controls.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", required=True)
    parser.add_argument("--comparison", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    draw(args.study, args.comparison, args.output)
