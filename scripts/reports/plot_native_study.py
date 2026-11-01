"""Plot saved native diagnostic results only; never fit models or open row-level data."""

import json
from pathlib import Path

import matplotlib.pyplot as plt


def main():
    study = json.loads(Path("reports/evidence/native-study/study.json").read_text())
    chosen = next(row for row in study["records"] if row["id"] == study["selected_diagnostic_id"])
    colors = {"logistic": "#16759b", "histogram": "#be702b", "extra_trees": "#47754a"}
    with plt.rc_context({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False}):
        fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4.5), layout="constrained")
        for name, color in colors.items():
            rows = [row for row in study["records"] if row["specification"]["model"] == name]
            left.scatter(
                [
                    row["metrics"]["validation"]["connections"]["false_positive_rate"] * 100
                    for row in rows
                ],
                [
                    row["metrics"]["validation"]["connections"]["attack_recall"] * 100
                    for row in rows
                ],
                color=color,
                s=50,
                alpha=0.7,
                label=name.replace("_", " "),
            )
        left.axvline(0.5, color="#b3262e", linestyle="--", linewidth=1)
        left.axhline(80, color="#b3262e", linestyle="--", linewidth=1)
        left.fill_between([0, 0.5], 80, 105, color="#47754a", alpha=0.12)
        left.set(
            xlim=(0, 3.5),
            ylim=(0, 105),
            xlabel="Benign connection false-positive rate (%)",
            ylabel="Attack connection recall (%)",
            title="Twelve candidates: none passes both screens",
        )
        left.text(0.08, 103, "Research target", va="top", fontsize=8, color="#47754a")
        result = chosen["metrics"]["validation"]["connections"]
        left.annotate(
            "Selected diagnostic\n98.37% recall / 2.89% FPR",
            xy=(result["false_positive_rate"] * 100, result["attack_recall"] * 100),
            xytext=(0.8, 53),
            arrowprops={"arrowstyle": "->", "color": "#333333"},
            fontsize=9,
        )
        left.legend(loc="lower left", frameon=False, fontsize=8)
        families = chosen["metrics"]["validation"]["families"]
        names = ["C&C", "PartOfAHorizontalPortScan", "DDoS"]
        values = [families[name]["attack_recall"] * 100 for name in names]
        bars = right.bar(
            ["C&C\n6,706 UIDs", "Scanning\n103 UIDs", "DDoS\n114 UIDs"],
            values,
            color=["#16759b", "#16759b", "#b3262e"],
            width=0.55,
        )
        for bar, value in zip(bars, values, strict=True):
            right.text(
                bar.get_x() + bar.get_width() / 2,
                value + 2,
                f"{value:.2f}%",
                ha="center",
                fontsize=10,
            )
        right.set(
            ylim=(0, 115),
            ylabel="Accepted attack connection recall (%)",
            title="Selected model: DDoS remains a blind spot",
        )
        right.text(
            0.84,
            0.27,
            "DDoS linkage:\n114 / 14,394 UIDs\n\nDetected:\n1 / 114 accepted",
            transform=right.transAxes,
            ha="center",
            fontsize=9,
        )
        fig.suptitle(
            "Native capture/time diagnostic — validation selected the model; no external test",
            fontsize=12,
        )
        output = Path("reports/figures/native-study.png")
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=150, facecolor="white")
        plt.close(fig)


if __name__ == "__main__":
    main()
