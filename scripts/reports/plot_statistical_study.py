"""Render the saved controlled core/statistical comparison; no data rows or model fitting."""

import json
from pathlib import Path

import matplotlib.pyplot as plt


def main():
    study = json.loads(Path("reports/evidence/statistical-study/study.json").read_text())
    ids = ["logistic-core11_control-flow_natural", "logistic-statistics27-flow_natural"]
    records = [next(row for row in study["records"] if row["id"] == name) for name in ids]
    labels = ["Core 11\nnatural weights", "Statistics 27\nnatural weights"]
    with plt.rc_context({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False}):
        fig, (left, right) = plt.subplots(1, 2, figsize=(9, 4.2), layout="constrained")
        ddos = [
            row["metrics"]["validation"]["families"]["DDoS"]["attack_recall"] * 100
            for row in records
        ]
        fpr = [
            row["metrics"]["validation"]["connections"]["false_positive_rate"] * 100
            for row in records
        ]
        left.bar(labels, ddos, color=["#657985", "#16759b"], width=0.55)
        right.bar(labels, fpr, color=["#657985", "#b3262e"], width=0.55)
        for index, value in enumerate(ddos):
            left.text(
                index,
                value + 2,
                f"{value:.2f}%\n{[1, 75][index]}/114 detected",
                ha="center",
                fontsize=9,
            )
        for index, value in enumerate(fpr):
            fp = records[index]["metrics"]["validation"]["connections"]["confusion_matrix"]["fp"]
            right.text(
                index, value + 0.12, f"{value:.2f}%\n{fp}/1,904 benign", ha="center", fontsize=9
            )
        left.set(
            ylim=(0, 100),
            ylabel="Accepted DDoS connection recall (%)",
            title="More DDoS detections",
        )
        left.text(
            0.5,
            0.90,
            "Only 114 of 14,394 publisher\nDDoS UIDs link to accepted flows",
            transform=left.transAxes,
            ha="center",
            fontsize=8,
        )
        right.axhline(0.5, linestyle="--", color="#333333", linewidth=1)
        right.text(0.03, 0.62, "0.5% research FPR ceiling", fontsize=8)
        right.set(
            ylim=(0, 5),
            ylabel="Benign connection false-positive rate (%)",
            title="More false positives",
        )
        fig.suptitle(
            "Same logistic settings, weights and capture partitions — no candidate qualifies",
            fontsize=11,
        )
        fig.savefig("reports/figures/statistical-study.png", dpi=150, facecolor="white")
        plt.close(fig)


if __name__ == "__main__":
    main()
