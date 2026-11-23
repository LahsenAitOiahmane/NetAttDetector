"""Plot aggregate independent family recall and coverage; never read real feature rows."""

import json
from pathlib import Path

import matplotlib.pyplot as plt


def main():
    result = json.loads(Path("reports/evidence/cesnet-frozen-test/evaluation.json").read_text())
    families = result["families"]
    labels = list(families)
    values = [100 * families[name]["pooled_positive_flow_recall"] for name in labels]
    with plt.rc_context({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False}):
        fig, ax = plt.subplots(figsize=(10, 5.5))
        ax.barh(
            labels,
            values,
            color=["#b24c39" if value < 80 else "#267b88" for value in values],
            height=0.58,
        )
        ax.invert_yaxis()
        for index, (name, value) in enumerate(zip(labels, values, strict=True)):
            row = families[name]
            text = f"{value:.2f}%  |  {row['detected_positive_flows']:,}/{row['eligible_positive_flows']:,} segments"
            if value >= 60:
                ax.text(2, index, text, va="center", color="white", fontsize=9)
            else:
                ax.text(value + 1.5, index, text, va="center", color="#282828", fontsize=9)
        ax.set(
            xlim=(0, 106),
            xlabel="Endpoint-matched positive native flow-segment recall (%)",
            title="Frozen independent C&C test: large differences across botnet families",
        )
        ax.set_xticks([0, 20, 40, 60, 80, 100])
        fig.text(
            0.02,
            0.025,
            "32 captures; 30 assessed. 22,240 unknown segments retained.\nNo compatible benign data: precision and false-positive rate are unassessed.",
            fontsize=9,
        )
        fig.subplots_adjust(bottom=0.18, top=0.90, left=0.10, right=0.98)
        target = Path("reports/figures/cesnet-frozen-test.png")
        target.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(target, dpi=160, facecolor="white")
        plt.close(fig)


if __name__ == "__main__":
    main()
