"""Render development diagnostics and a frozen-source result without selecting models."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def percent(value):
    return f"{100 * value:.2f}%"


def render(prepared, run, output, holdout=None, diagnostic=None):
    prepared, run, output = Path(prepared), Path(run), Path(output)
    audit = json.loads((prepared / "audit.json").read_text())
    diagnostics = json.loads((prepared / "diagnostics.json").read_text())
    freeze = json.loads((run / "freeze.json").read_text())
    report = json.loads(Path(holdout).read_text()) if holdout else None
    extra = json.loads(Path(diagnostic).read_text()) if diagnostic else None
    figures = output.parent / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), layout="constrained")
    fig.suptitle(
        "Development evidence: source shift and constrained validation", fontsize=17, weight="bold"
    )
    features = diagnostics["features"][:12][::-1]
    axes[0].barh(
        [x["feature"] for x in features], [x["ks_benign"] for x in features], color="#147d92"
    )
    axes[0].set_xlim(0, 1)
    axes[0].set_xlabel("Benign-traffic KS distance (0=identical empirical distributions)")
    axes[0].set_title("Same field names, different observed distributions")
    records = freeze["validation_leaderboard"]
    best_by_candidate = {}
    for record in records:
        name = record["spec"]["id"]
        recall = min(m["attack_recall"] for m in record["by_source"].values())
        best_by_candidate[name] = max(best_by_candidate.get(name, 0), recall)
    names = list(best_by_candidate)
    axes[1].barh(
        names[::-1],
        [best_by_candidate[n] for n in names[::-1]],
        color=[
            "#147d92" if n == freeze["selected"]["spec"]["id"] else "#df813e" for n in names[::-1]
        ],
    )
    axes[1].set_xlim(0, 1)
    axes[1].set_xlabel("Worst-source validation recall (best calibration variant)")
    axes[1].set_title("Validation recall: selected candidate in teal")
    fig.savefig(figures / "multisource-development.png", dpi=150, facecolor="white")
    plt.close(fig)

    text = "# Multi-source experiment and model card\n\n"
    text += "Registered protocol: version 2, amended from the development audit before any candidate fit or fresh holdout access.\n\n"
    text += "## Data audit\n\n| Source | Full rows scanned | Benign | Attack | Label/name disagreements |\n|---|---:|---:|---:|---:|\n"
    for name, p in audit["sources"].items():
        text += f"| {name} | {p['rows_scanned']:,} | {p['label_counts']['0']:,} | {p['label_counts']['1']:,} | {p['binary_label_vs_attack_name_disagreements']:,} |\n"
    full, stable = audit["global_audit"]["full"], audit["global_audit"]["stable"]
    text += f"\nFull-feature repeated rows: **{full['repeated_rows']:,}**. Full-feature conflicting groups: **{full['conflicting_feature_groups']:,}**, containing **{full['rows_in_conflicting_groups']:,}** rows. These checks cover both complete development inputs.\n\n"
    text += f"The reduced representation has {stable['rows_in_conflicting_groups']:,} rows in ambiguous groups. They are retained because dropping measurements can make distinct benign/attack observations indistinguishable; this is not automatically a label error. Reduced-feature groups stay together across development partitions.\n\n"
    text += f"The bounded cohort contains {audit['cohort_rows']:,} rows after removing {audit['sample_global_conflict_rows_removed']:,} sampled rows in globally conflicting full groups and {audit['sample_repeated_full_rows_removed']:,} sampled full duplicates. Sampling then deduplicating changes prevalence; full counts and cohort counts are recorded in the audit JSON.\n\n"
    text += "Feature dictionaries have matching hashes where reported; this confirms identical supplied definitions, not independently measured exporter parity. KS and source-conditional label associations describe shift without proving causation. All numeric nonfinite/coercion/range counts are in the audit. Fingerprints use 64-bit hashes of float32 representations, with a small theoretical collision risk.\n\n"
    text += "## Development selection\n\n"
    if extra:
        text += "### Original-model dependence diagnosis\n\nOn a capped development diagnostic sample, shuffling MIN_TTL changed the original model's average precision as follows. Positive drops indicate measured dependence; negative drops indicate the shuffle improved ranking. This sample may overlap original baseline fitting and is not independent performance evidence.\n\n| Source | Mean AP drop when MIN_TTL is shuffled | Repeats |\n|---|---:|---:|\n"
        for source, item in extra["permutation_importance"].items():
            ttl = next(v for v in item["features"] if v["feature"] == "MIN_TTL")
            text += f"| {source} | {ttl['mean_ap_drop']:.5f} | 3 |\n"
        text += "\nThis supports a source-specific shortcut explanation alongside measured shift, but correlated predictors and unverified exporter semantics prevent a complete causal explanation. The primary [dataset paper, section 4](https://arxiv.org/html/2101.11315v2#S4) also excluded TTL from its UNSW experiments because of label association. Our frozen selected full-feature model still includes TTL and retains this limitation. Removing the six-field set did not meet the registered constraints in development. A targeted TTL-only ablation is a future experiment, not an unreported post-test model change.\n\n"
        ambiguity = extra["reduced_representation_ambiguity"]
        text += f"The development cohort has {ambiguity['ambiguous_groups']:,} reduced-feature groups containing both labels. Any deterministic classifier restricted to that representation must make at least {ambiguity['minimum_empirical_errors_for_a_deterministic_reduced_classifier']:,} errors on these recorded rows. This finite-cohort counting bound does not predict future-network performance.\n\n"
    text += f"Selected: **{freeze['selected']['id']}**, threshold `{freeze['selected']['threshold']}`. Constraints met in each development source: **{freeze['selected']['constraints_met_each_source']}**.\n\n"
    text += "| Candidate / calibration | Each-source constraints | Worst-source precision | Worst-source recall | Mean Brier | Fit seconds | Model bytes |\n|---|---|---:|---:|---:|---:|---:|\n"
    for r in records:
        values = list(r["by_source"].values())
        text += f"| {r['id']} | {r['constraints_met_each_source']} | {percent(min(m['attack_precision'] for m in values))} | {percent(min(m['attack_recall'] for m in values))} | {np.mean([m['brier_score'] for m in values]):.5f} | {r['fit_seconds']:.2f} | {r['model_bytes']:,} |\n"
    text += "\nThese are model-selection results, not independent transfer estimates. Sigmoid calibration uses a disjoint calibration partition and never refits preprocessing/the classifier. Equal-source and equal-source/class weights affect training only.\n\n"
    text += "Twenty source/label-stratified group folds approximate 60/15/25 train/calibration/validation, with disjoint reduced-feature groups. They are not time-, host- or session-held-out partitions. Exact row counts and source/class composition are recorded in the freeze.\n\n"
    text += (
        "![Source distributions and validation comparison](figures/multisource-development.png)\n\n"
    )
    if extra:
        text += "### Illustrative validation trade-offs\n\nThese thresholds were explored on validation after freezing, without changing the selected artifact or testing alternatives on CSE. They illustrate constraint costs and are not new independent test results.\n\n| Illustrative precision target | Feasible | Worst-source precision | Worst-source recall | Maximum source FPR |\n|---|---|---:|---:|---:|\n"
        for item in extra["illustrative_validation_operating_points"]:
            values = list(item["by_source"].values())
            text += f"| {percent(item['precision_target'])} | {item['feasible']} | {percent(min(m['attack_precision'] for m in values))} | {percent(min(m['attack_recall'] for m in values))} | {percent(max(m['false_positive_rate'] for m in values))} |\n"
        text += "\nOther support/FPR confidence constraints remain enforced. The illustrative precision lower bound is min(0.95, target−0.03). Operational trade-offs require an agreed local alert budget.\n\n"
    if report:
        text += "## Frozen independent CSE-CIC result\n\n"
        text += f"Scanned {report['profile']['rows_scanned']:,} fresh-source rows. Evaluated {report['evaluated_rows']:,} sampled, deduplicated rows, excluding {report['all_development_overlap_rows_excluded']:,} full-feature matches found anywhere in development. Reduced-representation overlaps and ambiguity are reported and retained.\n\n"
        text += "| Frozen model | Accuracy | Attack precision | Attack recall | Benign FPR | Average precision |\n|---|---:|---:|---:|---:|---:|\n"
        for name, item in report["models"].items():
            m = item["metrics"]
            text += f"| {name} | {percent(m['accuracy'])} | {percent(m['attack_precision'])} | {percent(m['attack_recall'])} | {percent(m['false_positive_rate'])} | {m['average_precision']:.5f} |\n"
        new = report["models"]["multisource"]["metrics"]
        text += f"\nPoint precision/FPR targets met: **{report['point_targets_met']}**. This does not establish an acceptable production recall or alert/day budget. No parameter or threshold was changed after observing this result.\n\n"
        text += "| New-model metric | Conditional Wilson 95% interval |\n|---|---|\n"
        for name, interval in new["conditional_wilson_95"].items():
            if isinstance(interval, list):
                text += f"| {name} | {percent(interval[0])}–{percent(interval[1])} |\n"
        text += "\nThese intervals assume independent labeled rows and exclude capture dependence, label uncertainty and model-selection uncertainty.\n\n"
        text += "| Attack family | Attack rows | Recall | Conditional 95% interval |\n|---|---:|---:|---|\n"
        for name, item in report["models"]["multisource"]["attack_slices"].items():
            if item["attack_rows"]:
                low, high = item["attack_recall_conditional_wilson_95"]
                text += f"| {name} | {item['attack_rows']:,} | {percent(item['attack_recall'])} | {percent(low)}–{percent(high)} |\n"
        fig, axes = plt.subplots(1, 3, figsize=(12, 4), layout="constrained")
        fig.suptitle("Fresh CSE-CIC source — frozen comparison", fontsize=17, weight="bold")
        names = list(report["models"])
        labels = ["Multi-source" if n == "multisource" else "Original UNSW" for n in names]
        for ax, key, title in zip(
            axes,
            ["attack_precision", "attack_recall", "false_positive_rate"],
            ["Attack precision", "Attack recall", "Benign false-positive rate"],
            strict=True,
        ):
            values = [100 * report["models"][n]["metrics"][key] for n in names]
            bars = ax.bar(labels, values, color=["#147d92", "#df813e"])
            ax.bar_label(bars, labels=[f"{v:.2f}%" for v in values], padding=3)
            ax.set_ylim(0, max(max(values) * 1.25, 0.1))
            ax.set_title(title)
            ax.set_ylabel("Percent")
        fig.savefig(figures / "multisource-holdout.png", dpi=150, facecolor="white")
        plt.close(fig)
        text += "\n![Frozen independent source comparison](figures/multisource-holdout.png)\n\n"
    text += "## Resource scope and deployment status\n\n"
    text += f"Selection/training: {freeze['resources']['wall_seconds']:.2f} seconds, sampled peak process RSS {freeze['resources']['sampled_peak_process_rss_mib']:.1f} MiB, two configured threads. Selected bundle: {(run / 'model.joblib').stat().st_size:,} bytes. Warm inference median/p95: {freeze['inference']['median_batch_ms']:.2f}/{freeze['inference']['p95_batch_ms']:.2f} ms per {freeze['inference']['batch_rows']} rows. Parsing, export and queueing are excluded. Audit preparation and holdout scanning are separate measurements.\n\n"
    text += "This is a research artifact. Live exporter parity, host/time robustness, minority support, local shadow testing, operational alert budgets and commercial-use permissions remain unresolved release gates.\n"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", required=True)
    parser.add_argument("--run", required=True)
    parser.add_argument("--holdout")
    parser.add_argument("--diagnostic")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    render(args.prepared, args.run, args.output, args.holdout, args.diagnostic)
