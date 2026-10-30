"""Render the registered development-only study from saved JSON; never refit models."""

import argparse
import json
from pathlib import Path


def percent(value):
    return f"{100 * value:.2f}%" if value is not None else "—"


def render(study, output, cloud=None, comparison=None):
    local = json.loads((Path(study) / "study.json").read_text())
    other = json.loads((Path(cloud) / "study.json").read_text()) if cloud else None
    parity = json.loads(Path(comparison).read_text()) if comparison else None
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    resource = local["resources"]
    lines = [
        "# Development ablation and training-stability report",
        "",
        "This is development evidence from UNSW/ToN only. CSE was previously observed and failed; "
        "it was not read, tuned against or scored in this study. No new independent source was consumed. "
        "The 80% recall screen is a registered research target, not an agreed operational SLO.",
        "",
        f"Local source revision: `{local['git_revision']}`; dirty: `{local['git_worktree_dirty']}`. "
        f"Completed {len(local['records'])}/48 ablations, {len(local['controls'])}/8 controls and "
        f"{len(local['known_source_transfer'])}/4 known-source transfer fits in "
        f"{resource['wall_seconds']:.1f} seconds; sampled peak process RSS "
        f"{resource['sampled_peak_process_rss_mib']:.1f} MiB. "
        f"Start-new-fit deadline exhausted: `{local['budget_exhausted']}`.",
        "",
        "Thresholds were chosen on calibration observations and then frozen for validation. "
        "No probability-calibration layer was fitted. Source names are metadata; direct float32 "
        "weights give each source equal total contribution. Input order and matrices were canonicalized. "
        "Histogram boosting used 150 iterations with early stopping disabled; XGBoost used 150 "
        "trees with full row/column sampling. Model seed was 42; partition seeds were 7, 42 and 101.",
        "",
        "## Isolation and measurement",
        "",
        f"The common 30-field grouping retained all {sum(x['train']['rows'] + x['calibration']['rows'] + x['validation']['rows'] for x in list(local['partition_payloads'].values())[:1]):,} cohort rows in "
        f"{local['grouping']['groups']:,} groups. The largest group contained "
        f"{local['grouping']['largest_group_rows']:,} rows; "
        f"{local['grouping']['rows_in_label_ambiguous_groups']:,} rows belonged to groups with "
        "mixed labels. These observations were retained together, not removed to improve scores. "
        "Partitions therefore differ from the previous 34-field protocol. Hashes operate on float32 "
        "representations; theoretical collisions and near-duplicates remain limitations.",
        "",
        "| Seed | Train | Calibration | Validation |",
        "|---|---:|---:|---:|",
    ]
    for seed, parts in local["partition_payloads"].items():
        lines.append(
            f"| {seed} | {parts['train']['rows']:,} | {parts['calibration']['rows']:,} | {parts['validation']['rows']:,} |"
        )
    overlap = sum(any(r["transformed_overlap_rows"].values()) for r in local["records"])
    lines += [
        "",
        f"{overlap} ablations had exact post-preprocessing input overlap between partitions. "
        "Any such overlap fails the screen. This check does not establish capture/session independence.",
        "",
        "Zero TTL/duration values and counter/rate distributions are recorded in `measurements.json`. "
        "No zero value was reinterpreted as missing, and no labels were repaired. Matching published "
        "field dictionaries do not prove matching exporters, timeouts, units or direction semantics.",
        "",
        "## Feature-policy results",
        "",
        "Each value below is the worst source/partition seed except AP, which averages sources/seeds. "
        "The screen requires every source and seed to meet precision ≥98%, precision Wilson lower "
        "bound ≥95%, FPR Wilson upper bound ≤0.5%, ≥25 alerts and recall ≥80%, with a feasible "
        "calibration threshold and uncontaminated partitions. Conditional intervals assume independent "
        "labeled rows. Repeated development splits are dependent and do not give a generalization CI.",
        "",
        "| Model / policy | Fields | Min precision | Min recall | Max FPR | Mean AP | All seeds pass |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in local["summary"]:
        lines.append(
            f"| {row['model']} / {row['policy']} | {row['feature_count']} | {percent(row['worst_source_seed_precision'])} | {percent(row['worst_source_seed_recall'])} | {percent(row['max_source_seed_fpr'])} | {row['mean_average_precision']:.4f} | {row['passes_all_seeds']} |"
        )
    lines += [
        "",
        f"**{sum(r['passes_all_seeds'] for r in local['summary'])} of {len(local['summary'])} "
        "model/policy combinations passed every seed.** No model is promoted by a development screen.",
        "",
        "## Training controls",
        "",
        "Each control uses the same seed-42 rows/labels/weights and 100 XGBoost trees. "
        "Only row/column sampling (0.8/0.8 versus 1/1) and one/two CPU threads change. "
        "Each mode repeats twice. Stable tree bytes are retained as hashes; validation score "
        "arrays are retained in workflow/local artifacts.",
        "",
        "| Mode | Local repeat max probability difference |",
        "|---|---:|",
    ]
    for row in local["controls"]:
        if row["repeat"] == 1:
            lines.append(f"| {row['id']} | {row['max_repeat_probability_difference']:.9g} |")
    if other:
        lines += [
            "",
            f"Linux source revision: `{other['git_revision']}`; dirty: `{other['git_worktree_dirty']}`. "
            f"Completed {len(other['records'])} ablations; "
            f"{sum(r['passes_all_seeds'] for r in other['summary'])} combinations passed every seed. "
            f"Runtime {other['resources']['wall_seconds']:.1f} seconds, sampled peak process RSS "
            f"{other['resources']['sampled_peak_process_rss_mib']:.1f} MiB.",
        ]
    if parity:
        lines += [
            "",
            f"Cross-platform ordered raw payloads/labels/weights matched: "
            f"`{parity['same_canonical_inputs_order_labels_weights']}`. "
            "The comparison also requires identical post-preprocessing training and validation matrices "
            "and all eight controls. A score difference with these inputs fixed is a fitting/runtime "
            "difference; it does not establish a specific implementation cause or explain the old run.",
            "",
            "| Control | Windows/Linux max probability difference | Exact scores |",
            "|---|---:|---|",
        ]
        for row in parity["controls"]:
            lines.append(
                f"| {row['control']} | {row['max_probability_difference']:.9g} | {row['exact_score_match']} |"
            )
    lines += [
        "",
        "## Known-source transfer diagnostics",
        "",
        "Only the origin source's train/calibration rows fit the model and choose its threshold. "
        "The other source's validation observations are scored without fitting. Both sources were "
        "already used in research, so these are diagnostics, not independent test results.",
        "",
        "| Policy | Training → validation source | Precision | Recall | FPR | Origin calibration feasible |",
        "|---|---|---:|---:|---:|---|",
    ]
    for row in local["known_source_transfer"]:
        metric = row["metrics"]
        lines.append(
            f"| {row['policy']} | {row['training_source']} → {row['target_source']} | {percent(metric['attack_precision'])} | {percent(metric['attack_recall'])} | {percent(metric['false_positive_rate'])} | {row['calibration_feasible']} |"
        )
    lines += [
        "",
        "## Decision and remaining work",
        "",
        "Keep the research artifacts and the earlier failed external results. Use the measured "
        "policies to prioritize further research; do not choose a production subset from these "
        "development results alone. First recover capture/time/host metadata and verify actual "
        "flow-export measurements on controlled PCAP fixtures, then reserve a fresh source before "
        "fitting. Agree an operator alert budget before a passive shadow pilot. More trees or a larger "
        "random sample cannot by themselves establish measurement parity or new-source robustness.",
        "",
        "See the preregistered [protocol](../docs/STABILITY_PROTOCOL.md), "
        "[reproduction guide](../docs/STABILITY_OPERATIONS.md) and "
        "[ordered backlog](../docs/ROADMAP.md). Full per-source metrics, confidence intervals, "
        "attack-family slices, input signatures and resource measurements are in the saved JSON evidence.",
        "",
    ]
    output.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--cloud")
    parser.add_argument("--comparison")
    args = parser.parse_args()
    render(args.study, args.output, args.cloud, args.comparison)
