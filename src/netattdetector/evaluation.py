"""Binary IDS metrics and validation-only operating-point selection."""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


def wilson_bounds(successes, totals):
    """Vectorized 95% binomial intervals, conditional on independent labeled rows."""
    totals = np.asarray(totals, dtype=float)
    n = np.maximum(totals, 1)
    p = np.asarray(successes, dtype=float) / n
    z = 1.959963984540054
    center = (p + z * z / (2 * n)) / (1 + z * z / n)
    radius = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return np.where(
        (totals > 0) & (np.asarray(successes) > 0), np.maximum(0, center - radius), 0
    ), np.where((totals > 0) & (np.asarray(successes) < totals), np.minimum(1, center + radius), 1)


def wilson_interval(successes, total):
    if total == 0:
        return None
    low, high = wilson_bounds(successes, total)
    return [float(low), float(high)]


def metrics(y, scores, threshold):
    y = np.asarray(y)
    prediction = np.asarray(scores) >= threshold
    tn, fp, fn, tp = confusion_matrix(y, prediction, labels=[0, 1]).ravel()
    both = len(np.unique(y)) == 2
    return {
        "rows": len(y),
        "attack_prevalence": float(y.mean()),
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(y, prediction)) if both else None,
        "attack_precision": float(precision_score(y, prediction, zero_division=0)),
        "attack_recall": float(recall_score(y, prediction, zero_division=0)),
        "attack_f1": float(f1_score(y, prediction, zero_division=0)),
        "macro_f1": float(f1_score(y, prediction, labels=[0, 1], average="macro", zero_division=0)),
        "average_precision": float(average_precision_score(y, scores)) if both else None,
        "roc_auc": float(roc_auc_score(y, scores)) if both else None,
        "brier_score": float(brier_score_loss(y, scores)),
        "false_positive_rate": float(fp / (fp + tn)) if fp + tn else None,
        "false_alarms_per_10000_benign_flows": float(10000 * fp / (fp + tn)) if fp + tn else None,
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "conditional_wilson_95": {
            "attack_precision": wilson_interval(tp, tp + fp),
            "attack_recall": wilson_interval(tp, tp + fn),
            "false_positive_rate": wilson_interval(fp, fp + tn),
            "limitation": "Binomial intervals assume independent labels/rows; not capture-level or model-selection uncertainty.",
        },
    }


def choose_threshold(y, scores, min_precision=0.98, max_fpr=0.005):
    """Maximize recall subject to precision/FPR; explicitly flag infeasible targets."""
    y = np.asarray(y)
    if len(np.unique(y)) != 2:
        raise ValueError("Threshold validation requires both binary classes.")
    precision, recall, thresholds = precision_recall_curve(y, scores)
    precision, recall = precision[:-1], recall[:-1]
    positives = y.sum()
    negatives = len(y) - positives
    tp = recall * positives
    predicted = np.divide(tp, precision, out=np.zeros_like(tp), where=precision > 0)
    fpr = (predicted - tp) / negatives
    eligible = (precision >= min_precision) & (fpr <= max_fpr + 1e-12)
    if eligible.any():
        indices = np.flatnonzero(eligible)
        selected = max(indices, key=lambda i: (recall[i], precision[i], thresholds[i]))
    else:
        f1 = np.divide(
            2 * precision * recall,
            precision + recall,
            out=np.zeros_like(tp),
            where=(precision + recall) > 0,
        )
        selected = int(np.argmax(f1))
    return float(thresholds[selected]), bool(eligible.any())


def attack_slices(frame, scores, threshold):
    predictions = np.asarray(scores) >= threshold
    result = {}
    for category, indices in frame.groupby("__attack").indices.items():
        y = frame.iloc[indices]["__label"].to_numpy()
        pred = predictions[indices]
        result[str(category)] = {
            "support": len(indices),
            "attack_rows": int(y.sum()),
            "attack_recall": float(pred[y == 1].mean()) if (y == 1).any() else None,
            "benign_false_positive_rate": float(pred[y == 0].mean()) if (y == 0).any() else None,
            "attack_recall_conditional_wilson_95": wilson_interval(
                int(pred[y == 1].sum()), int((y == 1).sum())
            ),
        }
    return result
