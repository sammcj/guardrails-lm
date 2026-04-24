"""Threshold picking for asymmetric FP/FN costs.

A binary classifier's default 0.5 threshold assumes the two error types cost the
same. In production they rarely do: missing an attack (FN) might cost more than
flagging a benign prompt (FP), or vice versa. This module picks an operating
point from validation-set scores.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import numpy as np

Mode = Literal["f1", "cost", "fpr_budget"]

THRESHOLD_FILENAME = "threshold.json"


@dataclass(frozen=True)
class Recommendation:
    """Chosen operating point plus metrics at that threshold."""

    threshold: float
    precision: float
    recall: float
    f1: float
    fpr: float
    tpr: float
    accuracy: float
    mode: Mode
    criterion: str
    n: int
    data_source: str = "val"  # `val` | `val+ood_benign` | custom — aids later traceability


def _metrics_at(probs: np.ndarray, labels: np.ndarray, threshold: float) -> dict[str, float]:
    preds = (probs >= threshold).astype(int)
    tp = int(((preds == 1) & (labels == 1)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())
    pos = tp + fn
    neg = fp + tn
    total = pos + neg
    return {
        "precision": tp / (tp + fp) if (tp + fp) > 0 else 0.0,
        "recall": tp / pos if pos > 0 else 0.0,
        "tpr": tp / pos if pos > 0 else 0.0,
        "fpr": fp / neg if neg > 0 else 0.0,
        "accuracy": (tp + tn) / total if total > 0 else 0.0,
    }


def _f1(precision: float, recall: float) -> float:
    return 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0


def _candidate_thresholds(probs: np.ndarray) -> np.ndarray:
    """Use midpoints between sorted unique scores plus the endpoints."""
    uniq = np.unique(probs)
    if len(uniq) <= 1:
        return np.array([0.5])
    mids = (uniq[1:] + uniq[:-1]) / 2.0
    return np.concatenate([[0.0], mids, [1.0]])


def pick_by_f1(probs: np.ndarray, labels: np.ndarray) -> Recommendation:
    """Pick the threshold that maximises F1 on the validation split.

    This is the default calibration mode because F1 is robust to class imbalance
    (unlike raw accuracy) and requires no user-specified cost ratio.
    """
    thresholds = _candidate_thresholds(probs)
    best_t, best_f1 = 0.5, -1.0
    for t in thresholds:
        m = _metrics_at(probs, labels, float(t))
        f1 = _f1(m["precision"], m["recall"])
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    m = _metrics_at(probs, labels, best_t)
    return Recommendation(
        threshold=best_t,
        precision=m["precision"],
        recall=m["recall"],
        f1=_f1(m["precision"], m["recall"]),
        fpr=m["fpr"],
        tpr=m["tpr"],
        accuracy=m["accuracy"],
        mode="f1",
        criterion="max F1",
        n=len(labels),
    )


def pick_by_cost(
    probs: np.ndarray, labels: np.ndarray, cost_fp: float = 1.0, cost_fn: float = 1.0
) -> Recommendation:
    """Return the threshold that minimises cost_fp * FP + cost_fn * FN."""
    if cost_fp < 0 or cost_fn < 0:
        raise ValueError("costs must be non-negative")
    thresholds = _candidate_thresholds(probs)
    best_t = 0.5
    best_cost = float("inf")
    for t in thresholds:
        preds = (probs >= t).astype(int)
        fp = ((preds == 1) & (labels == 0)).sum()
        fn = ((preds == 0) & (labels == 1)).sum()
        cost = cost_fp * fp + cost_fn * fn
        if cost < best_cost:
            best_cost = float(cost)
            best_t = float(t)
    m = _metrics_at(probs, labels, best_t)
    return Recommendation(
        threshold=best_t,
        precision=m["precision"],
        recall=m["recall"],
        f1=_f1(m["precision"], m["recall"]),
        fpr=m["fpr"],
        tpr=m["tpr"],
        accuracy=m["accuracy"],
        mode="cost",
        criterion=f"cost_fp={cost_fp},cost_fn={cost_fn}",
        n=len(labels),
    )


def pick_by_fpr_budget(
    probs: np.ndarray,
    labels: np.ndarray,
    max_fpr: float,
    min_threshold: float = 0.3,
) -> Recommendation:
    """Return the lowest threshold whose FPR stays under `max_fpr`, subject to a floor.

    `min_threshold` prevents the common trap where in-distribution scores let the
    algorithm pick a near-zero threshold that satisfies the FPR budget on val but
    flags benign prompts out-of-distribution. 0.3 is a conservative default; pass
    0.0 to disable the floor.
    """
    if not 0.0 <= max_fpr <= 1.0:
        raise ValueError("max_fpr must be in [0, 1]")
    if not 0.0 <= min_threshold <= 1.0:
        raise ValueError("min_threshold must be in [0, 1]")
    thresholds = np.sort(_candidate_thresholds(probs))[::-1]  # high -> low
    chosen = 1.0
    clamped = False
    for t in thresholds:
        if t < min_threshold:
            chosen = max(chosen, min_threshold)
            clamped = True
            break
        fpr = _metrics_at(probs, labels, float(t))["fpr"]
        if fpr <= max_fpr:
            chosen = float(t)
        else:
            break
    m = _metrics_at(probs, labels, chosen)
    criterion = f"max_fpr={max_fpr},min_threshold={min_threshold}"
    if clamped:
        criterion += " (floor hit)"
    return Recommendation(
        threshold=chosen,
        precision=m["precision"],
        recall=m["recall"],
        f1=_f1(m["precision"], m["recall"]),
        fpr=m["fpr"],
        tpr=m["tpr"],
        accuracy=m["accuracy"],
        mode="fpr_budget",
        criterion=criterion,
        n=len(labels),
    )


def save(rec: Recommendation, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(rec), indent=2))


def load(path: Path) -> Recommendation | None:
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    return Recommendation(**data)
