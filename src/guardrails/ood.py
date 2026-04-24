"""Out-of-distribution evaluation against external safety datasets.

Covers the two failure modes InjectGuard metrics can't see:
- **Over-defense**: flagging benign prompts as unsafe (FPR on benign-only sets).
- **Distribution gap**: missing attacks that don't look like InjectGuard (miss rate on attack-only sets).

Datasets are loaded lazily. If a dataset is unavailable or has a different
schema to what the registry declares, the entry is skipped with a warning.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from datasets import concatenate_datasets, load_dataset
from transformers import PreTrainedTokenizerBase

from .config import Settings
from .model import load_for_inference

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class OODSet:
    """Declarative spec for a held-out evaluation dataset.

    `expected_label` pins the dataset to a single class when its examples are
    homogeneous (all benign or all attack). When `label_column` is set the
    dataset is treated as labelled and mixed metrics are reported.
    """

    name: str
    hf_dataset: str
    text_column: str
    split: str | tuple[str, ...] = "train"  # tuple values are loaded and concatenated
    label_column: str | None = None
    expected_label: int | None = None
    limit: int | None = 2000

    def __post_init__(self) -> None:
        if self.label_column is None and self.expected_label is None:
            raise ValueError(f"{self.name}: need label_column or expected_label")
        if self.label_column is not None and self.expected_label is not None:
            raise ValueError(f"{self.name}: set label_column OR expected_label, not both")


DEFAULT_REGISTRY: list[OODSet] = [
    # NotInject: benign prompts that contain injection trigger words (PIGuard authors).
    # We only use NotInject_three (113 rows) so NotInject_one/two remain available as
    # training augmentation without contaminating this benchmark.
    OODSet(
        name="NotInject_three (benign)",
        hf_dataset="leolee99/NotInject",
        text_column="prompt",
        expected_label=0,
        split="NotInject_three",
    ),
    OODSet(
        name="deepset-prompt-injections",
        hf_dataset="deepset/prompt-injections",
        text_column="text",
        label_column="label",
        split="test",
    ),
    OODSet(
        name="jackhhao-jailbreak-classification",
        hf_dataset="jackhhao/jailbreak-classification",
        text_column="prompt",
        label_column="type",
        split="test",
    ),
    OODSet(
        name="awesome-chatgpt-prompts (benign)",
        hf_dataset="fka/awesome-chatgpt-prompts",
        text_column="prompt",
        expected_label=0,
        split="train",
    ),
]


@dataclass(frozen=True)
class OODResult:
    name: str
    hf_dataset: str
    n: int
    mode: str
    accuracy: float | None
    fpr: float | None
    tpr: float | None
    mean_prob_unsafe: float
    note: str = ""


@dataclass(frozen=True)
class OODReport:
    threshold: float
    results: list[OODResult] = field(default_factory=list)


def _normalise_label(value: object) -> int | None:
    """Map common label encodings (int, 'safe'/'unsafe', 'jailbreak'/'benign') to 0/1."""
    if isinstance(value, int):
        return 1 if value != 0 else 0
    if isinstance(value, str):
        v = value.strip().lower()
        if v in {"1", "unsafe", "jailbreak", "injection", "attack", "malicious"}:
            return 1
        if v in {"0", "safe", "benign", "normal", "harmless"}:
            return 0
    return None


@torch.inference_mode()
def _score(
    tokenizer: PreTrainedTokenizerBase,
    model: torch.nn.Module,
    texts: list[str],
    settings: Settings,
) -> np.ndarray:
    out = np.empty(len(texts), dtype=np.float32)
    for start in range(0, len(texts), settings.eval_batch_size):
        chunk = texts[start : start + settings.eval_batch_size]
        enc = tokenizer(
            chunk,
            truncation=True,
            max_length=settings.max_seq_len,
            padding=True,
            return_tensors="pt",
        ).to(settings.device)
        logits = model(**enc).logits
        probs = torch.softmax(logits, dim=-1)[:, 1].float().cpu().numpy()
        out[start : start + len(chunk)] = probs
    return out


def _mixed_result(spec: OODSet, preds: np.ndarray, raw: list, mean_prob: float) -> OODResult:
    valid = [(p, lbl) for p, lbl in zip(preds, raw, strict=True) if lbl is not None]
    if not valid:
        return OODResult(
            name=spec.name,
            hf_dataset=spec.hf_dataset,
            n=0,
            mode="mixed",
            accuracy=None,
            fpr=None,
            tpr=None,
            mean_prob_unsafe=mean_prob,
            note="no recognisable labels",
        )
    preds_arr = np.array([p for p, _ in valid])
    labels_arr = np.array([lbl for _, lbl in valid])
    tp = int(((preds_arr == 1) & (labels_arr == 1)).sum())
    fp = int(((preds_arr == 1) & (labels_arr == 0)).sum())
    fn = int(((preds_arr == 0) & (labels_arr == 1)).sum())
    tn = int(((preds_arr == 0) & (labels_arr == 0)).sum())
    return OODResult(
        name=spec.name,
        hf_dataset=spec.hf_dataset,
        n=len(valid),
        mode="mixed",
        accuracy=float((preds_arr == labels_arr).mean()),
        fpr=fp / (fp + tn) if (fp + tn) > 0 else None,
        tpr=tp / (tp + fn) if (tp + fn) > 0 else None,
        mean_prob_unsafe=mean_prob,
    )


def _homogeneous_result(spec: OODSet, preds: np.ndarray, mean_prob: float) -> OODResult:
    expected = spec.expected_label
    correct = float((preds == expected).mean()) if expected is not None else 0.0
    benign = expected == 0
    return OODResult(
        name=spec.name,
        hf_dataset=spec.hf_dataset,
        n=len(preds),
        mode="benign-only" if benign else "attack-only",
        accuracy=correct,
        fpr=float((preds == 1).mean()) if benign else None,
        tpr=None if benign else float((preds == 1).mean()),
        mean_prob_unsafe=mean_prob,
    )


def _evaluate_set(
    spec: OODSet,
    tokenizer: PreTrainedTokenizerBase,
    model: torch.nn.Module,
    settings: Settings,
    threshold: float,
) -> OODResult | None:
    try:
        if isinstance(spec.split, tuple):
            parts = [load_dataset(spec.hf_dataset, split=s) for s in spec.split]
            ds = concatenate_datasets(parts)
        else:
            ds = load_dataset(spec.hf_dataset, split=spec.split)
    except Exception as exc:
        logger.warning("Skipping %s (%s): %s", spec.name, spec.hf_dataset, exc)
        return None

    if spec.limit is not None and len(ds) > spec.limit:
        ds = ds.shuffle(seed=settings.seed).select(range(spec.limit))

    if spec.text_column not in ds.column_names:
        logger.warning(
            "Skipping %s: text column %r not in %s", spec.name, spec.text_column, ds.column_names
        )
        return None
    if spec.label_column is not None and spec.label_column not in ds.column_names:
        logger.warning(
            "Skipping %s: label column %r not in %s",
            spec.name,
            spec.label_column,
            ds.column_names,
        )
        return None

    texts = list(ds[spec.text_column])
    probs = _score(tokenizer, model, texts, settings)
    preds = (probs >= threshold).astype(int)
    mean_prob = float(probs.mean())

    if spec.label_column is not None:
        raw_labels = [_normalise_label(v) for v in ds[spec.label_column]]
        return _mixed_result(spec, preds, raw_labels, mean_prob)
    return _homogeneous_result(spec, preds, mean_prob)


def score_benign_ood(
    settings: Settings, model_path: Path, registry: list[OODSet] | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Score every benign-only dataset in the registry.

    Returns `(probs, labels)` where labels are all zero — suitable for mixing into a
    calibration pool so thresholds are picked against a realistic benign distribution,
    not just in-distribution val.
    """
    specs = registry if registry is not None else DEFAULT_REGISTRY
    benign = [s for s in specs if s.expected_label == 0]
    tokenizer, model = load_for_inference(settings, model_path)

    all_probs: list[np.ndarray] = []
    for spec in benign:
        try:
            if isinstance(spec.split, tuple):
                parts = [load_dataset(spec.hf_dataset, split=s) for s in spec.split]
                ds = concatenate_datasets(parts)
            else:
                ds = load_dataset(spec.hf_dataset, split=spec.split)
        except Exception as exc:
            logger.warning("Skipping %s (%s): %s", spec.name, spec.hf_dataset, exc)
            continue
        if spec.limit is not None and len(ds) > spec.limit:
            ds = ds.shuffle(seed=settings.seed).select(range(spec.limit))
        if spec.text_column not in ds.column_names:
            continue
        texts = list(ds[spec.text_column])
        all_probs.append(_score(tokenizer, model, texts, settings))

    if not all_probs:
        return np.array([], dtype=np.float32), np.array([], dtype=np.int64)
    probs = np.concatenate(all_probs)
    labels = np.zeros(len(probs), dtype=np.int64)
    return probs, labels


def evaluate_ood(
    settings: Settings,
    model_path: Path,
    threshold: float,
    registry: list[OODSet] | None = None,
) -> OODReport:
    specs = registry if registry is not None else DEFAULT_REGISTRY
    tokenizer, model = load_for_inference(settings, model_path)
    results: list[OODResult] = []
    for spec in specs:
        result = _evaluate_set(spec, tokenizer, model, settings, threshold)
        if result is not None:
            results.append(result)
    return OODReport(threshold=threshold, results=results)
