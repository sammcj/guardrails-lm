"""Dataset loading, splitting, and tokenisation."""

from __future__ import annotations

import logging
from typing import Any, cast

from datasets import ClassLabel, Dataset, DatasetDict, concatenate_datasets, load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding, PreTrainedTokenizerBase

from .config import Settings

logger = logging.getLogger(__name__)

LABELS_COLUMN = "labels"  # HF Trainer expects this exact name to compute loss


def _ensure_classlabel(ds: Dataset, label_col: str) -> Dataset:
    """Cast the label column to `ClassLabel` so `stratify_by_column` works."""
    if isinstance(ds.features[label_col], ClassLabel):
        return ds
    values = sorted({str(v) for v in ds[label_col]})
    return ds.cast_column(label_col, ClassLabel(names=values))


def _stratified_split(ds: Dataset, test_size: float, seed: int, label_col: str) -> DatasetDict:
    """Stratified split that falls back to random split if stratification is impossible."""
    try:
        ds = _ensure_classlabel(ds, label_col)
        return ds.train_test_split(test_size=test_size, seed=seed, stratify_by_column=label_col)
    except (TypeError, ValueError) as exc:
        logger.warning("Stratified split failed (%s), falling back to random", exc)
        return ds.train_test_split(test_size=test_size, seed=seed)


def _looks_like_json_source(name: str) -> bool:
    return name.startswith(("http://", "https://", "file://")) or name.endswith((".json", ".jsonl"))


def _load_raw(name: str):
    """Accept either a Hugging Face Hub ID or a JSON/JSONL URL or local path."""
    if _looks_like_json_source(name):
        return load_dataset("json", data_files=name)
    return load_dataset(name)


def load_and_split(
    settings: Settings, val_size: float = 0.1, test_size: float = 0.1
) -> DatasetDict:
    """Load the dataset and produce a train/val/test split.

    If the dataset ships with a `test` split it's honoured as the held-out test set;
    otherwise a split is carved out of `train`. Splits are stratified where possible.
    """
    raw = _load_raw(settings.dataset)
    if isinstance(raw, Dataset):
        train, test = raw, None
    else:
        if "train" not in raw:
            raise ValueError(f"Dataset {settings.dataset!r} has no 'train' split")
        train = raw["train"]
        test = raw.get("test")

    label_col = settings.label_column
    if label_col not in train.column_names:
        raise ValueError(f"Label column {label_col!r} not in dataset columns {train.column_names}")

    if test is None:
        first = _stratified_split(train, val_size + test_size, settings.seed, label_col)
        relative = test_size / (val_size + test_size)
        second = _stratified_split(first["test"], relative, settings.seed, label_col)
        return DatasetDict(
            {"train": first["train"], "val": second["train"], "test": second["test"]}
        )

    split = _stratified_split(train, val_size, settings.seed, label_col)
    return DatasetDict({"train": split["train"], "val": split["test"], "test": test})


def build_tokenizer(settings: Settings) -> PreTrainedTokenizerBase:
    # AutoTokenizer.from_pretrained's declared return includes None and backend
    # stubs; narrow to the concrete base class for downstream call sites.
    return cast(PreTrainedTokenizerBase, AutoTokenizer.from_pretrained(settings.encoder))


def tokenise(
    ds: DatasetDict, tokenizer: PreTrainedTokenizerBase, settings: Settings
) -> DatasetDict:
    """Tokenise all splits. The label column is renamed to `labels` for the Trainer."""
    text_col = settings.text_column

    def _tok(batch: dict[str, Any]) -> Any:
        return tokenizer(batch[text_col], truncation=True, max_length=settings.max_seq_len)

    keep = {settings.label_column}
    drop = [c for c in ds["train"].column_names if c not in keep]
    tokenised = ds.map(_tok, batched=True, remove_columns=drop)
    if settings.label_column != LABELS_COLUMN:
        tokenised = tokenised.rename_column(settings.label_column, LABELS_COLUMN)
    return tokenised


def collator(tokenizer: PreTrainedTokenizerBase) -> DataCollatorWithPadding:
    return DataCollatorWithPadding(tokenizer=tokenizer)


NOTINJECT_TRAIN_SPLITS = ("NotInject_one", "NotInject_two")
NOTINJECT_EVAL_SPLIT = "NotInject_three"


def augment_with_notinject(ds: DatasetDict, settings: Settings) -> DatasetDict:
    """Append two NotInject splits to the training set as benign hard negatives.

    Leaves `NotInject_three` untouched so it remains a clean eval benchmark.
    No-op if loading fails (offline, network error), with a warning.
    """
    try:
        parts = [load_dataset("leolee99/NotInject", split=s) for s in NOTINJECT_TRAIN_SPLITS]
        hard_neg = concatenate_datasets(parts)
    except Exception as exc:
        logger.warning("augment_with_notinject: skipping — %s", exc)
        return ds

    text_col, label_col = settings.text_column, settings.label_column
    hard_neg = hard_neg.map(
        lambda row: {text_col: row["prompt"], label_col: 0},
        remove_columns=hard_neg.column_names,
    )
    # Align feature schema: project main train to (text, label) and cast hard_neg to match
    train_subset = ds["train"].select_columns([text_col, label_col])
    hard_neg = hard_neg.cast(train_subset.features)
    ds["train"] = concatenate_datasets([train_subset, hard_neg])
    logger.info("Augmented training set with %d NotInject benign samples", len(hard_neg))
    return ds


WILDJAILBREAK_DATASET = "allenai/wildjailbreak"
WILDJAILBREAK_BENIGN_TYPE = "adversarial_benign"


def augment_with_wildjailbreak(
    ds: DatasetDict, settings: Settings, n: int | None = None
) -> DatasetDict:
    """Append WildJailbreak adversarial-benign samples to training as benign hard negatives.

    Allen AI built these ~78k prompts as the contrastive partner for jailbreaks: they
    use the same trigger vocabulary ("ignore", "override", role-play framings) but
    have no harmful intent. Training on them should reduce the model's over-defense
    rate on benign prompts that contain attack-adjacent words.

    `n` caps the number of samples (None = all). Requires the caller to have
    accepted the AI2 Responsible Use license on HF Hub and authenticated via
    `hf auth login`.
    """
    try:
        raw = load_dataset(
            WILDJAILBREAK_DATASET,
            "train",
            delimiter="\t",
            keep_default_na=False,
            split="train",
        )
        raw = raw.filter(lambda r: r.get("data_type") == WILDJAILBREAK_BENIGN_TYPE)
    except Exception as exc:
        logger.warning("augment_with_wildjailbreak: skipping — %s", exc)
        return ds

    if n is not None and len(raw) > n:
        raw = raw.shuffle(seed=settings.seed).select(range(n))

    text_col, label_col = settings.text_column, settings.label_column
    # Source column name varies across dataset revisions: current ships `adversarial`,
    # some earlier revisions shipped `prompt`. Prefer `adversarial` when present.
    src_text = "adversarial" if "adversarial" in raw.column_names else "prompt"
    raw = raw.map(
        lambda row: {text_col: row[src_text], label_col: 0},
        remove_columns=raw.column_names,
    )
    train_subset = ds["train"].select_columns([text_col, label_col])
    raw = raw.cast(train_subset.features)
    ds["train"] = concatenate_datasets([train_subset, raw])
    logger.info("Augmented training set with %d WildJailbreak adversarial-benign samples", len(raw))
    return ds
