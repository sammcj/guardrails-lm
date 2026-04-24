"""Regression tests for the data pipeline. Use in-memory fake datasets to stay offline."""

from __future__ import annotations

from typing import cast
from unittest.mock import patch

import pytest
from datasets import ClassLabel, Dataset, DatasetDict, Features, Value
from transformers import PreTrainedTokenizerBase

from guardrails.config import Settings
from guardrails.data import (
    LABELS_COLUMN,
    _stratified_split,
    augment_with_notinject,
    augment_with_wildjailbreak,
    load_and_split,
    tokenise,
)


class _FakeTokenizer:
    """Minimal stand-in so tokenise() can run without downloading a real tokeniser."""

    def __call__(self, texts, truncation=True, max_length=None):
        if isinstance(texts, str):
            texts = [texts]
        return {
            "input_ids": [[1, 2, 3] for _ in texts],
            "attention_mask": [[1, 1, 1] for _ in texts],
        }


def _fake_dataset(label_col: str, labels: list[int], texts: list[str]) -> Dataset:
    features = Features({"prompt": Value("string"), label_col: ClassLabel(num_classes=2)})
    return Dataset.from_dict({"prompt": texts, label_col: labels}, features=features)


@pytest.fixture
def fake_train() -> Dataset:
    texts = [f"prompt {i}" for i in range(40)]
    labels = [i % 2 for i in range(40)]
    return _fake_dataset("label", labels, texts)


def _fake_loader(splits: DatasetDict):
    return lambda *args, **kwargs: splits


def test_load_and_split_produces_three_splits(fake_train, monkeypatch):
    monkeypatch.setenv("GUARDRAILS_DATASET", "fake/dataset")
    monkeypatch.setattr(
        "guardrails.data.load_dataset", _fake_loader(DatasetDict({"train": fake_train}))
    )
    splits = load_and_split(Settings(), val_size=0.25, test_size=0.25)
    assert set(splits) == {"train", "val", "test"}
    assert len(splits["train"]) + len(splits["val"]) + len(splits["test"]) == 40


def test_load_and_split_honours_existing_test_split(fake_train, monkeypatch):
    monkeypatch.setenv("GUARDRAILS_DATASET", "fake/dataset")
    test = _fake_dataset("label", [0, 1, 0, 1], ["a", "b", "c", "d"])
    monkeypatch.setattr(
        "guardrails.data.load_dataset",
        _fake_loader(DatasetDict({"train": fake_train, "test": test})),
    )
    splits = load_and_split(Settings())
    assert list(splits["test"]["prompt"]) == ["a", "b", "c", "d"]


def test_load_and_split_raises_on_missing_label_column(fake_train, monkeypatch):
    monkeypatch.setenv("GUARDRAILS_DATASET", "fake/dataset")
    bad = Dataset.from_dict({"prompt": ["x"], "type": [1]})
    monkeypatch.setattr("guardrails.data.load_dataset", _fake_loader(DatasetDict({"train": bad})))
    with pytest.raises(ValueError, match="Label column"):
        load_and_split(Settings())


def test_load_dispatches_url_dataset_to_json_loader(fake_train, monkeypatch):
    """Default dataset is a GitHub JSON URL; ensure we route through load_dataset('json', ...)."""
    calls: list[tuple[tuple, dict]] = []

    def _loader(*args, **kwargs):
        calls.append((args, kwargs))
        return DatasetDict({"train": fake_train})

    monkeypatch.setenv("GUARDRAILS_DATASET", "https://example.com/data.json")
    monkeypatch.setattr("guardrails.data.load_dataset", _loader)
    load_and_split(Settings())
    assert calls[0][0][0] == "json"
    assert calls[0][1]["data_files"] == "https://example.com/data.json"


def test_tokenise_renames_custom_label_column_to_labels(fake_train, monkeypatch):
    """If settings.label_column != 'labels', tokenise must rename it so the Trainer sees loss."""
    ds = _fake_dataset("type", [0, 1, 0, 1], ["a", "b", "c", "d"])
    monkeypatch.setenv("GUARDRAILS_LABEL_COLUMN", "type")
    settings = Settings()
    tokenised = tokenise(
        DatasetDict({"train": ds}), cast(PreTrainedTokenizerBase, _FakeTokenizer()), settings
    )
    assert LABELS_COLUMN in tokenised["train"].column_names
    assert "type" not in tokenised["train"].column_names


def test_tokenise_keeps_default_labels_column(fake_train):
    tokenised = tokenise(
        DatasetDict({"train": fake_train}),
        cast(PreTrainedTokenizerBase, _FakeTokenizer()),
        Settings(),
    )
    # Default label_column is 'label' which gets renamed to 'labels'
    assert LABELS_COLUMN in tokenised["train"].column_names


def test_augment_with_notinject_appends_benign_rows(fake_train, monkeypatch):
    """Augmentation should only add label=0 rows and leave val/test alone."""
    ni_one = Dataset.from_dict({"prompt": ["a", "b"]})
    ni_two = Dataset.from_dict({"prompt": ["c"]})

    def _loader(name, split=None):
        assert name == "leolee99/NotInject"
        if split == "NotInject_one":
            return ni_one
        if split == "NotInject_two":
            return ni_two
        raise ValueError(f"unexpected split {split}")

    monkeypatch.setattr("guardrails.data.load_dataset", _loader)
    ds_in = DatasetDict({"train": fake_train, "val": fake_train, "test": fake_train})
    ds_out = augment_with_notinject(ds_in, Settings())
    assert len(ds_out["train"]) == len(fake_train) + 3
    assert set(ds_out["train"]["label"][-3:]) == {0}
    # Val and test untouched
    assert len(ds_out["val"]) == len(fake_train)
    assert len(ds_out["test"]) == len(fake_train)


def test_augment_gracefully_skips_on_load_failure(fake_train, monkeypatch):
    def _loader(name, split=None):
        raise RuntimeError("offline")

    monkeypatch.setattr("guardrails.data.load_dataset", _loader)
    ds_in = DatasetDict({"train": fake_train})
    ds_out = augment_with_notinject(ds_in, Settings())
    assert len(ds_out["train"]) == len(fake_train)


def _fake_wildjailbreak(rows: list[dict]) -> Dataset:
    """Build a Dataset that looks like a WildJailbreak slice."""
    cols = {k: [r.get(k, "") for r in rows] for k in rows[0]}
    return Dataset.from_dict(cols)


def test_augment_with_wildjailbreak_appends_filtered_benign_rows(fake_train, monkeypatch):
    """Only adversarial_benign rows survive the filter; val/test untouched; labels are all 0."""
    raw = _fake_wildjailbreak(
        [
            {"vanilla": "v1", "adversarial": "wj-benign-1", "data_type": "adversarial_benign"},
            {"vanilla": "v2", "adversarial": "wj-harm-1", "data_type": "adversarial_harmful"},
            {"vanilla": "v3", "adversarial": "wj-benign-2", "data_type": "adversarial_benign"},
            {"vanilla": "v4", "adversarial": "wj-van-b", "data_type": "vanilla_benign"},
        ]
    )

    def _loader(name, *args, **kwargs):
        assert name == "allenai/wildjailbreak"
        return raw

    monkeypatch.setattr("guardrails.data.load_dataset", _loader)
    ds_in = DatasetDict({"train": fake_train, "val": fake_train, "test": fake_train})
    ds_out = augment_with_wildjailbreak(ds_in, Settings())
    assert len(ds_out["train"]) == len(fake_train) + 2
    added_texts = ds_out["train"]["prompt"][-2:]
    assert set(added_texts) == {"wj-benign-1", "wj-benign-2"}
    assert set(ds_out["train"]["label"][-2:]) == {0}
    assert len(ds_out["val"]) == len(fake_train)
    assert len(ds_out["test"]) == len(fake_train)


def test_augment_with_wildjailbreak_caps_to_n(fake_train, monkeypatch):
    raw = _fake_wildjailbreak(
        [{"adversarial": f"wj-{i}", "data_type": "adversarial_benign"} for i in range(5)]
    )
    monkeypatch.setattr("guardrails.data.load_dataset", lambda *a, **kw: raw)
    ds_out = augment_with_wildjailbreak(DatasetDict({"train": fake_train}), Settings(), n=2)
    assert len(ds_out["train"]) == len(fake_train) + 2


def test_augment_with_wildjailbreak_uses_prompt_column_when_adversarial_missing(
    fake_train, monkeypatch
):
    """Earlier dataset revisions shipped `prompt` instead of `adversarial`. Both must work."""
    raw = _fake_wildjailbreak(
        [
            {"prompt": "legacy-benign", "data_type": "adversarial_benign"},
            {"prompt": "legacy-harm", "data_type": "adversarial_harmful"},
        ]
    )
    monkeypatch.setattr("guardrails.data.load_dataset", lambda *a, **kw: raw)
    ds_out = augment_with_wildjailbreak(DatasetDict({"train": fake_train}), Settings())
    assert ds_out["train"]["prompt"][-1] == "legacy-benign"


def test_augment_with_wildjailbreak_gracefully_skips_on_load_failure(fake_train, monkeypatch):
    def _loader(*args, **kwargs):
        raise RuntimeError("gated or offline")

    monkeypatch.setattr("guardrails.data.load_dataset", _loader)
    ds_out = augment_with_wildjailbreak(DatasetDict({"train": fake_train}), Settings())
    assert len(ds_out["train"]) == len(fake_train)


def test_stratified_split_falls_back_on_non_classlabel():
    """Some HF datasets ship labels as Value("int64"). Stratify should not crash."""
    raw = Dataset.from_dict({"prompt": ["x"] * 10, "label": [i % 2 for i in range(10)]})
    # Don't cast: leave as Value("int64") so stratify raises internally
    with patch("guardrails.data._ensure_classlabel", side_effect=TypeError("simulated")):
        split = _stratified_split(raw, test_size=0.2, seed=0, label_col="label")
    assert set(split) == {"train", "test"}
