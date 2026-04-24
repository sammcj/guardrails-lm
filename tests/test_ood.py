"""Tests for the OOD registry. Network-free checks only."""

from __future__ import annotations

import pytest

from guardrails.ood import DEFAULT_REGISTRY, OODSet, _normalise_label


def test_default_registry_is_non_empty():
    assert len(DEFAULT_REGISTRY) >= 1


def test_each_entry_has_exactly_one_label_source():
    for spec in DEFAULT_REGISTRY:
        has_col = spec.label_column is not None
        has_expected = spec.expected_label is not None
        assert has_col ^ has_expected, f"{spec.name} must declare exactly one label source"


def test_oodset_rejects_missing_labels():
    with pytest.raises(ValueError):
        OODSet(name="x", hf_dataset="foo/bar", text_column="t")


def test_oodset_rejects_both_labels():
    with pytest.raises(ValueError):
        OODSet(
            name="x",
            hf_dataset="foo/bar",
            text_column="t",
            label_column="lbl",
            expected_label=0,
        )


def test_oodset_accepts_split_tuple():
    spec = OODSet(
        name="multi",
        hf_dataset="foo/bar",
        text_column="t",
        expected_label=0,
        split=("a", "b", "c"),
    )
    assert spec.split == ("a", "b", "c")


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (0, 0),
        (1, 1),
        (2, 1),
        ("safe", 0),
        ("SAFE", 0),
        ("benign", 0),
        ("unsafe", 1),
        ("jailbreak", 1),
        ("injection", 1),
        ("malicious", 1),
        ("something weird", None),
        (None, None),
    ],
)
def test_normalise_label_mappings(value, expected):
    assert _normalise_label(value) == expected
