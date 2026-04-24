"""Offline tests for the compare-checkpoints wiring.

Exercises the pure delta / formatting logic directly so no model weights or
network calls are needed.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from guardrails.compare import (
    METRIC_REGISTRY,
    MetricSpec,
    _active_specs,
    _display_label,
    _format_delta,
    compare_checkpoints,
    format_comparison,
    is_improvement,
)


def test_metric_direction_registry_covers_both_senses():
    """F1/accuracy/TPR should be higher-is-better; FPR lower-is-better."""
    by_label = {spec.label: spec for spec in METRIC_REGISTRY}
    assert by_label["F1"].direction == "higher"
    assert by_label["accuracy"].direction == "higher"
    assert by_label["deepset TPR"].direction == "higher"
    assert by_label["jackhhao TPR"].direction == "higher"
    assert by_label["NotInject FPR"].direction == "lower"
    assert by_label["awesome-chatgpt FPR"].direction == "lower"


@pytest.mark.parametrize(
    ("direction", "delta", "expected"),
    [
        ("higher", 0.01, True),
        ("higher", -0.01, False),
        ("higher", 0.0, False),
        ("lower", -0.05, True),
        ("lower", 0.05, False),
        ("lower", 0.0, False),
    ],
)
def test_is_improvement_honours_direction(direction, delta, expected):
    assert is_improvement(direction, delta) is expected


def test_format_delta_colours_improvements_green():
    """Higher F1 -> green; higher FPR -> red. Zero -> dim."""
    assert "green" in _format_delta("higher", 0.02)
    assert "red" in _format_delta("higher", -0.02)
    assert "green" in _format_delta("lower", -0.03)
    assert "red" in _format_delta("lower", 0.03)
    assert "dim" in _format_delta("higher", 0.0)
    assert _format_delta("higher", None) == "-"


def test_format_comparison_computes_correct_delta_values():
    base = Path("/fake/v2")
    challenger = Path("/fake/v3")
    metrics_by_path: dict[Path, dict[str, float | None]] = {
        base: {"f1": 0.900, "accuracy": 0.980},
        challenger: {"f1": 0.950, "accuracy": 0.970},
    }
    table = format_comparison(
        [base, challenger],
        metrics_by_path,
        baseline=base,
        skip_ood=True,
    )
    # The rendered table cells carry both the raw number and the coloured delta.
    rows_text = "\n".join(str(cell) for column in table.columns for cell in column._cells)
    assert "+0.0500" in rows_text  # f1 challenger delta
    assert "-0.0100" in rows_text  # accuracy challenger delta


def test_format_comparison_raises_if_baseline_not_in_paths():
    base = Path("/fake/v2")
    other = Path("/fake/v3")
    with pytest.raises(ValueError, match="baseline"):
        format_comparison(
            [base, other],
            {base: {}, other: {}},
            baseline=Path("/fake/nope"),
            skip_ood=True,
        )


def test_active_specs_skip_eval_keeps_only_ood_rows():
    specs = _active_specs(skip_eval=True, skip_ood=False)
    assert specs
    assert all(s.group == "ood" for s in specs)


def test_active_specs_skip_ood_keeps_only_eval_rows():
    specs = _active_specs(skip_eval=False, skip_ood=True)
    assert specs
    assert all(s.group == "eval" for s in specs)


def test_active_specs_keep_everything_by_default():
    specs = _active_specs(skip_eval=False, skip_ood=False)
    assert {s.group for s in specs} == {"eval", "ood"}


def test_format_comparison_emits_dash_for_missing_metric():
    """When eval is skipped for a path the eval rows should render as '-'."""
    base = Path("/fake/v2")
    other = Path("/fake/v3")
    metrics_by_path: dict[Path, dict[str, float | None]] = {
        base: {"f1": 0.9},
        other: {},  # missing, e.g. a skipped battery
    }
    table = format_comparison(
        [base, other],
        metrics_by_path,
        baseline=base,
        skip_ood=True,
    )
    rows_text = "\n".join(str(cell) for column in table.columns for cell in column._cells)
    assert "-" in rows_text


def test_compare_checkpoints_rejects_single_path(tmp_path):
    with pytest.raises(ValueError, match="at least two"):
        compare_checkpoints([tmp_path])


def test_compare_checkpoints_rejects_both_skips(tmp_path):
    a = tmp_path / "a"
    b = tmp_path / "b"
    a.mkdir()
    b.mkdir()
    with pytest.raises(ValueError, match="mutually exclusive"):
        compare_checkpoints([a, b], skip_eval=True, skip_ood=True)


def test_compare_checkpoints_raises_on_missing_path(tmp_path):
    real = tmp_path / "real"
    real.mkdir()
    missing = tmp_path / "ghost"
    with pytest.raises(FileNotFoundError, match="does not exist"):
        compare_checkpoints([real, missing])


def test_display_label_includes_parent_for_disambiguation():
    """Two paths both ending in /best must render as different column labels."""
    assert _display_label(Path("checkpoints/best")) == "checkpoints/best"
    assert _display_label(Path("checkpoints-v3/best")) == "checkpoints-v3/best"
    assert _display_label(Path("/abs/dir/best")) == "dir/best"
    assert _display_label(Path("just-one")) == "just-one"


def test_metric_spec_requires_known_direction():
    """Guard against typos creeping into the registry."""
    spec = MetricSpec(key="x", label="x", direction="higher", group="eval")
    assert spec.direction in {"higher", "lower"}
    assert spec.group in {"eval", "ood"}
