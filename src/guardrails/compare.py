"""Side-by-side comparison of saved checkpoints.

Running `eval` + `eval-ood` separately per checkpoint and eyeballing the numbers
is error-prone once there are more than two models to compare. This module
collects the same metric battery for each path and emits a single table with
deltas vs a baseline column.

Design note: the metric collection (model loading + scoring) and the
presentation (delta maths + table rendering) are kept separate so the latter
is testable without any model or network access.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from rich.console import Console
from rich.table import Table

from .config import Settings
from .eval import evaluate
from .infer import resolve_threshold
from .ood import evaluate_ood

Direction = Literal["higher", "lower"]


@dataclass(frozen=True)
class MetricSpec:
    """Declarative row in the comparison table.

    `direction` records whether a larger value is better (F1, TPR, accuracy) or
    worse (FPR, latency). Used to colour the delta column, not to sort.
    """

    key: str
    label: str
    direction: Direction
    group: Literal["eval", "ood"]


METRIC_REGISTRY: tuple[MetricSpec, ...] = (
    MetricSpec("accuracy", "accuracy", "higher", "eval"),
    MetricSpec("f1", "F1", "higher", "eval"),
    MetricSpec("precision", "precision", "higher", "eval"),
    MetricSpec("recall", "recall", "higher", "eval"),
    MetricSpec(
        "ood.NotInject_three (benign).fpr",
        "NotInject FPR",
        "lower",
        "ood",
    ),
    MetricSpec(
        "ood.awesome-chatgpt-prompts (benign).fpr",
        "awesome-chatgpt FPR",
        "lower",
        "ood",
    ),
    MetricSpec(
        "ood.deepset-prompt-injections.tpr",
        "deepset TPR",
        "higher",
        "ood",
    ),
    MetricSpec(
        "ood.jackhhao-jailbreak-classification.tpr",
        "jackhhao TPR",
        "higher",
        "ood",
    ),
)


def collect_metrics(
    settings: Settings,
    model_path: Path,
    skip_eval: bool = False,
    skip_ood: bool = False,
) -> dict[str, float | None]:
    """Run the eval + eval-ood battery for one checkpoint and flatten into a dict.

    Keys follow the `METRIC_REGISTRY` convention so `format_comparison` can look
    them up without knowing anything about the underlying dataclasses.
    """
    out: dict[str, float | None] = {}
    threshold = resolve_threshold(model_path, None)

    if not skip_eval:
        result = evaluate(settings, model_path, threshold=threshold)
        out["accuracy"] = result.accuracy
        out["f1"] = result.f1
        out["precision"] = result.precision
        out["recall"] = result.recall

    if not skip_ood:
        report = evaluate_ood(settings, model_path, threshold=threshold)
        for r in report.results:
            out[f"ood.{r.name}.fpr"] = r.fpr
            out[f"ood.{r.name}.tpr"] = r.tpr
            out[f"ood.{r.name}.accuracy"] = r.accuracy

    return out


def is_improvement(direction: Direction, delta: float) -> bool:
    """Given a metric direction and a signed delta, decide whether it's an improvement."""
    if direction == "higher":
        return delta > 0
    return delta < 0


def _format_delta(direction: Direction, delta: float | None) -> str:
    if delta is None:
        return "-"
    sign = "+" if delta >= 0 else ""
    improved = is_improvement(direction, delta)
    colour = "green" if improved else ("red" if delta != 0 else "dim")
    return f"[{colour}]{sign}{delta:.4f}[/{colour}]"


def _format_value(value: float | None) -> str:
    return "-" if value is None else f"{value:.4f}"


def _display_label(path: Path) -> str:
    """Column label for a checkpoint. Trained models all live at `*/best`, so
    showing just the leaf makes every column identical; include the parent dir
    so `checkpoints/best` and `checkpoints-v3/best` are distinguishable.
    """
    parts = path.parts
    if len(parts) >= 2:
        return f"{parts[-2]}/{parts[-1]}"
    return str(path)


def _active_specs(skip_eval: bool, skip_ood: bool) -> list[MetricSpec]:
    specs: list[MetricSpec] = []
    for spec in METRIC_REGISTRY:
        if spec.group == "eval" and skip_eval:
            continue
        if spec.group == "ood" and skip_ood:
            continue
        specs.append(spec)
    return specs


def format_comparison(
    paths: list[Path],
    metrics_by_path: dict[Path, dict[str, float | None]],
    baseline: Path,
    skip_eval: bool = False,
    skip_ood: bool = False,
) -> Table:
    """Render the side-by-side comparison table with deltas vs baseline.

    Kept as a pure function taking already-collected metric dicts so tests don't
    need to load any models.
    """
    if baseline not in paths:
        raise ValueError(f"baseline {baseline} not in paths {paths}")

    table = Table(title=f"Checkpoint comparison (baseline: {_display_label(baseline)})")
    table.add_column("metric")
    for path in paths:
        suffix = " (baseline)" if path == baseline else ""
        table.add_column(f"{_display_label(path)}{suffix}")
        if path != baseline:
            table.add_column("Δ vs baseline")

    base_metrics = metrics_by_path[baseline]
    for spec in _active_specs(skip_eval, skip_ood):
        row: list[str] = [spec.label]
        base_val = base_metrics.get(spec.key)
        for path in paths:
            value = metrics_by_path[path].get(spec.key)
            row.append(_format_value(value))
            if path != baseline:
                delta = value - base_val if (value is not None and base_val is not None) else None
                row.append(_format_delta(spec.direction, delta))
        table.add_row(*row)
    return table


def compare_checkpoints(
    paths: list[Path],
    baseline: Path | None = None,
    skip_eval: bool = False,
    skip_ood: bool = False,
    console: Console | None = None,
    settings: Settings | None = None,
) -> None:
    """Load each checkpoint once, run the batteries, print the table."""
    if len(paths) < 2:
        raise ValueError("need at least two checkpoint paths to compare")
    for p in paths:
        if not p.exists():
            raise FileNotFoundError(f"checkpoint path does not exist: {p}")
    if skip_eval and skip_ood:
        raise ValueError("--skip-eval and --skip-ood are mutually exclusive")

    baseline = baseline if baseline is not None else paths[0]
    console = console if console is not None else Console()
    settings = settings if settings is not None else Settings()

    metrics_by_path: dict[Path, dict[str, float | None]] = {}
    for path in paths:
        console.print(f"[dim]Scoring {path}...[/dim]")
        metrics_by_path[path] = collect_metrics(
            settings, path, skip_eval=skip_eval, skip_ood=skip_ood
        )

    table = format_comparison(
        paths,
        metrics_by_path,
        baseline=baseline,
        skip_eval=skip_eval,
        skip_ood=skip_ood,
    )
    console.print(table)
