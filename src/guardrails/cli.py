"""Command-line entry point."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, replace
from pathlib import Path
from typing import Annotated

import numpy as np
import typer
from rich.console import Console
from rich.table import Table

from .calibration import (
    THRESHOLD_FILENAME,
    pick_by_cost,
    pick_by_f1,
    pick_by_fpr_budget,
)
from .calibration import (
    save as save_threshold,
)
from .compare import compare_checkpoints
from .config import Settings
from .data import build_tokenizer, load_and_split
from .eval import benchmark_latency, evaluate, score_val, threshold_sweep
from .export import export_to_onnx
from .infer import Classifier, resolve_threshold
from .ood import evaluate_ood, score_benign_ood
from .server import serve as serve_fn
from .train import train as train_fn

app = typer.Typer(no_args_is_help=True, add_completion=False)
console = Console()

ModelPathOpt = Annotated[
    Path | None, typer.Option("--model-path", help="Path to a saved best-model directory")
]


def _resolve(model_path: Path | None) -> Path:
    return model_path if model_path is not None else Settings().best_model_dir


@app.command()
def inspect() -> None:
    """Print label balance and sequence-length distribution per split."""
    settings = Settings()
    ds = load_and_split(settings)
    tokenizer = build_tokenizer(settings)

    table = Table(title=f"{settings.dataset} (tokeniser: {settings.encoder})")
    for col in ("split", "n", "labels", "len p50", "len p95", "len p99", "max"):
        table.add_column(col)

    for split_name in ("train", "val", "test"):
        split = ds[split_name]
        labels = Counter(split[settings.label_column])
        texts = list(split[settings.text_column])
        encoded = tokenizer(texts, truncation=False)["input_ids"]
        lens = np.asarray([len(x) for x in encoded])
        table.add_row(
            split_name,
            str(len(split)),
            str(dict(labels)),
            str(int(np.percentile(lens, 50))),
            str(int(np.percentile(lens, 95))),
            str(int(np.percentile(lens, 99))),
            str(int(lens.max())),
        )
    console.print(table)


@app.command()
def train() -> None:
    """Fine-tune ModernBERT on the configured dataset."""
    settings = Settings()
    metrics = train_fn(settings)
    table = Table(title="Final validation metrics")
    table.add_column("metric")
    table.add_column("value")
    for k, v in metrics.items():
        if isinstance(v, float):
            table.add_row(k, f"{v:.4f}")
    console.print(table)
    console.print(f"[green]Best model saved to[/green] {settings.best_model_dir}")


@app.command()
def eval(
    model_path: ModelPathOpt = None,
    threshold: float | None = None,
) -> None:
    """Evaluate the saved model on the held-out test split.

    If `--threshold` is omitted, a saved threshold.json is used, else 0.5.
    """
    settings = Settings()
    path = _resolve(model_path)
    t = resolve_threshold(path, threshold)
    result = evaluate(settings, path, threshold=t)
    table = Table(title=f"Test metrics (n={result.n}, threshold={result.threshold:.2f})")
    table.add_column("metric")
    table.add_column("value")
    for k, v in asdict(result).items():
        if k in {"confusion", "n", "threshold"}:
            continue
        table.add_row(k, f"{v:.4f}")
    console.print(table)
    console.print(f"[bold]confusion[/bold] [[TN, FP], [FN, TP]]: {result.confusion}")


@app.command()
def sweep(model_path: ModelPathOpt = None, steps: int = 20) -> None:
    """Sweep classification threshold on the validation split."""
    settings = Settings()
    rows = threshold_sweep(settings, _resolve(model_path), steps=steps)
    table = Table(title="Threshold sweep (val split)")
    for col in ("threshold", "precision", "recall", "f1"):
        table.add_column(col)
    for r in rows:
        table.add_row(
            f"{r['threshold']:.3f}",
            f"{r['precision']:.4f}",
            f"{r['recall']:.4f}",
            f"{r['f1']:.4f}",
        )
    console.print(table)


@app.command()
def benchmark(model_path: ModelPathOpt = None, n: int = 500) -> None:
    """Measure inference latency over N single-prompt forward passes."""
    settings = Settings()
    result = benchmark_latency(settings, _resolve(model_path), n=n)
    table = Table(
        title=f"Latency (n={result.n}, device={settings.device}, precision={settings.precision})"
    )
    table.add_column("metric")
    table.add_column("ms")
    for k, v in asdict(result).items():
        if k == "n":
            continue
        table.add_row(k, f"{v:.2f}")
    console.print(table)


@app.command()
def classify(
    prompt: str,
    model_path: ModelPathOpt = None,
    threshold: float | None = None,
) -> None:
    """Classify a single prompt as safe or unsafe."""
    settings = Settings()
    clf = Classifier(_resolve(model_path), settings, threshold=threshold)
    result = clf.classify(prompt)
    colour = "red" if result.label == "unsafe" else "green"
    console.print(
        f"[bold {colour}]{result.label}[/bold {colour}] "
        f"(prob_unsafe={result.prob_unsafe:.3f}, score={result.score:.3f}, "
        f"threshold={clf.threshold:.3f})"
    )


@app.command()
def calibrate(
    model_path: ModelPathOpt = None,
    cost_fp: float | None = None,
    cost_fn: float | None = None,
    max_fpr: float | None = None,
    min_threshold: float = 0.3,
    include_ood_benign: bool = False,
    save: bool = True,
) -> None:
    """Pick an operating threshold on the validation split.

    Default (no flags) picks the F1-maximising threshold. F1 is robust to class
    imbalance so this does the right thing without you having to think about it.

    Override with explicit modes:
    - `--cost-fn 10 --cost-fp 1` for cost-weighted (miss is ten times worse).
    - `--max-fpr 0.01` for the recall-maximising threshold under a 1% FPR budget.
      `--min-threshold 0.3` (default) guards against the val distribution letting
      the algorithm pick a near-zero threshold that collapses OOD. Pass `0.0` to disable.

    `--include-ood-benign` augments the calibration pool with NotInject +
    awesome-chatgpt-prompts (any benign-only registry entries) so FPR is measured
    against a realistic benign distribution, not just in-distribution val.

    Saves to `<model>/threshold.json` unless `--no-save` is given.
    """
    settings = Settings()
    path = _resolve(model_path)
    probs, labels = score_val(settings, path)
    data_source = "val"

    if include_ood_benign:
        console.print("[dim]Scoring OOD benign sets for calibration...[/dim]")
        ood_probs, ood_labels = score_benign_ood(settings, path)
        if ood_probs.size > 0:
            probs = np.concatenate([probs, ood_probs])
            labels = np.concatenate([labels, ood_labels])
            data_source = f"val+ood_benign (n_ood={ood_probs.size})"
        else:
            console.print("[yellow]No OOD benign sets loaded.[/yellow]")

    if max_fpr is not None:
        rec = pick_by_fpr_budget(probs, labels, max_fpr=max_fpr, min_threshold=min_threshold)
    elif cost_fp is not None or cost_fn is not None:
        rec = pick_by_cost(
            probs,
            labels,
            cost_fp=cost_fp if cost_fp is not None else 1.0,
            cost_fn=cost_fn if cost_fn is not None else 1.0,
        )
    else:
        rec = pick_by_f1(probs, labels)

    rec = replace(rec, data_source=data_source)

    table = Table(
        title=f"Calibration ({rec.mode}: {rec.criterion}, n={rec.n}, data={rec.data_source})"
    )
    table.add_column("metric")
    table.add_column("value")
    for k in ("threshold", "precision", "recall", "f1", "fpr", "tpr", "accuracy"):
        table.add_row(k, f"{getattr(rec, k):.4f}")
    console.print(table)

    if save:
        save_threshold(rec, path / THRESHOLD_FILENAME)
        console.print(f"[green]Saved[/green] {path / THRESHOLD_FILENAME}")


@app.command("eval-ood")
def eval_ood(
    model_path: ModelPathOpt = None,
    threshold: float | None = None,
) -> None:
    """Evaluate against out-of-distribution safety datasets.

    Reports FPR on benign-only sets (over-defense) and TPR on attack-only sets
    (distribution gap). Datasets that fail to load are skipped.
    """
    settings = Settings()
    path = _resolve(model_path)
    t = resolve_threshold(path, threshold)
    report = evaluate_ood(settings, path, threshold=t)

    table = Table(title=f"OOD evaluation (threshold={report.threshold:.3f})")
    for col in ("dataset", "mode", "n", "acc", "fpr", "tpr", "mean_prob_unsafe", "note"):
        table.add_column(col)
    for r in report.results:
        table.add_row(
            r.name,
            r.mode,
            str(r.n),
            "-" if r.accuracy is None else f"{r.accuracy:.3f}",
            "-" if r.fpr is None else f"{r.fpr:.3f}",
            "-" if r.tpr is None else f"{r.tpr:.3f}",
            f"{r.mean_prob_unsafe:.3f}",
            r.note,
        )
    console.print(table)
    if not report.results:
        console.print("[yellow]No OOD datasets loaded (all skipped).[/yellow]")


@app.command("compare-checkpoints")
def compare_checkpoints_cmd(
    paths: Annotated[
        list[Path],
        typer.Argument(help="2+ model directories to compare (config.json + model.safetensors)"),
    ],
    baseline: Annotated[
        Path | None,
        typer.Option("--baseline", help="Reference column for deltas (defaults to first path)"),
    ] = None,
    skip_eval: Annotated[
        bool, typer.Option("--skip-eval", help="Skip in-distribution test metrics")
    ] = False,
    skip_ood: Annotated[
        bool, typer.Option("--skip-ood", help="Skip out-of-distribution battery")
    ] = False,
) -> None:
    """Compare two or more saved checkpoints side by side.

    Runs `eval` + `eval-ood` per path and prints a table with one column per
    checkpoint plus a delta column against the baseline. Green means
    improvement, red means regression (direction depends on the metric: higher
    F1 is good, lower FPR is good).
    """
    compare_checkpoints(
        paths=paths,
        baseline=baseline,
        skip_eval=skip_eval,
        skip_ood=skip_ood,
        console=console,
    )


@app.command("export-onnx")
def export_onnx_cmd(
    output: Annotated[Path, typer.Option("--output", help="Output directory for the ONNX bundle")],
    model_path: ModelPathOpt = None,
    opset: Annotated[int, typer.Option("--opset", help="ONNX opset version")] = 18,
    device: Annotated[str, typer.Option("--device", help="Trace device (cpu or cuda)")] = "cpu",
) -> None:
    """Export the fine-tuned classifier to ONNX for non-Python consumers.

    Produces a directory containing `model.onnx`, the tokenizer files,
    `config.json`, the calibrated `threshold.json` if present, and a README
    describing how to score a prompt with onnxruntime. Trace runs on CPU by
    default.
    """
    source = _resolve(model_path)
    out_dir = export_to_onnx(source, output, opset=opset, device=device)
    console.print(f"[green]Exported[/green] {source} -> {out_dir}")


@app.command()
def serve(
    host: Annotated[
        str | None,
        typer.Option("--host", help="Bind address (defaults to GUARDRAILS_SERVER_HOST)"),
    ] = None,
    port: Annotated[
        int | None,
        typer.Option("--port", help="Bind port (defaults to GUARDRAILS_SERVER_PORT)"),
    ] = None,
    reload: Annotated[
        bool,
        typer.Option("--reload", help="Auto-reload on source change (dev only)"),
    ] = False,
) -> None:
    """Run the HTTP classifier server.

    Exposes POST /v1/classify, POST /v1/classify/batch, GET /v1/info, GET /healthz.
    Listens on 127.0.0.1:8080 by default; put it behind a reverse proxy if you
    need auth, TLS, or rate limiting.
    """
    settings = Settings()
    console.print(
        f"[green]Serving[/green] classifier from {settings.best_model_dir} "
        f"at http://{host or settings.server_host}:{port or settings.server_port}"
    )
    serve_fn(settings=settings, host=host, port=port, reload=reload)


if __name__ == "__main__":
    app()
