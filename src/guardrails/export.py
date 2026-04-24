"""ONNX export so non-Python callers can run the classifier via onnxruntime.

Uses `torch.onnx` directly (no `optimum` dependency) to trace the fine-tuned
ModernBERT into a portable ONNX graph with dynamic batch + sequence dimensions.
The tokenizer, config, and threshold.json are copied alongside so consumers
have everything they need to reproduce the Python pipeline's verdict. No Python
runtime required beyond export time.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import cast

import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedTokenizerBase,
)

from .calibration import THRESHOLD_FILENAME

logger = logging.getLogger(__name__)

ONNX_FILENAME = "model.onnx"


def _dummy_inputs(
    tokenizer, seq_len: int = 16, device: str = "cpu"
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build a tiny batch so torch.onnx can trace shapes. Content doesn't matter."""
    enc = tokenizer(
        ["hello world"] * 2,
        padding="max_length",
        truncation=True,
        max_length=seq_len,
        return_tensors="pt",
    ).to(device)
    return enc["input_ids"], enc["attention_mask"]


def export_to_onnx(
    model_path: Path,
    output_dir: Path,
    opset: int = 18,
    device: str = "cpu",
) -> Path:
    """Export a fine-tuned checkpoint to ONNX.

    Writes `model.onnx` plus the tokenizer files, the HF config, the calibrated
    `threshold.json` (if present), and a consumer-facing README into `output_dir`.

    Tracing runs on CPU by default since ONNX is usually targeted at portable /
    non-GPU consumers. Pass `device="cuda"` if you want the trace to happen on GPU.
    """
    if not model_path.exists():
        raise FileNotFoundError(f"model path does not exist: {model_path}")
    if not (model_path / "config.json").exists():
        raise FileNotFoundError(
            f"{model_path} doesn't look like a fine-tuned checkpoint (no config.json)"
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Exporting %s -> %s (opset=%d, device=%s)", model_path, output_dir, opset, device)
    tokenizer = cast(PreTrainedTokenizerBase, AutoTokenizer.from_pretrained(str(model_path)))
    model = AutoModelForSequenceClassification.from_pretrained(str(model_path)).to(device)
    model.eval()

    input_ids, attention_mask = _dummy_inputs(tokenizer, device=device)
    onnx_path = output_dir / ONNX_FILENAME
    dynamic_axes = {
        "input_ids": {0: "batch", 1: "sequence"},
        "attention_mask": {0: "batch", 1: "sequence"},
        "logits": {0: "batch"},
    }
    with torch.inference_mode():
        torch.onnx.export(
            model,
            (input_ids, attention_mask),
            str(onnx_path),
            input_names=["input_ids", "attention_mask"],
            output_names=["logits"],
            dynamic_axes=dynamic_axes,
            opset_version=opset,
            do_constant_folding=True,
        )

    # Copy sidecars the consumer needs to reproduce scoring end-to-end.
    tokenizer.save_pretrained(str(output_dir))
    for name in ("config.json",):
        src = model_path / name
        if src.exists():
            shutil.copy2(src, output_dir / name)

    threshold_src = model_path / THRESHOLD_FILENAME
    if threshold_src.exists():
        shutil.copy2(threshold_src, output_dir / THRESHOLD_FILENAME)
        logger.info("Copied threshold.json alongside the ONNX export")
    else:
        logger.warning(
            "No threshold.json at %s; consumers will need to pick a threshold themselves",
            threshold_src,
        )

    _write_readme(output_dir, model_path, threshold_src.exists())
    return output_dir


def _write_readme(output_dir: Path, source_model: Path, has_threshold: bool) -> None:
    """Drop a README so whoever inherits the ONNX bundle knows how to use it."""
    threshold_note = (
        "Apply the `threshold` from `threshold.json` against the softmax probability "
        "of class 1 (`unsafe`) to reproduce the Python classifier's verdict."
        if has_threshold
        else "No threshold.json shipped. Pick one (0.5 is a neutral starting point) "
        "or calibrate on held-out data."
    )
    readme = f"""# Guardrails ONNX bundle

Exported from: `{source_model}`

## Files
- `model.onnx` — fine-tuned ModernBERT classifier (text-classification head, 2 classes).
- `tokenizer.json`, `tokenizer_config.json`, `special_tokens_map.json` — the tokenizer the model was trained with.
- `config.json` — HF model config; `id2label` encodes label names (0=safe, 1=unsafe).
- `threshold.json` — calibrated threshold + metadata (if present).

## Scoring a prompt
1. Tokenise with the shipped tokenizer (truncation, max_length ~1024).
2. Run the ONNX graph: inputs `input_ids` + `attention_mask`, output `logits` of shape (batch, 2).
3. Softmax over axis -1 -> `probs`. `probs[:, 1]` is the `unsafe` probability.
4. {threshold_note}

## Runtime hints
- `onnxruntime` covers Python, JS (via ort-web), Rust, Go, Java, C#, C++.
- CPU execution provider is fine for single-prompt latency at ModernBERT-base size.
"""
    (output_dir / "README.md").write_text(readme)
