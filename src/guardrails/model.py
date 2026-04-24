"""ModernBERT classification head built for MPS + SDPA."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, PreTrainedTokenizerBase

from .config import Precision, Settings

_DTYPE: dict[Precision, torch.dtype] = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "fp32": torch.float32,
}

ID2LABEL: dict[int, str] = {0: "safe", 1: "unsafe"}
LABEL2ID: dict[str, int] = {v: k for k, v in ID2LABEL.items()}


def resolve_dtype(precision: Precision) -> torch.dtype:
    return _DTYPE[precision]


def build_model(settings: Settings, num_labels: int = 2):
    """Load ModernBERT for training. Weights stay fp32; Trainer handles autocast."""
    model = AutoModelForSequenceClassification.from_pretrained(
        settings.encoder,
        num_labels=num_labels,
        attn_implementation="sdpa",
        dtype=torch.float32,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )
    model.to(settings.device)
    return model


def load_for_inference(
    settings: Settings, model_path: Path
) -> tuple[PreTrainedTokenizerBase, torch.nn.Module]:
    """Load a saved tokeniser + model ready for inference in the configured precision."""
    tokenizer = cast(PreTrainedTokenizerBase, AutoTokenizer.from_pretrained(str(model_path)))
    model = AutoModelForSequenceClassification.from_pretrained(
        str(model_path),
        dtype=resolve_dtype(settings.precision),
        attn_implementation="sdpa",
    )
    model.to(settings.device).eval()
    return tokenizer, model
