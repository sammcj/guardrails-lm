"""Tests that don't require network or model downloads."""

from __future__ import annotations

import torch

from guardrails.config import Settings
from guardrails.infer import LABELS
from guardrails.model import resolve_dtype


def test_resolve_dtype_covers_all_precisions() -> None:
    assert resolve_dtype("bf16") is torch.bfloat16
    assert resolve_dtype("fp16") is torch.float16
    assert resolve_dtype("fp32") is torch.float32


def test_infer_labels_are_binary() -> None:
    assert set(LABELS.keys()) == {0, 1}
    assert LABELS[0] == "safe"
    assert LABELS[1] == "unsafe"


def test_settings_roundtrip_via_precision_change(monkeypatch) -> None:
    monkeypatch.setenv("GUARDRAILS_PRECISION", "fp32")
    s = Settings()
    assert resolve_dtype(s.precision) is torch.float32
