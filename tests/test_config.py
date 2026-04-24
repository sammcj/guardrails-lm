"""Smoke tests for configuration loading."""

from __future__ import annotations

import pytest

from guardrails.config import Settings


def test_defaults_are_valid() -> None:
    s = Settings()
    assert s.encoder.startswith("answerdotai/")
    assert s.max_seq_len >= 64
    assert s.batch_size >= 1
    assert s.precision in {"bf16", "fp16", "fp32"}
    assert s.device in {"mps", "cuda", "cpu"}


def test_env_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GUARDRAILS_MAX_SEQ_LEN", "512")
    monkeypatch.setenv("GUARDRAILS_BATCH_SIZE", "8")
    monkeypatch.setenv("GUARDRAILS_PRECISION", "fp16")
    monkeypatch.setenv("GUARDRAILS_DEVICE", "cpu")
    s = Settings()
    assert s.max_seq_len == 512
    assert s.batch_size == 8
    assert s.precision == "fp16"
    assert s.device == "cpu"


def test_rejects_invalid_precision(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GUARDRAILS_PRECISION", "int4")
    with pytest.raises(ValueError):
        Settings()


def test_rejects_invalid_seq_len(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GUARDRAILS_MAX_SEQ_LEN", "32")
    with pytest.raises(ValueError):
        Settings()


def test_best_model_dir_is_under_output() -> None:
    s = Settings()
    assert s.best_model_dir.parts[-1] == "best"
    assert s.best_model_dir.parent == s.output_dir
