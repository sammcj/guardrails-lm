"""Tests for the train module. Network-free checks only."""

from __future__ import annotations

import pytest

from guardrails.config import Settings
from guardrails.train import _has_checkpoint


@pytest.fixture
def settings_with_tmp_output(tmp_path, monkeypatch) -> Settings:
    monkeypatch.setenv("GUARDRAILS_OUTPUT_DIR", str(tmp_path))
    return Settings()


def test_no_checkpoint_when_dir_missing(tmp_path, monkeypatch):
    monkeypatch.setenv("GUARDRAILS_OUTPUT_DIR", str(tmp_path / "nonexistent"))
    assert _has_checkpoint(Settings()) is False


def test_no_checkpoint_when_dir_empty(settings_with_tmp_output):
    assert _has_checkpoint(settings_with_tmp_output) is False


def test_no_checkpoint_when_only_best_exists(settings_with_tmp_output):
    (settings_with_tmp_output.output_dir / "best").mkdir()
    assert _has_checkpoint(settings_with_tmp_output) is False


def test_detects_checkpoint_directory(settings_with_tmp_output):
    (settings_with_tmp_output.output_dir / "checkpoint-100").mkdir()
    assert _has_checkpoint(settings_with_tmp_output) is True


def test_detects_any_of_multiple_checkpoints(settings_with_tmp_output):
    for step in (100, 200, 300):
        (settings_with_tmp_output.output_dir / f"checkpoint-{step}").mkdir()
    assert _has_checkpoint(settings_with_tmp_output) is True
