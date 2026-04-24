"""Offline tests for the ONNX export wiring.

Tracing a real transformer to ONNX would load ~570MB of weights and take
several seconds — tests stub both the model load and the torch.onnx trace
step. We verify argument plumbing, sidecar copy, and error paths, not the
shape of the traced graph itself.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from guardrails import export as export_module
from guardrails.export import ONNX_FILENAME, export_to_onnx


class _FakeTokenizer:
    @classmethod
    def from_pretrained(cls, path: str) -> _FakeTokenizer:
        return cls()

    def save_pretrained(self, path: str) -> None:
        p = Path(path)
        (p / "tokenizer.json").write_text("{}")
        (p / "tokenizer_config.json").write_text("{}")

    def __call__(self, *args, **kwargs):
        class _Batch(dict):
            def to(self, device):
                return self

        return _Batch(
            input_ids=torch.zeros((2, 16), dtype=torch.long),
            attention_mask=torch.ones((2, 16), dtype=torch.long),
        )


class _FakeModel:
    @classmethod
    def from_pretrained(cls, path: str) -> _FakeModel:
        return cls()

    def to(self, device: str) -> _FakeModel:
        return self

    def eval(self) -> _FakeModel:
        return self


def _stub_onnx_export(model, args, path, **kwargs) -> None:
    """Write a placeholder ONNX file without actually tracing anything."""
    Path(path).write_bytes(b"\x00ONNX\x00")


def _patch_heavy_deps(monkeypatch):
    """Replace the real model/tokenizer/onnx.export with offline stubs."""
    monkeypatch.setattr(export_module, "AutoTokenizer", _FakeTokenizer)
    monkeypatch.setattr(export_module, "AutoModelForSequenceClassification", _FakeModel)
    monkeypatch.setattr(export_module.torch.onnx, "export", _stub_onnx_export)


def _make_checkpoint(tmp_path: Path, with_threshold: bool = True) -> Path:
    ck = tmp_path / "best"
    ck.mkdir()
    (ck / "config.json").write_text(json.dumps({"model_type": "modernbert"}))
    (ck / "tokenizer.json").write_text("{}")
    (ck / "tokenizer_config.json").write_text("{}")
    if with_threshold:
        (ck / "threshold.json").write_text(json.dumps({"threshold": 0.987, "f1": 0.978}))
    return ck


def test_export_fails_when_model_path_missing(tmp_path, monkeypatch):
    _patch_heavy_deps(monkeypatch)
    with pytest.raises(FileNotFoundError, match="does not exist"):
        export_to_onnx(tmp_path / "ghost", tmp_path / "out")


def test_export_fails_when_checkpoint_has_no_config(tmp_path, monkeypatch):
    _patch_heavy_deps(monkeypatch)
    bad = tmp_path / "bad"
    bad.mkdir()  # no config.json
    with pytest.raises(FileNotFoundError, match=r"config\.json"):
        export_to_onnx(bad, tmp_path / "out")


def test_export_copies_threshold_json_alongside(tmp_path, monkeypatch):
    _patch_heavy_deps(monkeypatch)
    ck = _make_checkpoint(tmp_path)
    out = tmp_path / "onnx"
    export_to_onnx(ck, out)
    assert (out / ONNX_FILENAME).exists()
    assert (out / "config.json").exists()
    assert (out / "threshold.json").exists()
    assert json.loads((out / "threshold.json").read_text())["threshold"] == 0.987
    assert (out / "README.md").exists()


def test_export_without_threshold_still_succeeds(tmp_path, monkeypatch):
    _patch_heavy_deps(monkeypatch)
    ck = _make_checkpoint(tmp_path, with_threshold=False)
    out = tmp_path / "onnx"
    export_to_onnx(ck, out)
    assert (out / ONNX_FILENAME).exists()
    assert not (out / "threshold.json").exists()
    readme = (out / "README.md").read_text()
    assert "No threshold.json" in readme


def test_export_returns_output_directory(tmp_path, monkeypatch):
    _patch_heavy_deps(monkeypatch)
    ck = _make_checkpoint(tmp_path)
    out = tmp_path / "onnx"
    result = export_to_onnx(ck, out)
    assert result == out
