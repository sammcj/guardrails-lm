"""Runtime inference: single-prompt classification."""

from __future__ import annotations

import statistics
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import torch

from .calibration import THRESHOLD_FILENAME
from .calibration import load as load_threshold
from .config import Settings
from .model import ID2LABEL, load_for_inference

LABELS: dict[int, str] = ID2LABEL

# Window size for rolling per-bucket p50 tracking.
_ROLLING_WINDOW = 50
# Passes per bucket when running the startup micro-benchmark. Big enough for
# a stable p50, small enough to not balloon startup (~10 * p50_ms per bucket).
_BASELINE_PASSES = 15

# Powers-of-two bucket ladder. Pads each request up to the next bucket so MPS
# (and CUDA graph capture) can reuse compiled kernels across prompt lengths
# instead of recompiling for every unseen shape. Overridden per-classifier
# against `settings.max_seq_len`.
_BASE_SHAPE_BUCKETS: tuple[int, ...] = (16, 32, 64, 128, 256, 512, 1024)


def resolve_threshold(model_path: Path, override: float | None = None) -> float:
    """If a threshold.json sits alongside the model use it, else return the override or 0.5."""
    if override is not None:
        return override
    rec = load_threshold(model_path / THRESHOLD_FILENAME)
    return rec.threshold if rec is not None else 0.5


@dataclass(frozen=True)
class Classification:
    label: str
    score: float
    prob_unsafe: float
    tokens: int = 0  # number of real tokens in the prompt (excludes right-pad)
    bucket: int = 0  # shape bucket the prompt was padded to for MPS kernel reuse


@dataclass(frozen=True)
class Timings:
    """Wall-clock breakdown of a single classify() call in milliseconds.

    Device work is synchronised before each timestamp so `model_ms` measures
    just the forward pass (not kernel launch latency).
    """

    tokenize_ms: float
    model_ms: float
    postprocess_ms: float

    @property
    def total_ms(self) -> float:
        return self.tokenize_ms + self.model_ms + self.postprocess_ms


def _sync(device: str) -> None:
    if device == "mps" and hasattr(torch, "mps"):
        torch.mps.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()


class Classifier:
    """Thin wrapper over a fine-tuned ModernBERT classifier."""

    def __init__(
        self, model_path: Path, settings: Settings, threshold: float | None = None
    ) -> None:
        self.settings = settings
        self.threshold = resolve_threshold(model_path, threshold)
        self.tokenizer, self.model = load_for_inference(settings, model_path)
        max_len = settings.max_seq_len
        self._shape_buckets: tuple[int, ...] = tuple(b for b in _BASE_SHAPE_BUCKETS if b < max_len)
        if not self._shape_buckets or self._shape_buckets[-1] != max_len:
            self._shape_buckets = (*self._shape_buckets, max_len)
        pad_id = getattr(self.tokenizer, "pad_token_id", None)
        self._pad_id: int = int(pad_id) if pad_id is not None else 0
        # Hardware baseline measured once at startup in `warmup()`: the steady
        # state p50 model_ms per bucket on this machine, free of HTTP/browser
        # overhead. Frontend surfaces this alongside live per-request numbers
        # so users can see what the hardware is capable of vs what this click
        # paid.
        self.baseline_ms_by_bucket: dict[int, float] = {}
        # Rolling window of recent model_ms values per bucket. Updated on every
        # `classify_timed` call; exposed so the API can report a live p50 at
        # whatever bucket the current prompt landed in.
        self._rolling: dict[int, deque[float]] = {
            b: deque(maxlen=_ROLLING_WINDOW) for b in self._shape_buckets
        }

    def _bucket_for(self, length: int) -> int:
        for b in self._shape_buckets:
            if length <= b:
                return b
        return self._shape_buckets[-1]

    def _pad_to(self, enc: dict[str, torch.Tensor], target: int) -> dict[str, torch.Tensor]:
        input_ids = enc["input_ids"]
        attention_mask = enc["attention_mask"]
        cur = int(input_ids.shape[-1])
        if cur >= target:
            return enc
        pad_len = target - cur
        batch = input_ids.shape[0]
        pad_ids = input_ids.new_full((batch, pad_len), self._pad_id)
        pad_mask = attention_mask.new_zeros((batch, pad_len))
        return {
            "input_ids": torch.cat([input_ids, pad_ids], dim=-1),
            "attention_mask": torch.cat([attention_mask, pad_mask], dim=-1),
        }

    def _dummy_encoding(self, bucket: int) -> dict[str, torch.Tensor]:
        """Build a bucket-sized input via the real tokeniser → `_pad_to` path."""
        dummy = " ".join(["hello world"] * (self.settings.max_seq_len // 2))
        enc = self.tokenizer(dummy, truncation=True, max_length=bucket, return_tensors="pt")
        enc = self._pad_to(dict(enc), bucket)
        return {k: v.to(self.settings.device) for k, v in enc.items()}

    @torch.inference_mode()
    def warmup(self) -> None:
        """Pre-compile MPS/CUDA kernels for every shape bucket.

        Routes through the real hot path (tokeniser → `_pad_to` → model →
        softmax → cpu transfer) so MPS caches the exact kernel variant a real
        request will hit. Two passes per bucket — compile + allocator prime.
        Cheap (~400 ms on Apple Silicon); always safe to call at startup.
        """
        device = self.settings.device
        for bucket in self._shape_buckets:
            enc = self._dummy_encoding(bucket)
            for _ in range(2):
                logits = self.model(**enc).logits
                _ = torch.softmax(logits, dim=-1).float().cpu().numpy()
        _sync(device)

    @torch.inference_mode()
    def measure_baseline(self, passes: int = _BASELINE_PASSES) -> None:
        """Populate `baseline_ms_by_bucket` with the median `model_ms` per bucket.

        This is what the hardware genuinely delivers at steady state, free of
        HTTP/browser overhead. Pure overhead at startup (~1-2 s on Apple
        Silicon dominated by the largest bucket); gated behind
        `Settings.measure_baseline_on_startup` so production serving can skip
        it.
        """
        device = self.settings.device
        for bucket in self._shape_buckets:
            enc = self._dummy_encoding(bucket)
            samples: list[float] = []
            for _ in range(passes):
                _sync(device)
                t0 = time.perf_counter()
                _ = self.model(**enc).logits
                _sync(device)
                samples.append((time.perf_counter() - t0) * 1000)
            self.baseline_ms_by_bucket[bucket] = statistics.median(samples)

    def rolling_p50_ms(self, bucket: int) -> float | None:
        """Median of recent `model_ms` values at this bucket, or None if empty."""
        window = self._rolling.get(bucket)
        if not window:
            return None
        return statistics.median(window)

    @torch.inference_mode()
    def classify(self, prompt: str) -> Classification:
        result, _ = self.classify_timed(prompt)
        return result

    @torch.inference_mode()
    def classify_timed(self, prompt: str) -> tuple[Classification, Timings]:
        """Like `classify`, but also returns a tokenise/model/postprocess breakdown."""
        device = self.settings.device
        t0 = time.perf_counter()
        enc = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.settings.max_seq_len,
            return_tensors="pt",
        )
        token_count = int(enc["input_ids"].shape[-1])
        bucket = self._bucket_for(token_count)
        enc = self._pad_to(dict(enc), bucket)
        enc = {k: v.to(device) for k, v in enc.items()}
        _sync(device)
        t1 = time.perf_counter()
        logits = self.model(**enc).logits
        _sync(device)
        t2 = time.perf_counter()
        probs = torch.softmax(logits, dim=-1).float().cpu().numpy()[0]
        prob_unsafe = float(probs[1])
        idx = 1 if prob_unsafe >= self.threshold else 0
        result = Classification(
            label=LABELS[idx],
            score=float(probs[idx]),
            prob_unsafe=prob_unsafe,
            tokens=token_count,
            bucket=bucket,
        )
        t3 = time.perf_counter()
        model_ms = (t2 - t1) * 1000
        self._rolling[bucket].append(model_ms)
        return result, Timings(
            tokenize_ms=(t1 - t0) * 1000,
            model_ms=model_ms,
            postprocess_ms=(t3 - t2) * 1000,
        )

    @torch.inference_mode()
    def classify_batch(self, prompts: list[str]) -> list[Classification]:
        """Score a batch of prompts in a single forward pass per mini-batch.

        Chunked at `settings.eval_batch_size` so an oversized request doesn't
        exhaust MPS memory. Sequence length is rounded up to the next shape
        bucket to reuse compiled kernels.
        """
        if not prompts:
            return []
        out: list[Classification] = []
        for start in range(0, len(prompts), self.settings.eval_batch_size):
            chunk = prompts[start : start + self.settings.eval_batch_size]
            enc = self.tokenizer(
                chunk,
                truncation=True,
                max_length=self.settings.max_seq_len,
                padding=True,
                return_tensors="pt",
            )
            # Per-row real-token count = sum of attention mask before bucket padding.
            per_row_tokens = [int(x) for x in enc["attention_mask"].sum(dim=-1).tolist()]
            bucket = self._bucket_for(int(enc["input_ids"].shape[-1]))
            enc = self._pad_to(dict(enc), bucket)
            enc = {k: v.to(self.settings.device) for k, v in enc.items()}
            probs = torch.softmax(self.model(**enc).logits, dim=-1).float().cpu().numpy()
            for row, tok_count in zip(probs, per_row_tokens, strict=True):
                prob_unsafe = float(row[1])
                idx = 1 if prob_unsafe >= self.threshold else 0
                out.append(
                    Classification(
                        label=LABELS[idx],
                        score=float(row[idx]),
                        prob_unsafe=prob_unsafe,
                        tokens=tok_count,
                        bucket=bucket,
                    )
                )
        return out
