"""HTTP server exposing the classifier as an OpenAPI-documented REST endpoint.

Thin FastAPI wrapper. The classifier stays portable (use the Python `Classifier`
class directly if you're in-process, or this server for language-agnostic access).

Consumers decide what to do with the verdict: block, warn, log, route to a
second-stage judge. The server is verdict-only — no policy opinions.
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .config import Settings
from .infer import Classifier

logger = logging.getLogger(__name__)

_STATIC_DIR = Path(__file__).parent / "static"


class ClassifyRequest(BaseModel):
    prompt: str = Field(..., min_length=1)


class ClassifyResponse(BaseModel):
    unsafe: bool
    probability: float
    threshold: float
    latency_ms: float  # total server-side wall time (tokenise + model + post + FastAPI overhead)
    model_ms: float  # pure forward pass, device-synchronised
    tokenize_ms: float  # tokeniser + host-to-device transfer
    tokens: int  # real token count in the prompt (excludes bucket padding)
    bucket: int  # shape bucket this request was padded to
    baseline_ms: float | None = None  # startup-measured p50 model_ms at this bucket
    session_p50_ms: float | None = None  # rolling p50 across recent requests at this bucket


class BatchClassifyRequest(BaseModel):
    prompts: list[str] = Field(..., min_length=1)


class BatchClassifyResponse(BaseModel):
    results: list[ClassifyResponse]
    total_latency_ms: float


class InfoResponse(BaseModel):
    model_path: str
    threshold: float
    max_seq_len: int
    eval_batch_size: int
    device: str
    baseline_ms: dict[str, float]  # per-bucket p50 model_ms measured at startup


class ExampleGroup(BaseModel):
    id: str
    label: str
    prompts: list[str]


class ExamplesResponse(BaseModel):
    groups: list[ExampleGroup]


_EXAMPLE_GROUP_LABELS: dict[str, str] = {
    "benign": "Benign",
    "jailbreak": "Jailbreak / injection attempts",
}


def _load_example_groups() -> list[ExampleGroup]:
    root = _STATIC_DIR / "examples"
    if not root.is_dir():
        return []
    groups: list[ExampleGroup] = []
    for sub in sorted(p for p in root.iterdir() if p.is_dir()):
        prompts: list[str] = []
        for f in sorted(sub.glob("*.txt")):
            text = f.read_text(encoding="utf-8").strip()
            if text:
                prompts.append(text)
        if not prompts:
            continue
        label = _EXAMPLE_GROUP_LABELS.get(sub.name, sub.name.replace("-", " ").title())
        groups.append(ExampleGroup(id=sub.name, label=label, prompts=prompts))
    return groups


def _check_prompt_size(prompt: str, limit: int) -> None:
    if len(prompt) > limit:
        raise HTTPException(
            status_code=413,
            detail=f"prompt length {len(prompt)} exceeds server_max_prompt_chars={limit}",
        )


def build_app(settings: Settings | None = None, classifier: Classifier | None = None) -> FastAPI:
    """Construct the FastAPI app. `classifier` is injected in tests.

    In production we load the classifier on startup so the first request doesn't
    pay model-load latency.
    """
    settings = settings if settings is not None else Settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> Any:
        if app.state.classifier is None:
            logger.info("Loading classifier from %s", settings.best_model_dir)
            app.state.classifier = Classifier(settings.best_model_dir, settings)
        # Warm kernel cache so the first real request doesn't pay MPS/CUDA
        # graph-compile latency. Cheap (~400 ms on MPS) and runs only once.
        clf = app.state.classifier
        if clf is not None and getattr(settings, "warmup_on_startup", True):
            logger.info("Warming up shape buckets on %s", settings.device)
            t0 = time.perf_counter()
            clf.warmup()
            logger.info("Warmup complete in %.1f ms", (time.perf_counter() - t0) * 1000)
        # Optional: benchmark per-bucket hardware floor. Surfaces in the demo
        # UI as a "what this machine can do" reference alongside live numbers.
        # Pure overhead; disable via GUARDRAILS_MEASURE_BASELINE_ON_STARTUP=false
        # when running guardrails as a production backend.
        if clf is not None and getattr(settings, "measure_baseline_on_startup", True):
            logger.info("Measuring hardware baseline per bucket...")
            t0 = time.perf_counter()
            clf.measure_baseline()
            logger.info(
                "Baseline complete in %.1f ms: %s",
                (time.perf_counter() - t0) * 1000,
                ", ".join(f"{b}={v:.1f}" for b, v in clf.baseline_ms_by_bucket.items()),
            )
        yield

    app = FastAPI(
        title="guardrails classifier",
        version="0.1.0",
        description=(
            "Runtime classifier for prompt safety. Returns a verdict per prompt. "
            "Consumers integrate the verdict into their own request pipeline."
        ),
        lifespan=lifespan,
    )
    app.state.settings = settings
    app.state.classifier = classifier

    if _STATIC_DIR.is_dir():
        app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")

        @app.get("/", response_class=HTMLResponse, include_in_schema=False)
        def index() -> FileResponse:
            return FileResponse(_STATIC_DIR / "index.html")

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "ok" if app.state.classifier is not None else "loading"}

    @app.get("/v1/examples", response_model=ExamplesResponse)
    def examples() -> ExamplesResponse:
        return ExamplesResponse(groups=_load_example_groups())

    @app.get("/v1/info", response_model=InfoResponse)
    def info() -> InfoResponse:
        clf = app.state.classifier
        if clf is None:
            raise HTTPException(status_code=503, detail="classifier not loaded")
        return InfoResponse(
            model_path=str(settings.best_model_dir),
            threshold=clf.threshold,
            max_seq_len=settings.max_seq_len,
            eval_batch_size=settings.eval_batch_size,
            device=settings.device,
            baseline_ms={str(b): v for b, v in clf.baseline_ms_by_bucket.items()},
        )

    # Async so PyTorch work runs on the single event-loop thread. FastAPI
    # routes sync (`def`) endpoints through an anyio threadpool, which rotates
    # worker threads. MPS kernel caches are thread-local, so each new worker
    # re-lowers the graph on its first hit (~30-50 ms). Pinning to the loop
    # thread keeps the cache hot and makes HTTP latency match the in-process
    # Python path. The classifier call itself is O(10 ms) so the event loop
    # pause is fine for this demo scope.
    @app.post("/v1/classify", response_model=ClassifyResponse)
    async def classify(req: ClassifyRequest) -> ClassifyResponse:
        clf = app.state.classifier
        if clf is None:
            raise HTTPException(status_code=503, detail="classifier not loaded")
        _check_prompt_size(req.prompt, settings.server_max_prompt_chars)
        t0 = time.perf_counter()
        result, timings = clf.classify_timed(req.prompt)
        latency_ms = (time.perf_counter() - t0) * 1000
        logger.info(
            "classify tokens=%d bucket=%d model=%.1fms tokenize=%.1fms total=%.1fms",
            result.tokens,
            result.bucket,
            timings.model_ms,
            timings.tokenize_ms,
            latency_ms,
        )
        return ClassifyResponse(
            unsafe=result.label == "unsafe",
            probability=result.prob_unsafe,
            threshold=clf.threshold,
            latency_ms=latency_ms,
            model_ms=timings.model_ms,
            tokenize_ms=timings.tokenize_ms,
            tokens=result.tokens,
            bucket=result.bucket,
            baseline_ms=clf.baseline_ms_by_bucket.get(result.bucket),
            session_p50_ms=clf.rolling_p50_ms(result.bucket),
        )

    @app.post("/v1/classify/batch", response_model=BatchClassifyResponse)
    async def classify_batch(req: BatchClassifyRequest) -> BatchClassifyResponse:
        clf = app.state.classifier
        if clf is None:
            raise HTTPException(status_code=503, detail="classifier not loaded")
        for p in req.prompts:
            _check_prompt_size(p, settings.server_max_prompt_chars)
        t0 = time.perf_counter()
        results = clf.classify_batch(req.prompts)
        total_ms = (time.perf_counter() - t0) * 1000
        per_prompt_ms = total_ms / len(results) if results else 0.0
        return BatchClassifyResponse(
            results=[
                ClassifyResponse(
                    unsafe=r.label == "unsafe",
                    probability=r.prob_unsafe,
                    threshold=clf.threshold,
                    latency_ms=per_prompt_ms,
                    model_ms=0.0,  # batch pass is not broken down per-prompt
                    tokenize_ms=0.0,
                    tokens=r.tokens,
                    bucket=r.bucket,
                    baseline_ms=clf.baseline_ms_by_bucket.get(r.bucket),
                    session_p50_ms=clf.rolling_p50_ms(r.bucket),
                )
                for r in results
            ],
            total_latency_ms=total_ms,
        )

    return app


def serve(
    settings: Settings | None = None,
    host: str | None = None,
    port: int | None = None,
    reload: bool = False,
) -> None:
    """Run the server with uvicorn. Host/port default to settings."""
    settings = settings if settings is not None else Settings()
    if not settings.best_model_dir.exists():
        raise FileNotFoundError(
            f"model not found at {settings.best_model_dir}. Train or point "
            "GUARDRAILS_OUTPUT_DIR at a directory with best/config.json."
        )

    # Route our module logs through uvicorn's handler so warmup + load lines
    # appear in the server console (uvicorn configures its own loggers and
    # doesn't install a root handler).
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:     %(message)s")

    uvicorn.run(
        "guardrails.server:_app_factory",
        factory=True,
        host=host or settings.server_host,
        port=port or settings.server_port,
        reload=reload,
    )


def _app_factory() -> FastAPI:
    """Uvicorn entry point. Kept separate so `serve()` can pass kwargs without
    pickling the settings object into a worker.
    """
    return build_app()


__all__ = [
    "BatchClassifyRequest",
    "BatchClassifyResponse",
    "ClassifyRequest",
    "ClassifyResponse",
    "ExampleGroup",
    "ExamplesResponse",
    "InfoResponse",
    "build_app",
    "serve",
]
