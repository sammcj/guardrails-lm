"""Runtime configuration loaded from environment variables or a .env file."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

Precision = Literal["bf16", "fp16", "fp32"]
Device = Literal["mps", "cuda", "cpu"]


class Settings(BaseSettings):
    """Single source of truth for runtime configuration.

    Values are loaded from environment variables prefixed with `GUARDRAILS_`,
    falling back to defaults. A `.env` file in the project root is honoured.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="GUARDRAILS_",
        protected_namespaces=(),
        extra="ignore",
    )

    encoder: str = "answerdotai/ModernBERT-base"
    # PIGuard (formerly InjecGuard) 75k training set ships as JSON in the GitHub repo.
    # Override with any Hugging Face Hub ID or another JSON URL via .env if preferred.
    dataset: str = "https://raw.githubusercontent.com/leolee99/PIGuard/main/datasets/train.json"
    text_column: str = "prompt"
    label_column: str = "label"

    max_seq_len: int = Field(1024, ge=64, le=8192)
    batch_size: int = Field(16, ge=1)
    eval_batch_size: int = Field(32, ge=1)
    grad_accum_steps: int = Field(4, ge=1)
    learning_rate: float = Field(2e-5, gt=0)
    weight_decay: float = Field(0.01, ge=0)
    warmup_ratio: float = Field(0.1, ge=0, le=1)
    num_epochs: int = Field(2, ge=1)
    max_steps: int = Field(0, ge=0)  # >0 overrides num_epochs; useful for smoke tests
    group_by_length: bool = True  # sort batches by sequence length to cut padding waste
    augment_hard_negatives: bool = True  # append NotInject_one+two to training
    augment_wildjailbreak: bool = False  # gated dataset; user must accept the AI2 licence
    augment_wildjailbreak_n: int = Field(30000, ge=0)  # cap; full adversarial-benign set is ~78k

    precision: Precision = "bf16"
    device: Device = "mps"
    output_dir: Path = Path("checkpoints")
    seed: int = 42

    server_host: str = "127.0.0.1"  # 0.0.0.0 to listen on all interfaces
    server_port: int = Field(8080, ge=1, le=65535)
    server_max_prompt_chars: int = Field(100_000, ge=1)  # reject oversize prompts before tokenising
    warmup_on_startup: bool = True  # pre-compile shape buckets so first request isn't slow
    measure_baseline_on_startup: bool = True  # benchmark hardware floor for the /v1/info panel

    @property
    def best_model_dir(self) -> Path:
        return self.output_dir / "best"
