"""Fine-tune ModernBERT as a binary safe/unsafe classifier."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, cast

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from transformers import EvalPrediction, Trainer, TrainingArguments
from transformers.trainer_pt_utils import LengthGroupedSampler

if TYPE_CHECKING:
    from torch.utils.data import Dataset as TorchDataset

from .config import Settings
from .data import (
    augment_with_notinject,
    augment_with_wildjailbreak,
    build_tokenizer,
    collator,
    load_and_split,
    tokenise,
)
from .model import build_model

logger = logging.getLogger(__name__)


def compute_metrics(eval_pred: EvalPrediction) -> dict[str, float]:
    logits = eval_pred.predictions
    if isinstance(logits, tuple):
        logits = logits[0]
    preds = np.argmax(logits, axis=1)
    labels = eval_pred.label_ids
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "precision": float(precision_score(labels, preds, zero_division=0.0)),
        "recall": float(recall_score(labels, preds, zero_division=0.0)),
        "f1": float(f1_score(labels, preds, zero_division=0.0)),
        "f1_macro": float(f1_score(labels, preds, average="macro", zero_division=0.0)),
    }


def _training_args(settings: Settings) -> TrainingArguments:
    kwargs: dict[str, Any] = dict(
        output_dir=str(settings.output_dir),
        num_train_epochs=settings.num_epochs,
        per_device_train_batch_size=settings.batch_size,
        per_device_eval_batch_size=settings.eval_batch_size,
        gradient_accumulation_steps=settings.grad_accum_steps,
        learning_rate=settings.learning_rate,
        weight_decay=settings.weight_decay,
        warmup_ratio=settings.warmup_ratio,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        logging_steps=50,
        seed=settings.seed,
        dataloader_num_workers=2,
        report_to="none",
    )
    if settings.precision == "bf16":
        kwargs["bf16"] = True
    elif settings.precision == "fp16":
        kwargs["fp16"] = True
    if settings.max_steps > 0:
        kwargs["max_steps"] = settings.max_steps
    return TrainingArguments(**kwargs)


class _LengthGroupedTrainer(Trainer):
    """Trainer that groups batches by sequence length to reduce padding waste.

    `group_by_length` was removed from `TrainingArguments` in transformers 5.x,
    but `LengthGroupedSampler` is still available; this subclass wires it back up.
    """

    def _get_train_sampler(self, *args, **kwargs):
        if self.train_dataset is None:
            return None
        # HF `datasets.Dataset` duck-types as a torch map-style Dataset but isn't
        # a subclass of `torch.utils.data.Dataset`, so cast for the type checker.
        return LengthGroupedSampler(
            batch_size=self.args.per_device_train_batch_size
            * self.args.gradient_accumulation_steps,
            dataset=cast("TorchDataset[Any]", self.train_dataset),
            model_input_name="input_ids",
        )


def _has_checkpoint(settings: Settings) -> bool:
    """True if at least one `checkpoint-N` directory exists under output_dir."""
    if not settings.output_dir.exists():
        return False
    return any(p.name.startswith("checkpoint-") for p in settings.output_dir.iterdir())


def train(settings: Settings) -> dict[str, float]:
    """Run the fine-tuning loop. Resumes from the latest checkpoint if one exists.

    Returns the best-model eval metrics.
    """
    tokenizer = build_tokenizer(settings)
    ds = load_and_split(settings)
    if settings.augment_hard_negatives:
        ds = augment_with_notinject(ds, settings)
    if settings.augment_wildjailbreak:
        ds = augment_with_wildjailbreak(ds, settings, n=settings.augment_wildjailbreak_n)
    ds = tokenise(ds, tokenizer, settings)
    model = build_model(settings)

    trainer_cls = _LengthGroupedTrainer if settings.group_by_length else Trainer
    trainer = trainer_cls(
        model=model,
        args=_training_args(settings),
        train_dataset=ds["train"],
        eval_dataset=ds["val"],
        processing_class=tokenizer,
        data_collator=collator(tokenizer),
        compute_metrics=compute_metrics,
    )
    resume = _has_checkpoint(settings)
    if resume:
        logger.info("Resuming from latest checkpoint in %s", settings.output_dir)
    trainer.train(resume_from_checkpoint=resume)

    settings.best_model_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(settings.best_model_dir))
    tokenizer.save_pretrained(str(settings.best_model_dir))

    return trainer.evaluate()
