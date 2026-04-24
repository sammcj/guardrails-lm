# guardrails

## Project discipline

- **Never train on eval datasets**: `NotInject_three`, `XSTest`, `deepset/prompt-injections`, `WildJailbreak eval` are held out for OOD benchmarks. See `DEFAULT_REGISTRY` in `src/guardrails/ood.py`. Training on them contaminates the benchmark and invalidates every comparison downstream.
- **Checkpoints convention**: active model lives at `checkpoints/best/`. Preserve the previous version as `checkpoints-vN/best/` before retraining (`mv checkpoints checkpoints-vN && make train`). v1, v2 (active), v3 (preserved with negative result).
- **Augment from train-side splits only**: `NotInject_one` + `NotInject_two` for training; `NotInject_three` for eval. Mirror this discipline for any new augmentation source.

## Dataset gotchas

- **WildJailbreak loader**: requires `delimiter="\t"` and `keep_default_na=False` on `load_dataset`. Without them pyarrow type-infers a text column as `double` and crashes with a non-obvious error. Template: `augment_with_wildjailbreak` in `src/guardrails/data.py`.
- **HF `concatenate_datasets` schema alignment**: after normalising a new dataset to `(text_column, label)`, always `cast(train_subset.features)` before concatenating. Otherwise HF refuses with "Features can't be aligned". Template: `augment_with_notinject`.
- **Gated HF datasets**: WildJailbreak needs licence acceptance + `hf auth login` before `load_dataset` will work. All augmentation helpers must `logger.warning` and return `ds` unchanged on load failure, never raise — training should keep going with whatever did load.

## Experiment history

Read before proposing augmentation or retraining:

- README's "Experiments and next steps" section has the v3 WildJailbreak log — net negative (deepset TPR dropped 13 points, NotInject FPR slightly worse). Same section sketches the LLM-as-judge fallback recommendation.
- `docs/failure-analysis-v2.md` shows v2 learned a trigger-word shortcut. Missed attacks have 0 occurrences of "ignore" vs 6 in caught attacks; non-English and role-play framings are the biggest blind spots. Same-style augmentation won't fix this.

## Tests

- Fully offline. Mock `load_dataset`, use `_FakeTokenizer`, inject a stub `Classifier` into the FastAPI app via `build_app(classifier=...)`. Never add network calls. Templates: `tests/test_data.py`, `tests/test_server.py`.
- `make lint` scopes to `src/` and `tests/` only. Format `scripts/` manually with `uv run ruff format scripts/` when adding analysis scripts.

## Runtime gotchas

- **ONNX export opset**: requires `>= 18` on PyTorch 2.8+ because the emitted `Split` op uses `num_outputs`. Default in `export.py` is 18; don't lower it.
- **Pyright in `server.py`**: FastAPI route handlers and the uvicorn factory are decorator-registered or string-referenced, so Pyright flags them as "not accessed". These are false positives — ignore them. Real Pyright errors elsewhere should still be fixed.
- **`socksio` is a hard dep**: httpx fails to import if any `*_PROXY` env var contains `socks5h://` without it. Don't remove.

## Useful references

- Training + eval + OOD comparison workflow: `README.md`
- Threshold calibration modes (cost / F1 / FPR budget): `src/guardrails/calibration.py`
- Classifier surface exposed to library users: `src/guardrails/__init__.py` re-exports `Classifier`, `Classification`, `resolve_threshold`
- Before declaring augmentation/eval work done, run `make eval-ood` and diff vs `checkpoints-v2/best/` using `uv run guardrails compare-checkpoints checkpoints/best checkpoints-v2/best`.
