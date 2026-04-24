.PHONY: help install lint format test train eval benchmark inspect classify sweep calibrate eval-ood compare serve demo export-onnx clean

help:
	@echo "Targets: install lint format test train eval benchmark inspect classify sweep calibrate eval-ood compare serve demo export-onnx clean"

install:
	uv sync

lint:
	uv run ruff check src tests
	uv run ruff format --check src tests

format:
	uv run ruff format src tests
	uv run ruff check --fix src tests

test:
	uv run pytest

train:
	uv run guardrails train

eval:
	uv run guardrails eval

benchmark:
	uv run guardrails benchmark

inspect:
	uv run guardrails inspect

sweep:
	uv run guardrails sweep

calibrate:
	uv run guardrails calibrate

eval-ood:
	uv run guardrails eval-ood

compare:
	@echo "Usage: uv run guardrails compare-checkpoints PATH1 PATH2 [PATH3 ...] [--baseline PATH] [--skip-eval] [--skip-ood]"
	@echo "Example: uv run guardrails compare-checkpoints checkpoints-v2/ checkpoints/best/"

serve:
	uv run guardrails serve

demo:
	@HOST=$${HOST:-127.0.0.1}; PORT=$${PORT:-8080}; \
		echo "guardrails demo UI -> http://$$HOST:$$PORT/"; \
		uv run guardrails serve --host $$HOST --port $$PORT

export-onnx:
	@test -n "$(OUTPUT)" || (echo "Usage: make export-onnx OUTPUT=path/to/model.onnx" && exit 1)
	uv run guardrails export-onnx --output "$(OUTPUT)"

classify:
	@test -n "$(PROMPT)" || (echo "Usage: make classify PROMPT='your text'" && exit 1)
	uv run guardrails classify "$(PROMPT)"

clean:
	rm -rf checkpoints runs logs .pytest_cache .ruff_cache dist build *.egg-info
