.DEFAULT_GOAL := help
PY := .venv/bin/python
CLI := .venv/bin/sheldon

help:  ## show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-14s\033[0m %s\n",$$1,$$2}'

setup:  ## create .venv and install the package (editable)
	uv venv --python 3.12
	uv pip install -e ".[dev]"

smoke:  ## full pipeline offline against FakeLLM, ~1s, no API key
	$(CLI) smoke

test:  ## unit tests
	$(PY) -m pytest -q

lint:  ## ruff check + format check
	.venv/bin/ruff check sheldon tests
	.venv/bin/ruff format --check sheldon tests

fmt:  ## ruff format
	.venv/bin/ruff format sheldon tests
	.venv/bin/ruff check --fix sheldon tests

ls:  ## list benchmarks and methods
	$(CLI) ls

check:  ## report which benchmark data is missing
	$(CLI) check

pilot:  ## 10 tasks, all methods, one benchmark: BENCH=arc-agi-1
	$(CLI) run -b $(or $(BENCH),arc-agi-1) -m direct,best-of-n,sheldon -n 10 --tag pilot

report:  ## the table
	$(CLI) report

clean-runs:  ## delete all run artifacts (keeps the LLM cache)
	rm -rf runs/* && touch runs/.gitkeep

clean-cache:  ## delete the LLM cache (next run pays full price)
	rm -rf .cache

.PHONY: help setup smoke test lint fmt ls check pilot report clean-runs clean-cache
