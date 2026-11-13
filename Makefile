.DEFAULT_GOAL := help
PYTEST_ARGS ?=
.PHONY: help install install-train install-report hooks lint fmt typecheck test test-fast test-integration check check-fast

help: ## List targets
	@grep -E '^[a-z-]+:.*##' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-18s %s\n", $$1, $$2}'

install: ## Create the venv and install locked dependencies
	uv sync --frozen

install-train: ## Also install PyTorch for the benchmark (large download)
	uv sync --frozen --group train

install-seg: ## Also install torch and transformers for `fraktal streets --segment`
	uv sync --frozen --group seg

install-report: ## Also install plotting and PDF libraries for `fraktal report`
	uv sync --frozen --group report

hooks: ## Install the pre-push hook
	git config core.hooksPath .githooks

lint: ## Ruff lint and format check
	uv run ruff check .
	uv run ruff format --check .

fmt: ## Apply ruff formatting and safe fixes
	uv run ruff check --fix .
	uv run ruff format .

typecheck: ## mypy strict
	uv run mypy

test: ## Fast layer: no infra, no network (CI passes PYTEST_ARGS=--durations=30)
	uv run pytest -m "not integration" $(PYTEST_ARGS)

test-fast: ## As test, without the slow end-to-end tests: seconds, not minutes
	uv run pytest -m "not integration and not slow"

test-integration: ## Slower layer: subprocesses, external services, network
	uv run pytest -m integration

check-fast: lint typecheck ## The pre-push gate: ruff and mypy only. CI runs the tests

check: lint typecheck test ## Everything CI runs for lint, types and unit tests. Slow on the 8 GB machine
