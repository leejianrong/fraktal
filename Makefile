.DEFAULT_GOAL := help
.PHONY: help install hooks lint fmt typecheck test test-integration check

help: ## List targets
	@grep -E '^[a-z-]+:.*##' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-18s %s\n", $$1, $$2}'

install: ## Create the venv and install locked dependencies
	uv sync --frozen

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

test: ## Fast layer: no infra, no network
	uv run pytest -m "not integration"

test-integration: ## Slower layer: subprocesses, external services, network
	uv run pytest -m integration

check: lint typecheck test ## The pre-push gate; mirrors the cheap CI jobs
