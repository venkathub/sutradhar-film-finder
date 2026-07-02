# Sutradhar — task runner (DEC-P0-3). One-command access to the P0 workflows.
# `make` or `make help` lists targets. Targets with `##` comments are self-documented.
#
# Note: `up`/`down` use the repo-relative compose path, correct on a normal dev laptop
# (repo under $HOME). If Docker is snap-confined and the repo lives outside $HOME, see
# infra/README.md for the staging workaround.

COMPOSE ?= docker compose -f infra/docker-compose.yml

.DEFAULT_GOAL := help
.PHONY: help setup fmt lint typecheck test test-int check up down down-v db-migrate ingest-spine \
        smoke hf-check gpu-validate gpu-nuke

help: ## List available targets
	@grep -hE '^[a-zA-Z0-9_-]+:.*?## ' $(MAKEFILE_LIST) \
		| awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

setup: ## Install the locked Python environment (uv sync)
	uv sync

fmt: ## Auto-format the codebase (ruff format)
	uv run ruff format .

lint: ## Lint the codebase (ruff check)
	uv run ruff check .

typecheck: ## Static type check (mypy, strict)
	uv run mypy src

test: ## Run unit tests (integration excluded by default)
	uv run pytest

test-int: ## Run integration tests (requires `make up` first)
	uv run pytest -m integration

check: lint typecheck test ## Tier-1 gate: lint + typecheck + unit tests

up: ## Start the local stack (Postgres+pgvector, Redis) and wait for healthy
	$(COMPOSE) up -d --wait

down: ## Stop the local stack (keeps the pgdata volume)
	$(COMPOSE) down

down-v: ## Stop the local stack and DROP the pgdata volume
	$(COMPOSE) down -v

db-migrate: ## Apply graph-schema migrations (alembic upgrade head; needs `make up`)
	uv run alembic upgrade head

ingest-spine: ## Ingest the Wikidata spine for the seed slice (snapshot-first; needs db-migrate)
	uv run python data-pipeline/ingest_spine.py

smoke: ## LLM connectivity smoke test (green whether the GPU endpoint is up or off)
	uv run python -m sutradhar.serving.smoke

hf-check: ## Verify Hugging Face Hub auth (whoami via HF_TOKEN)
	uv run python -m sutradhar.serving.hf_check

gpu-validate: ## One-time ephemeral JarvisLabs create->serve->smoke->destroy validation
	uv run python infra/gpu/jarvis.py validate

gpu-nuke: ## Safety: destroy any stray tagged JarvisLabs instance (no leaked GPU)
	uv run python infra/gpu/jarvis.py nuke
