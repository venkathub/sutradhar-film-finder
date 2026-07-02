# Sutradhar — task runner (DEC-P0-3). One-command access to the P0 workflows.
# `make` or `make help` lists targets. Targets with `##` comments are self-documented.
#
# Note: `up`/`down` use the repo-relative compose path, correct on a normal dev laptop
# (repo under $HOME). If Docker is snap-confined and the repo lives outside $HOME, see
# infra/README.md for the staging workaround.

COMPOSE ?= docker compose -f infra/docker-compose.yml

.DEFAULT_GOAL := help
.PHONY: help setup fmt lint typecheck test test-int check up down down-v db-migrate ingest-spine enrich-tmdb load-akas fetch-plots rekey-titles build-graph extract-candidates review-candidates graph-report golden-validate graph-demo ingest-seed \
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

enrich-tmdb: ## Enrich versions from TMDB (titles/credits; needs ingest-spine + TMDB_API_KEY)
	uv run python data-pipeline/enrich_tmdb.py

load-akas: ## Load slice-filtered IMDb title.akas into version_title (needs ingest-spine)
	uv run python data-pipeline/load_akas.py

fetch-plots: ## Fetch revision-pinned Wikipedia plots into plot_texts (needs ingest-spine)
	uv run python data-pipeline/fetch_plots.py

rekey-titles: ## Re-key version_title with full transliteration + script detection (task 8)
	uv run python data-pipeline/rekey_titles.py

build-graph: ## Dub-vs-remake rule cross-check + dub-track edges + integrity report (task 9)
	uv run python data-pipeline/build_graph.py

extract-candidates: ## LLM candidate-edge extraction (GPU session; --offline replays artifact)
	uv run python data-pipeline/extract_candidates.py

review-candidates: ## Human review gate: confirm/reject candidates -> promotion (DEC-P1-6)
	uv run python data-pipeline/review_candidates.py --reviewer $${USER}

graph-report: ## Coverage per franchise + extraction lift + reproducibility stamp (task 13)
	uv run python data-pipeline/graph_report.py

ingest-seed: ingest-spine enrich-tmdb load-akas fetch-plots rekey-titles build-graph ## Full seed-slice ingestion chain (needs up + db-migrate + TMDB_API_KEY)

smoke: ## LLM connectivity smoke test (green whether the GPU endpoint is up or off)
	uv run python -m sutradhar.serving.smoke

hf-check: ## Verify Hugging Face Hub auth (whoami via HF_TOKEN)
	uv run python -m sutradhar.serving.hf_check

gpu-validate: ## One-time ephemeral JarvisLabs create->serve->smoke->destroy validation
	uv run python infra/gpu/jarvis.py validate

gpu-nuke: ## Safety: destroy any stray tagged JarvisLabs instance (no leaked GPU)
	uv run python infra/gpu/jarvis.py nuke

build-corpus: ## P2: gate-visible plot chunks + metadata cards -> chunks table (all ablation configs)
	uv run python rag-engine/build_corpus.py

gpu-embed: ## P2: ephemeral GPU embed+score session (export -> HF relay -> pull artifacts -> destroy)
	uv run python infra/gpu/jarvis.py embed

golden-validate: ## Validate golden fixtures against the live graph (task 14)
	uv run python evals/build_golden.py

graph-demo: ## 30-second demo: cited, relationship-labelled Drishyam version set
	uv run python data-pipeline/graph_demo.py
