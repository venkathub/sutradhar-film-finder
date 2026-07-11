# Sutradhar — task runner (DEC-P0-3). One-command access to the P0 workflows.
# `make` or `make help` lists targets. Targets with `##` comments are self-documented.
#
# Note: `up`/`down` use the repo-relative compose path, correct on a normal dev laptop
# (repo under $HOME). If Docker is snap-confined and the repo lives outside $HOME, see
# infra/README.md for the staging workaround.

COMPOSE ?= docker compose -f infra/docker-compose.yml

.DEFAULT_GOAL := help
.PHONY: help setup fmt lint typecheck test test-int check up down down-v mlflow-up mlflow-down mlflow-backfill mlflow-log-serving db-migrate ingest-spine enrich-tmdb load-akas fetch-plots rekey-titles build-graph ingest-training resolve-conflicts export-training-entities ft-snapshot build-ft-scaffold validate-dataset teach-dataset gpu-teacher ft-dryrun gpu-finetune ft-verdict extract-candidates review-candidates graph-report golden-validate graph-demo ingest-seed \
        smoke hf-check gpu-validate gpu-serve gpu-stop gpu-nuke api-up serving-benchmark injection-eval ui-install ui-gen ui-build ui-dev ui-test ui-e2e demo-up demo-down

help: ## List available targets
	@grep -hE '^[a-zA-Z0-9_-]+:.*?## ' $(MAKEFILE_LIST) \
		| awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

setup: ## Install the locked Python environment (uv sync)
	uv sync

fmt: ## Auto-format the codebase (ruff format)
	uv run ruff format .

lint: ## Lint + format check (ruff) — matches the Tier-1 CI job exactly
	uv run ruff check .
	uv run ruff format --check .

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

mlflow-up: ## Start self-hosted MLflow (idempotent: creates the `mlflow` DB if missing; DEC-P3-2)
	$(COMPOSE) up -d --wait postgres
	$(COMPOSE) exec -T postgres sh -c 'psql -U "$${POSTGRES_USER:-sutradhar}" -d "$${POSTGRES_DB:-sutradhar}" -tAc "SELECT 1 FROM pg_database WHERE datname='\''mlflow'\''" | grep -q 1 || createdb -U "$${POSTGRES_USER:-sutradhar}" mlflow'
	$(COMPOSE) --profile mlflow up -d --build --wait mlflow

mlflow-down: ## Stop the MLflow service (runs persist in Postgres + data/mlflow-artifacts/)
	$(COMPOSE) --profile mlflow stop mlflow

mlflow-backfill: ## P3: log the committed P2 retrieval run (Table 1) to MLflow (needs mlflow-up)
	uv run python -m sutradhar.obs.mlflow_log backfill-retrieval

mlflow-log-serving: ## P5: log the committed serving-benchmark window to MLflow (needs mlflow-up)
	uv run python -m sutradhar.obs.mlflow_log log-serving

langfuse-up: ## P3: idempotent from-scratch Langfuse bootstrap on AIC Cloud (DEC-P3-7; needs AICCLOUD_API_KEY)
	uv run python infra/langfuse/provision.py

db-migrate: ## Apply graph-schema migrations (alembic upgrade head; needs `make up`)
	uv run alembic upgrade head

seed-graph-ci: ## Seed the graph from RECORDED fixtures (offline; fresh-clone + Tier-2 path)
	uv run python data-pipeline/seed_graph_ci.py

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

ingest-training: ## P4 training-slice ingestion (D3, DEC-P4-3): disjointness gate, then the existing chain over data-pipeline/training_slice.yaml (separate snapshot roots)
	uv run pytest tests/test_ft_training_slice_disjoint.py -q
	uv run python data-pipeline/ingest_spine.py --slice data-pipeline/training_slice.yaml --snapshot-root data/raw/wikidata-training
	uv run python data-pipeline/enrich_tmdb.py --slice data-pipeline/training_slice.yaml --snapshot-root data/raw/tmdb-training
	uv run python data-pipeline/load_akas.py --slice data-pipeline/training_slice.yaml --snapshot-root data/raw/imdb-training
	uv run python data-pipeline/fetch_plots.py --snapshot-root data/raw/wikipedia-training --wikidata-snapshot-root data/raw/wikidata-training
	uv run python data-pipeline/rekey_titles.py
	uv run python data-pipeline/build_graph.py

resolve-conflicts: ## Apply human-reviewed conflict resolutions (audited YAML; layered gate §1.5)
	uv run python data-pipeline/resolve_conflicts.py

export-training-entities: ## Emit the D3 entity-disjointness fixture list from gate views (P4)
	uv run python finetune/export_training_entities.py

ft-snapshot: ## P4: export gate-view tool-result recordings for the scaffold generator (needs up)
	uv run python finetune/build_dataset.py snapshot

build-ft-scaffold: ## P4: generate the scaffold-only dataset from the committed snapshot (no DB)
	uv run python finetune/build_dataset.py scaffold

validate-dataset: ## P4: run every dataset validation layer (v0 schema, grounding, decontamination, quotas)
	uv run python finetune/build_dataset.py validate

teach-dataset: ## P4: teacher surface pass against TEACHER_BASE_URL (blank => clean skip)
	uv run python finetune/build_dataset.py teach

gpu-teacher: ## P4: ephemeral Sarvam-M session -> surface pass -> destroy (DEC-P4-1)
	uv run python infra/gpu/jarvis.py teacher

ft-dryrun: ## P4: NO-GPU rehearsal — scaffold -> mock teach -> validate -> render/mask -> config
	uv run python finetune/ft_dryrun.py

gpu-finetune: ## P4: THE one-time window — base capture -> QLoRA train -> after capture(s) -> judge -> destroy
	uv run python infra/gpu/jarvis.py finetune

ft-verdict: ## P4: 30-second demo — base-vs-QLoRA table + the frozen DEC-P4-8 keep/cut verdict (GPU off)
	uv run python finetune/ft_verdict.py

smoke: ## LLM connectivity smoke test (green whether the GPU endpoint is up or off)
	uv run python -m sutradhar.serving.smoke

api-up: ## P5: serve the orchestration API (GPU-off experience works with zero GPU/DB)
	uv run uvicorn --factory sutradhar.serving.app:create_app --host 0.0.0.0 --port $${API_PORT:-8080}

ui-install: ## P6: install the pinned UI toolchain (npm ci; Node 24 LTS, build-time only)
	cd ui/app && npm ci

ui-gen: ## P6: regenerate the v0 tool-label map (byte-derived from tool_schema.v0.json)
	python3 ui/app/scripts/gen_tool_labels.py

ui-build: ui-gen ## P6: type-check + build the UI to ui/app/dist (served by the API at /)
	cd ui/app && npm run build

ui-dev: ui-gen ## P6: Vite dev server (/api proxied to localhost:$${API_PORT:-8080})
	cd ui/app && npm run dev

ui-test: ## P6: UI component tests (Vitest 4 Browser Mode, headless chromium)
	cd ui/app && npm test

ui-e2e: ui-build ## P6: Playwright E2E — the seven named golden regressions on the rendered DOM (needs `make up`)
	cd ui/app && npx playwright test

demo-up: ## P6: THE 30-second zero-GPU demo — build+up (demo profile) -> migrate -> seed -> UI
	@test -f .env || cp .env.example .env
	$(COMPOSE) --profile demo up -d --build --wait postgres redis app
	$(COMPOSE) --profile demo exec -T app alembic upgrade head
	$(COMPOSE) --profile demo exec -T app python data-pipeline/seed_graph_ci.py
	@echo "Sutradhar UI -> http://localhost:$${API_PORT:-8080}/  (GPU off = offline + replay; export gpu-serve env + rerun for live)"

demo-down: ## P6: stop the demo stack (keeps the pgdata volume)
	$(COMPOSE) --profile demo down

hf-check: ## Verify Hugging Face Hub auth (whoami via HF_TOKEN)
	uv run python -m sutradhar.serving.hf_check

gpu-validate: ## One-time ephemeral JarvisLabs create->serve->smoke->destroy validation
	uv run python infra/gpu/jarvis.py validate

gpu-serve: ## P5: on-demand serve window — vLLM + embed/rerank sidecar, hold SERVE_HOLD_MINUTES, destroy
	uv run python infra/gpu/jarvis.py serve

serving-benchmark: ## P5: THE capture window — parity + injection ASR on/off + latency + relevancy -> sealed artifact
	uv run python infra/gpu/jarvis.py serving-benchmark

gpu-stop: ## P5: end the serve window from another terminal (destroys the tagged instance)
	uv run python infra/gpu/jarvis.py nuke

gpu-nuke: ## Safety: destroy any stray tagged JarvisLabs instance (no leaked GPU)
	uv run python infra/gpu/jarvis.py nuke

judge-worksheet: ## P3: build the ~24-item judge human-labelling worksheet from the committed gen run
	uv run python evals/judge_validate.py generate

judge-validate: ## P3: judge pass over the labelled worksheet -> kappa agreement report (needs JUDGE_BASE_URL)
	uv run python evals/judge_validate.py report

gpu-judge: ## P3: ephemeral judge session (serve JUDGE_MODEL + BGE-M3 -> kappa report -> destroy)
	uv run python infra/gpu/jarvis.py judge

generation-dryrun: ## P3: 30-second demo — scripted mock endpoint -> scored transcripts -> committed artifact
	uv run python evals/run_generation_eval.py --mode dry_run

injection-eval: ## P5: injection suite dry-run (defenses ON) -> ASR/FP/utility -> committed summary
	uv run python evals/run_injection_eval.py --defenses on

benchmark-generation: ## P4 window: authoritative generation benchmark against LLM_BASE_URL (Table 2)
	uv run python evals/run_generation_eval.py --mode live --with-judge --with-ragas

build-corpus: ## P2: gate-visible plot chunks + metadata cards -> chunks table (all ablation configs)
	uv run python rag-engine/build_corpus.py

gpu-embed: ## P2: ephemeral GPU embed+score session (export -> HF relay -> pull artifacts -> destroy)
	uv run python infra/gpu/jarvis.py embed

load-index: ## P2: load the pinned artifact run (RETRIEVAL_RUN) into chunk_embeddings
	uv run python rag-engine/load_index.py

retrieval-eval: ## P2: run the retrieval eval + ablation grid; write the committed run artifact
	uv run python evals/run_retrieval_eval.py

calibrate-no-match: ## P2: tune the NO_MATCH abstention threshold from the committed artifact
	uv run python evals/calibrate_no_match.py

rag-demo: ## P2: 30-second demo — recorded golden queries -> cited, labelled version sets (GPU off)
	uv run python rag-engine/demo.py

golden-validate: ## Validate golden fixtures against the live graph (task 14)
	uv run python evals/build_golden.py

graph-demo: ## 30-second demo: cited, relationship-labelled Drishyam version set
	uv run python data-pipeline/graph_demo.py
