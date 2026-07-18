# syntax=docker/dockerfile:1
# Sutradhar app image (P6 task 8, DEC-P6-5): ONE image = API + built UI assets.
#
# Stage 1 — UI build (node:24 Active LTS, BUILD-TIME ONLY — never deployed; the
#   ROADMAP compute-placement rule is untouched). Uses the committed, drift-gated
#   tool_labels.json — no generator run, no Python in this stage.
# Stage 2 — Python deps via the official Astral uv-in-Docker multi-stage pattern:
#   `uv sync --frozen --no-dev` with cache mounts against the SAME uv.lock the
#   laptop and CI use; the project installed non-editable.
# Stage 3 — slim runtime: the venv + the runtime artifact tree only. No uv, no
#   compilers, no node. Model endpoints stay env-driven and EMPTY by default —
#   GPU off is the first-class state (offline payload + replay work with zero GPU).

FROM node:24-slim AS ui-build
WORKDIR /build
COPY ui/app/package.json ui/app/package-lock.json ./
RUN npm ci --no-fund --no-audit
COPY ui/app/ ./
RUN npm run build

FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS py-build
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy
WORKDIR /app
# Deps first (cached layer), project second — the documented Astral pattern.
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --frozen --no-install-project --no-dev
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen --no-dev --no-editable

FROM python:3.12-slim-bookworm AS runtime
WORKDIR /app
ENV PATH="/app/.venv/bin:$PATH"
# P7 task 5 (DEC-P7-1 finding 10): non-root runtime user. The app only reads its
# artifact tree and listens on 8080 (>1024) — root grants nothing but blast radius.
RUN groupadd --system app && useradd --system --gid app --home-dir /app app
COPY --from=py-build /app/.venv /app/.venv
# Runtime artifact tree (repo-relative paths the app reads from its cwd):
COPY alembic.ini ./
COPY alembic ./alembic
COPY docs/phases/tool_schema.v0.json ./docs/phases/tool_schema.v0.json
COPY evals/prompts ./evals/prompts
COPY evals/generation_runs ./evals/generation_runs
COPY evals/retrieval_runs ./evals/retrieval_runs
# Offline seeding chain (`seed-graph-ci`): recorded fixtures, no network, no keys.
COPY data-pipeline/seed_graph_ci.py data-pipeline/seed_slice.yaml ./data-pipeline/
COPY tests/__init__.py ./tests/__init__.py
COPY tests/integration/__init__.py tests/integration/ci_review_pass.py ./tests/integration/
COPY tests/fixtures ./tests/fixtures
# The built UI, served same-origin at / (task 2 mount).
COPY --from=ui-build /build/dist ./ui/app/dist
USER app
EXPOSE 8080
# Image-level healthcheck (P7 task 5): compose keeps its own; this makes the image
# self-describing for any orchestrator. stdlib urllib — no curl in the slim image.
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:8080/api/health', timeout=2)"]
CMD ["uvicorn", "--factory", "sutradhar.serving.app:create_app", "--host", "0.0.0.0", "--port", "8080"]
