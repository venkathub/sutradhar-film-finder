# serving

Config subsystem, LLM connectivity client, and (later) the FastAPI orchestration API.

**Import package:** `sutradhar.serving`

## Planned architecture
- **P0 shell (this phase):** typed env-driven `Settings` (pydantic-settings, in
  `sutradhar.config`) — the single place the environment is read — with secret redaction and clear
  var-named errors for contextually-required secrets; plus an OpenAI-compatible `LLMClient` with a
  graceful health/smoke path (endpoint UP → token; endpoint OFF → clear status, exit 0, no crash),
  and an HF Hub `whoami` auth check. No neural model runs here — the smoke test is a *client* call
  to an on-demand vLLM endpoint reached via the env-driven `LLM_BASE_URL`.
- **P5:** FastAPI application routes, request/response guardrails, caching, and token/cost/latency
  tracking; vLLM serving adapter (optional llama.cpp/GGUF fallback). The endpoint is chosen by env
  var, never hardcoded.

## Status
**P0 builds the config + LLM client + smoke CLI shell only.** API orchestration is **P5**.
**P3 extends `LLMClient` with the tool-calling `chat()` path** used by the generation eval
harness (see below). **P5 (in progress) adds the orchestration API** — see the next section.

## FastAPI orchestration API (P5, DEC-P5-1)

`make api-up` (→ `uvicorn --factory sutradhar.serving.app:create_app`, port `API_PORT`) serves
the JSON chat surface (P5_SPEC §2.2). Pure FastAPI — the Spring Boot gateway was a deliberate
cut (DEC-P5-1). **The GPU-off experience works on a fresh clone with zero GPU and zero DB.**

| Route | Behaviour |
|---|---|
| `POST /api/chat` | GPU up → orchestrator turn (below). GPU off/error → the structured offline payload, **HTTP 200, never a 5xx** (DEC-P0-4 at the API layer) |
| `GET /api/status` | cached degradation state (one health probe per ~30 s TTL, DEC-P5-5) |
| `GET /api/health` | aggregate: api / db / redis / llm / embed / rerank (`EndpointStatus`-shaped) |
| `GET /api/replays` | replay discovery (P6 task 1): the pinned run's id/model/prompt-hash + replayable fixture ids — what the UI replay browser lists |
| `GET /api/replay/{fixture}` | committed pinned-run transcript (e.g. `GS-08a`) — the zero-GPU Papanasam story; 404 lists available fixtures |
| `GET /api/metrics` | token/cost/latency summary (P5 task 10) |

**One turn** (`sutradhar.serving.orchestrator`, the P3 driver loop lifted to live traffic):
session load (Redis, in-memory fallback — DEC-P5-2) → caps → agent loop (LLM ↔ the five v0
tools; every emitted call schema-validated before execution, errors fed back, ≤6 rounds) →
guardrails (datamarking spotlight on tool results, adversarial check, no-hallucinated-movie
output gate — DEC-P5-3, prompt bundle v1.1) → response assembly (`versions[]` with
`relationship`/`is_original`/`sources` passed through untouched; citations per version;
INTENT preamble parsed) → state save. LLM off/error mid-turn → offline payload, state not
persisted (clean retry). `search_by_plot` runs the real hybrid `Retriever` against the GPU
sidecar when `EMBED_BASE_URL`/`RERANK_BASE_URL`/`RETRIEVAL_RUN` are set (`make gpu-serve`
prints them); unconfigured → tool-error feedback while the graph tools keep answering.

**Trace (P6 task 1, DEC-P6-4):** every tool call in the loop also becomes a `TraceStep` on
`ChatResponse.trace[]` — `tool`/`arguments` as emitted, the `validate_emitted_call` outcome
(`valid`/`validation_error`), a **bounded** `result_summary` (kind/count/ids — never the tool
result blob; `versions[]`/`citations[]` already carry the user-facing content), and per-call
`latency_ms`. Additive: pre-P6 consumers ignore it. The UI trace view renders exactly what the
orchestrator already validated — no re-derived tool semantics.

**Offline evidence (P6 task 1):** the GPU-off payload's `evidence` block links the benchmark
doc, the replay route, and — when `DEMO_VIDEO_URL` is set (a GitHub Release asset, DEC-P6-3) —
the recorded demo video; unset ⇒ the key is omitted, never a dead link.

30-second demo (no GPU, no DB):

```
make api-up
curl -s localhost:8080/api/chat -H 'content-type: application/json' \
     -d '{"message": "wo film jisme baap evidence chhupa ke family ko bachata hai"}'
curl -s localhost:8080/api/replay/GS-08a
```

Tests: `tests/test_api.py` (HTTP surface, GPU-off + scripted GS-08a over HTTP),
`tests/test_orchestrator.py`, `tests/test_guardrails.py`, `tests/test_sessions.py`,
`tests/test_live_executor.py`, `tests/test_providers_http.py`; the six named golden regressions
through the API path in `tests/integration/test_api_golden_regressions.py`.

### Results — live serving-benchmark window (2026-07-05, `servewin-25c029d3`)

One `make serving-benchmark` run on an on-demand A100 (two ephemeral sessions, both destroyed;
sealed to `evals/serving_runs/`, logged to MLflow `sutradhar/serving`). Full numbers +
reproducibility stamp in `docs/BENCHMARKS.md` §"Serving & guardrails":

- **Injection ASR = 0.000 defenses-on** (vs 0.273 off), false-positive rate 0.000, utility-under-
  attack 0.727 — indirect prompt-injection defense (DEC-P5-3) holds against the live model.
- **API e2e latency p50/p95 = 4535 / 5395 ms**, **76 tok/s** through the full turn (+ a vLLM
  `/metrics` snapshot).
- **Live-path parity:** the winner retrieval cell re-validated through the live GPU providers —
  Recall@10 = 1.0, VSR GS-01/06 = 1.0 (identical to the committed P2 run; "swaps providers, not
  code").
- **answer_relevancy backfill = 0.571** (12/12 scored) over the pinned base run — discharges the
  P4 footnote-¹ gap.
- **Graceful degradation:** with the GPU off, `/api/chat` returns a structured HTTP 200 and
  `/api/replay/GS-08a` serves the recorded story — demonstrable with zero GPU.

## Chat with tool-calling (P3)

`LLMClient.chat(messages, tools=…, tool_choice=…)` is the OpenAI tool-calling round-trip the
P3 generation harness drives (P3_SPEC §1.3). It returns a `ChatResult`:

- `status` — the same DEC-P0-4 contract as `health()`: `up` / `off` (paused GPU — first-class,
  never an exception) / `error` (endpoint answered badly). The eval driver and the P5 API path
  can always rely on `chat()` **never raising for a down endpoint**.
- `message` — the raw OpenAI-wire assistant message dict, appended verbatim to the next turn's
  `messages` and recorded in the eval transcript.
- `tool_calls` — parsed `ToolCall`s (`id`, `name`, `arguments_raw`, `arguments`). Malformed
  argument JSON gives `arguments=None` (raw string preserved): a **scored** schema-validity
  failure for the driver, never a crash. Validation against `tool_schema.v0.json` happens in
  the driver (P3_SPEC §2.3), not here — the client stays schema-agnostic.
- `usage` (prompt/completion/total tokens — the Table 2 tokens/sec source), `latency_ms`,
  `finish_reason`, `detail`.

The `tools` array is generated from `docs/phases/tool_schema.v0.json` by the caller, never
hand-written (P3_SPEC §2.8). One injected `httpx.Client` backs both the raw `/health` GET and
the SDK, so a single `MockTransport` mocks the whole surface in tests (`tests/test_llm_chat.py`).

## LLM connectivity smoke (P0)

`make smoke` (→ `python -m sutradhar.serving.smoke`) probes `LLM_BASE_URL` and prints an
`EndpointStatus` (P0_SPEC §2.2). It is **green whether the endpoint is up or off** — the on-demand
GPU is normally paused, and "endpoint OFF" is a first-class success path, not a crash.

Probe sequence (OpenAI-compatible, DEC-P0-4): `GET /health` → `GET /v1/models` (confirm served id ==
`LLM_MODEL`) → `POST /v1/chat/completions` (`max_tokens=1`, capture token + latency).

| Status | Meaning | CLI exit |
|--------|---------|----------|
| `up` | endpoint answered; token + latency captured | `0` |
| `off` | connection refused / timeout (paused GPU) | `0` (graceful) |
| `error` | endpoint reachable but 5xx / malformed body | `1` |

`up`/`off` exit `0` per the spec's "green in both states". `error` exits `1` so a genuinely
misconfigured/broken endpoint is detectable in CI and scripts (P0_SPEC §2.2 pins only up/off = 0;
error exit code is this module's choice, logged here rather than in DECISIONS.md).

## HF Hub auth check (P0)

`make hf-check` (→ `python -m sutradhar.serving.hf_check`) verifies `HF_TOKEN` authenticates via
`HfApi().whoami()` (huggingface_hub v1.0). Prints the username on success (exit 0). A **missing**
token → clear `ConfigError` naming `HF_TOKEN` (exit 1); an **invalid** token → clear `HFAuthError`
noting the modern **API-v2 token** requirement (exit 1). The token value is never printed. The CLI
equivalent `hf auth whoami` is optional; the programmatic path is used so it stays CI-portable and
mockable.
