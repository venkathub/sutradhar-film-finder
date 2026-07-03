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
harness (see below).

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
