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
