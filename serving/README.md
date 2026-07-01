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
