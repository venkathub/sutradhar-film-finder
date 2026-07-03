"""Serving shell (P0): env-driven LLM connectivity smoke + HF auth check.

No neural model runs here — the smoke test is a *client* call to an on-demand vLLM
endpoint reached via ``LLM_BASE_URL`` (normally paused). P3 (tracing) and P5
(orchestration) extend :class:`LLMClient` rather than replace it.
"""

from sutradhar.serving.llm_client import ChatResult, EndpointStatus, LLMClient, ToolCall

# Note: hf_check and smoke are runnable modules (python -m ...); they are intentionally NOT
# imported here so `python -m sutradhar.serving.<mod>` doesn't trigger a runpy re-import warning.
# Import them directly, e.g. `from sutradhar.serving.hf_check import hf_whoami`.

__all__ = ["ChatResult", "EndpointStatus", "LLMClient", "ToolCall"]
