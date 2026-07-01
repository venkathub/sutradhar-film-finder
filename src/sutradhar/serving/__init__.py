"""Serving shell (P0): env-driven LLM connectivity smoke + HF auth check.

No neural model runs here — the smoke test is a *client* call to an on-demand vLLM
endpoint reached via ``LLM_BASE_URL`` (normally paused). P3 (tracing) and P5
(orchestration) extend :class:`LLMClient` rather than replace it.
"""

from sutradhar.serving.llm_client import EndpointStatus, LLMClient

__all__ = ["EndpointStatus", "LLMClient"]
