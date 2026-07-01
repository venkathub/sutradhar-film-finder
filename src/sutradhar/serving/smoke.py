"""``python -m sutradhar.serving.smoke`` — LLM connectivity smoke test (P0 task 6).

Green whether the on-demand GPU endpoint is up or off:
  - up   → prints the sampled token + latency, exit 0
  - off  → prints a clear "endpoint OFF" guidance message, exit 0 (graceful, no crash)
  - error→ prints the error detail, exit 1 (a genuinely misconfigured/broken endpoint)
"""

from __future__ import annotations

import json
import sys

from sutradhar.config import get_settings
from sutradhar.serving.llm_client import EndpointStatus, LLMClient

_EXIT_CODES = {"up": 0, "off": 0, "error": 1}


def run() -> EndpointStatus:
    settings = get_settings()
    return LLMClient(settings).health()


def main() -> int:
    status = run()
    icon = {"up": "✓", "off": "…", "error": "✗"}.get(status.status, "?")
    print(f"[{icon}] LLM endpoint status: {status.status.upper()} — {status.detail}")
    if status.status == "up":
        print(
            f"    model={status.model} sample_token={status.sample_token!r} "
            f"latency_ms={status.latency_ms}"
        )
    print(json.dumps(status.to_dict()))
    return _EXIT_CODES.get(status.status, 1)


if __name__ == "__main__":
    sys.exit(main())
