"""``python -m sutradhar.serving.hf_check`` — Hugging Face Hub auth check (P0 task 7).

Verifies the ``HF_TOKEN`` env var authenticates against the Hub via ``HfApi().whoami()``
(huggingface_hub v1.0, P0_SPEC §2.7). A missing token yields the clear var-named error from
``Settings.require`` (not a traceback); an invalid token yields a clear :class:`HFAuthError`.
The token itself is never printed.
"""

from __future__ import annotations

import sys

from huggingface_hub import HfApi

from sutradhar.config import ConfigError, Settings, get_settings


class HFAuthError(RuntimeError):
    """Raised when a present HF_TOKEN fails to authenticate against the Hub."""


def hf_whoami(settings: Settings | None = None) -> str:
    """Return the authenticated Hugging Face username, or raise a clear error.

    Raises:
        ConfigError: ``HF_TOKEN`` is absent/empty (message names the var).
        HFAuthError: the token is present but rejected by the Hub.
    """
    settings = settings or get_settings()
    token = settings.require("hf_token")  # ConfigError names HF_TOKEN if missing
    try:
        info = HfApi().whoami(token=token)
    except Exception as exc:  # noqa: BLE001 — surface a clean message, never a traceback/token
        raise HFAuthError(
            "Hugging Face auth failed for HF_TOKEN. Check the token is valid and is a modern "
            "(API-v2) token — legacy tokens 401 with huggingface_hub v1.0. "
            f"Underlying error: {type(exc).__name__}: {exc}"
        ) from exc
    name = info.get("name") if isinstance(info, dict) else None
    if not name:
        raise HFAuthError("Hugging Face whoami returned no username; token may be invalid.")
    return str(name)


def main() -> int:
    try:
        name = hf_whoami()
    except (ConfigError, HFAuthError) as exc:
        print(f"[✗] {exc}")
        return 1
    print(f"[✓] Authenticated to Hugging Face Hub as: {name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
