"""Env-driven configuration for Sutradhar.

The single source of runtime config: every knob is read from the environment here
and nowhere else (CLAUDE.md: "no secrets in code", "all model config swappable by env").
"""

from sutradhar.config.settings import ConfigError, Settings, get_settings

__all__ = ["ConfigError", "Settings", "get_settings"]
