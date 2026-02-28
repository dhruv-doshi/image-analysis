from __future__ import annotations

import os

from anthropic import Anthropic

_MODEL: str = os.getenv("LLM_MODEL", "claude-haiku-4-5-20251001")


def get_client() -> Anthropic:
    """Return an Anthropic client. Reads ANTHROPIC_API_KEY from environment."""
    return Anthropic()  # SDK auto-reads ANTHROPIC_API_KEY env var
