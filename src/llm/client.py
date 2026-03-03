from __future__ import annotations

import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()  # no-op if already loaded; ensures .env is read before os.getenv below

_MODEL: str = os.getenv("LLM_MODEL", "anthropic/claude-haiku-4-5-20251001")


def get_client() -> OpenAI:
    """Return an OpenAI client pointed at OpenRouter. Reads OPENROUTER_API_KEY."""
    return OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ["OPENROUTER_API_KEY"],
    )
