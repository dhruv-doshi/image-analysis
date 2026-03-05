from __future__ import annotations

import logging
import os
from collections.abc import Generator

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()  # no-op if already loaded; ensures .env is read before os.getenv below

_MODEL: str = os.getenv("LLM_MODEL", "anthropic/claude-haiku-4-5-20251001")

logger = logging.getLogger(__name__)


def get_client() -> OpenAI:
    """Return an OpenAI client pointed at OpenRouter. Reads OPENROUTER_API_KEY."""
    return OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ["OPENROUTER_API_KEY"],
    )


def synthesise_stream(tech, comp, exif, features) -> Generator[str, None, None]:
    """Stream LLM tokens for the analysis report; yields text chunks."""
    # Lazy imports to avoid circular dependency (synthesizer imports client)
    from src.llm.synthesizer import _SYSTEM_PROMPT, _build_payload  # noqa: PLC0415

    payload = _build_payload(tech, comp, exif, features)
    logger.info(
        "LLM stream request  model=%s  system_prompt=%d chars  user_payload=%d chars  max_tokens=4096",
        _MODEL,
        len(_SYSTEM_PROMPT),
        len(payload),
    )
    logger.debug("LLM stream user payload:\n%s", payload)

    client = get_client()
    finish_reason: str | None = None

    with client.chat.completions.create(
        model=_MODEL,
        max_tokens=4096,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": payload},
        ],
        stream=True,
    ) as stream:
        for chunk in stream:
            if not chunk.choices:
                continue
            choice = chunk.choices[0]
            if choice.finish_reason:
                finish_reason = choice.finish_reason
            delta = choice.delta.content
            if delta:
                yield delta

    logger.info("LLM stream done  finish_reason=%s", finish_reason)
