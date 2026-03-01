"""Anthropic backend — wraps the anthropic package (optional dep: pip install axon[anthropic])."""

from __future__ import annotations

import os
from functools import lru_cache


def _client():
    try:
        import anthropic
    except ImportError as exc:
        raise ImportError(
            "Anthropic provider requires the anthropic package. "
            "Install with: pip install axon[anthropic]"
        ) from exc
    return anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))


@lru_cache(maxsize=1)
def _cached_client():
    return _client()


def complete(prompt: str, model_name: str) -> str:
    """Single-turn completion via Anthropic Messages API."""
    client = _cached_client()
    message = client.messages.create(
        model=model_name,
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )
    return message.content[0].text if message.content else ""


def is_available() -> bool:
    """Return True if ANTHROPIC_API_KEY is set."""
    return bool(os.environ.get("ANTHROPIC_API_KEY"))
