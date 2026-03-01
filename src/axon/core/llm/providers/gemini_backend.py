"""Gemini backend — uses OpenAI-compatible endpoint (no extra dep beyond openai package).

Requires: pip install axon[openai]  (reuses the openai package)
Env var:  GEMINI_API_KEY
"""

from __future__ import annotations

import os
from functools import lru_cache

_GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"


def _client():
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise ImportError(
            "Gemini provider uses the openai package for transport. "
            "Install with: pip install axon[openai]"
        ) from exc
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        raise EnvironmentError(
            "GEMINI_API_KEY environment variable is not set. "
            "Get a key at https://aistudio.google.com/apikey"
        )
    return OpenAI(api_key=api_key, base_url=_GEMINI_BASE_URL)


@lru_cache(maxsize=1)
def _cached_client():
    return _client()


def complete(prompt: str, model_name: str) -> str:
    """Single-turn chat completion via Gemini's OpenAI-compatible endpoint.

    Strips markdown code fences defensively — gemini-2.5-flash-lite has a known
    bug where it sometimes wraps JSON in ```json fences even in plain-text mode.
    """
    client = _cached_client()
    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = response.choices[0].message.content or ""
    # Strip ```json ... ``` fences if present (known gemini-2.5-flash-lite quirk)
    raw = raw.strip()
    if raw.startswith("```"):
        lines = raw.splitlines()
        inner = lines[1:-1] if lines[-1].strip() == "```" else lines[1:]
        raw = "\n".join(inner).strip()
    return raw


def list_models() -> list[str]:
    """Return Gemini model names available via the API key."""
    import urllib.request
    import json

    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        return []
    try:
        url = f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}"
        with urllib.request.urlopen(url, timeout=5) as r:
            data = json.loads(r.read())
        return [m["name"].replace("models/", "") for m in data.get("models", [])]
    except Exception:
        return []


def is_available() -> bool:
    """Return True if GEMINI_API_KEY is set."""
    return bool(os.environ.get("GEMINI_API_KEY"))
