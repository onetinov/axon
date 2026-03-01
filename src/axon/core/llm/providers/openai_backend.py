"""OpenAI backend — wraps the openai package (optional dep: pip install axon[openai])."""

from __future__ import annotations

import os
from functools import lru_cache


def _client():
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise ImportError(
            "OpenAI provider requires the openai package. Install with: pip install axon[openai]"
        ) from exc
    return OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))


@lru_cache(maxsize=1)
def _cached_client():
    return _client()


def embed(texts: list[str], model_name: str, dimensions: int | None = None) -> list[list[float]]:
    """Embed *texts* using OpenAI's embedding API."""
    client = _cached_client()
    kwargs: dict = {"model": model_name, "input": texts}
    if dimensions is not None:
        kwargs["dimensions"] = dimensions
    response = client.embeddings.create(**kwargs)
    return [item.embedding for item in response.data]


def complete(prompt: str, model_name: str) -> str:
    """Single-turn chat completion via OpenAI."""
    client = _cached_client()
    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content or ""


def is_available() -> bool:
    """Return True if OPENAI_API_KEY is set."""
    return bool(os.environ.get("OPENAI_API_KEY"))
