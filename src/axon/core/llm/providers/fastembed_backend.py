"""FastEmbed embedding backend — wraps the fastembed package (already a core dep)."""

from __future__ import annotations

from functools import lru_cache


@lru_cache(maxsize=4)
def _get_model(model_name: str):
    from fastembed import TextEmbedding
    return TextEmbedding(model_name=model_name)


def embed(texts: list[str], model_name: str) -> list[list[float]]:
    """Embed *texts* using a fastembed local model."""
    model = _get_model(model_name)
    return [v.tolist() for v in model.embed(texts)]
