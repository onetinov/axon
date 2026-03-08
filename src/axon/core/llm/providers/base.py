"""Provider router — dispatches embed calls based on model prefix.

Model string format: ``{provider}/{model_name}``

Examples::

    fastembed/BAAI/bge-small-en-v1.5
    ollama/nomic-embed-text
    openai/text-embedding-3-small
"""

from __future__ import annotations

import os


def embed(texts: list[str], model: str) -> list[list[float]]:
    """Route embedding to the correct backend based on the model prefix.

    Args:
        texts: Texts to embed.
        model: Provider-qualified model string, e.g. ``ollama/nomic-embed-text``.

    Returns:
        List of embedding vectors, one per input text.
    """
    prefix, _, name = model.partition("/")

    if prefix == "fastembed":
        from axon.core.llm.providers.fastembed_backend import embed as _embed
        return _embed(texts, name)

    if prefix == "ollama":
        from axon.core.llm.providers.ollama_backend import embed as _embed
        return _embed(texts, name)

    if prefix == "openai":
        from axon.core.llm.providers.openai_backend import embed as _embed
        return _embed(texts, name)

    raise ValueError(
        f"Unknown embedding provider prefix {prefix!r} in model {model!r}. "
        f"Expected one of: fastembed, ollama, openai"
    )


def list_available_models() -> dict[str, list[dict]]:
    """Probe available backends and return embedding model availability.

    Returns:
        Dict with key ``\"embedding\"``, containing a list of dicts with
        keys ``\"model\"``, ``\"available\"``, ``\"note\"``.
    """
    embedding_models: list[dict] = []

    # --- fastembed (always available, local) ---
    embedding_models.append({
        "model": "fastembed/BAAI/bge-small-en-v1.5",
        "available": True,
        "note": "local, 384 dims, no API key needed",
    })

    # --- Ollama ---
    ollama_models: list[str] = []
    try:
        from axon.core.llm.providers.ollama_backend import list_models
        ollama_models = list_models()
    except Exception:
        pass

    for m in ["nomic-embed-text", "mxbai-embed-large"]:
        available = any(m in om for om in ollama_models)
        embedding_models.append({
            "model": f"ollama/{m}",
            "available": available,
            "note": "local" if available else "not found in local Ollama",
        })

    # --- OpenAI ---
    openai_key = bool(os.environ.get("OPENAI_API_KEY"))
    for m, note in [
        ("text-embedding-3-small", "~$0.02/M tokens, recommended for docs"),
        ("text-embedding-3-large", "~$0.13/M tokens"),
    ]:
        embedding_models.append({
            "model": f"openai/{m}",
            "available": openai_key,
            "note": note if openai_key else "OPENAI_API_KEY not set",
        })

    return {"embedding": embedding_models}
