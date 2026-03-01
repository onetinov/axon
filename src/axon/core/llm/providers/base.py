"""Provider router — dispatches embed/complete calls based on model prefix.

Model string format: ``{provider}/{model_name}``

Examples::

    fastembed/BAAI/bge-small-en-v1.5
    ollama/nomic-embed-text
    openai/text-embedding-3-small
    anthropic/claude-haiku-4-5
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


def complete(prompt: str, model: str) -> str:
    """Route completion to the correct backend based on the model prefix.

    Args:
        prompt: The user prompt.
        model: Provider-qualified model string, e.g. ``anthropic/claude-haiku-4-5``.

    Returns:
        Completion text.
    """
    prefix, _, name = model.partition("/")

    if prefix == "ollama":
        from axon.core.llm.providers.ollama_backend import complete as _complete
        return _complete(prompt, name)

    if prefix == "openai":
        from axon.core.llm.providers.openai_backend import complete as _complete
        return _complete(prompt, name)

    if prefix == "anthropic":
        from axon.core.llm.providers.anthropic_backend import complete as _complete
        return _complete(prompt, name)

    if prefix == "gemini":
        from axon.core.llm.providers.gemini_backend import complete as _complete
        return _complete(prompt, name)

    raise ValueError(
        f"Unknown completion provider prefix {prefix!r} in model {model!r}. "
        f"Expected one of: ollama, openai, anthropic, gemini"
    )


def list_available_models() -> dict[str, list[dict]]:
    """Probe available backends and return a structured availability report.

    Returns:
        Dict with keys ``"embedding"`` and ``"completion"``, each a list of
        dicts with keys ``"model"``, ``"available"``, ``"note"``.
    """
    embedding_models: list[dict] = []
    completion_models: list[dict] = []

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

    ollama_embed_candidates = ["nomic-embed-text", "mxbai-embed-large"]
    ollama_complete_candidates = ["qwen2.5", "llama3.2", "mistral"]

    for m in ollama_embed_candidates:
        available = any(m in om for om in ollama_models)
        embedding_models.append({
            "model": f"ollama/{m}",
            "available": available,
            "note": "local" if available else "not found in local Ollama",
        })

    for m in ollama_complete_candidates:
        available = any(m in om for om in ollama_models)
        completion_models.append({
            "model": f"ollama/{m}",
            "available": available,
            "note": "local" if available else "not found in local Ollama",
        })

    # --- OpenAI ---
    openai_key = bool(os.environ.get("OPENAI_API_KEY"))
    for m, note in [
        ("text-embedding-3-small", "~$0.02/M tokens"),
        ("text-embedding-3-large", "~$0.13/M tokens"),
    ]:
        embedding_models.append({
            "model": f"openai/{m}",
            "available": openai_key,
            "note": note if openai_key else "OPENAI_API_KEY not set",
        })
    for m, note in [
        ("gpt-4.1-nano", "non-reasoning, ~$0.10/$0.40 per 1M"),
        ("gpt-4.1-mini", "non-reasoning, ~$0.40/$1.60 per 1M"),
    ]:
        completion_models.append({
            "model": f"openai/{m}",
            "available": openai_key,
            "note": note if openai_key else "OPENAI_API_KEY not set",
        })

    # --- Anthropic ---
    anthropic_key = bool(os.environ.get("ANTHROPIC_API_KEY"))
    for m, note in [
        ("claude-haiku-4-5", "~$1.00/$5.00 per 1M, best edge-case quality"),
        ("claude-sonnet-4-6", "~$3.00/$15.00 per 1M"),
    ]:
        completion_models.append({
            "model": f"anthropic/{m}",
            "available": anthropic_key,
            "note": note if anthropic_key else "ANTHROPIC_API_KEY not set",
        })

    # --- Gemini ---
    gemini_key = bool(os.environ.get("GEMINI_API_KEY"))
    for m, note in [
        ("gemini-2.5-flash-lite", "~$0.10/$0.40 per 1M, recommended default"),
        ("gemini-2.5-flash",      "~$0.30/$2.50 per 1M"),
    ]:
        completion_models.append({
            "model": f"gemini/{m}",
            "available": gemini_key,
            "note": note if gemini_key else "GEMINI_API_KEY not set",
        })

    return {"embedding": embedding_models, "completion": completion_models}
