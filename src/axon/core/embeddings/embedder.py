"""Batch embedding pipeline for Axon knowledge graphs.

Takes a :class:`KnowledgeGraph`, generates natural-language descriptions for
each embeddable node, encodes them, and returns a list of :class:`NodeEmbedding`
objects ready for storage.

A single model is used for all node types (code + docs) to ensure all
embeddings share the same vector space and cross-label similarity works.

Model routing:
  - Bare model names (no ``/`` prefix) → fastembed local inference.
  - ``fastembed/…`` → fastembed (explicit prefix).
  - ``ollama/…`` → local Ollama server.
  - ``openai/…`` → OpenAI API.

Default model: ``nomic-ai/nomic-embed-text-v1.5``
  768-dim, 8192-token context window, trained on code + prose.
  Auto-upgrade to ``openai/text-embedding-3-small`` when OPENAI_API_KEY
  is present (1536-dim, higher quality, ~$0.02/M tokens).
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import TYPE_CHECKING

from axon.core.embeddings.text import build_class_method_index, generate_text
from axon.core.graph.graph import KnowledgeGraph
from axon.core.graph.model import NodeLabel
from axon.core.storage.base import NodeEmbedding

if TYPE_CHECKING:
    from fastembed import TextEmbedding


_DEFAULT_FASTEMBED_MODEL = "nomic-ai/nomic-embed-text-v1.5"
_DEFAULT_OPENAI_MODEL = "openai/text-embedding-3-small"


def check_model_available(model: str) -> tuple[bool, str]:
    """Check whether *model* is currently usable.

    Returns ``(available, reason)`` where *reason* is a human-readable
    explanation when ``available`` is ``False``.
    """
    prefix = model.partition("/")[0]

    if prefix == "openai" or (prefix not in _PROVIDER_PREFIXES and "/" not in model):
        # Bare nomic-style names go to fastembed — always available.
        if prefix not in _PROVIDER_PREFIXES:
            return True, ""
        if not os.environ.get("OPENAI_API_KEY"):
            return False, f"OPENAI_API_KEY is not set (required for {model})"
        return True, ""

    if prefix == "ollama":
        try:
            import urllib.request
            urllib.request.urlopen("http://localhost:11434/api/tags", timeout=3)
            return True, ""
        except Exception:
            return False, f"Ollama is not reachable at localhost:11434 (required for {model})"

    # fastembed — always available
    return True, ""


def default_embed_model() -> str:
    """Return the best available embedding model.

    Prefers OpenAI when ``OPENAI_API_KEY`` is set; otherwise falls back to
    the best local fastembed model.
    """
    if os.environ.get("OPENAI_API_KEY"):
        return _DEFAULT_OPENAI_MODEL
    return _DEFAULT_FASTEMBED_MODEL


@lru_cache(maxsize=4)
def _get_fastembed_model(model_name: str) -> "TextEmbedding":
    from fastembed import TextEmbedding

    return TextEmbedding(model_name=model_name)


# Backward-compatible alias used by tests and MCP tools.
_get_model = _get_fastembed_model


_PROVIDER_PREFIXES = frozenset({"fastembed", "ollama", "openai"})


def _embed_texts(texts: list[str], model: str, batch_size: int = 64) -> list[list[float]]:
    """Embed *texts* using *model*, routing to the correct backend.

    If the part before the first ``/`` is a known provider prefix the call is
    forwarded to the provider-adapter layer.  Otherwise *model* is treated as a
    bare fastembed model name.
    """
    prefix = model.partition("/")[0]
    if prefix in _PROVIDER_PREFIXES:
        from axon.core.llm.providers.base import embed as _provider_embed
        results: list[list[float]] = []
        for i in range(0, len(texts), batch_size):
            results.extend(_provider_embed(texts[i : i + batch_size], model))
        return results

    # Default: treat as bare fastembed model name.
    fastembed_model = _get_fastembed_model(model)
    return [v.tolist() for v in fastembed_model.embed(texts, batch_size=batch_size)]


# Labels worth embedding — skip Folder, Community, Process (structural only).
EMBEDDABLE_LABELS: frozenset[NodeLabel] = frozenset(
    {
        NodeLabel.FILE,
        NodeLabel.FUNCTION,
        NodeLabel.CLASS,
        NodeLabel.METHOD,
        NodeLabel.INTERFACE,
        NodeLabel.TYPE_ALIAS,
        NodeLabel.ENUM,
        NodeLabel.DOCUMENT,
        NodeLabel.SECTION,
    }
)


def embed_graph(
    graph: KnowledgeGraph,
    model_name: str | None = None,
    batch_size: int = 64,
) -> list[NodeEmbedding]:
    """Generate embeddings for all embeddable nodes using a single model.

    One model is used for all node types (code + docs) so that all vectors
    live in the same space and cross-label similarity works correctly.

    Args:
        graph: The knowledge graph whose nodes should be embedded.
        model_name: Embedding model to use.  When ``None``, :func:`default_embed_model`
            is called to auto-select (OpenAI if key present, else nomic fastembed).
        batch_size: Number of texts to encode per batch.

    Returns:
        A list of :class:`NodeEmbedding` instances.
    """
    model = model_name or default_embed_model()
    all_nodes = [n for n in graph.iter_nodes() if n.label in EMBEDDABLE_LABELS]

    if not all_nodes:
        return []

    class_method_idx = build_class_method_index(graph)
    texts = [generate_text(node, graph, class_method_idx) for node in all_nodes]
    vectors = _embed_texts(texts, model, batch_size)
    return [
        NodeEmbedding(node_id=node.id, embedding=vec)
        for node, vec in zip(all_nodes, vectors)
    ]


def embed_nodes(
    graph: KnowledgeGraph,
    node_ids: set[str],
    model_name: str | None = None,
    batch_size: int = 64,
) -> list[NodeEmbedding]:
    """Like :func:`embed_graph`, but only for the given *node_ids*."""
    if not node_ids:
        return []

    model = model_name or default_embed_model()
    nodes = [graph.get_node(nid) for nid in node_ids]
    nodes = [n for n in nodes if n is not None and n.label in EMBEDDABLE_LABELS]

    if not nodes:
        return []

    class_method_idx = build_class_method_index(graph)
    texts = [generate_text(n, graph, class_method_idx) for n in nodes]
    vectors = _embed_texts(texts, model, batch_size)
    return [NodeEmbedding(node_id=n.id, embedding=v) for n, v in zip(nodes, vectors)]
