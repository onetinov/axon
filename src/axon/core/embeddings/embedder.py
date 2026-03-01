"""Batch embedding pipeline for Axon knowledge graphs.

Takes a :class:`KnowledgeGraph`, generates natural-language descriptions for
each embeddable node, encodes them, and returns a list of :class:`NodeEmbedding`
objects ready for storage.

Model routing:
  - Bare model names (no ``/`` prefix) are treated as fastembed models.
  - Prefixed names (``ollama/…``, ``openai/…``, etc.) are routed through the
    provider adapter in :mod:`axon.core.llm.providers.base`.
"""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

from axon.core.embeddings.text import build_class_method_index, generate_text
from axon.core.graph.graph import KnowledgeGraph
from axon.core.graph.model import NodeLabel
from axon.core.storage.base import NodeEmbedding

if TYPE_CHECKING:
    from fastembed import TextEmbedding


@lru_cache(maxsize=4)
def _get_fastembed_model(model_name: str) -> "TextEmbedding":
    from fastembed import TextEmbedding

    return TextEmbedding(model_name=model_name)


# Backward-compatible alias used by tests and MCP tools.
_get_model = _get_fastembed_model


_PROVIDER_PREFIXES = frozenset({"fastembed", "ollama", "openai", "anthropic"})


def _embed_texts(texts: list[str], model: str, batch_size: int = 64) -> list[list[float]]:
    """Embed *texts* using *model*, routing to the correct backend.

    If the part before the first ``/`` is a known provider prefix the call is
    forwarded to the provider-adapter layer.  Otherwise *model* is treated as a
    bare fastembed model name (e.g. ``"BAAI/bge-small-en-v1.5"``).
    """
    prefix = model.partition("/")[0]
    if prefix in _PROVIDER_PREFIXES:
        from axon.core.llm.providers.base import embed as _provider_embed
        # Process in batches to avoid overwhelming remote APIs.
        results: list[list[float]] = []
        for i in range(0, len(texts), batch_size):
            results.extend(_provider_embed(texts[i : i + batch_size], model))
        return results

    # Default: treat as fastembed model name directly.
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

# Code-only embeddable labels (used when separating code vs doc embedding).
_CODE_LABELS: frozenset[NodeLabel] = frozenset(
    {
        NodeLabel.FILE,
        NodeLabel.FUNCTION,
        NodeLabel.CLASS,
        NodeLabel.METHOD,
        NodeLabel.INTERFACE,
        NodeLabel.TYPE_ALIAS,
        NodeLabel.ENUM,
    }
)

_DOC_LABELS: frozenset[NodeLabel] = frozenset({NodeLabel.DOCUMENT, NodeLabel.SECTION})


def embed_graph(
    graph: KnowledgeGraph,
    model_name: str = "BAAI/bge-small-en-v1.5",
    doc_embed_model: str | None = None,
    batch_size: int = 64,
) -> list[NodeEmbedding]:
    """Generate embeddings for all embeddable nodes in the graph.

    When *doc_embed_model* is provided, DOCUMENT and SECTION nodes are
    embedded separately using that model (which may be an Ollama or OpenAI
    model optimised for prose).  Code nodes always use *model_name*.

    Args:
        graph: The knowledge graph whose nodes should be embedded.
        model_name: Model for code nodes.  Defaults to
            ``"BAAI/bge-small-en-v1.5"`` (fastembed).
        doc_embed_model: Optional model for DOCUMENT/SECTION nodes.
            When ``None``, doc nodes use the same *model_name*.
        batch_size: Number of texts to encode per batch.

    Returns:
        A list of :class:`NodeEmbedding` instances.
    """
    all_nodes = [n for n in graph.iter_nodes() if n.label in EMBEDDABLE_LABELS]

    if not all_nodes:
        return []

    class_method_idx = build_class_method_index(graph)

    if doc_embed_model is None or doc_embed_model == model_name:
        # Single model for all nodes — simple path.
        texts = [generate_text(node, graph, class_method_idx) for node in all_nodes]
        vectors = _embed_texts(texts, model_name, batch_size)
        return [
            NodeEmbedding(node_id=node.id, embedding=vec)
            for node, vec in zip(all_nodes, vectors)
        ]

    # Separate models for code vs doc nodes.
    code_nodes = [n for n in all_nodes if n.label in _CODE_LABELS]
    doc_nodes = [n for n in all_nodes if n.label in _DOC_LABELS]

    results: list[NodeEmbedding] = []

    if code_nodes:
        texts = [generate_text(n, graph, class_method_idx) for n in code_nodes]
        vectors = _embed_texts(texts, model_name, batch_size)
        results.extend(
            NodeEmbedding(node_id=n.id, embedding=v)
            for n, v in zip(code_nodes, vectors)
        )

    if doc_nodes:
        texts = [generate_text(n, graph, class_method_idx) for n in doc_nodes]
        vectors = _embed_texts(texts, doc_embed_model, batch_size)
        results.extend(
            NodeEmbedding(node_id=n.id, embedding=v)
            for n, v in zip(doc_nodes, vectors)
        )

    return results


def embed_nodes(
    graph: KnowledgeGraph,
    node_ids: set[str],
    model_name: str = "BAAI/bge-small-en-v1.5",
    doc_embed_model: str | None = None,
    batch_size: int = 64,
) -> list[NodeEmbedding]:
    """Like :func:`embed_graph`, but only for the given *node_ids*."""
    if not node_ids:
        return []

    nodes = [graph.get_node(nid) for nid in node_ids]
    nodes = [n for n in nodes if n is not None and n.label in EMBEDDABLE_LABELS]

    if not nodes:
        return []

    class_method_idx = build_class_method_index(graph)

    if doc_embed_model is None or doc_embed_model == model_name:
        texts = [generate_text(n, graph, class_method_idx) for n in nodes]
        vectors = _embed_texts(texts, model_name, batch_size)
        return [NodeEmbedding(node_id=n.id, embedding=v) for n, v in zip(nodes, vectors)]

    code_nodes = [n for n in nodes if n.label in _CODE_LABELS]
    doc_nodes = [n for n in nodes if n.label in _DOC_LABELS]
    results: list[NodeEmbedding] = []

    if code_nodes:
        texts = [generate_text(n, graph, class_method_idx) for n in code_nodes]
        vectors = _embed_texts(texts, model_name, batch_size)
        results.extend(NodeEmbedding(node_id=n.id, embedding=v) for n, v in zip(code_nodes, vectors))

    if doc_nodes:
        texts = [generate_text(n, graph, class_method_idx) for n in doc_nodes]
        vectors = _embed_texts(texts, doc_embed_model, batch_size)
        results.extend(NodeEmbedding(node_id=n.id, embedding=v) for n, v in zip(doc_nodes, vectors))

    return results
