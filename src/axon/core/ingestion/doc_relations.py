"""Optional LLM-based doc relation extraction phase.

When ``--doc-relations`` is enabled, this phase sends each SECTION node's
content to a completion model and asks it to identify semantic relationships
with other sections (DISCUSSES, BLOCKS, SUPERSEDES).

The cheap tier (markdown link → REFERENCES edges) is handled at parse time
in the markdown parser and does not require this module.
"""

from __future__ import annotations

import json
import logging
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TYPE_CHECKING

from axon.core.graph.graph import KnowledgeGraph
from axon.core.graph.model import GraphRelationship, NodeLabel, RelType, generate_id

if TYPE_CHECKING:
    from axon.core.ingestion.parser_phase import FileParseData

logger = logging.getLogger(__name__)

_RELATION_PROMPT = """\
You are a technical documentation analyst. Given a documentation section and a list of \
other section titles from the same project, identify semantic relationships.

Current section:
  Title: {title}
  File:  {file_path}
  Content (excerpt):
{content}

Other sections (title | file_path):
{candidates}

Return a JSON array of relationships. Each element must have:
  - "relation": one of "discusses", "blocks", "supersedes"
  - "target_title": the exact title from the candidates list
  - "target_file": the EXACT file_path string copied verbatim from after the | character
    in the candidates list — do not reformat, abbreviate, or infer the path

Return only valid JSON, no markdown. Return [] if no relationships are found.
"""

_VALID_RELATIONS = {"discusses", "blocks", "supersedes"}

_MAX_RETRIES = 6
_RETRY_BASE_SECONDS = 2


def _complete_with_retry(prompt: str, model: str) -> str:
    """Call complete() with retry on rate-limit errors.

    Uses the ``Retry-After`` header when the provider includes it (OpenAI SDK
    and Anthropic SDK both surface it on RateLimitError).  Falls back to
    exponential backoff with jitter when the header is absent.

    Non-retriable errors (auth, bad request, etc.) are re-raised immediately.
    """
    from axon.core.llm.providers.base import complete

    last_exc: Exception | None = None
    for attempt in range(_MAX_RETRIES):
        try:
            return complete(prompt, model)
        except Exception as exc:
            msg = str(exc).lower()
            is_rate_limit = any(
                tok in msg for tok in ("429", "rate limit", "rate_limit", "quota", "too many")
            )
            if not is_rate_limit:
                raise  # auth errors, bad requests, etc. — fail fast

            # Try to honour the Retry-After header if the SDK exposes it.
            wait: float | None = None
            response = getattr(exc, "response", None)
            if response is not None:
                headers = getattr(response, "headers", {})
                ra = headers.get("retry-after") or headers.get("Retry-After")
                if ra is not None:
                    try:
                        wait = float(ra)
                    except (TypeError, ValueError):
                        pass

            if wait is None:
                wait = float(_RETRY_BASE_SECONDS ** attempt)

            wait += random.uniform(0.1, 1.0)  # jitter
            logger.info(
                "Rate limited (attempt %d/%d); retrying in %.1fs — %s",
                attempt + 1,
                _MAX_RETRIES,
                wait,
                exc,
            )
            time.sleep(wait)
            last_exc = exc

    raise RuntimeError(
        f"Max retries ({_MAX_RETRIES}) exceeded for doc-relations completion"
    ) from last_exc


def process_doc_relations(
    graph: KnowledgeGraph,
    completion_model: str,
    max_candidates: int = 30,
) -> int:
    """Extract semantic relationships between SECTION nodes using an LLM.

    For each SECTION node, sends its content and a sample of other section
    titles to the completion model and creates DISCUSSES / BLOCKS / SUPERSEDES
    edges in the graph.

    Args:
        graph: The knowledge graph (must have SECTION nodes already populated).
        completion_model: Provider-qualified model string, e.g.
            ``"ollama/qwen2.5"`` or ``"anthropic/claude-haiku-4-5"``.
        max_candidates: Maximum number of candidate sections to include in
            each prompt (keeps token usage bounded).

    Returns:
        Total number of relationships created.
    """
    sections = list(graph.get_nodes_by_label(NodeLabel.SECTION))
    if not sections:
        return 0

    # Build a lookup: (name, file_path) → node_id
    section_lookup: dict[tuple[str, str], str] = {
        (n.name, n.file_path): n.id for n in sections
    }

    # Candidate list for the prompt (title | file_path)
    all_candidates = [f"{n.name} | {n.file_path}" for n in sections]

    # Only process sections that have content.
    sections_with_content = [s for s in sections if s.content]
    total = len(sections_with_content)
    logger.info("Doc relations: processing %d sections with model %s", total, completion_model)

    # Build work items up front (prompt + section metadata).
    WorkItem = tuple  # (section, prompt)
    work: list[WorkItem] = []
    for section in sections_with_content:
        candidates = [
            c for c in all_candidates
            if not c.startswith(f"{section.name} | {section.file_path}")
        ][:max_candidates]
        if not candidates:
            continue
        prompt = _RELATION_PROMPT.format(
            title=section.name,
            file_path=section.file_path,
            content=section.content[:800],
            candidates="\n".join(candidates),
        )
        work.append((section, prompt))

    completed = 0
    created = 0

    def _call(item: WorkItem) -> tuple:
        section, prompt = item
        return section, _complete_with_retry(prompt, completion_model)

    # LLM calls are I/O-bound — run in parallel; graph writes stay on main thread.
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(_call, item): item for item in work}
        for future in as_completed(futures):
            completed += 1
            section, _ = futures[future]
            try:
                section, raw = future.result()
                relations = _parse_relations(raw)
            except Exception as exc:
                logger.warning(
                    "Doc relations: [%d/%d] failed for '%s': %s",
                    completed, total, section.name, exc,
                )
                continue

            logger.info(
                "Doc relations: [%d/%d] %s — %d relation(s)",
                completed, total, section.name, len(relations),
            )

            for rel in relations:
                rel_type_str = rel.get("relation", "")
                target_title = rel.get("target_title", "")
                target_file = rel.get("target_file", "")

                if rel_type_str not in _VALID_RELATIONS:
                    continue

                target_id = section_lookup.get((target_title, target_file))
                if target_id is None:
                    continue

                rel_type = _rel_type_for(rel_type_str)
                rel_id = f"{rel_type_str}:{section.id}->{target_id}"

                graph.add_relationship(
                    GraphRelationship(
                        id=rel_id,
                        type=rel_type,
                        source=section.id,
                        target=target_id,
                    )
                )
                created += 1

    logger.info("Doc relations: created %d relationships", created)
    return created


def _parse_relations(raw: str) -> list[dict]:
    """Parse the LLM's JSON response, tolerating common formatting issues."""
    raw = raw.strip()
    # Strip markdown code fences if present.
    if raw.startswith("```"):
        lines = raw.splitlines()
        raw = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    try:
        data = json.loads(raw)
        if isinstance(data, list):
            return data
    except (json.JSONDecodeError, ValueError):
        pass
    return []


def _rel_type_for(name: str) -> RelType:
    return {
        "discusses": RelType.DISCUSSES,
        "blocks": RelType.BLOCKS,
        "supersedes": RelType.SUPERSEDES,
    }[name]
