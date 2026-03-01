"""Phase 3: Code parsing for Axon.

Takes file entries from the walker, parses each one with the appropriate
tree-sitter parser, and adds symbol nodes (Function, Class, Method, Interface,
TypeAlias, Enum) to the knowledge graph with DEFINES relationships from File
to Symbol.
"""

from __future__ import annotations

import logging
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any

from axon.core.graph.graph import KnowledgeGraph
from axon.core.graph.model import (
    GraphNode,
    GraphRelationship,
    NodeLabel,
    RelType,
    generate_id,
)
from axon.core.ingestion.walker import FileEntry
from axon.core.parsers.base import LanguageParser, ParseResult

logger = logging.getLogger(__name__)

_KIND_TO_LABEL: dict[str, NodeLabel] = {
    "function": NodeLabel.FUNCTION,
    "class": NodeLabel.CLASS,
    "method": NodeLabel.METHOD,
    "interface": NodeLabel.INTERFACE,
    "type_alias": NodeLabel.TYPE_ALIAS,
    "enum": NodeLabel.ENUM,
    "section": NodeLabel.SECTION,
}

@dataclass
class FileParseData:
    """Parse results for a single file, kept for later phases."""

    file_path: str
    language: str
    parse_result: ParseResult

_PARSER_CACHE: dict[str, LanguageParser] = {}

def get_parser(language: str) -> LanguageParser:
    """Return the appropriate parser for *language*.

    Parser instances are cached per language to avoid repeated instantiation.

    Args:
        language: One of ``"python"``, ``"typescript"``, ``"javascript"``,
            or ``"markdown"``.

    Returns:
        A :class:`LanguageParser` instance ready to parse source code or docs.

    Raises:
        ValueError: If *language* is not supported.
    """
    cached = _PARSER_CACHE.get(language)
    if cached is not None:
        return cached

    if language == "python":
        from axon.core.parsers.python_lang import PythonParser

        parser = PythonParser()

    elif language == "typescript":
        from axon.core.parsers.typescript import TypeScriptParser

        parser = TypeScriptParser(dialect="typescript")

    elif language == "javascript":
        from axon.core.parsers.typescript import TypeScriptParser

        parser = TypeScriptParser(dialect="javascript")

    elif language == "markdown":
        from axon.core.parsers.markdown import MarkdownParser

        parser = MarkdownParser()

    else:
        raise ValueError(
            f"Unsupported language {language!r}. "
            f"Expected one of: python, typescript, javascript, markdown"
        )

    _PARSER_CACHE[language] = parser
    return parser

def parse_file(file_path: str, content: str, language: str) -> FileParseData:
    """Parse a single file and return structured parse data.

    If parsing fails for any reason the returned :class:`FileParseData` will
    contain an empty :class:`ParseResult` so that downstream phases can
    safely skip it.

    Args:
        file_path: Relative path to the file (used for identification).
        content: Raw source code of the file.
        language: Language identifier (``"python"``, ``"typescript"``, etc.).

    Returns:
        A :class:`FileParseData` carrying the parse result.
    """
    try:
        parser = get_parser(language)
        result = parser.parse(content, file_path)
    except Exception:
        logger.warning("Failed to parse %s (%s), skipping", file_path, language, exc_info=True)
        result = ParseResult()

    return FileParseData(file_path=file_path, language=language, parse_result=result)

def process_parsing(
    files: list[FileEntry],
    graph: KnowledgeGraph,
    max_workers: int = 8,
) -> list[FileParseData]:
    """Parse every file and populate the knowledge graph with symbol nodes.

    Parsing is done in parallel using a thread pool (tree-sitter releases
    the GIL during C parsing).  Graph mutation remains sequential since
    :class:`KnowledgeGraph` is not thread-safe.

    For each symbol discovered during parsing a graph node is created with
    the appropriate label (Function, Class, Method, etc.) and a DEFINES
    relationship is added from the owning File node to the new symbol node.

    Args:
        files: File entries produced by the walker phase.
        graph: The knowledge graph to populate.  File nodes are expected to
            already exist (created by the structure phase).
        max_workers: Maximum number of threads for parallel parsing.

    Returns:
        A list of :class:`FileParseData` objects that carry the full parse
        results (imports, calls, heritage, type_refs) for use by later phases.
    """
    # Phase 1: Parse all files in parallel.
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        all_parse_data = list(
            executor.map(
                lambda f: parse_file(f.path, f.content, f.language),
                files,
            )
        )

    # Phase 2: Graph mutation (sequential — not thread-safe).
    for file_entry, parse_data in zip(files, all_parse_data):
        if file_entry.language == "markdown":
            _populate_doc_nodes(file_entry, parse_data, graph)
        else:
            _populate_code_nodes(file_entry, parse_data, graph)

    return all_parse_data


def _populate_code_nodes(
    file_entry: "FileEntry",
    parse_data: FileParseData,
    graph: KnowledgeGraph,
) -> None:
    """Populate symbol nodes and DEFINES edges for a code file."""
    file_id = generate_id(NodeLabel.FILE, file_entry.path)
    exported_names: set[str] = set(parse_data.parse_result.exports)

    # Build class -> base class names for storing on class nodes.
    class_bases: dict[str, list[str]] = {}
    for cls_name, kind, parent_name in parse_data.parse_result.heritage:
        if kind == "extends":
            class_bases.setdefault(cls_name, []).append(parent_name)

    for symbol in parse_data.parse_result.symbols:
        label = _KIND_TO_LABEL.get(symbol.kind)
        if label is None:
            logger.warning(
                "Unknown symbol kind %r for %s in %s, skipping",
                symbol.kind,
                symbol.name,
                file_entry.path,
            )
            continue

        # For methods, use "ClassName.method_name" as the symbol name
        # to disambiguate methods across different classes.
        symbol_name = (
            f"{symbol.class_name}.{symbol.name}"
            if symbol.kind == "method" and symbol.class_name
            else symbol.name
        )

        symbol_id = generate_id(label, file_entry.path, symbol_name)

        props: dict[str, Any] = {}
        if symbol.decorators:
            props["decorators"] = symbol.decorators
        if symbol.kind == "class" and symbol.name in class_bases:
            props["bases"] = class_bases[symbol.name]

        is_exported = symbol.name in exported_names

        graph.add_node(
            GraphNode(
                id=symbol_id,
                label=label,
                name=symbol.name,
                file_path=file_entry.path,
                start_line=symbol.start_line,
                end_line=symbol.end_line,
                content=symbol.content,
                signature=symbol.signature,
                class_name=symbol.class_name,
                language=file_entry.language,
                is_exported=is_exported,
                properties=props,
            )
        )

        rel_id = f"defines:{file_id}->{symbol_id}"
        graph.add_relationship(
            GraphRelationship(
                id=rel_id,
                type=RelType.DEFINES,
                source=file_id,
                target=symbol_id,
            )
        )


def _populate_doc_nodes(
    file_entry: "FileEntry",
    parse_data: FileParseData,
    graph: KnowledgeGraph,
) -> None:
    """Populate SECTION nodes, CONTAINS edges, and REFERENCES edges for a markdown document.

    DOCUMENT node is expected to already exist (created by the structure phase).
    Builds the heading hierarchy: DOCUMENT → H1 sections → H2 sections → H3 …
    using a stack-based algorithm.

    Also creates REFERENCES edges from explicit markdown cross-file links
    ``[text](other.md)`` to the target DOCUMENT node, anchored to the innermost
    section that contains the link's line number.
    """
    doc_id = generate_id(NodeLabel.DOCUMENT, file_entry.path)

    # Stack: list of (heading_level, section_node_id)
    heading_stack: list[tuple[int, str]] = []
    # Collected in order for enclosing-section lookups when wiring REFERENCES.
    sections_by_line: list[tuple[int, int, str]] = []  # (start, end, section_id)

    for symbol in parse_data.parse_result.symbols:
        if symbol.kind != "section":
            continue

        # Heading level is stored in the signature field as "1", "2", etc.
        try:
            level = int(symbol.signature)
        except (ValueError, TypeError):
            level = 1

        # Unique name: include line number to handle repeated headings.
        symbol_name = f"{symbol.name}:{symbol.start_line}"
        section_id = generate_id(NodeLabel.SECTION, file_entry.path, symbol_name)

        graph.add_node(
            GraphNode(
                id=section_id,
                label=NodeLabel.SECTION,
                name=symbol.name,
                file_path=file_entry.path,
                start_line=symbol.start_line,
                end_line=symbol.end_line,
                content=symbol.content,
                signature=symbol.signature,  # heading level
                language=file_entry.language,
            )
        )

        # Pop the stack until we find a section with a strictly lower level.
        while heading_stack and heading_stack[-1][0] >= level:
            heading_stack.pop()

        # Parent is the nearest ancestor section, or the document itself.
        parent_id = heading_stack[-1][1] if heading_stack else doc_id

        rel_id = f"contains:{parent_id}->{section_id}"
        graph.add_relationship(
            GraphRelationship(
                id=rel_id,
                type=RelType.CONTAINS,
                source=parent_id,
                target=section_id,
            )
        )

        heading_stack.append((level, section_id))
        sections_by_line.append((symbol.start_line, symbol.end_line, section_id))

    # --- REFERENCES edges from explicit markdown cross-file links ---
    for call in parse_data.parse_result.calls:
        # Resolve the link target relative to this file's directory and
        # normalise away any ".." components (os.path.normpath handles this
        # without hitting the filesystem).
        raw = str(PurePosixPath(file_entry.path).parent / call.name)
        target_path = os.path.normpath(raw).replace(os.sep, "/")
        target_doc_id = generate_id(NodeLabel.DOCUMENT, target_path)

        # Only create the edge if the target document was actually indexed.
        if target_doc_id not in graph._nodes:
            continue

        # Anchor the edge to the innermost section containing the link's line,
        # or to the document itself if the link is in the preamble (before any
        # heading).
        source_id = doc_id
        for start, end, sid in sections_by_line:
            if start <= call.line <= end:
                source_id = sid
                break

        rel_id = f"references:{source_id}->{target_doc_id}"
        graph.add_relationship(
            GraphRelationship(
                id=rel_id,
                type=RelType.REFERENCES,
                source=source_id,
                target=target_doc_id,
            )
        )
