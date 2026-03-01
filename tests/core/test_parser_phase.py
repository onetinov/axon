"""Tests for the parsing processor (Phase 3)."""

from __future__ import annotations

import pytest

from axon.core.graph.graph import KnowledgeGraph
from axon.core.graph.model import NodeLabel, RelType, generate_id, GraphNode
from axon.core.ingestion.parser_phase import (
    FileParseData,
    get_parser,
    parse_file,
    process_parsing,
)
from axon.core.ingestion.walker import FileEntry
from axon.core.parsers.python_lang import PythonParser
from axon.core.parsers.typescript import TypeScriptParser


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def graph() -> KnowledgeGraph:
    """Return a KnowledgeGraph pre-populated with File nodes for test files."""
    g = KnowledgeGraph()

    # Python file node
    g.add_node(
        GraphNode(
            id=generate_id(NodeLabel.FILE, "src/utils.py"),
            label=NodeLabel.FILE,
            name="utils.py",
            file_path="src/utils.py",
            language="python",
        )
    )

    # TypeScript file node
    g.add_node(
        GraphNode(
            id=generate_id(NodeLabel.FILE, "src/app.ts"),
            label=NodeLabel.FILE,
            name="app.ts",
            file_path="src/app.ts",
            language="typescript",
        )
    )

    return g


PYTHON_CODE = """\
class UserService:
    def get_user(self, user_id: int) -> str:
        return str(user_id)

    def delete_user(self, user_id: int) -> None:
        pass

def helper(x: int) -> int:
    return x + 1
"""

TYPESCRIPT_CODE = """\
interface Config {
    host: string;
    port: number;
}

class App {
    start(): void {}
}

function run(config: Config): void {
    const app = new App();
    app.start();
}
"""

JAVASCRIPT_CODE = """\
function add(a, b) {
    return a + b;
}
"""


def _make_file_entry(
    path: str, content: str, language: str
) -> FileEntry:
    return FileEntry(path=path, content=content, language=language)


# ---------------------------------------------------------------------------
# get_parser tests
# ---------------------------------------------------------------------------


class TestGetParserPython:
    """get_parser returns PythonParser for 'python'."""

    def test_get_parser_python(self) -> None:
        parser = get_parser("python")
        assert isinstance(parser, PythonParser)


class TestGetParserTypeScript:
    """get_parser returns TypeScriptParser for 'typescript'."""

    def test_get_parser_typescript(self) -> None:
        parser = get_parser("typescript")
        assert isinstance(parser, TypeScriptParser)
        assert parser.dialect == "typescript"


class TestGetParserJavaScript:
    """get_parser returns TypeScriptParser with 'javascript' dialect."""

    def test_get_parser_javascript(self) -> None:
        parser = get_parser("javascript")
        assert isinstance(parser, TypeScriptParser)
        assert parser.dialect == "javascript"


class TestGetParserUnsupported:
    """get_parser raises ValueError for unknown languages."""

    def test_get_parser_unsupported(self) -> None:
        with pytest.raises(ValueError, match="Unsupported language"):
            get_parser("rust")


# ---------------------------------------------------------------------------
# parse_file tests
# ---------------------------------------------------------------------------


class TestParseFilePython:
    """parse_file parses Python source and returns correct symbols."""

    def test_parse_file_python(self) -> None:
        data = parse_file("src/utils.py", PYTHON_CODE, "python")

        assert isinstance(data, FileParseData)
        assert data.file_path == "src/utils.py"
        assert data.language == "python"

        symbol_names = [s.name for s in data.parse_result.symbols]
        assert "UserService" in symbol_names
        assert "get_user" in symbol_names
        assert "delete_user" in symbol_names
        assert "helper" in symbol_names

    def test_method_has_class_name(self) -> None:
        data = parse_file("src/utils.py", PYTHON_CODE, "python")
        methods = [s for s in data.parse_result.symbols if s.kind == "method"]
        for m in methods:
            assert m.class_name == "UserService"


class TestParseFileTypeScript:
    """parse_file parses TypeScript source and returns correct symbols."""

    def test_parse_file_typescript(self) -> None:
        data = parse_file("src/app.ts", TYPESCRIPT_CODE, "typescript")

        assert isinstance(data, FileParseData)
        assert data.file_path == "src/app.ts"
        assert data.language == "typescript"

        symbol_names = [s.name for s in data.parse_result.symbols]
        assert "Config" in symbol_names
        assert "App" in symbol_names
        assert "run" in symbol_names


# ---------------------------------------------------------------------------
# process_parsing tests
# ---------------------------------------------------------------------------


class TestProcessParsingCreatesFunctionNodes:
    """process_parsing creates Function nodes in the graph."""

    def test_process_parsing_creates_function_nodes(
        self, graph: KnowledgeGraph
    ) -> None:
        files = [_make_file_entry("src/utils.py", PYTHON_CODE, "python")]
        process_parsing(files, graph)

        func_nodes = graph.get_nodes_by_label(NodeLabel.FUNCTION)
        func_names = {n.name for n in func_nodes}
        assert "helper" in func_names

    def test_function_node_properties(self, graph: KnowledgeGraph) -> None:
        files = [_make_file_entry("src/utils.py", PYTHON_CODE, "python")]
        process_parsing(files, graph)

        func_id = generate_id(NodeLabel.FUNCTION, "src/utils.py", "helper")
        node = graph.get_node(func_id)
        assert node is not None
        assert node.name == "helper"
        assert node.file_path == "src/utils.py"
        assert node.start_line > 0
        assert node.end_line >= node.start_line
        assert "def helper" in node.content
        assert node.signature != ""


class TestProcessParsingCreatesClassNodes:
    """process_parsing creates Class nodes in the graph."""

    def test_process_parsing_creates_class_nodes(
        self, graph: KnowledgeGraph
    ) -> None:
        files = [_make_file_entry("src/utils.py", PYTHON_CODE, "python")]
        process_parsing(files, graph)

        class_nodes = graph.get_nodes_by_label(NodeLabel.CLASS)
        class_names = {n.name for n in class_nodes}
        assert "UserService" in class_names

    def test_class_node_has_content(self, graph: KnowledgeGraph) -> None:
        files = [_make_file_entry("src/utils.py", PYTHON_CODE, "python")]
        process_parsing(files, graph)

        class_id = generate_id(NodeLabel.CLASS, "src/utils.py", "UserService")
        node = graph.get_node(class_id)
        assert node is not None
        assert "class UserService" in node.content


class TestProcessParsingCreatesMethodNodes:
    """process_parsing creates Method nodes with class_name set."""

    def test_process_parsing_creates_method_nodes(
        self, graph: KnowledgeGraph
    ) -> None:
        files = [_make_file_entry("src/utils.py", PYTHON_CODE, "python")]
        process_parsing(files, graph)

        method_nodes = graph.get_nodes_by_label(NodeLabel.METHOD)
        method_names = {n.name for n in method_nodes}
        assert "get_user" in method_names
        assert "delete_user" in method_names

    def test_method_nodes_have_class_name(
        self, graph: KnowledgeGraph
    ) -> None:
        files = [_make_file_entry("src/utils.py", PYTHON_CODE, "python")]
        process_parsing(files, graph)

        method_nodes = graph.get_nodes_by_label(NodeLabel.METHOD)
        for method in method_nodes:
            assert method.class_name == "UserService"

    def test_method_node_id_uses_class_dot_method(
        self, graph: KnowledgeGraph
    ) -> None:
        files = [_make_file_entry("src/utils.py", PYTHON_CODE, "python")]
        process_parsing(files, graph)

        method_id = generate_id(
            NodeLabel.METHOD, "src/utils.py", "UserService.get_user"
        )
        node = graph.get_node(method_id)
        assert node is not None
        assert node.name == "get_user"


class TestProcessParsingCreatesDefinesRelationships:
    """process_parsing creates DEFINES relationships from File to Symbol."""

    def test_process_parsing_creates_defines_relationships(
        self, graph: KnowledgeGraph
    ) -> None:
        files = [_make_file_entry("src/utils.py", PYTHON_CODE, "python")]
        process_parsing(files, graph)

        defines_rels = graph.get_relationships_by_type(RelType.DEFINES)
        assert len(defines_rels) > 0

        file_id = generate_id(NodeLabel.FILE, "src/utils.py")
        # All DEFINES relationships should originate from the file node.
        for rel in defines_rels:
            assert rel.source == file_id

    def test_defines_relationship_targets_symbol(
        self, graph: KnowledgeGraph
    ) -> None:
        files = [_make_file_entry("src/utils.py", PYTHON_CODE, "python")]
        process_parsing(files, graph)

        defines_rels = graph.get_relationships_by_type(RelType.DEFINES)
        target_ids = {rel.target for rel in defines_rels}

        # The function node should be a target.
        func_id = generate_id(NodeLabel.FUNCTION, "src/utils.py", "helper")
        assert func_id in target_ids

        # The class node should be a target.
        class_id = generate_id(NodeLabel.CLASS, "src/utils.py", "UserService")
        assert class_id in target_ids

    def test_defines_relationship_id_format(
        self, graph: KnowledgeGraph
    ) -> None:
        files = [_make_file_entry("src/utils.py", PYTHON_CODE, "python")]
        process_parsing(files, graph)

        defines_rels = graph.get_relationships_by_type(RelType.DEFINES)
        for rel in defines_rels:
            assert rel.id.startswith("defines:")
            assert "->" in rel.id


class TestProcessParsingReturnsParseData:
    """process_parsing returns FileParseData for use by later phases."""

    def test_process_parsing_returns_parse_data(
        self, graph: KnowledgeGraph
    ) -> None:
        files = [
            _make_file_entry("src/utils.py", PYTHON_CODE, "python"),
            _make_file_entry("src/app.ts", TYPESCRIPT_CODE, "typescript"),
        ]
        result = process_parsing(files, graph)

        assert len(result) == 2
        assert all(isinstance(d, FileParseData) for d in result)

    def test_parse_data_carries_imports(
        self, graph: KnowledgeGraph
    ) -> None:
        code_with_import = "import os\n\ndef main():\n    pass\n"
        graph.add_node(
            GraphNode(
                id=generate_id(NodeLabel.FILE, "src/main.py"),
                label=NodeLabel.FILE,
                name="main.py",
                file_path="src/main.py",
                language="python",
            )
        )
        files = [_make_file_entry("src/main.py", code_with_import, "python")]
        result = process_parsing(files, graph)

        assert len(result[0].parse_result.imports) > 0

    def test_parse_data_carries_calls(
        self, graph: KnowledgeGraph
    ) -> None:
        code_with_call = "def foo():\n    bar()\n"
        graph.add_node(
            GraphNode(
                id=generate_id(NodeLabel.FILE, "src/caller.py"),
                label=NodeLabel.FILE,
                name="caller.py",
                file_path="src/caller.py",
                language="python",
            )
        )
        files = [_make_file_entry("src/caller.py", code_with_call, "python")]
        result = process_parsing(files, graph)

        call_names = [c.name for c in result[0].parse_result.calls]
        assert "bar" in call_names


class TestProcessParsingHandlesError:
    """process_parsing handles bad content gracefully without crashing."""

    def test_process_parsing_handles_error(
        self, graph: KnowledgeGraph
    ) -> None:
        # Provide an unsupported language to trigger the error path.
        graph.add_node(
            GraphNode(
                id=generate_id(NodeLabel.FILE, "src/bad.rs"),
                label=NodeLabel.FILE,
                name="bad.rs",
                file_path="src/bad.rs",
                language="rust",
            )
        )
        files = [_make_file_entry("src/bad.rs", "fn main() {}", "rust")]
        result = process_parsing(files, graph)

        # Should still return a FileParseData with empty result.
        assert len(result) == 1
        assert result[0].parse_result.symbols == []
        assert result[0].parse_result.imports == []

    def test_error_does_not_affect_other_files(
        self, graph: KnowledgeGraph
    ) -> None:
        graph.add_node(
            GraphNode(
                id=generate_id(NodeLabel.FILE, "src/bad.rs"),
                label=NodeLabel.FILE,
                name="bad.rs",
                file_path="src/bad.rs",
                language="rust",
            )
        )
        files = [
            _make_file_entry("src/bad.rs", "fn main() {}", "rust"),
            _make_file_entry("src/utils.py", PYTHON_CODE, "python"),
        ]
        result = process_parsing(files, graph)

        assert len(result) == 2
        # The Rust file should have empty symbols.
        assert result[0].parse_result.symbols == []
        # The Python file should parse successfully.
        assert len(result[1].parse_result.symbols) > 0


class TestProcessParsingTypeScript:
    """process_parsing handles TypeScript interface and class nodes."""

    def test_creates_interface_nodes(self, graph: KnowledgeGraph) -> None:
        files = [_make_file_entry("src/app.ts", TYPESCRIPT_CODE, "typescript")]
        process_parsing(files, graph)

        iface_nodes = graph.get_nodes_by_label(NodeLabel.INTERFACE)
        iface_names = {n.name for n in iface_nodes}
        assert "Config" in iface_names

    def test_creates_ts_class_and_method_nodes(
        self, graph: KnowledgeGraph
    ) -> None:
        files = [_make_file_entry("src/app.ts", TYPESCRIPT_CODE, "typescript")]
        process_parsing(files, graph)

        class_nodes = graph.get_nodes_by_label(NodeLabel.CLASS)
        class_names = {n.name for n in class_nodes}
        assert "App" in class_names

        method_nodes = graph.get_nodes_by_label(NodeLabel.METHOD)
        method_names = {n.name for n in method_nodes}
        assert "start" in method_names


# ---------------------------------------------------------------------------
# Markdown / doc node tests
# ---------------------------------------------------------------------------

# Two synthetic markdown files that cross-reference each other.
# docs/guide.md links to docs/reference.md and to the top-level README.md.
GUIDE_MD = """\
# Guide

This is the main guide. See [Reference](reference.md) for the API.

## Installation

Install with pip. Also read the [README](../README.md).

## Usage

Call the library like so.
"""

REFERENCE_MD = """\
# Reference

Full API reference. See the [Guide](guide.md) for examples.

## Functions

Details here.
"""

README_MD = """\
# Project

Top-level readme.
"""


def _doc_graph() -> KnowledgeGraph:
    """Return a graph pre-populated with DOCUMENT nodes for all three test files."""
    g = KnowledgeGraph()
    for path, name in [
        ("docs/guide.md", "guide.md"),
        ("docs/reference.md", "reference.md"),
        ("README.md", "README.md"),
    ]:
        g.add_node(
            GraphNode(
                id=generate_id(NodeLabel.DOCUMENT, path),
                label=NodeLabel.DOCUMENT,
                name=name,
                file_path=path,
                language="markdown",
            )
        )
    return g


class TestProcessParsingDocSectionNodes:
    """process_parsing creates SECTION nodes for markdown files."""

    def test_creates_section_nodes(self) -> None:
        g = _doc_graph()
        files = [_make_file_entry("docs/guide.md", GUIDE_MD, "markdown")]
        process_parsing(files, g)

        sections = list(g.get_nodes_by_label(NodeLabel.SECTION))
        names = {s.name for s in sections}
        assert "Guide" in names
        assert "Installation" in names
        assert "Usage" in names

    def test_section_node_has_line_numbers(self) -> None:
        g = _doc_graph()
        files = [_make_file_entry("docs/guide.md", GUIDE_MD, "markdown")]
        process_parsing(files, g)

        sections = {s.name: s for s in g.get_nodes_by_label(NodeLabel.SECTION)}
        guide = sections["Guide"]
        assert guide.start_line == 1
        assert guide.end_line >= guide.start_line

    def test_section_signature_is_heading_level(self) -> None:
        g = _doc_graph()
        files = [_make_file_entry("docs/guide.md", GUIDE_MD, "markdown")]
        process_parsing(files, g)

        sections = {s.name: s for s in g.get_nodes_by_label(NodeLabel.SECTION)}
        assert sections["Guide"].signature == "1"
        assert sections["Installation"].signature == "2"
        assert sections["Usage"].signature == "2"


class TestProcessParsingDocContainsHierarchy:
    """process_parsing builds the DOCUMENT → SECTION CONTAINS hierarchy."""

    def test_document_contains_h1_section(self) -> None:
        g = _doc_graph()
        files = [_make_file_entry("docs/guide.md", GUIDE_MD, "markdown")]
        process_parsing(files, g)

        contains = list(g.get_relationships_by_type(RelType.CONTAINS))
        doc_id = generate_id(NodeLabel.DOCUMENT, "docs/guide.md")
        # At least one CONTAINS edge must originate from the document node.
        doc_sources = [r for r in contains if r.source == doc_id]
        assert len(doc_sources) >= 1

    def test_h1_contains_h2_sections(self) -> None:
        g = _doc_graph()
        files = [_make_file_entry("docs/guide.md", GUIDE_MD, "markdown")]
        process_parsing(files, g)

        contains = list(g.get_relationships_by_type(RelType.CONTAINS))
        sections = {s.name: s for s in g.get_nodes_by_label(NodeLabel.SECTION)}

        h1_id = sections["Guide"].id
        h2_targets = {r.target for r in contains if r.source == h1_id}

        installation_id = sections["Installation"].id
        assert installation_id in h2_targets

    def test_no_contains_between_sibling_sections(self) -> None:
        g = _doc_graph()
        files = [_make_file_entry("docs/guide.md", GUIDE_MD, "markdown")]
        process_parsing(files, g)

        contains = list(g.get_relationships_by_type(RelType.CONTAINS))
        sections = {s.name: s for s in g.get_nodes_by_label(NodeLabel.SECTION)}

        # "Usage" should NOT be a child of "Installation" (they're siblings).
        install_id = sections["Installation"].id
        usage_id = sections["Usage"].id
        install_children = {r.target for r in contains if r.source == install_id}
        assert usage_id not in install_children


class TestProcessParsingDocReferences:
    """process_parsing creates REFERENCES edges from markdown cross-file links."""

    def test_creates_references_edge_to_indexed_doc(self) -> None:
        g = _doc_graph()
        files = [_make_file_entry("docs/guide.md", GUIDE_MD, "markdown")]
        process_parsing(files, g)

        refs = list(g.get_relationships_by_type(RelType.REFERENCES))
        assert len(refs) >= 1

        # All REFERENCES targets should be DOCUMENT nodes.
        target_ids = {r.target for r in refs}
        for tid in target_ids:
            node = g.get_node(tid)
            assert node is not None
            assert node.label == NodeLabel.DOCUMENT

    def test_references_edge_targets_correct_document(self) -> None:
        g = _doc_graph()
        files = [_make_file_entry("docs/guide.md", GUIDE_MD, "markdown")]
        process_parsing(files, g)

        refs = list(g.get_relationships_by_type(RelType.REFERENCES))
        target_ids = {r.target for r in refs}

        ref_doc_id = generate_id(NodeLabel.DOCUMENT, "docs/reference.md")
        readme_doc_id = generate_id(NodeLabel.DOCUMENT, "README.md")

        assert ref_doc_id in target_ids
        assert readme_doc_id in target_ids

    def test_references_edge_anchored_to_enclosing_section(self) -> None:
        g = _doc_graph()
        files = [_make_file_entry("docs/guide.md", GUIDE_MD, "markdown")]
        process_parsing(files, g)

        refs = list(g.get_relationships_by_type(RelType.REFERENCES))
        ref_doc_id = generate_id(NodeLabel.DOCUMENT, "docs/reference.md")

        # The link to reference.md is in the "Guide" section body (line 3).
        guide_section = next(
            s for s in g.get_nodes_by_label(NodeLabel.SECTION) if s.name == "Guide"
        )
        edge_to_ref = next(r for r in refs if r.target == ref_doc_id)
        assert edge_to_ref.source == guide_section.id

    def test_no_references_edge_for_unindexed_target(self) -> None:
        """Links to files not in the graph are silently skipped."""
        g = _doc_graph()
        # reference.md links to guide.md — but guide.md is NOT in this graph.
        g._nodes = {
            k: v for k, v in g._nodes.items()
            if "reference" in k or "README" in k
        }
        files = [_make_file_entry("docs/reference.md", REFERENCE_MD, "markdown")]
        process_parsing(files, g)

        refs = list(g.get_relationships_by_type(RelType.REFERENCES))
        # guide.md is not indexed, so no edge should point to it.
        guide_doc_id = generate_id(NodeLabel.DOCUMENT, "docs/guide.md")
        assert all(r.target != guide_doc_id for r in refs)

    def test_both_files_produce_mutual_references(self) -> None:
        """When both docs are indexed, both directions of the link appear."""
        g = _doc_graph()
        files = [
            _make_file_entry("docs/guide.md", GUIDE_MD, "markdown"),
            _make_file_entry("docs/reference.md", REFERENCE_MD, "markdown"),
        ]
        process_parsing(files, g)

        refs = list(g.get_relationships_by_type(RelType.REFERENCES))
        guide_doc_id = generate_id(NodeLabel.DOCUMENT, "docs/guide.md")
        ref_doc_id = generate_id(NodeLabel.DOCUMENT, "docs/reference.md")

        targets = {r.target for r in refs}
        assert guide_doc_id in targets   # reference.md → guide.md
        assert ref_doc_id in targets     # guide.md → reference.md
