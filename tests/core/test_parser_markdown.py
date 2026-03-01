"""Tests for the MarkdownParser."""

from __future__ import annotations

import pytest

from axon.core.parsers.markdown import MarkdownParser


@pytest.fixture()
def parser() -> MarkdownParser:
    return MarkdownParser()


class TestBasicParsing:
    def test_empty_doc(self, parser: MarkdownParser) -> None:
        result = parser.parse("", "empty.md")
        assert result.symbols == []

    def test_single_heading(self, parser: MarkdownParser) -> None:
        md = "# Hello\nsome body"
        result = parser.parse(md, "doc.md")
        assert len(result.symbols) == 1
        s = result.symbols[0]
        assert s.kind == "section"
        assert s.name == "Hello"
        assert s.signature == "1"
        assert s.start_line == 1
        assert s.end_line == 2

    def test_heading_levels(self, parser: MarkdownParser) -> None:
        md = "# H1\n## H2\n### H3\n"
        result = parser.parse(md, "doc.md")
        levels = [s.signature for s in result.symbols]
        assert levels == ["1", "2", "3"]

    def test_end_line_set_correctly(self, parser: MarkdownParser) -> None:
        md = "# First\nline a\nline b\n# Second\nline c\n"
        result = parser.parse(md, "doc.md")
        assert result.symbols[0].end_line == 3  # lines 1-3
        assert result.symbols[1].end_line == 5  # lines 4-5

    def test_section_content_captured(self, parser: MarkdownParser) -> None:
        md = "# Title\nbody text here\n"
        result = parser.parse(md, "doc.md")
        assert "body text here" in result.symbols[0].content


class TestCodeFenceSkipping:
    def test_heading_in_backtick_fence_skipped(self, parser: MarkdownParser) -> None:
        md = "# Real Heading\n```bash\n# fake heading\n```\n# Another Real\n"
        result = parser.parse(md, "doc.md")
        names = [s.name for s in result.symbols]
        assert "Real Heading" in names
        assert "Another Real" in names
        assert "fake heading" not in names

    def test_heading_in_tilde_fence_skipped(self, parser: MarkdownParser) -> None:
        md = "# Real\n~~~python\n# not a heading\n~~~\n"
        result = parser.parse(md, "doc.md")
        names = [s.name for s in result.symbols]
        assert "Real" in names
        assert "not a heading" not in names

    def test_multiple_code_fences(self, parser: MarkdownParser) -> None:
        md = (
            "# Intro\n"
            "```\n# ignored\n```\n"
            "# Middle\n"
            "```\n# also ignored\n```\n"
            "# End\n"
        )
        result = parser.parse(md, "doc.md")
        names = [s.name for s in result.symbols]
        assert names == ["Intro", "Middle", "End"]

    def test_unclosed_fence_skips_to_end(self, parser: MarkdownParser) -> None:
        md = "# Before\n```\n# inside\n# still inside\n"
        result = parser.parse(md, "doc.md")
        names = [s.name for s in result.symbols]
        assert names == ["Before"]


class TestCrossFileLinks:
    def test_md_link_becomes_call(self, parser: MarkdownParser) -> None:
        md = "# Intro\nSee [CONTRIBUTING](CONTRIBUTING.md) for details.\n"
        result = parser.parse(md, "doc.md")
        assert len(result.calls) == 1
        assert result.calls[0].name == "CONTRIBUTING.md"

    def test_non_md_link_ignored(self, parser: MarkdownParser) -> None:
        md = "# Intro\nSee [docs](https://example.com) for info.\n"
        result = parser.parse(md, "doc.md")
        assert result.calls == []

    def test_multiple_links(self, parser: MarkdownParser) -> None:
        md = "# Intro\nSee [A](a.md) and [B](b.md).\n"
        result = parser.parse(md, "doc.md")
        targets = {c.name for c in result.calls}
        assert targets == {"a.md", "b.md"}
