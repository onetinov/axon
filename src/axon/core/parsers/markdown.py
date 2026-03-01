"""Markdown parser for Axon doc intelligence.

Produces heading-delimited SECTION nodes with line numbers and parses
explicit markdown links as REFERENCES candidates.  No tree-sitter needed —
a simple line scanner handles ATX headings (``#``, ``##``, ``###`` …).
"""

from __future__ import annotations

import re

from axon.core.parsers.base import CallInfo, LanguageParser, ParseResult, SymbolInfo

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)")
_LINK_RE = re.compile(r"\[([^\]]+)\]\(([^)]+\.md[^)]*)\)")


class MarkdownParser(LanguageParser):
    """Parse a markdown document into heading-delimited sections.

    Each ATX heading (``#`` … ``######``) produces a :class:`SymbolInfo` with:

    - ``kind = "section"``
    - ``name`` = heading text (stripped)
    - ``start_line`` / ``end_line`` = 1-based line range of the section body
    - ``content`` = raw text of the section body
    - ``signature`` = heading level as a string (``"1"`` … ``"6"``)

    Explicit markdown cross-file links ``[text](other.md)`` are returned as
    :class:`CallInfo` objects so downstream phases can create REFERENCES edges.
    """

    def parse(self, content: str, file_path: str) -> ParseResult:
        result = ParseResult()
        lines = content.splitlines()
        total_lines = len(lines)

        # --- Pass 1: locate all headings (skip lines inside fenced code blocks) ---
        heading_positions: list[tuple[int, str, int]] = []  # (level, name, 1-based line)
        in_code_fence = False
        fence_marker = ""
        for i, line in enumerate(lines, 1):
            stripped = line.lstrip()
            if not in_code_fence:
                if stripped.startswith("```") or stripped.startswith("~~~"):
                    in_code_fence = True
                    fence_marker = stripped[:3]
                    continue
                m = _HEADING_RE.match(line)
                if m:
                    level = len(m.group(1))
                    name = m.group(2).strip()
                    heading_positions.append((level, name, i))
            else:
                if stripped.startswith(fence_marker):
                    in_code_fence = False
                    fence_marker = ""

        # --- Pass 2: assign end lines and build SymbolInfo objects ---
        for idx, (level, name, start) in enumerate(heading_positions):
            end = (
                heading_positions[idx + 1][2] - 1
                if idx + 1 < len(heading_positions)
                else total_lines
            )
            # Section body starts after the heading line
            body_lines = lines[start:end]  # lines[start] is the first line *after* heading
            section_content = "\n".join(body_lines).strip()

            result.symbols.append(
                SymbolInfo(
                    kind="section",
                    name=name,
                    start_line=start,
                    end_line=end,
                    content=section_content,
                    signature=str(level),  # heading level stored in signature field
                )
            )

        # --- Pass 3: parse explicit cross-file markdown links ---
        for m in _LINK_RE.finditer(content):
            target = m.group(2)
            # Compute approximate line number from character offset
            line_num = content[: m.start()].count("\n") + 1
            result.calls.append(
                CallInfo(
                    name=target,
                    line=line_num,
                )
            )

        return result
