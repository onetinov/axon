# Axon Doc Intelligence Layer

## What This Is

A parallel documentation intelligence layer added on top of Axon's existing code graph.
Indexes markdown files as `DOCUMENT` / `SECTION` nodes in the same KuzuDB graph as code,
with line-number-accurate section boundaries, embedding-based semantic search, and an
optional LLM completion pass that extracts implicit semantic relationships between sections.

**Primary use case**: a massive `README.md` or architecture doc, impractical to read in
full, becomes consumable. Each heading-delimited section gets its own node with line numbers,
its own embedding, and `CONTAINS` edges linking the H1→H2→H3 hierarchy. A query for
"authentication" returns the relevant section from line 23,000 of the file rather than the
whole document.

## What Was Built

### New node types
- `NodeLabel.DOCUMENT` — one per `.md` file (analogous to `FILE`)
- `NodeLabel.SECTION` — one per heading-delimited section (analogous to `FUNCTION`)

### New relationship types
- `RelType.CONTAINS` — DOCUMENT→SECTION and SECTION→SECTION (heading hierarchy)
- `RelType.REFERENCES` — SECTION→DOCUMENT/SECTION from explicit markdown links
- `RelType.DISCUSSES` — LLM-extracted implicit semantic relationship
- `RelType.BLOCKS` — LLM-extracted blocking/dependency relationship
- `RelType.SUPERSEDES` — LLM-extracted supersession relationship

### New files
| File | Purpose |
|---|---|
| `src/axon/config/doc_config.py` | `DocConfig` dataclass persisted to `meta.json` |
| `src/axon/core/parsers/markdown.py` | Heading-delimited section parser with code-fence awareness |
| `src/axon/core/ingestion/doc_relations.py` | LLM relation extraction phase (parallel, retrying) |
| `src/axon/core/llm/providers/base.py` | Provider router — dispatches by model prefix |
| `src/axon/core/llm/providers/fastembed_backend.py` | Local fastembed (no extra deps) |
| `src/axon/core/llm/providers/ollama_backend.py` | Local Ollama via urllib (no extra deps) |
| `src/axon/core/llm/providers/openai_backend.py` | OpenAI embedding + completion |
| `src/axon/core/llm/providers/anthropic_backend.py` | Anthropic completion |
| `src/axon/core/llm/providers/gemini_backend.py` | Gemini via OpenAI-compat endpoint |

### Modified files
- `model.py` — new NodeLabel + RelType values
- `pipeline.py` — threads DocConfig through all phases; calls doc-relations phase
- `walker.py` — extends walked extensions with `.md` when doc mode enabled
- `watcher.py` — reads DocConfig from `meta.json`; routes `.md` through doc pipeline
- `structure.py` — creates DOCUMENT nodes for `.md` files
- `parser_phase.py` — routes `language == "markdown"` to MarkdownParser
- `embedder.py` — separate embed model for doc vs code nodes
- `text.py` — `_text_for_document()` and `_text_for_section()` generators
- `kuzu_backend.py` — schema migration: adds missing FROM/TO pairs to CodeRelation table
- `tools.py` — `handle_doc_search()`, `handle_doc_context()`
- `server.py` — registers `axon_doc_search`, `axon_doc_context` MCP tools
- `cli/main.py` — `--include-docs`, `--doc-relations`, `--doc-model`, `--embed-model`; `axon models` command
- `pyproject.toml` — optional extras: `[openai]`, `[anthropic]`, `[google]`

## Usage

```bash
# Index docs alongside code (embedding only — recommended starting point)
axon analyze . --include-docs --embed-model openai/text-embedding-3-small --full

# Add LLM semantic relationship extraction (~2 min for 400 sections)
axon analyze . --include-docs \
  --embed-model openai/text-embedding-3-small \
  --doc-relations \
  --doc-model gemini/gemini-2.5-flash-lite \
  --full

# List available models and API key status
axon models
```

## Provider Model Strings

| String | Type | Cost (1M tokens) | Notes |
|---|---|---|---|
| `fastembed/BAAI/bge-small-en-v1.5` | embed | free | local, default for code |
| `ollama/nomic-embed-text` | embed | free | local, prose-optimised |
| `openai/text-embedding-3-small` | embed | $0.02 | recommended for docs |
| `ollama/qwen2.5` | completion | free | known path hallucination issues |
| `gemini/gemini-2.5-flash-lite` | completion | $0.10/$0.40 | **recommended default** |
| `openai/gpt-4.1-nano` | completion | $0.10/$0.40 | non-reasoning, clean output |
| `anthropic/claude-haiku-4-5` | completion | $1.00/$5.00 | best quality |

Install extras as needed:
```bash
uv tool install --editable ".[openai]"   # OpenAI embed + Gemini completion
uv tool install --editable ".[anthropic]" # Anthropic completion
```

## Live Test Results

Tested against a 50-file markdown corpus (~400 sections). Results after full
`--include-docs --doc-relations` scan:

| Edge type | Count |
|---|---|
| discusses | 1,442 |
| contains | 483 |
| blocks | 58 |
| references | 14 |
| supersedes | 2 |

The `axon_doc_search` tool correctly synthesised cross-file answers from section
embeddings, returning line-number-precise results across 6+ files per query. The
`axon_doc_context` tool surfaced correct document neighbourhood but BLOCKS edge
traversal required fallback to embedding search (see known issues below).

---

## Known Issues / Remaining Work

### 1. `axon_doc_context` — section title matching is fragile
**Problem**: Section lookup by title is an exact or near-exact string match. Slightly
different phrasing returns the wrong section node, giving incorrect edge neighbourhood.
**Fix**: Use FTS (full-text search) or embedding nearest-neighbour to identify the section
node, not string equality.

### 2. `axon_doc_context` — BLOCKS/DISCUSSES edges not surfaced explicitly
**Problem**: The tool returns section content and nearby sections but does not clearly
label which neighbours are connected via `BLOCKS` vs `DISCUSSES` vs `REFERENCES` edges.
The LLM falls back to embedding search to reconstruct blocking relationships that are
already stored as graph edges.
**Fix**: The tool output should explicitly list:
```
DISCUSSES → [section title] (file:start-end)
BLOCKS (inbound) ← [section title] (file:start-end)
BLOCKS (outbound) → [section title] (file:start-end)
```
Inbound edges are especially important: "what does the vault block?" requires
traversing INBOUND edges on the vault node, not outbound.

### 3. CLI logging level — doc-relations progress not visible by default
**Problem**: `logger.info()` calls in `doc_relations.py` are suppressed at default
log level, so the per-section progress `[1/372]` output is invisible during a scan.
**Fix**: Either elevate to `logger.warning()`, add a `--verbose` flag, or use rich
progress bar tied to the pipeline `progress_callback`.

### 4. `axon_doc_context` DISCUSSES edges — signal vs noise validation
**Problem**: 1,442 DISCUSSES edges from 402 sections is dense. Unclear what fraction
are precise (genuinely related sections with different vocabulary) vs trivially derived
from shared vocabulary that embedding would also capture.
**Fix**: Run the same `axon_doc_context` queries against an embedding-only index and
compare the neighbour sets. Any neighbours that appear in the graph but not in the
top-K embedding results are the unique contribution of the completion pass.

### 5. litellm-scale run not yet attempted
The ThreadPoolExecutor (10 workers) parallelism is implemented and correct for API
rate limits, but has not been run against the full ~5,000 section litellm corpus.
Expected duration: ~25 min at 10 workers. Rate-limit retry with `Retry-After` header
is implemented.

### 6. `uv tool install` requires explicit `[openai]` extra
`uv tool install --editable .` installs without the OpenAI package, causing both
`openai/` embedding and `gemini/` completion to fail silently (caught and logged as
warnings). The install docs should make `.[openai]` the default recommendation.

### 7. MarkdownParser — setext headings not supported
Parser handles ATX headings (`#`, `##`) but not setext style (`===`, `---` underlines).
Rare in modern docs but present in some projects.

### 8. No embedding-only → relations upgrade path
To add doc-relations to an existing embedding-only index, you currently must `--full`
re-index. A targeted `axon doc-relations` sub-command that runs only the LLM phase
against existing SECTION nodes would be faster and avoid re-embedding.
