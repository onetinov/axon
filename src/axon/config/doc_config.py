"""Configuration dataclass for the doc intelligence layer."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DocConfig:
    """Configuration for indexing markdown documentation files.

    Persisted to/from meta.json alongside existing repo metadata so
    ``axon serve --watch`` reads it on startup and knows whether to
    watch and process ``.md`` files.
    """

    enabled: bool = False
    embed_model: str = "ollama/nomic-embed-text"    # prose-optimised default
    completion_model: str | None = None              # None = skip LLM relations
    doc_relations: bool = False                      # whether to run LLM phase

    def to_dict(self) -> dict:
        """Serialise to a plain dict for JSON storage."""
        return {
            "enabled": self.enabled,
            "embed_model": self.embed_model,
            "completion_model": self.completion_model,
            "doc_relations": self.doc_relations,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "DocConfig":
        """Deserialise from a plain dict (e.g. from meta.json)."""
        return cls(
            enabled=data.get("enabled", False),
            embed_model=data.get("embed_model", "ollama/nomic-embed-text"),
            completion_model=data.get("completion_model"),
            doc_relations=data.get("doc_relations", False),
        )
