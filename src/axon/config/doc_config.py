"""Configuration dataclass for the doc intelligence layer."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DocConfig:
    """Configuration for indexing markdown documentation files.

    Persisted to/from meta.json alongside existing repo metadata so
    ``axon serve --watch`` reads it on startup and knows whether to
    watch and process ``.md`` files.

    The embedding model is a pipeline-level concern (applies to code + docs
    uniformly) and is stored separately in meta.json as ``"embed_model"``.
    """

    enabled: bool = False

    def to_dict(self) -> dict:
        """Serialise to a plain dict for JSON storage."""
        return {"enabled": self.enabled}

    @classmethod
    def from_dict(cls, data: dict) -> "DocConfig":
        """Deserialise from a plain dict (e.g. from meta.json)."""
        return cls(enabled=data.get("enabled", False))
