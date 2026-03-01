"""Ollama backend — uses urllib (stdlib, no new deps) to call a local Ollama server."""

from __future__ import annotations

import json
import os
import urllib.request
from typing import Any


def _ollama_host() -> str:
    return os.environ.get("OLLAMA_HOST", "http://localhost:11434")


def embed(texts: list[str], model_name: str) -> list[list[float]]:
    """Embed *texts* using the Ollama /api/embed endpoint (batch)."""
    host = _ollama_host()
    url = f"{host}/api/embed"
    payload = json.dumps({"model": model_name, "input": texts}).encode()
    req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = json.loads(resp.read())
    # Ollama returns {"embeddings": [[...]]}
    return data["embeddings"]


def complete(prompt: str, model_name: str) -> str:
    """Single-turn completion via Ollama /api/generate (non-streaming)."""
    host = _ollama_host()
    url = f"{host}/api/generate"
    payload = json.dumps({"model": model_name, "prompt": prompt, "stream": False}).encode()
    req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=180) as resp:
        data = json.loads(resp.read())
    return data.get("response", "")


def list_models() -> list[str]:
    """Return available model names from the local Ollama instance."""
    host = _ollama_host()
    try:
        with urllib.request.urlopen(f"{host}/api/tags", timeout=5) as resp:
            data: Any = json.loads(resp.read())
        return [m["name"] for m in data.get("models", [])]
    except Exception:
        return []
