"""File system walker for discovering and reading source files in a repository."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from axon.config.ignore import should_ignore
from axon.config.languages import get_language, is_supported

if TYPE_CHECKING:
    from axon.config.doc_config import DocConfig

@dataclass
class FileEntry:
    """A source file discovered during walking."""

    path: str  # relative path from repo root (e.g., "src/auth/validate.py")
    content: str  # full file content
    language: str  # "python", "typescript", "javascript"

def discover_files(
    repo_path: Path,
    gitignore_patterns: list[str] | None = None,
    extra_extensions: dict[str, str] | None = None,
) -> list[Path]:
    """Discover supported source file paths without reading their content.

    Walks *repo_path* recursively and returns paths that are not ignored and
    have a supported language extension.  Useful for incremental indexing where
    you want to check paths before reading.

    Parameters
    ----------
    repo_path:
        Root directory of the repository to walk.
    gitignore_patterns:
        Optional list of gitignore-style patterns (e.g. from
        :func:`axon.config.ignore.load_gitignore`).
    extra_extensions:
        Optional ``{".ext": "language"}`` mapping to augment supported
        extensions for this call only (does not modify the global registry).

    Returns
    -------
    list[Path]
        List of absolute :class:`Path` objects for each discovered file.
    """
    repo_path = repo_path.resolve()
    discovered: list[Path] = []

    for file_path in repo_path.rglob("*"):
        if not file_path.is_file():
            continue

        relative = file_path.relative_to(repo_path)

        if should_ignore(str(relative), gitignore_patterns):
            continue

        suffix = file_path.suffix
        if extra_extensions and suffix in extra_extensions:
            discovered.append(file_path)
            continue

        if not is_supported(file_path):
            continue

        discovered.append(file_path)

    return discovered

def read_file(
    repo_path: Path,
    file_path: Path,
    extra_extensions: dict[str, str] | None = None,
) -> FileEntry | None:
    """Read a single file and return a :class:`FileEntry`, or ``None`` on failure.

    Returns ``None`` when the file cannot be decoded as UTF-8 (binary files),
    when the file is empty, or when an OS-level error occurs.
    """
    relative = file_path.relative_to(repo_path)

    try:
        content = file_path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, ValueError, OSError):
        return None

    if not content:
        return None

    language = get_language(file_path)
    if language is None and extra_extensions:
        language = extra_extensions.get(file_path.suffix)
    if language is None:
        return None

    return FileEntry(
        path=str(relative),
        content=content,
        language=language,
    )

def walk_repo(
    repo_path: Path,
    gitignore_patterns: list[str] | None = None,
    max_workers: int = 8,
    doc_config: "DocConfig | None" = None,
) -> list[FileEntry]:
    """Walk a repository and return all supported source files with their content.

    Discovers files using the same filtering logic as :func:`discover_files`,
    then reads their content in parallel using a :class:`ThreadPoolExecutor`.

    When *doc_config* is provided and ``doc_config.enabled`` is ``True``,
    ``.md`` files are also included without modifying the global
    :data:`~axon.config.languages.SUPPORTED_EXTENSIONS` registry.

    Parameters
    ----------
    repo_path:
        Root directory of the repository to walk.
    gitignore_patterns:
        Optional list of gitignore-style patterns (e.g. from
        :func:`axon.config.ignore.load_gitignore`).
    max_workers:
        Maximum number of threads for parallel file reading.  Defaults to 8.
    doc_config:
        Optional doc configuration.  When ``enabled``, ``.md`` files are
        included in the walk.

    Returns
    -------
    list[FileEntry]
        Sorted (by path) list of :class:`FileEntry` objects for every
        discovered source file.
    """
    repo_path = repo_path.resolve()

    extra_extensions: dict[str, str] | None = None
    if doc_config is not None and doc_config.enabled:
        extra_extensions = {".md": "markdown"}

    file_paths = discover_files(repo_path, gitignore_patterns, extra_extensions=extra_extensions)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = executor.map(
            lambda fp: read_file(repo_path, fp, extra_extensions=extra_extensions),
            file_paths,
        )

    entries = [entry for entry in results if entry is not None]
    entries.sort(key=lambda e: e.path)
    return entries
