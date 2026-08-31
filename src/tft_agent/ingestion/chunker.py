"""Contract for future deterministic document chunking."""

from typing import Protocol


class TextChunker(Protocol):
    """Split normalized source text into deterministic retrieval units."""

    def chunk(self, text: str) -> list[str]: ...
