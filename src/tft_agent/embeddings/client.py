"""Embedding client abstraction independent of retrieval storage."""

from typing import Protocol


class EmbeddingClient(Protocol):
    model_name: str
    model_revision: str | None
    dimension: int

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...
