"""Lazy local multilingual E5 embedding adapter."""

from typing import Any


class LocalE5EmbeddingClient:
    """Load one SentenceTransformer per composition root and apply E5 prefixes."""

    def __init__(self, model_name: str, device: str = "cpu") -> None:
        self.model_name = model_name
        self.model_revision: str | None = None
        self.dimension = 384
        self._device = device
        self._model: Any | None = None

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._encode([f"passage: {text}" for text in texts])

    def embed_query(self, text: str) -> list[float]:
        return self._encode([f"query: {text}"])[0]

    def _encode(self, texts: list[str]) -> list[list[float]]:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(
                self.model_name,
                device=self._device,
                local_files_only=True,
            )
            dimension = self._model.get_embedding_dimension()
            if dimension is None:
                raise RuntimeError("embedding model did not report its dimension")
            self.dimension = dimension
        vectors = self._model.encode(texts, normalize_embeddings=True)
        return [vector.tolist() for vector in vectors]
