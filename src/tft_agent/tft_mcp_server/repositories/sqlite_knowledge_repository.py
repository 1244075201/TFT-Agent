"""Small-corpus SQLite vector retrieval implementation."""

import asyncio
import sqlite3
from array import array
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path

from tft_agent.embeddings.client import EmbeddingClient
from tft_agent.tft_mcp_server.repositories.migrations import apply_migrations
from tft_agent.tft_mcp_server.schemas import KnowledgeChunk, KnowledgeChunkRecord, KnowledgeDocument


class SQLiteKnowledgeRepository:
    def __init__(self, path: Path, embeddings: EmbeddingClient, threshold: float) -> None:
        self._path, self._embeddings, self._threshold = path, embeddings, threshold

    def initialize_sync(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self._path) as connection:
            apply_migrations(connection)

    async def replace_document(
        self, document: KnowledgeDocument, chunks: Sequence[KnowledgeChunkRecord]
    ) -> None:
        await asyncio.to_thread(self._replace_sync, document, list(chunks))

    async def search(
        self,
        query: str,
        limit: int,
        game_patch: str | None = None,
        set_name: str | None = None,
        document_type: str | None = None,
    ) -> list[KnowledgeChunk]:
        vector = await asyncio.to_thread(self._embeddings.embed_query, query)
        return await asyncio.to_thread(
            self._search_sync, vector, limit, game_patch, set_name, document_type
        )

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self._path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def _replace_sync(
        self, document: KnowledgeDocument, chunks: list[KnowledgeChunkRecord]
    ) -> None:
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                "DELETE FROM knowledge_documents WHERE document_id=?", (document.document_id,)
            )
            data = document.model_dump(mode="json")
            connection.execute(
                "INSERT INTO knowledge_documents VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    data["document_id"],
                    data["title"],
                    data["content"],
                    data["source"],
                    data["source_type"],
                    data["document_type"],
                    data["game_patch"],
                    data["set_name"],
                    data["language"],
                    data["published_at"],
                    data["retrieved_at"],
                    data["url"],
                ),
            )
            connection.executemany(
                "INSERT INTO knowledge_chunks VALUES (?,?,?,?,?,?,?,?,?,?)",
                [
                    (
                        c.chunk_id,
                        c.document_id,
                        c.chunk_index,
                        c.heading,
                        c.text,
                        array("f", c.embedding).tobytes(),
                        c.embedding_model,
                        c.embedding_model_revision,
                        c.embedding_dimension,
                        c.embedded_at.isoformat(),
                    )
                    for c in chunks
                ],
            )

    def _search_sync(
        self,
        query: list[float],
        limit: int,
        game_patch: str | None,
        set_name: str | None,
        document_type: str | None,
    ) -> list[KnowledgeChunk]:
        clauses = ["c.embedding_model=?", "c.embedding_dimension=?"]
        params: list[object] = [self._embeddings.model_name, len(query)]
        for column, value in (
            ("d.game_patch", game_patch),
            ("d.set_name", set_name),
            ("d.document_type", document_type),
        ):
            if value is not None:
                clauses.append(f"{column}=?")
                params.append(value)
        sql = (
            "SELECT c.*, d.* FROM knowledge_chunks c JOIN knowledge_documents d "
            f"ON d.document_id=c.document_id WHERE {' AND '.join(clauses)}"
        )
        with self._connect() as connection:
            rows = connection.execute(sql, params).fetchall()
        ranked = sorted(
            ((_dot(query, list(array("f", row["embedding"]))), row) for row in rows),
            key=lambda item: item[0],
            reverse=True,
        )
        return [_chunk(row, score) for score, row in ranked[:limit] if score >= self._threshold]


def _dot(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right, strict=True))


def _chunk(row: sqlite3.Row, score: float) -> KnowledgeChunk:
    return KnowledgeChunk(
        chunk_id=row["chunk_id"],
        document_id=row["document_id"],
        title=row["title"],
        heading=row["heading"],
        text=row["text"],
        source=row["source"],
        source_type=row["source_type"],
        document_type=row["document_type"],
        game_patch=row["game_patch"],
        set_name=row["set_name"],
        language=row["language"],
        url=row["url"],
        published_at=_dt(row["published_at"]),
        retrieved_at=datetime.fromisoformat(row["retrieved_at"]),
        embedding_model=row["embedding_model"],
        relevance_score=score,
    )


def _dt(value: str | None) -> datetime | None:
    return datetime.fromisoformat(value) if value else None
