"""Small-corpus SQLite vector retrieval implementation."""

import asyncio
import json
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
        source_type: str | None = None,
        strategy_comp_name: str | None = None,
        strategy_topic: str | None = None,
    ) -> list[KnowledgeChunk]:
        vector = await asyncio.to_thread(self._embeddings.embed_query, query)
        return await asyncio.to_thread(
            self._search_sync,
            vector,
            limit,
            game_patch,
            set_name,
            document_type,
            source_type,
            strategy_comp_name,
            strategy_topic,
        )

    def capability_summary(self) -> str:
        """Describe only the source/document types and strategy titles actually stored."""

        with self._connect() as connection:
            rows = connection.execute(
                "SELECT source_type, document_type, title FROM knowledge_documents "
                "ORDER BY source_type, document_type, title"
            ).fetchall()
        if not rows:
            return "The local knowledge corpus is empty."
        coverage: dict[str, set[str]] = {}
        strategy_titles: list[str] = []
        for row in rows:
            coverage.setdefault(row["source_type"], set()).add(row["document_type"])
            if row["source_type"] == "community_strategy":
                strategy_titles.append(row["title"])
        groups = [
            f"{source_type}: {', '.join(sorted(document_types))}"
            for source_type, document_types in sorted(coverage.items())
        ]
        if strategy_titles:
            groups.append("limited strategy guides: " + "; ".join(strategy_titles))
        return "Local corpus coverage — " + " | ".join(groups)

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
                """INSERT INTO knowledge_documents (
                    document_id, title, content, source, source_type, document_type,
                    game_patch, set_name, language, published_at, retrieved_at, url, updated_at,
                    strategy_comp_names, strategy_entities, strategy_topics
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
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
                    data["updated_at"],
                    json.dumps(data["strategy_comp_names"], ensure_ascii=False),
                    json.dumps(data["strategy_entities"], ensure_ascii=False),
                    json.dumps(data["strategy_topics"], ensure_ascii=False),
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
        source_type: str | None,
        strategy_comp_name: str | None,
        strategy_topic: str | None,
    ) -> list[KnowledgeChunk]:
        clauses = ["c.embedding_model=?", "c.embedding_dimension=?"]
        params: list[object] = [self._embeddings.model_name, len(query)]
        for column, value in (
            ("d.game_patch", game_patch),
            ("d.set_name", set_name),
            ("d.document_type", document_type),
            ("d.source_type", source_type),
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
        if strategy_comp_name is not None:
            wanted_comp = strategy_comp_name.casefold()
            rows = [
                row
                for row in rows
                if wanted_comp
                in {item.casefold() for item in json.loads(row["strategy_comp_names"])}
            ]
        if strategy_topic is not None:
            wanted_topic = strategy_topic.casefold()
            rows = [
                row
                for row in rows
                if wanted_topic in {item.casefold() for item in json.loads(row["strategy_topics"])}
            ]
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
        updated_at=_dt(row["updated_at"]),
        strategy_comp_names=json.loads(row["strategy_comp_names"]),
        strategy_entities=json.loads(row["strategy_entities"]),
        strategy_topics=json.loads(row["strategy_topics"]),
        retrieved_at=datetime.fromisoformat(row["retrieved_at"]),
        embedding_model=row["embedding_model"],
        relevance_score=score,
    )


def _dt(value: str | None) -> datetime | None:
    return datetime.fromisoformat(value) if value else None
