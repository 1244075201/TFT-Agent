from datetime import UTC, datetime
from pathlib import Path

import pytest

from tft_agent.embeddings.local_e5 import LocalE5EmbeddingClient
from tft_agent.ingestion.chunker import HeadingAwareChunker
from tft_agent.tft_mcp_server.repositories.sqlite_knowledge_repository import (
    SQLiteKnowledgeRepository,
)
from tft_agent.tft_mcp_server.schemas import KnowledgeChunkRecord, KnowledgeDocument
from tft_agent.tft_mcp_server.tools.knowledge import KnowledgeTools


class FakeEmbeddings:
    model_name = "fake-v1"
    model_revision = "test"
    dimension = 2

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0] if "Wisps" in text else [0.0, 1.0] for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return [1.0, 0.0] if "精灵" in text else [0.0, 1.0]


def document(document_id: str, patch: str) -> KnowledgeDocument:
    return KnowledgeDocument(
        document_id=document_id,
        title="Wisp reference",
        content="# Set Mechanic\n\nWisps appear in every other shop.",
        source="Riot Games",
        source_type="riot_official",
        document_type="set_overview",
        game_patch=patch,
        set_name="Enchanted Wilds",
        language="en",
        published_at=datetime(2026, 8, 25, tzinfo=UTC),
        retrieved_at=datetime(2026, 9, 1, tzinfo=UTC),
        url="https://example.test/riot",
    )


def record(doc: KnowledgeDocument, vector: list[float]) -> KnowledgeChunkRecord:
    return KnowledgeChunkRecord(
        chunk_id=f"{doc.document_id}:0",
        document_id=doc.document_id,
        title=doc.title,
        heading="Set Mechanic",
        text="Set Mechanic\nWisps appear in every other shop.",
        chunk_index=0,
        source=doc.source,
        source_type=doc.source_type,
        document_type=doc.document_type,
        game_patch=doc.game_patch,
        set_name=doc.set_name,
        language=doc.language,
        url=doc.url,
        published_at=doc.published_at,
        retrieved_at=doc.retrieved_at,
        embedding=vector,
        embedding_model="fake-v1",
        embedding_model_revision="test",
        embedding_dimension=2,
        embedded_at=datetime.now(UTC),
    )


def test_chunker_preserves_heading_and_metadata_context() -> None:
    chunks = HeadingAwareChunker(target_tokens=50, overlap_tokens=5).chunk(
        "# Patch 18.1\n\n## Systems\n\nThe Carousel returns."
    )

    assert chunks[0].heading == "Patch 18.1 > Systems"
    assert chunks[0].text.startswith("Patch 18.1 > Systems")


@pytest.mark.asyncio
async def test_repository_filters_patch_before_similarity(tmp_path: Path) -> None:
    repo = SQLiteKnowledgeRepository(tmp_path / "test.sqlite3", FakeEmbeddings(), 0.7)
    repo.initialize_sync()
    current, old = document("current", "18.1"), document("old", "17.9")
    await repo.replace_document(current, [record(current, [1.0, 0.0])])
    await repo.replace_document(old, [record(old, [1.0, 0.0])])

    results = await repo.search("精灵机制", 4, game_patch="18.1")

    assert [item.document_id for item in results] == ["current"]
    assert results[0].source_type == "riot_official"


@pytest.mark.asyncio
async def test_repository_threshold_returns_no_answer(tmp_path: Path) -> None:
    repo = SQLiteKnowledgeRepository(tmp_path / "test.sqlite3", FakeEmbeddings(), 0.7)
    repo.initialize_sync()
    doc = document("unrelated", "18.1")
    await repo.replace_document(doc, [record(doc, [1.0, 0.0])])

    assert await repo.search("unrelated query", 4, game_patch="18.1") == []


@pytest.mark.asyncio
async def test_knowledge_tool_stops_on_no_answer(tmp_path: Path) -> None:
    repo = SQLiteKnowledgeRepository(tmp_path / "test.sqlite3", FakeEmbeddings(), 0.7)
    repo.initialize_sync()
    doc = document("unrelated", "18.1")
    await repo.replace_document(doc, [record(doc, [1.0, 0.0])])

    result = await KnowledgeTools(repo).search_game_knowledge(
        "call-1",
        {"query": "unrelated query", "game_patch": "18.1"},
    )

    assert result.success is False
    assert result.error is not None
    assert result.error.code == "knowledge_not_found"
    assert result.error.resolution == "terminal"
    assert result.metadata.game_patch == "18.1"


def test_e5_adapter_applies_query_and_passage_prefixes() -> None:
    seen: list[str] = []

    class Model:
        def encode(self, texts: list[str], normalize_embeddings: bool) -> list[object]:
            seen.extend(texts)
            return [Vector() for _ in texts]

    class Vector:
        def tolist(self) -> list[float]:
            return [1.0, 0.0]

    client = LocalE5EmbeddingClient("fixture")
    client._model = Model()  # type: ignore[assignment]
    client.embed_query("中文问题")
    client.embed_documents(["English passage"])

    assert seen == ["query: 中文问题", "passage: English passage"]
