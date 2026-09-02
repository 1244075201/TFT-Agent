import sqlite3
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
    model_revision: str | None = "test"
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
        updated_at=doc.updated_at,
        strategy_comp_names=doc.strategy_comp_names,
        strategy_entities=doc.strategy_entities,
        strategy_topics=doc.strategy_topics,
        retrieved_at=doc.retrieved_at,
        embedding=vector,
        embedding_model="fake-v1",
        embedding_model_revision="test",
        embedding_dimension=2,
        embedded_at=datetime.now(UTC),
    )


def test_chunker_preserves_heading_and_metadata_context() -> None:
    chunks = HeadingAwareChunker(target_tokens=50, overlap_tokens=5).chunk(
        "# Patch 18.1\n\n## Systems\n\nThe Carousel returns.", "Riot Patch Notes"
    )

    assert chunks[0].heading == "Patch 18.1 > Systems"
    assert chunks[0].text.startswith("Riot Patch Notes\nPatch 18.1 > Systems")


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
    client._model = Model()
    client.embed_query("中文问题")
    client.embed_documents(["English passage"])

    assert seen == ["query: 中文问题", "passage: English passage"]


@pytest.mark.asyncio
async def test_strategy_metadata_filtering_and_idempotent_replace(tmp_path: Path) -> None:
    repo = SQLiteKnowledgeRepository(tmp_path / "test.sqlite3", FakeEmbeddings(), 0.7)
    repo.initialize_sync()
    strategy = document("strategy", "18.1").model_copy(
        update={
            "title": "Blossom Ahri strategy",
            "source": "Community Guide",
            "source_type": "community_strategy",
            "document_type": "strategy_guide",
            "updated_at": datetime(2026, 8, 30, tzinfo=UTC),
        }
    )
    official = document("official", "18.1")
    await repo.replace_document(strategy, [record(strategy, [1.0, 0.0])])
    await repo.replace_document(strategy, [record(strategy, [1.0, 0.0])])
    await repo.replace_document(official, [record(official, [1.0, 0.0])])

    results = await repo.search(
        "精灵策略",
        4,
        game_patch="18.1",
        set_name="Enchanted Wilds",
        document_type="strategy_guide",
        source_type="community_strategy",
    )

    assert [item.document_id for item in results] == ["strategy"]
    assert results[0].updated_at == datetime(2026, 8, 30, tzinfo=UTC)
    with sqlite3.connect(tmp_path / "test.sqlite3") as connection:
        assert connection.execute(
            "SELECT count(*) FROM knowledge_documents WHERE document_id='strategy'"
        ).fetchone() == (1,)
        assert connection.execute(
            "SELECT count(*) FROM knowledge_chunks WHERE document_id='strategy'"
        ).fetchone() == (1,)


@pytest.mark.asyncio
async def test_wrong_patch_strategy_and_hard_negative_return_no_answer(tmp_path: Path) -> None:
    repo = SQLiteKnowledgeRepository(tmp_path / "test.sqlite3", FakeEmbeddings(), 0.7)
    repo.initialize_sync()
    strategy = document("strategy", "18.1").model_copy(
        update={"source_type": "community_strategy", "document_type": "strategy_guide"}
    )
    await repo.replace_document(strategy, [record(strategy, [1.0, 0.0])])

    assert await repo.search("精灵策略", 4, game_patch="17.9") == []
    assert await repo.search("ranked rewards", 4, game_patch="18.1") == []


def test_capability_summary_is_derived_from_stored_documents(tmp_path: Path) -> None:
    repo = SQLiteKnowledgeRepository(tmp_path / "test.sqlite3", FakeEmbeddings(), 0.7)
    repo.initialize_sync()
    strategy = document("strategy", "18.1").model_copy(
        update={
            "title": "Blossom Ahri strategy",
            "source_type": "community_strategy",
            "document_type": "strategy_guide",
        }
    )
    import asyncio

    asyncio.run(repo.replace_document(strategy, [record(strategy, [1.0, 0.0])]))

    summary = repo.capability_summary()
    assert "community_strategy: strategy_guide" in summary
    assert "Blossom Ahri strategy" in summary


@pytest.mark.asyncio
async def test_strategy_entity_and_topic_filters_precede_similarity(tmp_path: Path) -> None:
    repo = SQLiteKnowledgeRepository(tmp_path / "test.sqlite3", FakeEmbeddings(), 0.7)
    repo.initialize_sync()
    strategy = document("strategy", "18.1").model_copy(
        update={
            "source_type": "community_strategy",
            "document_type": "strategy_guide",
            "strategy_comp_names": ["Blossom Ahri"],
            "strategy_entities": ["Ahri", "Sett"],
            "strategy_topics": ["roll_decision", "core_quality"],
        }
    )
    await repo.replace_document(strategy, [record(strategy, [1.0, 0.0])])

    assert (
        await repo.search(
            "Solar Kayle怎么玩",
            4,
            game_patch="18.1",
            source_type="community_strategy",
            document_type="strategy_guide",
            strategy_comp_name="Solar Kayle",
        )
        == []
    )
    assert (
        await repo.search(
            "Blossom Ahri怎么站位",
            4,
            game_patch="18.1",
            source_type="community_strategy",
            document_type="strategy_guide",
            strategy_comp_name="Blossom Ahri",
            strategy_topic="positioning",
        )
        == []
    )
    matched = await repo.search(
        "精灵 Blossom Ahri核心质量",
        4,
        game_patch="18.1",
        source_type="community_strategy",
        document_type="strategy_guide",
        strategy_comp_name="blossom ahri",
        strategy_topic="core_quality",
    )
    assert [item.document_id for item in matched] == ["strategy"]
