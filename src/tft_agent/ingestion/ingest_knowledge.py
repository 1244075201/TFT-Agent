"""Offline controlled-snapshot knowledge ingestion command."""

import argparse
import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path

from tft_agent.core.config import Settings
from tft_agent.embeddings.local_e5 import LocalE5EmbeddingClient
from tft_agent.ingestion.chunker import HeadingAwareChunker
from tft_agent.tft_mcp_server.repositories.sqlite_knowledge_repository import (
    SQLiteKnowledgeRepository,
)
from tft_agent.tft_mcp_server.schemas import KnowledgeChunkRecord, KnowledgeDocument


async def ingest(path: Path, database: Path, settings: Settings) -> tuple[int, int]:
    documents = [
        KnowledgeDocument.model_validate(item)
        for item in json.loads(path.read_text(encoding="utf-8"))
    ]
    embeddings = LocalE5EmbeddingClient(settings.embedding_model, settings.embedding_device)
    repository = SQLiteKnowledgeRepository(database, embeddings, settings.retrieval_threshold)
    repository.initialize_sync()
    chunker = HeadingAwareChunker()
    total = 0
    for document in documents:
        chunks = chunker.chunk(document.content, document.title)
        vectors = await asyncio.to_thread(
            embeddings.embed_documents, [chunk.text for chunk in chunks]
        )
        now = datetime.now(UTC)
        records = [
            KnowledgeChunkRecord(
                chunk_id=f"{document.document_id}:{index}",
                document_id=document.document_id,
                title=document.title,
                heading=chunk.heading,
                text=chunk.text,
                chunk_index=index,
                source=document.source,
                source_type=document.source_type,
                document_type=document.document_type,
                game_patch=document.game_patch,
                set_name=document.set_name,
                language=document.language,
                url=document.url,
                published_at=document.published_at,
                updated_at=document.updated_at,
                retrieved_at=document.retrieved_at,
                relevance_score=None,
                embedding=vector,
                embedding_model=embeddings.model_name,
                embedding_model_revision=embeddings.model_revision,
                embedding_dimension=len(vector),
                embedded_at=now,
            )
            for index, (chunk, vector) in enumerate(zip(chunks, vectors, strict=True))
        ]
        await repository.replace_document(document, records)
        total += len(records)
    return len(documents), total


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest controlled TFT knowledge snapshots")
    parser.add_argument(
        "--input", type=Path, default=Path("data_sources/knowledge/riot_set18.json")
    )
    parser.add_argument("--database", type=Path)
    args = parser.parse_args()
    settings = Settings()
    documents, chunks = asyncio.run(
        ingest(args.input, args.database or settings.database_path, settings)
    )
    print(f"ingested_documents={documents} chunks={chunks}")
