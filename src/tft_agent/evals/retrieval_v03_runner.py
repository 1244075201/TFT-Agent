"""Deterministic local retrieval evaluation without an LLM judge."""

import argparse
import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import Field

from tft_agent.core.config import Settings
from tft_agent.core.schemas import StrictModel
from tft_agent.embeddings.local_e5 import LocalE5EmbeddingClient
from tft_agent.tft_mcp_server.repositories.sqlite_knowledge_repository import (
    SQLiteKnowledgeRepository,
)


class RetrievalCase(StrictModel):
    id: str
    query: str
    game_patch: str | None = None
    set_name: str | None = None
    document_type: str | None = None
    source_type: str | None = None
    strategy_comp_name: str | None = None
    strategy_topic: str | None = None
    metric_group: str | None = None
    expected_document_ids: list[str]
    expected_source_types: list[str] = Field(default_factory=list)
    no_answer: bool = False


class RetrievalDataset(StrictModel):
    dataset_version: str
    cases: list[RetrievalCase]


async def run(args: argparse.Namespace) -> None:
    settings = Settings()
    dataset = RetrievalDataset.model_validate_json(args.dataset.read_text(encoding="utf-8"))
    embedding = LocalE5EmbeddingClient(settings.embedding_model, settings.embedding_device)
    repository = SQLiteKnowledgeRepository(
        settings.database_path, embedding, settings.retrieval_threshold
    )
    results: list[dict[str, Any]] = []
    reciprocal: list[float] = []
    no_answer_correct = 0
    for case in dataset.cases:
        chunks = await repository.search(
            case.query,
            5,
            game_patch=case.game_patch,
            set_name=case.set_name,
            document_type=case.document_type,
            source_type=args.source_type_override or case.source_type,
            strategy_comp_name=case.strategy_comp_name,
            strategy_topic=case.strategy_topic,
        )
        ids = [chunk.document_id for chunk in chunks]
        rank = next(
            (index + 1 for index, value in enumerate(ids) if value in case.expected_document_ids),
            None,
        )
        if not case.no_answer:
            reciprocal.append(1 / rank if rank else 0)
        if case.no_answer and not chunks:
            no_answer_correct += 1
        results.append(
            {
                "id": case.id,
                "documents": ids,
                "scores": [chunk.relevance_score for chunk in chunks],
                "source_types": [chunk.source_type for chunk in chunks],
                "rank": rank,
                "no_answer_correct": case.no_answer and not chunks,
            }
        )
    answer_cases = [r for r, c in zip(results, dataset.cases, strict=True) if not c.no_answer]
    payload = {
        "dataset_version": dataset.dataset_version,
        "timestamp": datetime.now(UTC).isoformat(),
        "model": settings.embedding_model,
        "threshold": settings.retrieval_threshold,
        "case_count": len(results),
        "hit_at_1": sum(r["rank"] == 1 for r in answer_cases) / len(answer_cases),
        "hit_at_3": sum(r["rank"] is not None and r["rank"] <= 3 for r in answer_cases)
        / len(answer_cases),
        "hit_at_5": sum(r["rank"] is not None and r["rank"] <= 5 for r in answer_cases)
        / len(answer_cases),
        "mrr": sum(reciprocal) / len(answer_cases),
        "no_answer_precision": no_answer_correct / sum(case.no_answer for case in dataset.cases),
        "false_retrievals": sum(
            case.no_answer and bool(result["documents"])
            for case, result in zip(dataset.cases, results, strict=True)
        ),
        "entity_match_recall": _group_rate(results, dataset.cases, "entity_positive", True),
        "entity_false_positive": _group_rate(results, dataset.cases, "entity_negative", True),
        "topic_false_positive": _group_rate(results, dataset.cases, "topic_negative", True),
        "hard_negative_precision": _group_rate(results, dataset.cases, "hard_negative", False),
        "knowledge_not_found_correctness": _no_answer_rate(results, dataset.cases),
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {key: value for key, value in payload.items() if key != "results"}, ensure_ascii=False
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path("evals/retrieval_v0_3_cases.json"))
    parser.add_argument("--output", type=Path, default=Path("eval_results/retrieval_v03.json"))
    parser.add_argument("--source-type-override")
    asyncio.run(run(parser.parse_args()))


def _group_rate(
    results: list[dict[str, Any]],
    cases: list[RetrievalCase],
    group: str,
    expect_result: bool,
) -> float | None:
    selected = [
        bool(result["documents"]) == expect_result
        for result, case in zip(results, cases, strict=True)
        if case.metric_group == group
    ]
    return sum(selected) / len(selected) if selected else None


def _no_answer_rate(results: list[dict[str, Any]], cases: list[RetrievalCase]) -> float | None:
    selected = [
        not bool(result["documents"])
        for result, case in zip(results, cases, strict=True)
        if case.no_answer
    ]
    return sum(selected) / len(selected) if selected else None


if __name__ == "__main__":
    main()
