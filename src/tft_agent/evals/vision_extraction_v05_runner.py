"""Controlled-image extraction benchmark with safety-first vision metrics."""

import argparse
import asyncio
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from tft_agent.core.config import Settings
from tft_agent.core.schemas import StrictModel
from tft_agent.vision.deepseek_client import DeepSeekVisionClient
from tft_agent.vision.exceptions import VisionExtractionError
from tft_agent.vision.schemas import VisionObservation

DEFAULT_DATASET = Path("evals/vision_extraction_v0_5_cases.json")


class VisionCase(StrictModel):
    id: str
    slice: str
    image: Path
    truth: dict[str, Any]


class VisionDataset(StrictModel):
    dataset_version: str
    provenance: str
    cases: list[VisionCase]


def score_observation(predicted: VisionObservation, truth: VisionObservation) -> dict[str, Any]:
    exact = {
        field: int(getattr(predicted, field) == getattr(truth, field))
        for field in ("stage", "level", "hp", "gold")
    }
    board = _unit_scores(predicted.board_units, truth.board_units)
    bench = _unit_scores(predicted.bench_units, truth.bench_units, count_sensitive=True)
    item_pred = (predicted.item_components or []) + (predicted.completed_items or [])
    item_truth = (truth.item_components or []) + (truth.completed_items or [])
    items = _set_scores(item_pred, item_truth)
    augments = _set_scores(predicted.augments or [], truth.augments or [])
    visible_fields = (
        "stage",
        "level",
        "hp",
        "gold",
        "board_units",
        "bench_units",
        "item_components",
        "completed_items",
        "augments",
    )
    hallucinations = sum(
        getattr(truth, field) is None and getattr(predicted, field) is not None
        for field in visible_fields
    )
    false_fills = sum(
        field in truth.not_visible_fields and getattr(predicted, field) is not None
        for field in visible_fields
    )
    return {
        "exact": exact,
        "board": board,
        "bench": bench,
        "items": items,
        "augments": augments,
        "hallucinations": hallucinations,
        "not_visible_false_fills": false_fills,
        "field_opportunities": len(visible_fields),
        "not_visible_opportunities": len(truth.not_visible_fields),
    }


def aggregate(scores: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(scores)
    exact = {
        field: sum(score["exact"][field] for score in scores) / count
        for field in ("stage", "level", "hp", "gold")
    }
    return {
        **{f"{field}_exact_accuracy": value for field, value in exact.items()},
        "board": _aggregate_set(scores, "board"),
        "bench": _aggregate_set(scores, "bench"),
        "items": _aggregate_set(scores, "items"),
        "augments": _aggregate_set(scores, "augments"),
        "star_level_accuracy": _ratio(scores, "board", "star_correct", "star_total"),
        "bench_count_accuracy": _ratio(scores, "bench", "count_correct", "count_total"),
        "hallucinated_field_rate": sum(s["hallucinations"] for s in scores)
        / sum(s["field_opportunities"] for s in scores),
        "not_visible_false_fill_rate": sum(s["not_visible_false_fills"] for s in scores)
        / max(1, sum(s["not_visible_opportunities"] for s in scores)),
    }


def _unit_scores(predicted: Any, truth: Any, count_sensitive: bool = False) -> dict[str, int]:
    pred_units = predicted or []
    truth_units = truth or []
    pred_names = Counter(unit.name for unit in pred_units if unit.name)
    truth_names = Counter(unit.name for unit in truth_units if unit.name)
    tp = sum((pred_names & truth_names).values())
    truth_by_name = {unit.name: unit for unit in truth_units if unit.name}
    star_pairs = [unit for unit in pred_units if unit.name in truth_by_name and unit.star_level]
    count_pairs = [unit for unit in pred_units if unit.name in truth_by_name]
    return {
        "tp": tp,
        "predicted": sum(pred_names.values()),
        "truth": sum(truth_names.values()),
        "star_correct": sum(
            unit.star_level == truth_by_name[unit.name].star_level for unit in star_pairs
        ),
        "star_total": len(star_pairs),
        "count_correct": sum(unit.count == truth_by_name[unit.name].count for unit in count_pairs)
        if count_sensitive
        else 0,
        "count_total": len(count_pairs) if count_sensitive else 0,
    }


def _set_scores(predicted: list[str], truth: list[str]) -> dict[str, int]:
    pred = Counter(item.casefold() for item in predicted)
    gold = Counter(item.casefold() for item in truth)
    return {
        "tp": sum((pred & gold).values()),
        "predicted": sum(pred.values()),
        "truth": sum(gold.values()),
    }


def _aggregate_set(scores: list[dict[str, Any]], key: str) -> dict[str, float]:
    tp = sum(score[key]["tp"] for score in scores)
    predicted = sum(score[key]["predicted"] for score in scores)
    truth = sum(score[key]["truth"] for score in scores)
    precision = tp / predicted if predicted else 1.0
    recall = tp / truth if truth else 1.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
    }


def _ratio(scores: list[dict[str, Any]], key: str, numerator: str, denominator: str) -> float:
    top = sum(score[key][numerator] for score in scores)
    bottom = sum(score[key][denominator] for score in scores)
    return top / bottom if bottom else 1.0


async def run(args: argparse.Namespace) -> None:
    settings = Settings()
    dataset = VisionDataset.model_validate_json(args.dataset.read_text(encoding="utf-8"))
    client = DeepSeekVisionClient(
        settings.require_deepseek_api_key().get_secret_value(),
        str(settings.deepseek_base_url).rstrip("/"),
        settings.vision_model,
    )
    results: list[dict[str, Any]] = []
    try:
        for case in dataset.cases:
            if args.case and case.id not in args.case:
                continue
            truth = VisionObservation.model_validate(
                {
                    **case.truth,
                    "observation_id": f"truth-{case.id}",
                    "image_source": case.image.name,
                }
            )
            try:
                extraction = await client.extract_game_state(case.image, args.detail)
            except VisionExtractionError as exc:
                results.append(
                    {
                        "id": case.id,
                        "slice": case.slice,
                        "error": {"code": exc.code, "message": str(exc)},
                    }
                )
                print(f"{case.id} valid=False error={exc.code}")
                continue
            score = score_observation(extraction.observation, truth)
            results.append(
                {
                    "id": case.id,
                    "slice": case.slice,
                    "observation": extraction.observation.model_dump(mode="json"),
                    "trace": extraction.trace.model_dump(mode="json"),
                    "score": score,
                }
            )
            print(f"{case.id} valid=True latency_ms={extraction.trace.latency_ms:.1f}")
    finally:
        await client.close()
    score_values = [item["score"] for item in results if "score" in item]
    payload = {
        "dataset_version": dataset.dataset_version,
        "provenance": dataset.provenance,
        "timestamp": datetime.now(UTC).isoformat(),
        "detail": args.detail,
        "case_count": len(results),
        "valid_cases": sum("score" in item for item in results),
        "metrics": aggregate(score_values) if score_values else None,
        "results": results,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = (
        args.output_dir
        / f"vision_extraction_v05_{args.detail}_{datetime.now(UTC):%Y%m%d_%H%M%S}.json"
    )
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"output={output}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run V0.5 controlled vision extraction eval")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, default=Path("eval_results"))
    parser.add_argument("--case", action="append")
    parser.add_argument("--detail", choices=["low", "high", "original", "auto"], default="high")
    asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    main()
