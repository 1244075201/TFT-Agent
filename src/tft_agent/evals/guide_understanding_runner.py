"""Opt-in real-provider intent evaluation with explicit human-review targets."""

from __future__ import annotations

import argparse
import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from tft_agent.app.semantic_retrieval import RetrievalPlannerClient
from tft_agent.core.config import Settings


async def run(cases: list[dict[str, Any]], settings: Settings) -> list[dict[str, Any]]:
    settings.require_deepseek_api_key()
    planner = RetrievalPlannerClient(settings)
    results = []
    try:
        for case in cases:
            try:
                outcome = await planner.plan(str(case["query"]))
                plan = outcome.plan
                checks = {
                    "validated": not outcome.used_fallback,
                    "no_inferred_possession": not plan.observed_state,
                    "followup": not case.get("followup") or plan.followup,
                    "freshness": not case.get("freshness_required") or plan.freshness_required,
                    "entities": all(
                        name in plan.mentioned_entities for name in case.get("entities", [])
                    ),
                }
                results.append(
                    {
                        **case,
                        "outcome": outcome.model_dump(mode="json"),
                        "contract_checks": checks,
                        "human_review": "pending",
                    }
                )
            except Exception as exc:
                results.append(
                    {**case, "error_type": type(exc).__name__, "human_review": "provider_failed"}
                )
    finally:
        await planner.close()
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Billable, opt-in real-model intent evaluation")
    parser.add_argument("--cases", type=Path, default=Path("evals/guide_understanding_cases.json"))
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.limit < 1:
        parser.error("--limit must be positive")
    dataset = json.loads(args.cases.read_text(encoding="utf-8"))
    settings = Settings()
    results = asyncio.run(run(dataset["cases"][: args.limit], settings))
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    output = args.output or Path("eval_results") / f"guide_understanding_{timestamp}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "dataset_version": dataset["version"],
                "provider_backed": True,
                "model": settings.model,
                "results": results,
                "note": "Contract checks are not semantic accuracy; human review is required.",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Saved {len(results)} provider attempts to {output}; human review pending.")


if __name__ == "__main__":
    main()
