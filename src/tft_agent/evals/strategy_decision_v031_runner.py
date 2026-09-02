"""Deterministic checks plus saved final text for the V0.3.1 strategy baseline."""

import argparse
import asyncio
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

from tft_agent.agent.parser import GameStateParser
from tft_agent.agent.runtime import AgentRuntime
from tft_agent.agent.state import AgentState
from tft_agent.core.config import Settings
from tft_agent.core.schemas import StrictModel
from tft_agent.llm.deepseek_client import DeepSeekLLMClient
from tft_agent.mcp_client.stdio_client import StdioMCPClient

DEFAULT_DATASET = Path("evals/strategy_decision_v0_3_1_cases.json")


class StrategyCase(StrictModel):
    id: str
    slice: str
    prompt: str
    expected_tools: list[str]
    allowed_tools: list[str]
    required_source_type: str | None
    required_document_id: str | None
    expected_status: str
    expect_knowledge_not_found: bool


class StrategyDataset(StrictModel):
    dataset_version: str
    manual_quality_rubric: list[str]
    cases: list[StrategyCase]


async def run_case(
    case: StrategyCase, settings: Settings, llm: DeepSeekLLMClient, mcp: StdioMCPClient
) -> dict[str, object]:
    game_state = GameStateParser().parse_text(case.prompt)
    state = AgentState(
        game_state=game_state,
        goal=case.prompt,
        current_patch=settings.current_game_patch,
        max_steps=settings.max_steps,
    )
    state = await AgentRuntime(llm, mcp).run(state)
    sequence = [record.tool_call.name for record in state.tool_history]
    chunks = [
        chunk
        for record in state.tool_history
        if record.tool_call.name == "search_game_knowledge"
        and isinstance(record.result.data, dict)
        and isinstance(record.result.data.get("results"), list)
        for chunk in record.result.data["results"]
        if isinstance(chunk, dict)
    ]
    source_types = sorted(
        {value for chunk in chunks if isinstance((value := chunk.get("source_type")), str)}
    )
    document_ids = sorted(
        {value for chunk in chunks if isinstance((value := chunk.get("document_id")), str)}
    )
    errors = [
        record.result.error.code for record in state.tool_history if record.result.error is not None
    ]
    usage = [call.token_usage for call in state.trace.llm_calls if call.token_usage is not None]
    validation = state.trace.numeric_claim_validation
    qualitative = state.trace.qualitative_recommendation_validation
    detected_unsupported = validation.unsupported_claims if validation else 0
    final_unsupported = (
        0
        if validation is not None and (validation.correction_succeeded or validation.fallback_used)
        else detected_unsupported
    )
    arguments = state.trace.tool_argument_validation
    tool_latency = sum(record.latency_ms for record in state.tool_history)
    retrieval_latency = sum(
        record.latency_ms
        for record in state.tool_history
        if record.tool_call.name == "search_game_knowledge"
    )
    tools_ok = all(tool in sequence for tool in case.expected_tools) and all(
        tool in case.allowed_tools for tool in sequence
    )
    source_ok = case.required_source_type is None or case.required_source_type in source_types
    document_ok = case.required_document_id is None or case.required_document_id in document_ids
    not_found = "knowledge_not_found" in errors
    not_found_ok = not case.expect_knowledge_not_found or not_found
    recommendation = state.final_response.strategy_recommendation if state.final_response else None
    recommendation_ok = case.required_document_id is None or recommendation is not None
    passed = (
        state.status.value == case.expected_status
        and tools_ok
        and source_ok
        and document_ok
        and not_found_ok
        and arguments.unsupported_argument_claims == 0
        and arguments.mismatched_argument_claims == 0
        and arguments.invalid_reference_argument_claims == 0
        and final_unsupported == 0
        and recommendation_ok
    )
    return {
        "id": case.id,
        "slice": case.slice,
        "passed": passed,
        "prompt": case.prompt,
        "game_state": game_state.model_dump(mode="json", exclude_none=True),
        "expected_tools": case.expected_tools,
        "allowed_tools": case.allowed_tools,
        "tool_sequence": sequence,
        "tool_arguments": [record.tool_call.arguments for record in state.tool_history],
        "final_status": state.status.value,
        "failure_reason": state.failure_reason,
        "knowledge_error_codes": errors,
        "source_types": source_types,
        "document_ids": document_ids,
        "sources": state.final_response.sources if state.final_response else [],
        "retrieved_chunks": chunks,
        "final_text": state.final_response.recommendation if state.final_response else None,
        "detected_unsupported_numeric_claims": detected_unsupported,
        "unsupported_numeric_claims": final_unsupported,
        "numeric_correction_attempted": validation.correction_attempted if validation else False,
        "numeric_correction_succeeded": (validation.correction_succeeded if validation else False),
        "numeric_fallback_used": validation.fallback_used if validation else False,
        "qualitative_status": qualitative.status.value if qualitative else None,
        "qualitative_evidence_refs": qualitative.evidence_refs if qualitative else [],
        "qualitative_recommendation_preserved": (
            qualitative.preserved_after_repair if qualitative else False
        ),
        "strategy_recommendation": (
            recommendation.model_dump(mode="json") if recommendation else None
        ),
        "facts_only_fallback": bool(
            validation and validation.fallback_used and recommendation is None
        ),
        "unauthorized_tool_arguments": (
            arguments.unsupported_argument_claims
            + arguments.mismatched_argument_claims
            + arguments.invalid_reference_argument_claims
        ),
        "agent_steps": state.step_count,
        "llm_calls": len(state.trace.llm_calls),
        "tool_latency_ms": tool_latency,
        "retrieval_tool_latency_ms": retrieval_latency,
        "total_latency_ms": state.trace.total_latency_ms,
        "input_tokens": sum(item.prompt_tokens for item in usage) if usage else None,
        "output_tokens": sum(item.completion_tokens for item in usage) if usage else None,
        "total_tokens": sum(item.total_tokens for item in usage) if usage else None,
        "manual_review_required": True,
    }


async def run(args: argparse.Namespace) -> None:
    settings = Settings()
    settings.require_deepseek_api_key()
    dataset = StrategyDataset.model_validate_json(args.dataset.read_text(encoding="utf-8"))
    cases = [
        case
        for case in dataset.cases
        if (args.case is None or case.id == args.case)
        and (args.slice is None or case.slice == args.slice)
    ][: args.limit]
    mcp = StdioMCPClient(
        sys.executable,
        [
            "-m",
            "tft_agent.tft_mcp_server.server",
            "--database",
            str(settings.database_path.resolve()),
        ],
        Path.cwd(),
    )
    llm = DeepSeekLLMClient.from_settings(settings)
    results: list[dict[str, object]] = []
    try:
        async with mcp:
            for index, case in enumerate(cases, start=1):
                result = await run_case(case, settings, llm, mcp)
                results.append(result)
                print(
                    f"[{index:02d}/{len(cases):02d}] {case.id} "
                    f"passed={result['passed']} tools={result['tool_sequence']}"
                )
    finally:
        await llm.close()
    payload = {
        "dataset_version": dataset.dataset_version,
        "timestamp": datetime.now(UTC).isoformat(),
        "model": settings.model,
        "case_count": len(results),
        "passed": sum(bool(result["passed"]) for result in results),
        "unsupported_numeric_claims": sum(
            value
            for result in results
            if isinstance((value := result["unsupported_numeric_claims"]), int)
        ),
        "unauthorized_tool_arguments": sum(
            value
            for result in results
            if isinstance((value := result["unauthorized_tool_arguments"]), int)
        ),
        "corrections_attempted": sum(
            bool(result.get("numeric_correction_attempted")) for result in results
        ),
        "corrections_succeeded": sum(
            bool(result.get("numeric_correction_succeeded")) for result in results
        ),
        "fallbacks_used": sum(bool(result.get("numeric_fallback_used")) for result in results),
        "qualitative_recommendations_preserved": sum(
            bool(result.get("qualitative_recommendation_preserved")) for result in results
        ),
        "facts_only_fallbacks": sum(bool(result.get("facts_only_fallback")) for result in results),
        "manual_quality_rubric": dataset.manual_quality_rubric,
        "results": results,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / f"strategy_decision_v031_{datetime.now(UTC):%Y%m%d_%H%M%S}.json"
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"passed={payload['passed']}/{len(results)} output={output}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run billed V0.3.1 strategy evaluation")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, default=Path("eval_results"))
    parser.add_argument("--case")
    parser.add_argument("--slice")
    parser.add_argument("--limit", type=int, default=1_000)
    asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    main()
