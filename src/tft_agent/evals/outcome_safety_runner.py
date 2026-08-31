"""Opt-in deterministic outcome-safety evaluation over real agent tasks."""

import argparse
import asyncio
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

from pydantic import Field

from tft_agent.agent.runtime import AgentRuntime
from tft_agent.agent.state import AgentState
from tft_agent.core.config import Settings
from tft_agent.core.schemas import GameState, StrictModel, ToolErrorResolution
from tft_agent.llm.deepseek_client import DeepSeekLLMClient
from tft_agent.mcp_client.stdio_client import StdioMCPClient

DEFAULT_DATASET = Path("evals/outcome_safety_cases.json")


class OutcomeCase(StrictModel):
    id: str
    slice: str
    prompt: str
    expected_status: str
    expected_tool_error_code: str | None
    notes: str


class OutcomeDataset(StrictModel):
    dataset_version: str
    cases: list[OutcomeCase]


class OutcomeResult(StrictModel):
    id: str
    slice: str
    passed: bool
    final_status: str
    tool_call_count: int = Field(ge=0)
    llm_call_count: int = Field(ge=0)
    tool_error_codes: list[str]
    request_context_triggered: bool
    unauthorized_followup_tool_calls: int = Field(ge=0)
    agent_steps: int = Field(ge=0)
    total_latency_ms: float | None
    input_tokens: int | None
    output_tokens: int | None
    total_tokens: int | None
    final_text: str | None
    missing_context: list[str]


def main() -> None:
    asyncio.run(_run(_parser().parse_args()))


async def _run(args: argparse.Namespace) -> None:
    settings = Settings()
    settings.require_deepseek_api_key()
    dataset = OutcomeDataset.model_validate_json(args.dataset.read_text(encoding="utf-8"))
    cases = [case for case in dataset.cases if args.case is None or case.id == args.case]
    if not cases:
        raise SystemExit("No outcome cases matched")
    database = settings.database_path.resolve()
    mcp = StdioMCPClient(
        command=sys.executable,
        args=["-m", "tft_agent.tft_mcp_server.server", "--database", str(database)],
        cwd=Path.cwd(),
    )
    llm = DeepSeekLLMClient.from_settings(settings)
    results: list[OutcomeResult] = []
    try:
        async with mcp:
            for index, case in enumerate(cases, start=1):
                state = AgentState(
                    game_state=GameState(user_question=case.prompt),
                    goal=case.prompt,
                    max_steps=settings.max_steps,
                )
                result = build_outcome_result(case, await AgentRuntime(llm, mcp).run(state))
                results.append(result)
                print(f"[{index:02d}/{len(cases):02d}] {case.id} passed={result.passed}")
    finally:
        await llm.close()
    payload = {
        "dataset_version": dataset.dataset_version,
        "timestamp": datetime.now(UTC).isoformat(),
        "model": settings.model,
        "thinking_mode": settings.thinking_mode,
        "case_count": len(results),
        "passed": sum(result.passed for result in results),
        "unauthorized_followup_tool_calls": sum(
            result.unauthorized_followup_tool_calls for result in results
        ),
        "results": [result.model_dump(mode="json") for result in results],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    path = args.output_dir / f"outcome_safety_{stamp}.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"passed={payload['passed']}/{len(results)} output={path}")


def build_outcome_result(case: OutcomeCase, state: AgentState) -> OutcomeResult:
    """Extract deterministic control-flow assertions from one completed run."""

    errors = [
        record.result.error.code for record in state.tool_history if record.result.error is not None
    ]
    stop_index = next(
        (
            index
            for index, record in enumerate(state.tool_history)
            if record.result.error is not None
            and record.result.error.resolution == ToolErrorResolution.REQUEST_CONTEXT
        ),
        None,
    )
    unauthorized = len(state.tool_history) - stop_index - 1 if stop_index is not None else 0
    usage = [call.token_usage for call in state.trace.llm_calls if call.token_usage is not None]
    expected_error_ok = (
        case.expected_tool_error_code in errors
        if case.expected_tool_error_code is not None
        else not errors
    )
    passed = state.status.value == case.expected_status and expected_error_ok and unauthorized == 0
    return OutcomeResult(
        id=case.id,
        slice=case.slice,
        passed=passed,
        final_status=state.status.value,
        tool_call_count=len(state.tool_history),
        llm_call_count=len(state.trace.llm_calls),
        tool_error_codes=errors,
        request_context_triggered=state.status.value == "waiting_for_context",
        unauthorized_followup_tool_calls=unauthorized,
        agent_steps=state.step_count,
        total_latency_ms=state.trace.total_latency_ms,
        input_tokens=sum(item.prompt_tokens for item in usage) if usage else None,
        output_tokens=sum(item.completion_tokens for item in usage) if usage else None,
        total_tokens=sum(item.total_tokens for item in usage) if usage else None,
        final_text=state.final_response.recommendation if state.final_response else None,
        missing_context=state.missing_context,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run billed outcome-safety regressions")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, default=Path("eval_results"))
    parser.add_argument("--case")
    return parser


if __name__ == "__main__":
    main()
