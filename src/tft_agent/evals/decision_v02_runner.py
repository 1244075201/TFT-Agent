"""Opt-in deterministic V0.2 tool-sequence and outcome evaluation."""

import argparse
import asyncio
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

from pydantic import Field

from tft_agent.agent.parser import GameStateParser
from tft_agent.agent.runtime import AgentRuntime
from tft_agent.agent.state import AgentState
from tft_agent.core.config import Settings
from tft_agent.core.schemas import StrictModel
from tft_agent.llm.deepseek_client import DeepSeekLLMClient
from tft_agent.mcp_client.stdio_client import StdioMCPClient

DEFAULT_DATASET = Path("evals/decision_v0_2_cases.json")


class DecisionCase(StrictModel):
    id: str
    slice: str
    prompt: str
    expected_tool_sequence: list[str]
    expected_status: str
    notes: str
    order_required: bool = False


class DecisionDataset(StrictModel):
    dataset_version: str
    cases: list[DecisionCase]


class DecisionResult(StrictModel):
    id: str
    slice: str
    passed: bool
    expected_tool_sequence: list[str]
    actual_tool_sequence: list[str]
    order_required: bool
    final_status: str
    tool_success: list[bool]
    tool_error_codes: list[str]
    agent_steps: int = Field(ge=0)
    llm_calls: int = Field(ge=0)
    total_latency_ms: float | None
    tool_latency_ms: float = Field(ge=0)
    input_tokens: int | None
    output_tokens: int | None
    total_tokens: int | None
    game_state: dict[str, object]
    final_text: str | None
    missing_context: list[str]


def build_decision_result(case: DecisionCase, state: AgentState) -> DecisionResult:
    """Create deterministic assertions without judging strategy prose."""

    actual = [record.tool_call.name for record in state.tool_history]
    usage = [call.token_usage for call in state.trace.llm_calls if call.token_usage is not None]
    errors = [
        record.result.error.code for record in state.tool_history if record.result.error is not None
    ]
    return DecisionResult(
        id=case.id,
        slice=case.slice,
        passed=(
            (
                actual == case.expected_tool_sequence
                if case.order_required
                else sorted(actual) == sorted(case.expected_tool_sequence)
            )
            and state.status.value == case.expected_status
            and all(record.result.success for record in state.tool_history)
        ),
        expected_tool_sequence=case.expected_tool_sequence,
        actual_tool_sequence=actual,
        order_required=case.order_required,
        final_status=state.status.value,
        tool_success=[record.result.success for record in state.tool_history],
        tool_error_codes=errors,
        agent_steps=state.step_count,
        llm_calls=len(state.trace.llm_calls),
        total_latency_ms=state.trace.total_latency_ms,
        tool_latency_ms=sum(record.latency_ms for record in state.tool_history),
        input_tokens=sum(item.prompt_tokens for item in usage) if usage else None,
        output_tokens=sum(item.completion_tokens for item in usage) if usage else None,
        total_tokens=sum(item.total_tokens for item in usage) if usage else None,
        game_state=state.trace.game_state_snapshot,
        final_text=state.final_response.recommendation if state.final_response else None,
        missing_context=state.missing_context,
    )


async def _run(args: argparse.Namespace) -> None:
    settings = Settings()
    settings.require_deepseek_api_key()
    dataset = DecisionDataset.model_validate_json(args.dataset.read_text(encoding="utf-8"))
    cases = [
        case
        for case in dataset.cases
        if (args.case is None or case.id == args.case)
        and (args.slice is None or case.slice == args.slice)
    ][: args.limit]
    if not cases:
        raise SystemExit("No V0.2 decision cases matched")
    mcp = StdioMCPClient(
        command=sys.executable,
        args=[
            "-m",
            "tft_agent.tft_mcp_server.server",
            "--database",
            str(settings.database_path.resolve()),
        ],
        cwd=Path.cwd(),
    )
    llm = DeepSeekLLMClient.from_settings(settings)
    results: list[DecisionResult] = []
    try:
        async with mcp:
            for index, case in enumerate(cases, start=1):
                game_state = GameStateParser().parse_text(case.prompt)
                state = AgentState(
                    game_state=game_state, goal=case.prompt, max_steps=settings.max_steps
                )
                result = build_decision_result(case, await AgentRuntime(llm, mcp).run(state))
                results.append(result)
                print(
                    f"[{index:02d}/{len(cases):02d}] {case.id} "
                    f"passed={result.passed} tools={result.actual_tool_sequence}"
                )
    finally:
        await llm.close()
    payload = {
        "dataset_version": dataset.dataset_version,
        "timestamp": datetime.now(UTC).isoformat(),
        "model": settings.model,
        "thinking_mode": settings.thinking_mode,
        "case_count": len(results),
        "passed": sum(result.passed for result in results),
        "results": [result.model_dump(mode="json") for result in results],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / f"decision_v02_{datetime.now(UTC):%Y%m%d_%H%M%S}.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"passed={payload['passed']}/{len(results)} output={path}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run billed V0.2 decision regressions")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, default=Path("eval_results"))
    parser.add_argument("--case")
    parser.add_argument("--slice")
    parser.add_argument("--limit", type=int, default=1_000)
    return parser


def main() -> None:
    asyncio.run(_run(_parser().parse_args()))


if __name__ == "__main__":
    main()
