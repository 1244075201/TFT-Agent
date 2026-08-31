"""Opt-in deterministic evaluation of numerical decision grounding."""

import argparse
import asyncio
import json
import re
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

DEFAULT_DATASET = Path("evals/grounded_decision_v0_2_1_cases.json")


class GroundingCase(StrictModel):
    id: str
    slice: str
    prompt: str
    expected_status: str
    allow_waiting_for_context: bool = False
    required_tools: list[str]
    allowed_shop_levels: list[int]
    forbidden_patterns: list[str]
    exact_hit_patterns: list[str]
    notes: str


class GroundingDataset(StrictModel):
    dataset_version: str
    cases: list[GroundingCase]


class GroundingResult(StrictModel):
    case_id: str
    slice: str
    passed: bool
    final_status: str
    tool_sequence: list[str]
    queried_shop_levels: list[int]
    game_state_numerical_fields: dict[str, int | float]
    supported_numerical_values: list[int | float]
    final_text: str | None
    unsupported_numeric_claim_detected: bool
    chinese_numeral_unsupported_expression: bool = False
    unqueried_level_probability_claim: bool
    exact_hit_probability_hallucination: bool
    unauthorized_tool_argument: bool
    repeated_same_level_tool_call: bool
    tool_call_explosion: bool
    latency_ms: float | None
    input_tokens: int | None
    output_tokens: int | None
    total_tokens: int | None
    numeric_claims: int = Field(ge=0)
    supported_claims: int = Field(ge=0)
    unsupported_claims: int = Field(ge=0)
    mismatched_claims: int = Field(ge=0)
    invalid_reference_claims: int = Field(ge=0)
    validation_latency_ms: float = Field(ge=0)
    correction_attempted: bool
    correction_succeeded: bool
    correction_latency_ms: float | None
    correction_tokens: int | None
    fallback_used: bool
    initial_answer: str | None
    claim_validation: list[dict[str, object]]
    proposed_tool_calls: int = Field(default=0, ge=0)
    executed_tool_calls: int = Field(default=0, ge=0)
    argument_claim_count: int = Field(default=0, ge=0)
    supported_argument_claims: int = Field(default=0, ge=0)
    unsupported_argument_claims: int = Field(default=0, ge=0)
    mismatched_argument_claims: int = Field(default=0, ge=0)
    invalid_reference_argument_claims: int = Field(default=0, ge=0)
    blocked_tool_calls: int = Field(default=0, ge=0)
    request_context_due_to_argument_grounding: int = Field(default=0, ge=0)


def evaluate_grounding(case: GroundingCase, state: AgentState) -> GroundingResult:
    """Apply case-scoped deterministic checks; production output is never censored."""

    text = state.final_response.recommendation if state.final_response else ""
    sequence = [record.tool_call.name for record in state.tool_history]
    levels = [
        level
        for record in state.tool_history
        if record.tool_call.name == "get_shop_odds"
        and isinstance((level := record.tool_call.arguments.get("level")), int)
    ]
    game_numbers = {
        key: value
        for key, value in state.game_state.model_dump(exclude_none=True).items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    }
    supported = list(game_numbers.values())
    for record in state.tool_history:
        supported.extend(_numbers(record.result.data))
    unsupported = any(
        re.search(pattern, text, re.IGNORECASE) for pattern in case.forbidden_patterns
    )
    chinese_quantity = _has_chinese_quantity_expression(text)
    unqueried = _has_unqueried_level_probability(text, set(levels))
    exact_hit = any(re.search(pattern, text, re.IGNORECASE) for pattern in case.exact_hit_patterns)
    unauthorized = any(level not in case.allowed_shop_levels for level in levels)
    repeated = len(levels) != len(set(levels))
    explosion = len(state.tool_history) > max(3, len(case.required_tools) + 1)
    required_present = all(tool in sequence for tool in case.required_tools)
    status_ok = state.status.value == case.expected_status or (
        case.allow_waiting_for_context and state.status.value == "waiting_for_context"
    )
    passed = (
        status_ok
        and required_present
        and not unsupported
        and not chinese_quantity
        and not unqueried
        and not exact_hit
        and not unauthorized
        and not repeated
        and not explosion
    )
    usage = [call.token_usage for call in state.trace.llm_calls if call.token_usage is not None]
    validation = state.trace.numeric_claim_validation
    argument_validation = state.trace.tool_argument_validation
    correction_usage = [
        call.token_usage
        for call in state.trace.llm_calls
        if call.outcome == "numeric_correction" and call.token_usage is not None
    ]
    return GroundingResult(
        case_id=case.id,
        slice=case.slice,
        passed=passed,
        final_status=state.status.value,
        tool_sequence=sequence,
        queried_shop_levels=levels,
        game_state_numerical_fields=game_numbers,
        supported_numerical_values=sorted(set(supported)),
        final_text=text or None,
        unsupported_numeric_claim_detected=unsupported,
        chinese_numeral_unsupported_expression=chinese_quantity,
        unqueried_level_probability_claim=unqueried,
        exact_hit_probability_hallucination=exact_hit,
        unauthorized_tool_argument=unauthorized,
        repeated_same_level_tool_call=repeated,
        tool_call_explosion=explosion,
        latency_ms=state.trace.total_latency_ms,
        input_tokens=sum(item.prompt_tokens for item in usage) if usage else None,
        output_tokens=sum(item.completion_tokens for item in usage) if usage else None,
        total_tokens=sum(item.total_tokens for item in usage) if usage else None,
        numeric_claims=validation.total_claims if validation else 0,
        supported_claims=validation.supported_claims if validation else 0,
        unsupported_claims=validation.unsupported_claims if validation else 0,
        mismatched_claims=validation.mismatched_claims if validation else 0,
        invalid_reference_claims=validation.invalid_reference_claims if validation else 0,
        validation_latency_ms=validation.validation_latency_ms if validation else 0,
        correction_attempted=validation.correction_attempted if validation else False,
        correction_succeeded=validation.correction_succeeded if validation else False,
        correction_latency_ms=validation.correction_latency_ms if validation else None,
        correction_tokens=(
            sum(item.total_tokens for item in correction_usage) if correction_usage else None
        ),
        fallback_used=validation.fallback_used if validation else False,
        initial_answer=validation.initial_answer if validation else None,
        claim_validation=(
            [claim.model_dump(mode="json") for claim in validation.claims] if validation else []
        ),
        proposed_tool_calls=argument_validation.proposed_tool_calls,
        executed_tool_calls=argument_validation.executed_tool_calls,
        argument_claim_count=argument_validation.argument_claim_count,
        supported_argument_claims=argument_validation.supported_argument_claims,
        unsupported_argument_claims=argument_validation.unsupported_argument_claims,
        mismatched_argument_claims=argument_validation.mismatched_argument_claims,
        invalid_reference_argument_claims=argument_validation.invalid_reference_argument_claims,
        blocked_tool_calls=argument_validation.blocked_tool_calls,
        request_context_due_to_argument_grounding=(
            argument_validation.request_context_due_to_argument_grounding
        ),
    )


def _has_unqueried_level_probability(text: str, queried_levels: set[int]) -> bool:
    pattern = re.compile(r"(?P<level>10|[1-9])\s*级.{0,40}?(?P<odds>\d+(?:\.\d+)?)\s*%", re.DOTALL)
    return any(int(match.group("level")) not in queried_levels for match in pattern.finditer(text))


def _has_chinese_quantity_expression(text: str) -> bool:
    """Eval-only multilingual regression detector, never a production censor."""

    patterns = [
        r"(?:一两|两三|三五|十来|几十).{0,4}(?:轮|波|金币|金)",
        r"几(?:轮|波|金币|块)",
    ]
    return any(re.search(pattern, text) for pattern in patterns)


def _numbers(value: object) -> list[int | float]:
    if isinstance(value, bool) or value is None:
        return []
    if isinstance(value, (int, float)):
        return [value]
    if isinstance(value, dict):
        return [number for item in value.values() for number in _numbers(item)]
    if isinstance(value, list):
        return [number for item in value for number in _numbers(item)]
    return []


async def _run(args: argparse.Namespace) -> None:
    settings = Settings()
    settings.require_deepseek_api_key()
    dataset = GroundingDataset.model_validate_json(args.dataset.read_text(encoding="utf-8"))
    cases = [
        case
        for case in dataset.cases
        if (args.case is None or case.id == args.case)
        and (args.slice is None or case.slice == args.slice)
    ][: args.limit]
    if not cases:
        raise SystemExit("No grounded-decision cases matched")
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
    results: list[GroundingResult] = []
    try:
        async with mcp:
            for index, case in enumerate(cases, start=1):
                game_state = GameStateParser().parse_text(case.prompt)
                state = AgentState(
                    game_state=game_state, goal=case.prompt, max_steps=settings.max_steps
                )
                result = evaluate_grounding(case, await AgentRuntime(llm, mcp).run(state))
                results.append(result)
                print(
                    f"[{index:02d}/{len(cases):02d}] {case.id} "
                    f"passed={result.passed} tools={result.tool_sequence}"
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
        "unsupported_numeric_claims": sum(
            result.unsupported_numeric_claim_detected for result in results
        ),
        "chinese_numeral_unsupported_expressions": sum(
            result.chinese_numeral_unsupported_expression for result in results
        ),
        "unqueried_level_claims": sum(
            result.unqueried_level_probability_claim for result in results
        ),
        "exact_hit_hallucinations": sum(
            result.exact_hit_probability_hallucination for result in results
        ),
        "unauthorized_tool_arguments": sum(result.unauthorized_tool_argument for result in results),
        "blocked_tool_calls": sum(result.blocked_tool_calls for result in results),
        "corrections_attempted": sum(result.correction_attempted for result in results),
        "corrections_succeeded": sum(result.correction_succeeded for result in results),
        "fallbacks_used": sum(result.fallback_used for result in results),
        "results": [result.model_dump(mode="json") for result in results],
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / f"grounded_decision_{datetime.now(UTC):%Y%m%d_%H%M%S}.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"passed={payload['passed']}/{len(results)} output={path}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run billed grounded-decision regressions")
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
