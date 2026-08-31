"""Opt-in real-provider runner for the tool-selection golden set."""

import argparse
import asyncio
import hashlib
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

from tft_agent.agent.runtime import AgentRuntime
from tft_agent.agent.state import AgentState
from tft_agent.core.config import Settings
from tft_agent.core.schemas import GameState
from tft_agent.evals.metrics import classify, summarize, summarize_by_slice
from tft_agent.evals.schemas import BenchmarkResult, CaseResult, GoldenSet, ToolSelectionCase
from tft_agent.llm.deepseek_client import DeepSeekLLMClient
from tft_agent.llm.prompts import SYSTEM_PROMPT_VERSION, build_system_prompt
from tft_agent.mcp_client.stdio_client import StdioMCPClient

DEFAULT_DATASET = Path("evals/tool_selection_cases.json")
DEFAULT_RESULTS_DIR = Path("eval_results")


def main() -> None:
    """Run an explicitly requested, billed real-provider evaluation."""

    asyncio.run(_run(_parser().parse_args()))


async def _run(args: argparse.Namespace) -> None:
    settings = Settings()
    settings.require_deepseek_api_key()
    dataset = load_golden_set(args.dataset)
    selected = select_cases(dataset.cases, args.slice, args.case, args.limit)
    if not selected:
        raise SystemExit("No evaluation cases matched the requested filters")
    database = settings.database_path.resolve()
    if not database.is_file():
        raise SystemExit(f"Structured database is unavailable: {database}")
    mcp = StdioMCPClient(
        command=sys.executable,
        args=["-m", "tft_agent.tft_mcp_server.server", "--database", str(database)],
        cwd=Path.cwd(),
    )
    llm = DeepSeekLLMClient.from_settings(settings)
    try:
        async with mcp:
            tools = await mcp.list_tools()
            results: list[CaseResult] = []
            for index, case in enumerate(selected, start=1):
                result = await run_case(case, settings, llm, mcp)
                results.append(result)
                print_case(index, len(selected), result)
    finally:
        await llm.close()
    timestamp = datetime.now(UTC)
    prompt = build_system_prompt(tools)
    benchmark = BenchmarkResult(
        benchmark_name=args.benchmark_name,
        model=settings.model,
        thinking_mode=settings.thinking_mode,
        timestamp=timestamp,
        system_prompt_version=SYSTEM_PROMPT_VERSION,
        system_prompt_hash=hashlib.sha256(prompt.encode()).hexdigest(),
        tool_description_version=args.tool_description_version,
        tool_descriptions={tool.name: tool.description for tool in tools},
        dataset_version=dataset.dataset_version,
        dataset_policy=dataset.policy,
        overall_metrics=summarize(results),
        per_slice_metrics=summarize_by_slice(results),
        cases=results,
    )
    output = save_result(benchmark, args.output_dir)
    print_summary(benchmark, output)


async def run_case(
    case: ToolSelectionCase,
    settings: Settings,
    llm: DeepSeekLLMClient,
    mcp: StdioMCPClient,
) -> CaseResult:
    """Execute one independent task and extract visible trace data."""

    state = AgentState(
        game_state=GameState(user_question=case.prompt),
        goal=case.prompt,
        max_steps=settings.max_steps,
    )
    result = await AgentRuntime(llm, mcp).run(state)
    names = [record.tool_call.name for record in result.tool_history]
    actual_called = bool(names)
    usages = [call.token_usage for call in result.trace.llm_calls if call.token_usage is not None]
    expected_name = case.expected_tool_name
    return CaseResult(
        id=case.id,
        slice=case.slice,
        prompt=case.prompt,
        expected_behavior=case.expected_tool_behavior,
        expected_tool_name=expected_name,
        actual_behavior="call" if actual_called else "no_call",
        actual_tool_names=names,
        classification=classify(case, actual_called),
        tool_name_match=(expected_name in names) if expected_name is not None else None,
        final_status=result.status.value,
        tool_error_codes=[
            record.result.error.code
            for record in result.tool_history
            if record.result.error is not None
        ],
        final_text=result.final_response.recommendation if result.final_response else None,
        agent_steps=result.step_count,
        tool_calls=len(result.tool_history),
        llm_calls=len(result.trace.llm_calls),
        total_latency_ms=result.trace.total_latency_ms,
        input_tokens=sum(usage.prompt_tokens for usage in usages) if usages else None,
        output_tokens=sum(usage.completion_tokens for usage in usages) if usages else None,
        total_tokens=sum(usage.total_tokens for usage in usages) if usages else None,
    )


def load_golden_set(path: Path) -> GoldenSet:
    """Validate the dataset before making billable requests."""

    return GoldenSet.model_validate_json(path.read_text(encoding="utf-8"))


def select_cases(
    cases: Sequence[ToolSelectionCase],
    slice_name: str | None,
    case_id: str | None,
    limit: int | None,
) -> list[ToolSelectionCase]:
    """Filter deterministically in dataset order."""

    selected = [
        case
        for case in cases
        if (slice_name is None or case.slice == slice_name)
        and (case_id is None or case.id == case_id)
    ]
    return selected[:limit] if limit is not None else selected


def save_result(result: BenchmarkResult, output_dir: Path) -> Path:
    """Persist a timestamped, secret-free artifact."""

    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = result.timestamp.strftime("%Y%m%d_%H%M%S")
    path = output_dir / f"tool_selection_{result.benchmark_name}_{stamp}.json"
    path.write_text(result.model_dump_json(indent=2), encoding="utf-8")
    return path


def print_case(index: int, total: int, result: CaseResult) -> None:
    print(
        f"[{index:02d}/{total:02d}] {result.id} {result.classification} "
        f"expected={result.expected_behavior} actual={result.actual_behavior}"
    )


def print_summary(result: BenchmarkResult, output: Path) -> None:
    metrics = result.overall_metrics
    print(
        f"accuracy={metrics.accuracy:.3f} precision={_display(metrics.precision)} "
        f"recall={_display(metrics.recall)} fp={metrics.false_positives} "
        f"fn={metrics.false_negatives} output={output}"
    )


def _display(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.3f}"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run billed DeepSeek tool-selection evaluation")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--slice")
    parser.add_argument("--case")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--benchmark-name", default="baseline_a")
    parser.add_argument("--tool-description-version", default="champion_v1")
    return parser


if __name__ == "__main__":
    main()
