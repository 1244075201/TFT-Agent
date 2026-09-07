"""Provider-backed V0.4 board-context evaluation runner."""

import argparse
import asyncio
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from tft_agent.agent.parser import GameStateParser
from tft_agent.agent.runtime import AgentRuntime
from tft_agent.agent.state import AgentState
from tft_agent.core.config import Settings
from tft_agent.core.evidence import build_evidence_registry
from tft_agent.core.schemas import StrictModel
from tft_agent.llm.deepseek_client import DeepSeekLLMClient
from tft_agent.mcp_client.stdio_client import StdioMCPClient

DEFAULT_DATASET = Path("evals/board_decision_v0_4_cases.json")


class BoardCase(StrictModel):
    id: str
    slice: str
    prompt: str
    expected_state: dict[str, Any]
    expected_tools: list[str]
    allowed_tools: list[str]
    expected_status: str


class BoardDataset(StrictModel):
    dataset_version: str
    policy: str
    cases: list[BoardCase]


async def run_case(
    case: BoardCase, settings: Settings, llm: DeepSeekLLMClient, mcp: StdioMCPClient
) -> dict[str, Any]:
    game_state = GameStateParser().parse_text(case.prompt)
    state = AgentState(
        game_state=game_state,
        goal=case.prompt,
        current_patch=settings.current_game_patch,
        max_steps=settings.max_steps,
    )
    parsed = build_evidence_registry(state)
    expected = {f"game_state.{path}": value for path, value in case.expected_state.items()}
    parser_ok = all(parsed.get(path) == value for path, value in expected.items())
    state = await AgentRuntime(llm, mcp).run(state)
    sequence = [record.tool_call.name for record in state.tool_history]
    tools_ok = all(tool in sequence for tool in case.expected_tools) and all(
        tool in case.allowed_tools for tool in sequence
    )
    return {
        "id": case.id,
        "slice": case.slice,
        "passed": parser_ok and tools_ok and state.status.value == case.expected_status,
        "parser_ok": parser_ok,
        "expected_state": case.expected_state,
        "game_state": game_state.model_dump(mode="json", exclude_none=True),
        "tool_sequence": sequence,
        "final_status": state.status.value,
        "final_text": state.final_response.recommendation if state.final_response else None,
        "llm_calls": len(state.trace.llm_calls),
        "total_latency_ms": state.trace.total_latency_ms,
    }


async def run(args: argparse.Namespace) -> None:
    settings = Settings()
    settings.require_deepseek_api_key()
    dataset = BoardDataset.model_validate_json(args.dataset.read_text(encoding="utf-8"))
    cases = [case for case in dataset.cases if args.case is None or case.id == args.case]
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
    results: list[dict[str, Any]] = []
    try:
        async with mcp:
            for case in cases:
                result = await run_case(case, settings, llm, mcp)
                results.append(result)
                print(f"{case.id} passed={result['passed']} tools={result['tool_sequence']}")
    finally:
        await llm.close()
    payload = {
        "dataset_version": dataset.dataset_version,
        "timestamp": datetime.now(UTC).isoformat(),
        "model": settings.model,
        "passed": sum(bool(item["passed"]) for item in results),
        "case_count": len(results),
        "results": results,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / f"board_decision_v04_{datetime.now(UTC):%Y%m%d_%H%M%S}.json"
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"passed={payload['passed']}/{len(results)} output={output}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run billed V0.4 board-context evaluation")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, default=Path("eval_results"))
    parser.add_argument("--case")
    asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    main()
