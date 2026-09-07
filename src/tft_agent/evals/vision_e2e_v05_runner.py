"""Small screenshot-to-existing-agent V0.5 integration and consistency check."""

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
from tft_agent.core.schemas import GameState
from tft_agent.llm.deepseek_client import DeepSeekLLMClient
from tft_agent.mcp_client.stdio_client import StdioMCPClient
from tft_agent.vision.deepseek_client import DeepSeekVisionClient
from tft_agent.vision.workflow import observe_and_merge

QUESTION = "目标是4费卡。结合当前局面，我应该继续D还是升人口？"
IMAGE = Path("evals/vision_fixtures/full_board.png")
MANY_IMAGE = Path("evals/vision_fixtures/many_units.png")
MANY_QUESTION = "结合截图中可见的当前局面，给出有限的定性调整建议。"


async def _agent_run(state: AgentState, settings: Settings, llm: DeepSeekLLMClient) -> AgentState:
    async with StdioMCPClient(
        sys.executable,
        [
            "-m",
            "tft_agent.tft_mcp_server.server",
            "--database",
            str(settings.database_path.resolve()),
        ],
        Path.cwd(),
    ) as mcp:
        return await AgentRuntime(llm, mcp).run(state)


async def run() -> None:
    settings = Settings()
    api_key = settings.require_deepseek_api_key().get_secret_value()
    vision = DeepSeekVisionClient(
        api_key, str(settings.deepseek_base_url).rstrip("/"), settings.vision_model
    )
    llm = DeepSeekLLMClient.from_settings(settings)
    vision_state = AgentState(
        game_state=GameStateParser().parse_text(QUESTION),
        goal=QUESTION,
        current_patch=settings.current_game_patch,
    )
    try:
        await observe_and_merge(vision_state, IMAGE, vision, settings.vision_detail)
        vision_result = await _agent_run(vision_state, settings, llm)
        manual_state = AgentState(
            game_state=GameState.model_validate(
                {
                    **GameStateParser().parse_text(QUESTION).model_dump(mode="python"),
                    "stage": "4-2",
                    "level": 7,
                    "hp": 35,
                    "gold": 40,
                    "board": [
                        {"name": "Ahri", "star_level": 1},
                        {"name": "Sett", "star_level": 2},
                        {"name": "Garen", "star_level": 2},
                    ],
                    "bench": [
                        {"name": "Ahri"},
                        {"name": "Ahri"},
                        {"name": "Riven"},
                    ],
                    "item_components": ["Tear", "Sword"],
                    "completed_items": ["Blue Buff"],
                    "augments": ["Jeweled Lotus", "Cluttered Mind"],
                }
            ),
            goal=QUESTION,
            current_patch=settings.current_game_patch,
        )
        manual_result = await _agent_run(manual_state, settings, llm)
        many_state = AgentState(
            game_state=GameStateParser().parse_text(MANY_QUESTION),
            goal=MANY_QUESTION,
            current_patch=settings.current_game_patch,
        )
        await observe_and_merge(many_state, MANY_IMAGE, vision, settings.vision_detail)
        many_result = await _agent_run(many_state, settings, llm)
    finally:
        await vision.close()
        await llm.close()
    payload: dict[str, Any] = {
        "timestamp": datetime.now(UTC).isoformat(),
        "input": {"image": str(IMAGE), "question": QUESTION},
        "vision_observation": vision_state.vision_observations[0].model_dump(mode="json"),
        "merged_game_state": vision_state.game_state.model_dump(mode="json", exclude_none=True),
        "vision_trace": vision_state.vision_traces[0].model_dump(mode="json"),
        "vision_path": _result(vision_result),
        "manual_path": _result(manual_result),
        "consistency": {
            "level": vision_state.game_state.level == manual_state.game_state.level,
            "hp": vision_state.game_state.hp == manual_state.game_state.hp,
            "gold": vision_state.game_state.gold == manual_state.game_state.gold,
            "board_names": [unit.name for unit in vision_state.game_state.board]
            == [unit.name for unit in manual_state.game_state.board],
            "tool_sequence": [record.tool_call.name for record in vision_result.tool_history]
            == [record.tool_call.name for record in manual_result.tool_history],
            "tool_families": {record.tool_call.name for record in vision_result.tool_history}
            == {record.tool_call.name for record in manual_result.tool_history},
        },
        "tool_explosion_case": {
            "unit_count": len(many_state.game_state.board) + len(many_state.game_state.bench),
            **_result(many_result),
            "champion_info_calls": sum(
                record.tool_call.name == "get_champion_info" for record in many_result.tool_history
            ),
        },
    }
    output = Path("eval_results") / f"vision_e2e_v05_{datetime.now(UTC):%Y%m%d_%H%M%S}.json"
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"vision_tools={payload['vision_path']['tool_sequence']}")
    print(f"manual_tools={payload['manual_path']['tool_sequence']}")
    print(f"many_unit_tools={payload['tool_explosion_case']['tool_sequence']}")
    print(f"consistency={payload['consistency']} output={output}")


def _result(state: AgentState) -> dict[str, Any]:
    return {
        "status": state.status.value,
        "tool_sequence": [record.tool_call.name for record in state.tool_history],
        "strategy_evidence": [
            chunk
            for record in state.tool_history
            if record.tool_call.name == "search_game_knowledge"
            and isinstance(record.result.data, dict)
            for chunk in record.result.data.get("results", [])
        ],
        "final_answer": state.final_response.recommendation if state.final_response else None,
        "agent_latency_ms": state.trace.total_latency_ms,
        "agent_tokens": sum(
            trace.token_usage.total_tokens
            for trace in state.trace.llm_calls
            if trace.token_usage is not None
        ),
    }


def main() -> None:
    asyncio.run(run())


if __name__ == "__main__":
    main()
