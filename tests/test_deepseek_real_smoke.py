"""Opt-in, billed DeepSeek scenarios; excluded from default test runs."""

import json
import os
import sqlite3
import sys
from pathlib import Path
from uuid import uuid4

import pytest

from tft_agent.agent.runtime import AgentRuntime
from tft_agent.agent.state import AgentState
from tft_agent.core.config import Settings
from tft_agent.core.exceptions import ConfigurationError
from tft_agent.core.schemas import AgentStatus, GameState
from tft_agent.llm.deepseek_client import DeepSeekLLMClient
from tft_agent.mcp_client.stdio_client import StdioMCPClient

pytestmark = pytest.mark.real_llm


def _enabled_settings() -> Settings:
    if os.getenv("RUN_LLM_INTEGRATION_TESTS") != "1":
        pytest.skip("set RUN_LLM_INTEGRATION_TESTS=1 to run billed provider tests")
    settings = Settings()
    try:
        settings.require_deepseek_api_key()
    except ConfigurationError:
        pytest.skip("DEEPSEEK_API_KEY is not configured")
    if not settings.database_path.is_file():
        pytest.skip("synchronized SQLite database is unavailable")
    return settings


async def _run(question: str, settings: Settings) -> AgentState:
    database_path = settings.database_path.resolve()
    client = StdioMCPClient(
        command=sys.executable,
        args=[
            "-m",
            "tft_agent.tft_mcp_server.server",
            "--database",
            str(database_path),
        ],
        cwd=Path(__file__).parents[1],
    )
    state = AgentState(
        game_state=GameState(user_question=question),
        goal=question,
        max_steps=settings.max_steps,
    )
    llm_client = DeepSeekLLMClient.from_settings(settings)
    try:
        async with client:
            return await AgentRuntime(llm_client, client).run(state)
    finally:
        await llm_client.close()


def _unique_champion(database_path: Path) -> tuple[str, int]:
    with sqlite3.connect(database_path) as connection:
        row = connection.execute(
            """SELECT name, MIN(cost) FROM champions
            WHERE data_version = (SELECT data_version FROM champion_syncs
                ORDER BY retrieved_at DESC LIMIT 1) AND name = TRIM(name)
            GROUP BY name COLLATE NOCASE HAVING COUNT(*) = 1 ORDER BY name LIMIT 1"""
        ).fetchone()
    assert row is not None
    return str(row[0]), int(row[1])


def _emit_trace(case: str, result: AgentState) -> None:
    """Print only visible, non-sensitive observability fields for an opt-in run."""

    print(
        json.dumps(
            {
                "case": case,
                "status": result.status.value,
                "agent_steps": result.step_count,
                "llm_call_count": len(result.trace.llm_calls),
                "llm_calls": [call.model_dump(mode="json") for call in result.trace.llm_calls],
                "tool_call_count": len(result.tool_history),
                "available_tool_names": result.trace.available_tool_names,
                "final_used_tools": result.trace.final_used_tools,
                "final_sources": result.trace.final_sources,
                "tool_calls": [
                    {
                        "name": record.tool_call.name,
                        "latency_ms": record.latency_ms,
                        "success": record.result.success,
                        "error_code": record.result.error.code if record.result.error else None,
                    }
                    for record in result.tool_history
                ],
                "total_latency_ms": result.trace.total_latency_ms,
                "final_recommendation": (
                    result.final_response.recommendation if result.final_response else None
                ),
            },
            ensure_ascii=False,
        )
    )


@pytest.mark.asyncio
async def test_case_a_factual_question_uses_tool() -> None:
    settings = _enabled_settings()
    name, cost = _unique_champion(settings.database_path)

    result = await _run(f"{name} 是几费卡？", settings)
    _emit_trace("A", result)

    assert result.status == AgentStatus.COMPLETED
    assert len(result.tool_history) >= 1
    assert result.tool_history[0].result.success is True
    assert result.final_response is not None
    assert str(cost) in result.final_response.recommendation


@pytest.mark.asyncio
async def test_case_b_meta_question_does_not_use_tool() -> None:
    settings = _enabled_settings()

    result = await _run("你现在可以帮助我做什么？", settings)
    _emit_trace("B", result)

    assert result.status == AgentStatus.COMPLETED
    assert len(result.tool_history) == 0


@pytest.mark.asyncio
async def test_case_c_missing_champion_uses_failure_without_fabricating() -> None:
    settings = _enabled_settings()
    missing_name = f"不存在棋子-{uuid4()}"

    result = await _run(f"{missing_name} 是几费卡？", settings)
    _emit_trace("C", result)

    assert result.status == AgentStatus.COMPLETED
    assert len(result.tool_history) >= 1
    tool_result = result.tool_history[0].result
    assert tool_result.success is False
    assert tool_result.error is not None
    assert tool_result.error.code == "CHAMPION_NOT_FOUND"
    assert result.final_response is not None


@pytest.mark.asyncio
async def test_hardening_a_cost_is_grounded_without_inventing_set() -> None:
    settings = _enabled_settings()

    result = await _run("Alistar 是几费卡？它属于哪个 Set？", settings)
    _emit_trace("hardening-A", result)

    assert result.status == AgentStatus.COMPLETED
    assert len(result.tool_history) >= 1
    tool_result = result.tool_history[0].result
    assert tool_result.success is True
    assert isinstance(tool_result.data, dict)
    assert result.final_response is not None
    answer = result.final_response.recommendation
    assert str(tool_result.data["cost"]) in answer
    assert not any(f"set {number}" in answer.lower() for number in range(1, 30))


@pytest.mark.asyncio
async def test_hardening_b_capabilities_follow_registry() -> None:
    settings = _enabled_settings()

    result = await _run("你现在具体有哪些可靠能力？", settings)
    _emit_trace("hardening-B", result)

    assert result.status == AgentStatus.COMPLETED
    assert result.tool_history == []
    assert result.trace.available_tool_names == ["get_champion_info"]
    assert result.final_response is not None


@pytest.mark.asyncio
async def test_hardening_c_item_advice_is_capability_honest() -> None:
    settings = _enabled_settings()

    result = await _run("Alistar 最强装备是什么？", settings)
    _emit_trace("hardening-C", result)

    assert result.status == AgentStatus.COMPLETED
    assert result.final_response is not None
    actual_tools = list(dict.fromkeys(record.tool_call.name for record in result.tool_history))
    assert actual_tools in ([], ["get_champion_info"])
    assert result.final_response.used_tools == actual_tools
    assert result.trace.final_used_tools == actual_tools
