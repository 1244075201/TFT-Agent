import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

import pytest

from tft_agent.agent.runtime import AgentRuntime
from tft_agent.agent.state import AgentState
from tft_agent.core.schemas import (
    AgentAction,
    AgentDecision,
    AgentStatus,
    FinalResponse,
    GameState,
    ToolCall,
)
from tft_agent.ingestion.sources.riot_data_dragon import (
    RiotChampionDocument,
    RiotDataDragonSource,
)
from tft_agent.mcp_client.client import MCPToolDefinition
from tft_agent.mcp_client.stdio_client import StdioMCPClient
from tft_agent.tft_mcp_server.repositories.sqlite_game_data_repository import (
    SQLiteGameDataRepository,
)
from tft_agent.tft_mcp_server.schemas import ShopOddsRecord

FIXTURE_DOCUMENT = {
    "type": "tft-champion",
    "version": "fixture-version",
    "data": {
        "TFT_Fixture_Unique": {
            "id": "TFT_Fixture_Unique",
            "name": "Unique Hero",
            "tier": 3,
            "cost": 0,
            "image": {
                "full": "TFT_Fixture_Unique.png",
                "sprite": "tft-champion0.png",
                "group": "champion",
                "x": 0,
                "y": 0,
                "w": 48,
                "h": 48,
            },
        },
        "TFT_ModeA_Duplicate": {
            "id": "TFT_ModeA_Duplicate",
            "name": "Duplicate Hero",
            "tier": 1,
            "cost": 0,
            "image": {
                "full": "TFT_ModeA_Duplicate.png",
                "sprite": "tft-champion0.png",
                "group": "champion",
                "x": 48,
                "y": 0,
                "w": 48,
                "h": 48,
            },
        },
        "TFT_ModeB_Duplicate": {
            "id": "TFT_ModeB_Duplicate",
            "name": "Duplicate Hero",
            "tier": 2,
            "cost": 0,
            "image": {
                "full": "TFT_ModeB_Duplicate.png",
                "sprite": "tft-champion0.png",
                "group": "champion",
                "x": 96,
                "y": 0,
                "w": 48,
                "h": 48,
            },
        },
    },
}


async def seed_database(database_path: Path) -> None:
    source = RiotDataDragonSource()
    document = RiotChampionDocument.model_validate(FIXTURE_DOCUMENT)
    timestamp = datetime.now(UTC)
    records = source.normalize_champions(document, timestamp)
    repository = SQLiteGameDataRepository(database_path)
    await repository.initialize()
    await repository.replace_champions(records, document.version, timestamp)
    await repository.replace_shop_odds(
        [
            ShopOddsRecord(
                level=7,
                cost_1_probability=0.16,
                cost_2_probability=0.30,
                cost_3_probability=0.43,
                cost_4_probability=0.10,
                cost_5_probability=0.01,
                game_patch="fixture-patch",
                source_version="fixture-shop-version",
                source="test-fixture",
                source_type="test_fixture",
                retrieved_at=timestamp,
            )
        ]
    )


def make_client(database_path: Path) -> StdioMCPClient:
    return StdioMCPClient(
        command=sys.executable,
        args=[
            "-m",
            "tft_agent.tft_mcp_server.server",
            "--database",
            str(database_path),
        ],
        cwd=Path(__file__).parents[1],
    )


@pytest.mark.asyncio
async def test_stdio_list_and_unique_not_found_and_ambiguous(tmp_path: Path) -> None:
    database_path = tmp_path / "tft.sqlite3"
    await seed_database(database_path)

    async with make_client(database_path) as client:
        tools = await client.list_tools()
        assert [tool.name for tool in tools] == [
            "get_champion_info",
            "get_shop_odds",
            "search_game_knowledge",
        ]
        assert set(tools[0].input_schema["properties"]) == {
            "name",
            "champion_id",
            "data_version",
        }
        assert set(tools[1].input_schema["properties"]) == {"level", "game_patch"}

        odds = await client.call_tool(
            ToolCall(id="odds", name="get_shop_odds", arguments={"level": 7})
        )
        assert odds.success is True
        assert isinstance(odds.data, dict)
        assert odds.data["odds"]["cost_4"] == 0.10
        assert odds.metadata.game_patch == "fixture-patch"

        unique = await client.call_tool(
            ToolCall(id="unique", name="get_champion_info", arguments={"name": "Unique Hero"})
        )
        assert unique.success is True
        assert isinstance(unique.data, dict)
        assert unique.data["champion_id"] == "TFT_Fixture_Unique"
        assert unique.metadata.data_version == "fixture-version"

        missing = await client.call_tool(
            ToolCall(id="missing", name="get_champion_info", arguments={"name": "Missing"})
        )
        assert missing.success is False
        assert missing.error is not None
        assert missing.error.code == "CHAMPION_NOT_FOUND"

        ambiguous = await client.call_tool(
            ToolCall(
                id="ambiguous",
                name="get_champion_info",
                arguments={"name": "Duplicate Hero"},
            )
        )
        assert ambiguous.success is False
        assert ambiguous.error is not None
        assert ambiguous.error.code == "AMBIGUOUS_CHAMPION"
        assert isinstance(ambiguous.data, dict)
        assert len(ambiguous.data["candidates"]) == 2

        disambiguated = await client.call_tool(
            ToolCall(
                id="disambiguated",
                name="get_champion_info",
                arguments={
                    "name": "Duplicate Hero",
                    "champion_id": "TFT_ModeA_Duplicate",
                },
            )
        )
        assert disambiguated.success is True
        assert isinstance(disambiguated.data, dict)
        assert disambiguated.data["champion_id"] == "TFT_ModeA_Duplicate"


class FakeLLM:
    def __init__(self) -> None:
        self.calls = 0

    async def generate(
        self, state: AgentState, tools: Sequence[MCPToolDefinition]
    ) -> AgentDecision:
        self.calls += 1
        if self.calls == 1:
            return AgentDecision(
                action=AgentAction.TOOL_CALL,
                tool_call=ToolCall(
                    id="runtime-call",
                    name="get_champion_info",
                    arguments={"name": "Unique Hero"},
                ),
            )
        observation_data = state.observations[-1].data
        assert isinstance(observation_data, dict)
        assert observation_data["champion_id"] == "TFT_Fixture_Unique"
        return AgentDecision(
            action=AgentAction.FINAL,
            final_response=FinalResponse(recommendation="Use the official fixture record"),
        )


@pytest.mark.asyncio
async def test_agent_runtime_with_real_stdio_mcp(tmp_path: Path) -> None:
    database_path = tmp_path / "tft.sqlite3"
    await seed_database(database_path)

    async with make_client(database_path) as client:
        state = AgentState(game_state=GameState(), goal="Look up Unique Hero")
        result = await AgentRuntime(FakeLLM(), client).run(state)

    assert result.status == AgentStatus.COMPLETED
    assert result.step_count == 2
    assert result.observations[0].success is True
