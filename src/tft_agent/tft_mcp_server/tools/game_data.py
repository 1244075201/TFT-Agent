"""Exact game-data tools; these deliberately contain no strategic recommendations."""

from typing import Any

from tft_agent.core.schemas import ToolError, ToolErrorResolution, ToolMetadata, ToolResult
from tft_agent.tft_mcp_server.repositories.game_data_repository import GameDataRepository
from tft_agent.tft_mcp_server.tools.base import found_result, not_found_result


class GameDataTools:
    """Expose narrow champion, item, trait, and shop-odds lookups."""

    def __init__(self, repository: GameDataRepository) -> None:
        self._repository = repository

    async def get_champion_info(self, call_id: str, arguments: dict[str, Any]) -> ToolResult:
        return await self.lookup_champion(
            call_id=call_id,
            name=_required_string(arguments, "name"),
            champion_id=_optional_string(arguments, "champion_id"),
            data_version=_optional_string(arguments, "data_version"),
        )

    async def lookup_champion(
        self,
        call_id: str,
        name: str,
        champion_id: str | None = None,
        data_version: str | None = None,
    ) -> ToolResult:
        """Resolve a normalized champion while preserving ambiguity."""

        candidates = await self._repository.find_champions(name, champion_id, data_version)
        if not candidates:
            return ToolResult(
                tool_call_id=call_id,
                success=False,
                metadata=ToolMetadata(data_version=data_version),
                error=ToolError(
                    code="CHAMPION_NOT_FOUND",
                    message=f"No champion named {name!r} matched the supplied filters",
                    retryable=False,
                ),
            )
        if len(candidates) > 1:
            return ToolResult(
                tool_call_id=call_id,
                success=False,
                data={
                    "candidates": [
                        {
                            "champion_id": candidate.champion_id,
                            "name": candidate.name,
                            "cost": candidate.cost,
                            "data_version": candidate.data_version,
                        }
                        for candidate in candidates
                    ]
                },
                metadata=ToolMetadata(data_version=data_version),
                error=ToolError(
                    code="AMBIGUOUS_CHAMPION",
                    message=f"Multiple champions are named {name!r}; supply champion_id",
                    retryable=False,
                    resolution=ToolErrorResolution.REQUEST_CONTEXT,
                ),
            )
        candidate = candidates[0]
        return ToolResult(
            tool_call_id=call_id,
            success=True,
            data=candidate.model_dump(mode="json"),
            metadata=ToolMetadata(
                data_version=candidate.data_version,
                source=candidate.source,
                source_type="structured_official_data",
                retrieved_at=candidate.retrieved_at,
            ),
        )

    async def get_item_info(self, call_id: str, arguments: dict[str, Any]) -> ToolResult:
        patch = _optional_string(arguments, "patch")
        name = _required_string(arguments, "name")
        value = await self._repository.get_item(name, patch)
        return (
            found_result(call_id, value.model_dump(), patch)
            if value
            else not_found_result(call_id, f"item {name}", patch)
        )

    async def get_trait_info(self, call_id: str, arguments: dict[str, Any]) -> ToolResult:
        patch = _optional_string(arguments, "patch")
        name = _required_string(arguments, "name")
        value = await self._repository.get_trait(name, patch)
        return (
            found_result(call_id, value.model_dump(), patch)
            if value
            else not_found_result(call_id, f"trait {name}", patch)
        )

    async def get_shop_odds(self, call_id: str, arguments: dict[str, Any]) -> ToolResult:
        level = arguments.get("level")
        if not isinstance(level, int) or isinstance(level, bool):
            raise ValueError("level must be an integer")
        game_patch = _optional_string(arguments, "game_patch")
        value = await self._repository.get_shop_odds(level, game_patch)
        if value is None:
            return ToolResult(
                tool_call_id=call_id,
                success=False,
                metadata=ToolMetadata(game_patch=game_patch),
                error=ToolError(
                    code="SHOP_ODDS_NOT_FOUND",
                    message=f"No synchronized shop odds found for level {level}",
                ),
            )
        return ToolResult(
            tool_call_id=call_id,
            success=True,
            data={
                "level": value.level,
                "odds": {
                    "cost_1": value.cost_1_probability,
                    "cost_2": value.cost_2_probability,
                    "cost_3": value.cost_3_probability,
                    "cost_4": value.cost_4_probability,
                    "cost_5": value.cost_5_probability,
                },
                "source_version": value.source_version,
            },
            metadata=ToolMetadata(
                game_patch=value.game_patch,
                source=value.source,
                source_type=value.source_type,
                retrieved_at=value.retrieved_at,
            ),
        )

    async def search_datatft_comps(
        self, call_id: str, query: str | None, game_patch: str, limit: int = 5
    ) -> ToolResult:
        """Return only synchronized DataTFT records for the exact requested patch."""

        if not 1 <= limit <= 8:
            raise ValueError("limit must be between 1 and 8")
        records = await self._repository.search_comps(query, game_patch, limit)
        if not records:
            return ToolResult(
                tool_call_id=call_id,
                success=False,
                metadata=ToolMetadata(game_patch=game_patch, source="datatft"),
                error=ToolError(
                    code="DATATFT_DATA_NOT_FOUND",
                    message=(
                        "No DataTFT snapshot matches the requested patch; "
                        "do not infer meta statistics."
                    ),
                ),
            )
        newest = records[0].snapshot
        return ToolResult(
            tool_call_id=call_id,
            success=True,
            data={"results": [record.model_dump(mode="json") for record in records]},
            metadata=ToolMetadata(
                game_patch=newest.game_patch,
                source="datatft",
                source_type="community_statistics",
                retrieved_at=newest.retrieved_at,
                freshness=newest.status.value,
            ),
        )


def _required_string(arguments: dict[str, Any], key: str) -> str:
    value = arguments.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} must be a non-empty string")
    return value


def _optional_string(arguments: dict[str, Any], key: str) -> str | None:
    value = arguments.get(key)
    if value is not None and not isinstance(value, str):
        raise ValueError(f"{key} must be a string when provided")
    return value
