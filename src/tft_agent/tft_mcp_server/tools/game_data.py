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
        patch = _optional_string(arguments, "patch")
        level = arguments.get("level")
        if not isinstance(level, int) or isinstance(level, bool):
            raise ValueError("level must be an integer")
        value = await self._repository.get_shop_odds(level, patch)
        return (
            found_result(call_id, value.model_dump(), patch)
            if value
            else not_found_result(call_id, f"shop odds for level {level}", patch)
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
