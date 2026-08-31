"""Common helpers for consistent repository-backed tool results."""

from typing import Any

from tft_agent.core.schemas import ToolError, ToolMetadata, ToolResult


def found_result(call_id: str, data: Any, patch: str | None) -> ToolResult:
    return ToolResult(
        tool_call_id=call_id,
        success=True,
        data=data,
        metadata=ToolMetadata(game_patch=patch),
    )


def not_found_result(call_id: str, entity: str, patch: str | None) -> ToolResult:
    return ToolResult(
        tool_call_id=call_id,
        success=False,
        metadata=ToolMetadata(game_patch=patch),
        error=ToolError(code="not_found", message=f"{entity} was not found", retryable=False),
    )
