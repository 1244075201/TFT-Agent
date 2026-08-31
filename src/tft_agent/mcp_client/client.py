"""MCP client interfaces used by runtime, independent of transport library."""

from typing import Any, Protocol

from pydantic import Field

from tft_agent.core.schemas import StrictModel, ToolCall, ToolResult


class MCPToolDefinition(StrictModel):
    """Tool metadata and JSON Schema advertised by an MCP server."""

    name: str = Field(min_length=1)
    description: str = Field(min_length=1)
    input_schema: dict[str, Any]


class MCPClient(Protocol):
    """Discover and invoke remote tools through an interchangeable transport."""

    async def list_tools(self) -> list[MCPToolDefinition]: ...

    async def call_tool(self, tool_call: ToolCall) -> ToolResult: ...
