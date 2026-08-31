"""Official MCP SDK v2 stdio adapter behind the project client protocol."""

from collections.abc import Sequence
from pathlib import Path
from types import TracebackType

from mcp import Client, StdioServerParameters
from mcp.client.stdio import stdio_client

from tft_agent.core.schemas import ToolCall, ToolError, ToolResult
from tft_agent.mcp_client.client import MCPToolDefinition


class StdioMCPClient:
    """Own one MCP subprocess connection and close it deterministically."""

    def __init__(
        self,
        command: str,
        args: Sequence[str],
        cwd: Path | None = None,
        env: dict[str, str] | None = None,
    ) -> None:
        parameters = StdioServerParameters(
            command=command,
            args=list(args),
            cwd=cwd,
            env=env,
        )
        self._client = Client(stdio_client(parameters))
        self._connected = False

    async def __aenter__(self) -> "StdioMCPClient":
        await self._client.__aenter__()
        self._connected = True
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool | None:
        self._connected = False
        await self._client.__aexit__(exc_type, exc, traceback)
        return None

    async def list_tools(self) -> list[MCPToolDefinition]:
        client = self._require_connection()
        result = await client.list_tools()
        return [
            MCPToolDefinition(
                name=tool.name,
                description=tool.description or tool.name,
                input_schema=tool.input_schema,
            )
            for tool in result.tools
        ]

    async def call_tool(self, tool_call: ToolCall) -> ToolResult:
        client = self._require_connection()
        result = await client.call_tool(tool_call.name, tool_call.arguments)
        if result.is_error or result.structured_content is None:
            message = "MCP tool execution failed"
            for block in result.content:
                text = getattr(block, "text", None)
                if isinstance(text, str):
                    message = text
                    break
            return ToolResult(
                tool_call_id=tool_call.id,
                success=False,
                error=ToolError(code="MCP_TOOL_ERROR", message=message, retryable=False),
            )
        payload = dict(result.structured_content)
        payload["tool_call_id"] = tool_call.id
        return ToolResult.model_validate(payload)

    def _require_connection(self) -> Client:
        if not self._connected:
            raise RuntimeError("StdioMCPClient must be used inside 'async with'")
        return self._client
