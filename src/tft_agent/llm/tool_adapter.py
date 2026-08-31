"""Translate MCP definitions into generic function-calling schemas."""

from typing import Any

from tft_agent.mcp_client.client import MCPToolDefinition


def to_function_schema(tool: MCPToolDefinition) -> dict[str, Any]:
    """Convert one MCP tool definition to a provider-neutral function schema."""

    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.input_schema,
        },
    }
