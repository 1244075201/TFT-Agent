"""Bounded agent loop and decision routing."""

import logging
from time import perf_counter

from jsonschema import ValidationError as JSONSchemaValidationError
from jsonschema import validate

from tft_agent.agent.state import AgentState
from tft_agent.core.schemas import (
    AgentAction,
    AgentStatus,
    FinalResponse,
    ToolCall,
    ToolCallRecord,
    ToolError,
    ToolErrorResolution,
    ToolResult,
    utc_now,
)
from tft_agent.llm.client import LLMClient
from tft_agent.mcp_client.client import MCPClient, MCPToolDefinition

logger = logging.getLogger(__name__)


class AgentRuntime:
    """Route typed decisions while enforcing termination and observability."""

    def __init__(self, llm_client: LLMClient, mcp_client: MCPClient) -> None:
        self._llm_client = llm_client
        self._mcp_client = mcp_client

    async def run(self, state: AgentState) -> AgentState:
        """Run until a terminal decision, failure, or configured step limit."""

        if state.status != AgentStatus.RUNNING:
            return state

        try:
            tools = await self._mcp_client.list_tools()
            state.trace.available_tool_names = [tool.name for tool in tools]
            while state.step_count < state.max_steps:
                decision = await self._llm_client.generate(state, tools)
                state.step_count += 1

                if decision.action == AgentAction.FINAL:
                    state.final_response = self._with_runtime_metadata(
                        decision.final_response, state
                    )  # decision validation guarantees a response
                    return self._finish(state, AgentStatus.COMPLETED)

                if decision.action == AgentAction.REQUEST_CONTEXT:
                    state.missing_context = decision.missing_context
                    return self._finish(state, AgentStatus.WAITING_FOR_CONTEXT)

                # AgentDecision validation guarantees this payload exists.
                assert decision.tool_call is not None
                result = await self._execute_tool(state, decision.tool_call, tools)
                resolved = self._resolve_tool_failure(state, result)
                if resolved is not None:
                    return resolved

            return self._finish(state, AgentStatus.MAX_STEPS_REACHED)
        except Exception as exc:  # adapter failures terminate the bounded run cleanly
            logger.exception("Agent runtime failed")
            state.failure_reason = str(exc)
            return self._finish(state, AgentStatus.FAILED)

    async def _execute_tool(
        self,
        state: AgentState,
        tool_call: ToolCall,
        tools: list[MCPToolDefinition],
    ) -> ToolResult:
        started_at = utc_now()
        started_clock = perf_counter()
        definition = next((tool for tool in tools if tool.name == tool_call.name), None)
        if definition is None:
            result = ToolResult(
                tool_call_id=tool_call.id,
                success=False,
                error=ToolError(
                    code="TOOL_NOT_AVAILABLE",
                    message=f"Tool is not available: {tool_call.name}",
                    retryable=False,
                ),
            )
        else:
            try:
                validate(instance=tool_call.arguments, schema=definition.input_schema)
                result = await self._mcp_client.call_tool(tool_call)
            except JSONSchemaValidationError as exc:
                result = ToolResult(
                    tool_call_id=tool_call.id,
                    success=False,
                    error=ToolError(
                        code="INVALID_TOOL_ARGUMENTS",
                        message=exc.message,
                        retryable=False,
                    ),
                )
            except Exception as exc:
                logger.warning("Tool call %s failed: %s", tool_call.name, exc)
                result = ToolResult(
                    tool_call_id=tool_call.id,
                    success=False,
                    error=ToolError(
                        code="tool_execution_failed",
                        message=str(exc),
                        retryable=False,
                    ),
                )
        finished_at = utc_now()
        state.observations.append(result)
        state.tool_history.append(
            ToolCallRecord(
                tool_call=tool_call,
                result=result,
                started_at=started_at,
                finished_at=finished_at,
                latency_ms=(perf_counter() - started_clock) * 1000,
            )
        )
        return result

    def _resolve_tool_failure(
        self,
        state: AgentState,
        result: ToolResult,
    ) -> AgentState | None:
        """Enforce provider-neutral error control semantics after recording evidence."""

        if result.success:
            return None
        assert result.error is not None
        if result.error.resolution == ToolErrorResolution.REQUEST_CONTEXT:
            state.missing_context = self._context_requirements(result)
            return self._finish(state, AgentStatus.WAITING_FOR_CONTEXT)
        if result.error.resolution == ToolErrorResolution.TERMINAL:
            response = FinalResponse(recommendation=result.error.message)
            state.final_response = self._with_runtime_metadata(response, state)
            return self._finish(state, AgentStatus.COMPLETED)
        state.failure_reason = (
            f"Tool requested retry but automatic retry is not implemented: {result.error.message}"
        )
        return self._finish(state, AgentStatus.FAILED)

    @staticmethod
    def _context_requirements(result: ToolResult) -> list[str]:
        assert result.error is not None
        requirements = [result.error.message]
        if isinstance(result.data, dict) and isinstance(result.data.get("candidates"), list):
            for candidate in result.data["candidates"]:
                if isinstance(candidate, dict):
                    fields = ", ".join(
                        f"{key}={value}" for key, value in candidate.items() if value is not None
                    )
                    requirements.append(f"candidate: {fields}")
        return requirements

    @staticmethod
    def _with_runtime_metadata(
        response: FinalResponse | None,
        state: AgentState,
    ) -> FinalResponse:
        assert response is not None
        used_tools = list(dict.fromkeys(record.tool_call.name for record in state.tool_history))
        sources = list(
            dict.fromkeys(
                result.metadata.source
                for result in state.observations
                if result.metadata.source is not None
            )
        )
        enriched = response.model_copy(update={"used_tools": used_tools, "sources": sources})
        state.trace.final_used_tools = used_tools
        state.trace.final_sources = sources
        return enriched

    @staticmethod
    def _finish(state: AgentState, status: AgentStatus) -> AgentState:
        state.status = status
        state.trace.status = status
        state.trace.ended_at = utc_now()
        return state
