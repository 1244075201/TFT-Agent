"""Current-task scalar evidence registry shared by control boundaries."""

from typing import Any

from tft_agent.agent.state import AgentState

EvidenceValue = int | float | str
EvidenceRegistry = dict[str, EvidenceValue]


def build_evidence_registry(state: AgentState) -> EvidenceRegistry:
    """Index scalar facts and observations from only the current task."""

    registry: EvidenceRegistry = {}
    game_state = state.game_state.model_dump(mode="json", exclude_none=True)
    _register_scalars(registry, game_state, "game_state")
    registry["task.goal"] = state.goal
    if state.current_patch is not None:
        registry["agent.game_patch"] = state.current_patch
    for record in state.tool_history:
        prefix = f"tool.{record.tool_call.id}"
        if isinstance(record.result.data, dict):
            _register_scalars(registry, record.result.data, f"{prefix}.data")
        metadata = record.result.metadata.model_dump(mode="json", exclude_none=True)
        _register_scalars(registry, metadata, f"{prefix}.metadata")
    return registry


def _register_scalars(
    registry: EvidenceRegistry,
    value: dict[str, Any],
    prefix: str,
) -> None:
    for key, item in value.items():
        path = f"{prefix}.{key}"
        if isinstance(item, dict):
            _register_scalars(registry, item, path)
        elif isinstance(item, (int, float, str)) and not isinstance(item, bool):
            registry[path] = item
