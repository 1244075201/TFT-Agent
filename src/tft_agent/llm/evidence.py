"""Small evidence-summary helpers for grounded provider context and traces."""

from typing import Any

from tft_agent.agent.state import AgentState

EvidenceRegistry = dict[str, int | float | str]


def build_evidence_context(state: AgentState) -> dict[str, Any]:
    """Separate explicit user facts from authoritative tool observations."""

    user_facts = state.game_state.model_dump(
        mode="json", exclude_none=True, exclude={"user_question"}
    )
    tool_evidence = [
        {
            "tool": record.tool_call.name,
            "arguments": record.tool_call.arguments,
            "success": record.result.success,
            "data": record.result.data,
            "provenance": record.result.metadata.model_dump(mode="json", exclude_none=True),
        }
        for record in state.tool_history
    ]
    return {
        "user_facts": user_facts,
        "tool_evidence": tool_evidence,
        "allowed_inference_policy": (
            "Qualitative decision reasoning is allowed when uncertainty is explicit."
        ),
        "unsupported_precision_policy": (
            "A specific number is factual only when present above or deterministically calculated "
            "from values above. Otherwise omit it, stay qualitative, or query useful evidence."
        ),
    }


def evidence_field_paths(context: dict[str, Any]) -> list[str]:
    """Return source-qualified field paths for lightweight observability."""

    paths = [f"user_facts.{field}" for field in context["user_facts"]]
    for index, observation in enumerate(context["tool_evidence"]):
        tool = observation["tool"]
        data = observation["data"]
        if isinstance(data, dict):
            paths.extend(_leaf_paths(data, f"tool_evidence[{index}].{tool}.data"))
    return paths


def build_evidence_registry(state: AgentState) -> EvidenceRegistry:
    """Index scalar facts from only the current task under stable provenance references."""

    registry: EvidenceRegistry = {}
    game_state = state.game_state.model_dump(
        mode="json", exclude_none=True, exclude={"user_question"}
    )
    _register_scalars(registry, game_state, "game_state")
    for record in state.tool_history:
        prefix = f"tool.{record.tool_call.id}"
        if isinstance(record.result.data, dict):
            _register_scalars(registry, record.result.data, f"{prefix}.data")
        metadata = record.result.metadata.model_dump(mode="json", exclude_none=True)
        _register_scalars(registry, metadata, f"{prefix}.metadata")
    return registry


def _leaf_paths(value: dict[str, Any], prefix: str) -> list[str]:
    paths: list[str] = []
    for key, item in value.items():
        path = f"{prefix}.{key}"
        if isinstance(item, dict):
            paths.extend(_leaf_paths(item, path))
        else:
            paths.append(path)
    return paths


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
