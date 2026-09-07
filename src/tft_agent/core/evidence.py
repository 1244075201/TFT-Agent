"""Current-task scalar evidence registry shared by control boundaries."""

from typing import Any

from tft_agent.agent.state import AgentState

EvidenceValue = int | float | str
EvidenceRegistry = dict[str, EvidenceValue]
EvidenceAliasRegistry = dict[str, str]


def build_evidence_registry(state: AgentState) -> EvidenceRegistry:
    """Index scalar facts and observations from only the current task."""

    registry: EvidenceRegistry = {}
    game_state = state.game_state.model_dump(mode="json", exclude_none=True)
    _register_scalars(registry, game_state, "game_state")
    registry["task.goal"] = state.goal
    if state.current_patch is not None:
        registry["agent.game_patch"] = state.current_patch
    # Safe repository-routing vocabulary is capability evidence, not a game fact.
    # Keeping it in the registry lets the existing argument validator authorize
    # metadata filters without trusting model-invented free-form values.
    registry.update(
        {
            "agent.filters.riot_official.source_type": "riot_official",
            "agent.filters.community_strategy.source_type": "community_strategy",
            "agent.filters.patch_notes.document_type": "patch_notes",
            "agent.filters.set_overview.document_type": "set_overview",
            "agent.filters.strategy_guide.document_type": "strategy_guide",
            "agent.filters.roll_decision.strategy_topic": "roll_decision",
            "agent.filters.leveling.strategy_topic": "leveling",
            "agent.filters.economy.strategy_topic": "economy",
            "agent.filters.core_quality.strategy_topic": "core_quality",
            "agent.filters.comp_selection.strategy_topic": "comp_selection",
            "agent.filters.items.strategy_topic": "items",
            "agent.filters.positioning.strategy_topic": "positioning",
            "agent.filters.transition.strategy_topic": "transition",
            "agent.filters.ranked_rewards.strategy_topic": "ranked_rewards",
        }
    )
    for record in state.tool_history:
        prefix = f"tool.{record.tool_call.id}"
        if isinstance(record.result.data, dict):
            _register_scalars(registry, record.result.data, f"{prefix}.data")
        metadata = record.result.metadata.model_dump(mode="json", exclude_none=True)
        _register_scalars(registry, metadata, f"{prefix}.metadata")
        if record.tool_call.name == "search_game_knowledge" and isinstance(
            record.result.data, dict
        ):
            chunks = record.result.data.get("results")
            if isinstance(chunks, list):
                for chunk in chunks:
                    if not isinstance(chunk, dict) or not isinstance(chunk.get("chunk_id"), str):
                        continue
                    knowledge_prefix = f"knowledge.{record.tool_call.id}.{chunk['chunk_id']}"
                    _register_scalars(registry, chunk, knowledge_prefix)
    return registry


def build_evidence_alias_registry(state: AgentState) -> EvidenceAliasRegistry:
    """Assign short, current-task aliases to citeable evidence paths."""

    registry = build_evidence_registry(state)
    knowledge_call_ids = {
        record.tool_call.id
        for record in state.tool_history
        if record.tool_call.name == "search_game_knowledge"
    }
    refs: list[str] = [
        path
        for path in registry
        if path.startswith("game_state.")
        or (
            path.startswith("tool.")
            and ".data." in path
            and path.split(".", 2)[1] not in knowledge_call_ids
        )
    ]
    for record in state.tool_history:
        if record.tool_call.name != "search_game_knowledge" or not isinstance(
            record.result.data, dict
        ):
            continue
        chunks = record.result.data.get("results")
        if not isinstance(chunks, list):
            continue
        refs.extend(
            f"knowledge.{record.tool_call.id}.{chunk['chunk_id']}"
            for chunk in chunks
            if isinstance(chunk, dict) and isinstance(chunk.get("chunk_id"), str)
        )
    return {f"E{index}": ref for index, ref in enumerate(dict.fromkeys(refs), start=1)}


def _register_scalars(
    registry: EvidenceRegistry,
    value: Any,
    prefix: str,
) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            _register_scalars(registry, item, f"{prefix}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _register_scalars(registry, item, f"{prefix}.{index}")
    elif isinstance(value, (int, float, str)) and not isinstance(value, bool):
        registry[prefix] = value
