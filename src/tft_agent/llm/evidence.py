"""Small evidence-summary helpers for grounded provider context and traces."""

from typing import Any

from tft_agent.agent.state import AgentState
from tft_agent.core.evidence import (
    EvidenceRegistry,
    build_evidence_alias_registry,
    build_evidence_registry,
)


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
    structured_evidence = [
        item for item in tool_evidence if item["tool"] != "search_game_knowledge"
    ]
    official_knowledge: list[dict[str, Any]] = []
    community_strategy_evidence: list[dict[str, Any]] = []
    for item in tool_evidence:
        if item["tool"] != "search_game_knowledge" or not isinstance(item["data"], dict):
            continue
        results = item["data"].get("results")
        if not isinstance(results, list):
            continue
        for result in results:
            if not isinstance(result, dict):
                continue
            if result.get("source_type") == "riot_official":
                official_knowledge.append(result)
            elif result.get("source_type") == "community_strategy":
                community_strategy_evidence.append(result)
    aliases = build_evidence_alias_registry(state)
    return {
        "user_facts": user_facts,
        "tool_evidence": tool_evidence,
        "structured_evidence": structured_evidence,
        "official_knowledge": official_knowledge,
        "community_strategy_evidence": community_strategy_evidence,
        "available_evidence_aliases": {
            alias: _alias_description(ref, state) for alias, ref in aliases.items()
        },
        "qualitative_reasoning": {
            "allowed": True,
            "must_not_impersonate_current_strategy_evidence": True,
        },
        "allowed_inference_policy": (
            "Qualitative decision reasoning is allowed when uncertainty is explicit."
        ),
        "unsupported_precision_policy": (
            "A specific number is factual only when present above or deterministically calculated "
            "from values above. Otherwise omit it, stay qualitative, or query useful evidence."
        ),
    }


def _alias_description(ref: str, state: AgentState) -> dict[str, Any]:
    registry = build_evidence_registry(state)
    if ref in registry:
        return {"kind": ref.split(".", 1)[0], "value": registry[ref]}
    for record in state.tool_history:
        if not ref.startswith(f"knowledge.{record.tool_call.id}.") or not isinstance(
            record.result.data, dict
        ):
            continue
        chunks = record.result.data.get("results")
        if not isinstance(chunks, list):
            continue
        for chunk in chunks:
            if isinstance(chunk, dict) and ref.endswith(f".{chunk.get('chunk_id')}"):
                return {
                    "kind": "knowledge",
                    "title": chunk.get("title"),
                    "heading": chunk.get("heading"),
                    "source_type": chunk.get("source_type"),
                    "game_patch": chunk.get("game_patch"),
                }
    return {"kind": "evidence"}


def evidence_field_paths(context: dict[str, Any]) -> list[str]:
    """Return source-qualified field paths for lightweight observability."""

    paths = _leaf_paths(context["user_facts"], "user_facts")
    for index, observation in enumerate(context["tool_evidence"]):
        tool = observation["tool"]
        data = observation["data"]
        if isinstance(data, dict):
            paths.extend(_leaf_paths(data, f"tool_evidence[{index}].{tool}.data"))
    return paths


def _leaf_paths(value: Any, prefix: str) -> list[str]:
    paths: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            paths.extend(_leaf_paths(item, f"{prefix}.{key}"))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            paths.extend(_leaf_paths(item, f"{prefix}.{index}"))
    else:
        paths.append(prefix)
    return paths


__all__ = [
    "EvidenceRegistry",
    "build_evidence_alias_registry",
    "build_evidence_registry",
]
