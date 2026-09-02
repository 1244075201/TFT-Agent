"""Small evidence-summary helpers for grounded provider context and traces."""

from typing import Any

from tft_agent.agent.state import AgentState
from tft_agent.core.evidence import EvidenceRegistry, build_evidence_registry


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
    return {
        "user_facts": user_facts,
        "tool_evidence": tool_evidence,
        "structured_evidence": structured_evidence,
        "official_knowledge": official_knowledge,
        "community_strategy_evidence": community_strategy_evidence,
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


def evidence_field_paths(context: dict[str, Any]) -> list[str]:
    """Return source-qualified field paths for lightweight observability."""

    paths = [f"user_facts.{field}" for field in context["user_facts"]]
    for index, observation in enumerate(context["tool_evidence"]):
        tool = observation["tool"]
        data = observation["data"]
        if isinstance(data, dict):
            paths.extend(_leaf_paths(data, f"tool_evidence[{index}].{tool}.data"))
    return paths


def _leaf_paths(value: dict[str, Any], prefix: str) -> list[str]:
    paths: list[str] = []
    for key, item in value.items():
        path = f"{prefix}.{key}"
        if isinstance(item, dict):
            paths.extend(_leaf_paths(item, path))
        else:
            paths.append(path)
    return paths


__all__ = ["EvidenceRegistry", "build_evidence_registry"]
