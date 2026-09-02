"""Deterministic provenance checks for structured strategy recommendations."""

from dataclasses import dataclass

from tft_agent.agent.state import AgentState
from tft_agent.core.schemas import QualitativeClaimStatus, StrategyRecommendation


@dataclass(frozen=True)
class QualitativeValidation:
    status: QualitativeClaimStatus

    @property
    def valid(self) -> bool:
        return self.status == QualitativeClaimStatus.SUPPORTED


class QualitativeRecommendationValidator:
    """Validate references and retrieval gates without judging recommendation prose."""

    def validate(
        self, recommendation: StrategyRecommendation | None, state: AgentState
    ) -> QualitativeValidation:
        if recommendation is None:
            return QualitativeValidation(QualitativeClaimStatus.SUPPORTED)
        available: dict[str, tuple[dict[str, object], dict[str, object]]] = {}
        for record in state.tool_history:
            if record.tool_call.name != "search_game_knowledge":
                continue
            data = record.result.data
            if not isinstance(data, dict) or not isinstance(data.get("results"), list):
                continue
            for chunk in data["results"]:
                if isinstance(chunk, dict) and isinstance(chunk.get("chunk_id"), str):
                    ref = f"knowledge.{record.tool_call.id}.{chunk['chunk_id']}"
                    available[ref] = (chunk, record.tool_call.arguments)
        for ref in recommendation.evidence_refs:
            evidence = available.get(ref)
            if evidence is None:
                base_ref = next(
                    (
                        candidate
                        for candidate, (chunk, _) in available.items()
                        if ref.startswith(f"{candidate}.")
                        and ref.removeprefix(f"{candidate}.") in chunk
                    ),
                    None,
                )
                evidence = available.get(base_ref) if base_ref is not None else None
            if evidence is None:
                return QualitativeValidation(QualitativeClaimStatus.INVALID_REFERENCE)
            chunk, arguments = evidence
            if (
                chunk.get("source_type") != "community_strategy"
                or chunk.get("document_type") != "strategy_guide"
            ):
                return QualitativeValidation(QualitativeClaimStatus.WRONG_SOURCE_TYPE)
            patch = chunk.get("game_patch")
            if (
                patch is not None
                and state.current_patch is not None
                and patch != state.current_patch
            ):
                return QualitativeValidation(QualitativeClaimStatus.WRONG_VERSION)
            requested_comp = arguments.get("strategy_comp_name")
            comp_names = chunk.get("strategy_comp_names")
            if isinstance(requested_comp, str) and (
                not isinstance(comp_names, list)
                or requested_comp.casefold() not in {str(item).casefold() for item in comp_names}
            ):
                return QualitativeValidation(QualitativeClaimStatus.IRRELEVANT_EVIDENCE)
            requested_topic = arguments.get("strategy_topic")
            topics = chunk.get("strategy_topics")
            if isinstance(requested_topic, str) and (
                not isinstance(topics, list)
                or requested_topic.casefold() not in {str(item).casefold() for item in topics}
            ):
                return QualitativeValidation(QualitativeClaimStatus.IRRELEVANT_EVIDENCE)
        return QualitativeValidation(QualitativeClaimStatus.SUPPORTED)
