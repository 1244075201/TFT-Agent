"""Bind visible citations to exact evidence, never a shared overview URL."""
from __future__ import annotations

import json
from typing import Any

from tft_agent.app.pipeline import AnswerContext, EvidenceRecord, evidence_ref_id
from tft_agent.core.schemas import EvidenceCitation, FinalResponse


def citation(item: EvidenceRecord) -> EvidenceCitation:
    return EvidenceCitation(
        evidence_id=item.evidence_id, label=item.display_name,
        game_patch=item.source_patch or item.requested_patch,
        comp_id=item.comp_id or (item.canonical_entity_id
                                 if item.entity_type == "composition" else None),
        snapshot_id=item.snapshot_id,
        evidence_kind=item.evidence_kind or (
            "mechanic_fact" if "definition" in item.section else "author_advice"
        ),
        source_url=item.url,
        # Existing DataTFT records only have overview URLs. Do not invent deep links.
        detail_url=item.url if item.section == "web_strategy" else None,
        retrieved_at=item.retrieved_at,
        body=(item.semantic_content if isinstance(item.semantic_content, str)
              else json.dumps(item.semantic_content, ensure_ascii=False)),
    )


def validate_claims(data: Any, context: AnswerContext) -> tuple[list[str], list[EvidenceCitation]]:
    if not isinstance(data, list):
        return [], []
    evidence = {item.evidence_id: item for item in context.evidence
                if item.source_patch == context.retrieval_plan.game_patch
                and (not item.snapshot_id
                     or item.snapshot_id == context.retrieval_plan.active_snapshot_id)}
    evidence.update({evidence_ref_id(item): item for item in list(evidence.values())})
    names = {item.display_name: item.comp_id or item.canonical_entity_id
             for item in context.evidence if item.entity_type == "composition"}
    texts: list[str] = []
    citations: dict[str, EvidenceCitation] = {}
    for claim in data[:16]:
        if not isinstance(claim, dict):
            continue
        text, ids, comp = claim.get("text"), claim.get("evidence_ids"), claim.get("comp_id")
        if not isinstance(text, str) or not text.strip() or not isinstance(ids, list) or not ids:
            continue
        if any(not isinstance(i, str) or i not in evidence for i in ids):
            continue
        refs = [evidence[i] for i in ids]
        ref_comps = {r.comp_id or r.canonical_entity_id for r in refs
                     if r.entity_type == "composition" or r.comp_id}
        if comp and (comp not in ref_comps or ref_comps - {comp}):
            continue
        if any(name in text and identifier not in ref_comps for name, identifier in names.items()):
            continue
        texts.append(text.strip())
        for ref in refs:
            citations.setdefault(ref.evidence_id, citation(ref))
    return texts, list(citations.values())


def attach_citations(response: FinalResponse, evidence: list[EvidenceRecord]) -> FinalResponse:
    if response.citations:
        return response.model_copy(update={"sources": _source_labels(response.citations)})
    visible = "\n".join([response.recommendation, *response.reasoning, *response.next_steps])
    selected = [item for item in evidence if item.display_name in visible]
    unique = {item.evidence_id: citation(item) for item in selected}
    return response.model_copy(update={
        "citations": list(unique.values()),
        "sources": _source_labels(list(unique.values())),
    })


def _source_labels(citations: list[EvidenceCitation]) -> list[str]:
    return list(dict.fromkeys(
        f"{value.label} · {value.game_patch}（网站总览） — {value.source_url}"
        if not value.detail_url else f"{value.label} — {value.detail_url}"
        for value in citations
    ))
