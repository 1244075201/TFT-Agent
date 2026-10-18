import json

import pytest

from tft_agent.app.citations import attach_citations, validate_claims
from tft_agent.app.pipeline import (
    AnswerContext,
    EvidenceRecord,
    ProductMode,
    build_retrieval_plan,
    evidence_ref_id,
    understand_query,
)
from tft_agent.core.schemas import ChatEvent, FinalResponse, GameState
from tft_agent.llm.grounded_decision import normalize_decision_output


def context() -> AnswerContext:
    state = GameState()
    understanding = understand_query("强势阵容有哪些", state, ProductMode.GENERAL_GUIDE)
    plan = build_retrieval_plan(understanding, state, "18.2", 18, "test")
    plan.active_snapshot_id = "snap"
    return AnswerContext(question="强势阵容有哪些", understanding=understanding,
                         game_state=state, history=[], retrieval_plan=plan,
                         evidence=[EvidenceRecord(
                             evidence_id=name, source="DataTFT", requested_patch="18.2",
                             source_patch="18.2", patch_match="exact", entity_type="composition",
                             canonical_entity_id=name, comp_id=name, display_name=name,
                             section="core", semantic_content=body, relevance_score=1,
                             snapshot_id="snap", url="https://www.datatft.com/comps/tier/",
                         ) for name, body in [("德莱文95", "德莱文运营"),
                                              ("6护卫蛇女", "蛇女运营"), ("螳螂", "螳螂攻略")]])


def test_shared_overview_does_not_collapse_different_composition_citations() -> None:
    response = attach_citations(FinalResponse(recommendation="德莱文95、6护卫蛇女"),
                                context().evidence)
    assert [c.label for c in response.citations] == ["德莱文95", "6护卫蛇女"]
    assert len(response.sources) == 2
    assert all(c.detail_url is None for c in response.citations)


@pytest.mark.parametrize("claims", [
    [{"text": "德莱文95很强", "comp_id": "德莱文95", "evidence_ids": ["6护卫蛇女"]}],
    [{"text": "德莱文95很强", "evidence_ids": ["missing"]}],
    [{"text": "德莱文95很强", "evidence_ids": []}],
])
def test_wrong_or_absent_evidence_cannot_support_recommendation(claims) -> None:
    assert validate_claims(claims, context()) == ([], [])


def test_cross_snapshot_and_patch_refs_rejected() -> None:
    ctx = context()
    ctx.evidence[0].snapshot_id = "other"
    ctx.evidence[1].source_patch = "18.1"
    claims = [{"text": c.display_name, "evidence_ids": [c.evidence_id]} for c in ctx.evidence[:2]]
    assert validate_claims(claims, ctx) == ([], [])


def test_real_output_contract_renders_only_supported_claims() -> None:
    output = normalize_decision_output(json.dumps({
        "recommendation": "不应出现的无证据总述",
        "claims": [{"text": "螳螂攻略", "comp_id": "螳螂", "evidence_ids": ["螳螂"]}],
    }), context())
    assert output.response.recommendation == "螳螂攻略"
    assert output.response.citations[0].label == "螳螂"


def test_opaque_provider_reference_maps_back_to_original_evidence() -> None:
    ctx = context()
    ref = evidence_ref_id(ctx.evidence[2])
    output = normalize_decision_output(json.dumps({
        "recommendation": "螳螂", "claims": [
            {"text": "螳螂攻略", "comp_id": "螳螂", "evidence_ids": [ref]},
        ],
    }), ctx)
    assert output.response.citations[0].evidence_id == "螳螂"


def test_click_opens_exact_saved_body_not_overview(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtCore import QUrl
    from PySide6.QtWidgets import QApplication, QTextBrowser

    from tft_agent.desktop.chat_panel import ChatPanel

    app = QApplication.instance() or QApplication([])
    panel = ChatPanel("test")
    response = attach_citations(FinalResponse(recommendation="螳螂"), context().evidence)
    panel.show_event(ChatEvent(type="answer", message="螳螂", response=response))
    key = next(iter(panel.evidence_details))
    panel.transcript.anchorClicked.emit(QUrl("evidence:" + key))
    app.processEvents()
    text = panel.evidence_dialogs[0].findChild(QTextBrowser).toPlainText()
    assert "螳螂攻略" in text and "德莱文运营" not in text
    for dialog in panel.evidence_dialogs:
        dialog.close()
    panel.close()


def test_long_answer_stays_inside_scrollable_transcript(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    from tft_agent.desktop.chat_panel import ChatPanel

    app = QApplication.instance() or QApplication([])
    panel = ChatPanel("test")
    panel.resize(700, 400)
    panel.show()
    app.processEvents()
    height = panel.transcript.height()
    answer = "完整阵容与运营建议。" * 40 + "\n"
    answer *= 30
    panel.show_event(ChatEvent(type="answer", message=answer))
    app.processEvents()
    assert panel.status.text() == "回答完成"
    assert answer.strip() in panel.transcript.toPlainText()
    assert panel.transcript.height() == height
    assert panel.transcript.verticalScrollBar().maximum() > 0
    assert panel.send_button.geometry().bottom() < panel.height()
    panel.show_event(ChatEvent(type="error", message=answer))
    app.processEvents()
    assert panel.status.text() == "请求失败"
    assert panel.transcript.height() == height
    panel.close()
