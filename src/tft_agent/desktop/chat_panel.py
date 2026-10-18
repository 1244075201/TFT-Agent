"""Reusable chat transcript, composer, and request controls."""

from __future__ import annotations

import hashlib
import html
from urllib.parse import urlparse

from PySide6.QtCore import Qt, QUrl, Signal
from PySide6.QtGui import QDesktopServices, QKeyEvent
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QTextBrowser,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from tft_agent.core.schemas import ChatEvent, EvidenceCitation


class ChatInput(QTextEdit):
    send_requested = Signal()

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter) and not (
            event.modifiers() & Qt.KeyboardModifier.ShiftModifier
        ):
            self.send_requested.emit()
            event.accept()
            return
        super().keyPressEvent(event)


class ChatPanel(QWidget):
    send_requested = Signal(str)
    cancel_requested = Signal()

    def __init__(self, placeholder: str) -> None:
        super().__init__()
        self.transcript = QTextBrowser()
        self.transcript.setOpenLinks(False)
        self.transcript.anchorClicked.connect(self.open_citation)
        self.evidence_details: dict[str, EvidenceCitation] = {}
        self.evidence_dialogs: list[QDialog] = []
        self.input = ChatInput()
        self.input.setPlaceholderText(placeholder)
        self.input.setMaximumHeight(100)
        self.status = QLabel("就绪")
        self.status.setTextFormat(Qt.TextFormat.PlainText)
        self.status.setFixedHeight(self.status.fontMetrics().height() + 6)
        self.status.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
        self.send_button = QPushButton("发送")
        self.cancel_button = QPushButton("取消")
        actions = QHBoxLayout()
        actions.addWidget(self.status, 1)
        actions.addWidget(self.send_button)
        actions.addWidget(self.cancel_button)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.transcript, 1)
        layout.addWidget(self.input)
        layout.addLayout(actions)
        self.send_button.clicked.connect(self._send)
        self.input.send_requested.connect(self._send)
        self.cancel_button.clicked.connect(self.cancel_requested.emit)

    def _send(self) -> None:
        text = self.input.toPlainText().strip()
        if text:
            self.send_requested.emit(text)

    def begin_request(self, text: str) -> None:
        self.transcript.append(f"<b>你：</b>{html.escape(text)}")
        self.input.clear()
        self.set_busy(True)

    def show_event(self, event: ChatEvent) -> None:
        statuses = {"answer": "回答完成", "error": "请求失败", "cancelled": "已取消"}
        summary = " ".join(event.message.split())
        short_summary = summary[:60] + ("…" if len(summary) > 60 else "")
        self.status.setText(statuses.get(event.type, short_summary))
        self.status.setToolTip(event.message if event.type != "answer" else "")
        if event.type == "error":
            message = html.escape(event.message).replace("\n", "<br>")
            self.transcript.append(f"<b>请求失败：</b>{message}")
        if event.type == "answer":
            parts = [event.message]
            if event.response:
                if event.response.reasoning:
                    parts.append("理由：\n" + "\n".join(event.response.reasoning))
                if event.response.next_steps:
                    parts.append("后续行动：\n" + "\n".join(event.response.next_steps))
            message = html.escape("\n\n".join(parts)).replace("\n", "<br>")
            self.transcript.append(f"<b>Agent：</b>{message}")
            if event.response:
                links = []
                for evidence in event.response.citations:
                    key = hashlib.sha256(evidence.model_dump_json().encode()).hexdigest()
                    self.evidence_details[key] = evidence
                    links.append(f'<a href="evidence:{key}">{html.escape(evidence.label)}'
                                 f' · {html.escape(evidence.game_patch)}</a>')
                for source in ([] if links else event.response.sources[:6]):
                    label, separator, url = source.rpartition(" — ")
                    if separator and urlparse(url).scheme in {"http", "https"}:
                        links.append(f'<a href="{html.escape(url, quote=True)}">'
                                     f'{html.escape(label)}</a>')
                if links:
                    self.transcript.append("参考资料：" + " · ".join(links))
            self.status.setToolTip(
                f"Retrieval Plan: {event.retrieval_plan}\n"
                f"Context gaps: {event.context_gaps}\nLLM calls: {event.llm_call_count}"
            )

    def open_citation(self, url: QUrl) -> None:
        if url.scheme() in {"http", "https"}:
            QDesktopServices.openUrl(url)
            return
        if url.scheme() != "evidence":
            return
        evidence = self.evidence_details.get(url.path())
        if evidence is None:
            return
        if evidence.detail_url and urlparse(evidence.detail_url).scheme in {"http", "https"}:
            QDesktopServices.openUrl(QUrl(evidence.detail_url))
            return
        dialog = QDialog(self)
        dialog.setWindowTitle(evidence.label + " · 证据详情")
        dialog.resize(680, 520)
        browser = QTextBrowser(dialog)
        browser.setOpenExternalLinks(True)
        metadata = (f"版本：{evidence.game_patch}\n资料类型：{evidence.evidence_kind}\n"
                    f"抓取时间：{evidence.retrieved_at or '未记录'}\n"
                    f"阵容 ID：{evidence.comp_id or '无'}\n"
                    f"快照：{evidence.snapshot_id or '无'}\n")
        body = html.escape(metadata + "\n" + evidence.body).replace("\n", "<br>")
        if urlparse(evidence.source_url).scheme in {"http", "https"}:
            body += (f'<p><a href="{html.escape(evidence.source_url, quote=True)}">'
                     '原始来源（网站总览，不保证定位本阵容）</a></p>')
        browser.setHtml(body)
        QVBoxLayout(dialog).addWidget(browser)
        self.evidence_dialogs.append(dialog)
        dialog.show()

    def set_busy(self, busy: bool) -> None:
        self.send_button.setEnabled(not busy)
        self.cancel_button.setEnabled(busy)

    def clear_history(self) -> None:
        self.evidence_details.clear()
        self.transcript.clear()
        self.input.clear()
        self.status.setText("就绪")
        self.status.setToolTip("")
