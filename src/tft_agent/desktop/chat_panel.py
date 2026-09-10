"""Reusable chat transcript, composer, and request controls."""

from __future__ import annotations

import html

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QKeyEvent
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTextBrowser,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from tft_agent.core.schemas import ChatEvent


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
        self.input = ChatInput()
        self.input.setPlaceholderText(placeholder)
        self.input.setMaximumHeight(100)
        self.status = QLabel("就绪")
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
        self.status.setText(event.message)
        if event.type == "answer":
            message = html.escape(event.message).replace("\n", "<br>")
            self.transcript.append(f"<b>Agent：</b>{message}")
            self.status.setToolTip(
                f"Retrieval Plan: {event.retrieval_plan}\n"
                f"Context gaps: {event.context_gaps}\nLLM calls: {event.llm_call_count}"
            )

    def set_busy(self, busy: bool) -> None:
        self.send_button.setEnabled(not busy)
        self.cancel_button.setEnabled(busy)

    def clear_history(self) -> None:
        self.transcript.clear()
