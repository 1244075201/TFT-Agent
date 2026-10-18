"""The two persistent desktop product workspaces."""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from tft_agent.core.schemas import ChatEvent, ChatRequest, GameState
from tft_agent.desktop.catalog import CanonicalEntityCatalog, CompositionCard
from tft_agent.desktop.chat_panel import ChatPanel
from tft_agent.desktop.requests import game_state_summary, guide_request, match_request
from tft_agent.desktop.state_editor import GameStateEditor


class CompositionBrowser(QWidget):
    question_requested = Signal(str)

    def __init__(self, cards: list[CompositionCard]) -> None:
        super().__init__()
        grid = QGridLayout(self)
        if not cards:
            grid.addWidget(QLabel("阵容数据暂缺"), 0, 0)
        for index, card in enumerate(cards):
            box = QFrame()
            box.setFrameShape(QFrame.Shape.StyledPanel)
            layout = QVBoxLayout(box)
            layout.addWidget(QLabel(f"{card.tier or '—'}  {card.display_name}"))
            cores = "、".join(card.core_units) if card.core_units else "数据暂缺"
            layout.addWidget(QLabel(f"核心：{cores}"))
            layout.addWidget(QLabel(card.summary or "攻略摘要暂缺"))
            button = QPushButton("查看阵容卡片")
            button.clicked.connect(
                lambda _checked=False, name=card.display_name:
                self.question_requested.emit(f"{name}核心棋子是谁？")
            )
            layout.addWidget(button)
            grid.addWidget(box, index // 3, index % 3)


class GuideChatPage(QWidget):
    send_requested = Signal(str)
    cancel_requested = Signal()
    new_conversation_requested = Signal()

    def __init__(self, patch: str, cards: list[CompositionCard]) -> None:
        super().__init__()
        self.title = QLabel(
            f"<h2>攻略问答</h2>查询阵容、装备、强化、转职和一般运营知识　Patch {patch}"
        )
        self.new_conversation = QPushButton("新对话")
        header = QHBoxLayout()
        header.addWidget(self.title, 1)
        header.addWidget(self.new_conversation)
        shortcuts = QHBoxLayout()
        for label, question in (
            ("当前版本阵容推荐", "当前版本有哪些强势阵容？"),
            ("S 级阵容榜", "显示当前版本S级阵容榜"),
            ("装备 / 强化百科", "当前版本常见装备和强化有哪些？"),
        ):
            button = QPushButton(label)
            button.clicked.connect(
                lambda _checked=False, value=question: self._send_question(value)
            )
            shortcuts.addWidget(button)
        self.browser = CompositionBrowser(cards)
        self.chat = ChatPanel("输入攻略问题；Enter 发送，Shift+Enter 换行")
        layout = QVBoxLayout(self)
        layout.addLayout(header)
        layout.addLayout(shortcuts)
        layout.addWidget(self.browser)
        layout.addWidget(self.chat, 1)
        self.browser.question_requested.connect(self._send_question)
        self.chat.send_requested.connect(self.send_requested.emit)
        self.chat.cancel_requested.connect(self.cancel_requested.emit)
        self.new_conversation.clicked.connect(self.new_conversation_requested.emit)

    def _send_question(self, question: str) -> None:
        self.chat.input.setPlainText(question)
        self.send_requested.emit(question)

    def update_catalog(self, patch: str, cards: list[CompositionCard]) -> None:
        self.title.setText(
            f"<h2>攻略问答</h2>查询阵容、装备、强化、转职和一般运营知识　Patch {patch}"
        )
        previous = self.browser
        self.browser = CompositionBrowser(cards)
        layout = self.layout()
        if layout is not None:
            layout.replaceWidget(previous, self.browser)
        self.browser.question_requested.connect(self._send_question)
        previous.deleteLater()

    def request(self, question: str) -> ChatRequest:
        return guide_request(question)


class MatchGuidancePage(QWidget):
    send_requested = Signal(str)
    cancel_requested = Signal()
    save_requested = Signal()
    clear_requested = Signal()
    screenshot_requested = Signal()

    def __init__(self, catalog: CanonicalEntityCatalog) -> None:
        super().__init__()
        self.summary = QLabel("当前局面：尚未记录")
        self.summary.setStyleSheet("padding:8px;background:#edf2f7")
        self.editor = GameStateEditor(catalog)
        self.chat = ChatPanel("输入当前决策问题")
        self.evidence = QLabel("当前回答依据\n· 当前局面\n· 当前版本攻略\n· Knowledge gaps")
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.addWidget(QLabel("<h3>对局 Chat</h3>"))
        right_layout.addWidget(self.chat, 1)
        right_layout.addWidget(self.evidence)
        splitter = QSplitter()
        splitter.addWidget(self.editor)
        splitter.addWidget(right)
        splitter.setSizes([430, 700])
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("<h2>对局指导</h2>"))
        layout.addWidget(self.summary)
        layout.addWidget(splitter, 1)
        self.chat.send_requested.connect(self.send_requested.emit)
        self.chat.cancel_requested.connect(self.cancel_requested.emit)
        self.editor.save_requested.connect(self.save_requested.emit)
        self.editor.clear_requested.connect(self.clear_requested.emit)
        self.editor.screenshot_requested.connect(self.screenshot_requested.emit)

    def request(self, question: str) -> ChatRequest:
        return match_request(question, self.editor.game_state(question))

    def set_game_state(self, state: GameState) -> None:
        self.editor.set_game_state(state)
        self.refresh_summary(state)

    def refresh_summary(self, state: GameState) -> None:
        self.summary.setText("当前局面：" + game_state_summary(state))

    def show_event(self, event: ChatEvent) -> None:
        self.chat.show_event(event)
        if event.type == "observation" and event.game_state is not None:
            self.editor.conflict.setText("截图识别到其他值时保留手动选择。")
