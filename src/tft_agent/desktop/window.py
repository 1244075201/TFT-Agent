"""Top-level window, navigation, and coordination between persistent workspaces."""

from __future__ import annotations

import asyncio
from pathlib import Path

from PySide6.QtCore import QTimer, Signal
from PySide6.QtWidgets import (
    QButtonGroup,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from tft_agent.app.pipeline import ProductMode
from tft_agent.app.service import AgentService
from tft_agent.core.schemas import ChatEvent, GameState
from tft_agent.desktop.catalog import CanonicalEntityCatalog
from tft_agent.desktop.controller import DesktopRequestController
from tft_agent.desktop.pages import GuideChatPage, MatchGuidancePage


class WorkspaceNavigation(QWidget):
    workspace_selected = Signal(int)

    def __init__(self, data_status: str) -> None:
        super().__init__()
        self.guide = QPushButton("攻略问答")
        self.match = QPushButton("对局指导")
        self.guide.setCheckable(True)
        self.match.setCheckable(True)
        self.group = QButtonGroup(self)
        self.group.setExclusive(True)
        self.group.addButton(self.guide)
        self.group.addButton(self.match)
        self.guide.setChecked(True)
        self.setStyleSheet("QPushButton:checked{background:#4c6ef5;color:white;font-weight:bold}")
        self.guide.clicked.connect(lambda: self.select(0))
        self.match.clicked.connect(lambda: self.select(1))
        layout = QVBoxLayout(self)
        layout.addWidget(self.guide)
        layout.addWidget(self.match)
        layout.addStretch()
        self.data_status = QLabel(data_status)
        self.data_status.setWordWrap(True)
        layout.addWidget(self.data_status)
        layout.addWidget(QLabel("设置 / 关于"))
        self.setFixedWidth(170)

    def select(self, index: int) -> None:
        self.guide.setChecked(index == 0)
        self.match.setChecked(index == 1)
        self.workspace_selected.emit(index)


class DesktopMainWindow(QMainWindow):
    def __init__(self, service: AgentService | None = None) -> None:
        super().__init__()
        self.service = service or AgentService()
        self.controller = DesktopRequestController(self.service)
        self.pending_screenshot: Path | None = None
        self.setWindowTitle("云顶数据指导 Agent")
        self.resize(1180, 780)
        self.setMinimumSize(1024, 680)
        try:
            data_status = asyncio.run(self.service.data_status())
            catalog = asyncio.run(self.service.entity_catalog())
            cards = asyncio.run(self.service.composition_cards())
        except Exception as exc:
            data_status = f"数据读取失败：{exc}"
            catalog = CanonicalEntityCatalog([])
            cards = []
        active = self.service.active_context
        set_number = active.set_number if active else self.service.settings.current_set_number
        game_patch = active.game_patch if active else self.service.settings.current_game_patch
        header = QLabel(
            f"<h2>云顶数据指导 Agent</h2>Set {set_number} · Patch {game_patch}"
        )
        self.header = header
        self.displayed_context = active
        self.navigation = WorkspaceNavigation(data_status)
        self.guide_page = GuideChatPage(game_patch, cards)
        self.match_page = MatchGuidancePage(catalog)
        self.stack = QStackedWidget()
        self.stack.addWidget(self.guide_page)
        self.stack.addWidget(self.match_page)
        body = QHBoxLayout()
        body.addWidget(self.navigation)
        body.addWidget(self.stack, 1)
        root = QWidget()
        layout = QVBoxLayout(root)
        layout.addWidget(header)
        layout.addLayout(body, 1)
        self.setCentralWidget(root)
        self._connect_signals()
        self.update_timer = QTimer(self)
        self.update_timer.setInterval(10000)
        self.update_timer.timeout.connect(self._refresh_data_status)
        self.update_timer.start()
        try:
            state = asyncio.run(self.service.load_game_state("desktop-match"))
            self.match_page.set_game_state(state)
        except Exception:
            self.match_page.set_game_state(GameState())

    def _refresh_data_status(self) -> None:
        if self.controller.busy or self.service.data_updates.lock.locked():
            return
        try:
            self.navigation.data_status.setText(asyncio.run(self.service.data_status()))
            context = self.service.active_context
            if context and context != self.displayed_context:
                cards = asyncio.run(self.service.composition_cards())
                self.guide_page.update_catalog(context.game_patch, cards)
                self.header.setText(
                    f"<h2>云顶数据指导 Agent</h2>Set {context.set_number} "
                    f"· Patch {context.game_patch}"
                )
                self.displayed_context = context
        except Exception:
            self.navigation.data_status.setText("数据更新状态暂不可用")

    def _connect_signals(self) -> None:
        self.navigation.workspace_selected.connect(self.stack.setCurrentIndex)
        self.guide_page.send_requested.connect(lambda text: self._send("guide", text))
        self.match_page.send_requested.connect(lambda text: self._send("match", text))
        self.guide_page.cancel_requested.connect(self.controller.cancel)
        self.guide_page.new_conversation_requested.connect(
            self._new_guide_conversation
        )
        self.match_page.cancel_requested.connect(self.controller.cancel)
        self.match_page.save_requested.connect(self._save_state)
        self.match_page.clear_requested.connect(self._clear_state)
        self.match_page.screenshot_requested.connect(self._choose_screenshot)
        self.controller.chat_event.connect(self._show_event)
        self.controller.request_finished.connect(self._request_finished)

    def _send(self, workspace: str, text: str) -> None:
        page = self.guide_page if workspace == "guide" else self.match_page
        chat = page.chat
        if self.controller.busy:
            chat.status.setText("另一工作区的请求仍在处理；可先取消。")
            return
        try:
            request = page.request(text)
        except ValueError as exc:
            chat.status.setText(f"输入无效：{exc}")
            return
        if workspace == "match" and self.pending_screenshot is not None:
            request = request.model_copy(update={"screenshot_path": self.pending_screenshot})
            self.pending_screenshot = None
        if self.controller.submit(workspace, request):
            chat.begin_request(text)

    def _show_event(self, workspace: str, value: object) -> None:
        event = ChatEvent.model_validate(value)
        if workspace == "guide":
            self.guide_page.chat.show_event(event)
        else:
            self.match_page.show_event(event)
            if event.type == "answer":
                self.match_page.refresh_summary(self.match_page.editor.game_state())

    def _request_finished(self, workspace: str) -> None:
        page = self.guide_page if workspace == "guide" else self.match_page
        page.chat.set_busy(False)

    def _save_state(self) -> None:
        try:
            state = self.match_page.editor.game_state()
            asyncio.run(self.service.save_game_state("desktop-match", state))
            self.match_page.set_game_state(state)
            self.match_page.chat.status.setText("局面已保存")
        except ValueError as exc:
            self.match_page.chat.status.setText(f"局面无效：{exc}")

    def _clear_state(self) -> None:
        state = GameState()
        asyncio.run(self.service.save_game_state("desktop-match", state))
        self.match_page.set_game_state(state)
        self.match_page.chat.status.setText("本局已清空")

    def _choose_screenshot(self) -> None:
        name, _ = QFileDialog.getOpenFileName(
            self, "选择截图", "", "Images (*.png *.jpg *.jpeg)"
        )
        if name:
            self.pending_screenshot = Path(name)
            self.match_page.chat.status.setText(f"已附加截图：{self.pending_screenshot.name}")

    def clear_current_history(self) -> None:
        if self.stack.currentIndex() == 0:
            asyncio.run(self.service.clear_product_session(
                "desktop-guide", ProductMode.GENERAL_GUIDE
            ))
            self.guide_page.chat.clear_history()
        else:
            asyncio.run(self.service.clear_product_session(
                "desktop-match", ProductMode.MATCH_GUIDANCE
            ))
            self.match_page.chat.clear_history()

    def _new_guide_conversation(self) -> None:
        if self.controller.busy:
            self.guide_page.chat.status.setText("请先等待当前回答完成或取消请求。")
            return
        asyncio.run(self.service.clear_general_guide_conversation("desktop-guide"))
        self.guide_page.chat.clear_history()
