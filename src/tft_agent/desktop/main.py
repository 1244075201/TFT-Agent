"""Windows desktop chat frontend with opt-in experimental screen observation."""

from __future__ import annotations

import asyncio
import html
import tempfile
from pathlib import Path

from tft_agent.app.service import AgentService
from tft_agent.core.schemas import ChatEvent, ChatRequest, GameState
from tft_agent.screen.monitor import MSSScreenMonitor


def main() -> None:
    try:
        from PySide6.QtCore import (
            QObject,
            Qt,
            QThread,
            QTimer,
            Signal,
        )
        from PySide6.QtWidgets import (
            QApplication,
            QFileDialog,
            QHBoxLayout,
            QLabel,
            QMainWindow,
            QMessageBox,
            QPushButton,
            QTextBrowser,
            QTextEdit,
            QVBoxLayout,
            QWidget,
        )
    except ImportError as exc:
        raise SystemExit("桌面界面需要 PySide6 和 mss，请运行：uv sync --all-extras") from exc

    from tft_agent.desktop.catalog import CanonicalEntityCatalog
    from tft_agent.desktop.pages import ProductPages

    class AskWorker(QObject):
        chat_event = Signal(object)
        finished = Signal()

        def __init__(self, service: AgentService, request: ChatRequest) -> None:
            super().__init__()
            self.service, self.request = service, request
            self.loop: asyncio.AbstractEventLoop | None = None
            self.task: asyncio.Task[None] | None = None

        def run(self) -> None:
            async def consume() -> None:
                async for item in self.service.ask(self.request):
                    self.chat_event.emit(item)

            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)
            self.task = self.loop.create_task(consume())
            try:
                self.loop.run_until_complete(self.task)
            except asyncio.CancelledError:
                self.chat_event.emit(ChatEvent(type="cancelled", message="请求已取消"))
            finally:
                self.task = None
                self.loop.close()
                self.loop = None
                self.finished.emit()

        def cancel(self) -> None:
            if self.loop is not None and self.task is not None:
                self.loop.call_soon_threadsafe(self.task.cancel)

    class ChatInput(QTextEdit):
        send_requested = Signal()

        def keyPressEvent(self, event: object) -> None:
            key = event.key()  # type: ignore[attr-defined]
            modifiers = event.modifiers()  # type: ignore[attr-defined]
            if key in (Qt.Key.Key_Return, Qt.Key.Key_Enter) and not (
                modifiers & Qt.KeyboardModifier.ShiftModifier
            ):
                self.send_requested.emit()
                event.accept()  # type: ignore[attr-defined]
                return
            super().keyPressEvent(event)  # type: ignore[arg-type]

    class MainWindow(QMainWindow):
        def __init__(self) -> None:
            super().__init__()
            self.service = AgentService()
            self.session_id = "desktop-default"
            self.screenshot: Path | None = None
            self.monitor: MSSScreenMonitor | None = None
            self.worker_thread: QThread | None = None
            self.worker: AskWorker | None = None
            self.busy = False
            self.generated_capture: Path | None = None
            self.setWindowTitle("云顶数据指导 Agent")
            self.resize(920, 720)
            root = QWidget()
            layout = QVBoxLayout(root)
            self.banner = QLabel(
                "合规模式：赛前资料、手动截图解释与赛后复盘。实验实时模式默认关闭。"
            )
            self.banner.setStyleSheet("padding:8px;background:#fff3cd;color:#664d03")
            self.chat = QTextBrowser()
            try:
                data_status = asyncio.run(self.service.data_status())
            except Exception as exc:
                data_status = f"DataTFT 状态读取失败：{exc}"
            self.state = QLabel(data_status)
            self.context = QLabel("当前局面：尚未记录")
            try:
                catalog = asyncio.run(self.service.entity_catalog())
            except Exception:
                catalog = CanonicalEntityCatalog([])
            self.pages = ProductPages(catalog)
            self.state_inputs = self.pages.state_inputs
            self.pages.mode_changed.connect(self.set_product_mode)
            try:
                saved_state = asyncio.run(self.service.load_game_state(self.session_id))
                self.populate_game_state(saved_state)
            except Exception:
                pass
            self.input = ChatInput()
            self.input.setPlaceholderText("描述当前局面或提问；Enter 发送，Shift+Enter 换行…")
            self.input.setMaximumHeight(120)
            buttons = QHBoxLayout()
            attach = QPushButton("选择截图")
            paste = QPushButton("粘贴截图")
            send = QPushButton("发送")
            cancel = QPushButton("取消")
            self.live = QPushButton("启动实验监控")
            clear_state = QPushButton("清空局面")
            clear = QPushButton("清除本地历史")
            for widget in (attach, paste, send, cancel, self.live, clear_state, clear):
                buttons.addWidget(widget)
            layout.addWidget(self.banner)
            layout.addWidget(self.state)
            layout.addWidget(self.context)
            layout.addWidget(self.pages)
            layout.addWidget(self.chat)
            layout.addWidget(self.input)
            layout.addLayout(buttons)
            self.setCentralWidget(root)
            attach.clicked.connect(self.choose_screenshot)
            paste.clicked.connect(self.paste_screenshot)
            send.clicked.connect(self.send)
            self.input.send_requested.connect(self.send)
            cancel.clicked.connect(self.cancel_request)
            clear.clicked.connect(self.clear_history)
            clear_state.clicked.connect(self.clear_game_state)
            self.live.clicked.connect(self.toggle_live)
            self.timer = QTimer(self)
            self.timer.setInterval(2500)
            self.timer.timeout.connect(self.poll_screen)

        def set_product_mode(self, mode: str) -> None:
            if mode == "comp_browser":
                self.input.setPlainText("显示当前版本阵容列表")
                self.state.setText("阵容推荐：版本阵容浏览 / 一图流")
            elif mode == "general_guide":
                self.input.clear()
                self.state.setText("通用攻略：与当前对局状态及会话隔离")
            else:
                self.input.clear()
                self.state.setText("对局指导：结合当前 GameState 给出建议")

        def populate_game_state(self, state: GameState) -> None:
            for key, editor in self.state_inputs.items():
                value = getattr(state, key)
                if isinstance(value, list):
                    editor.setText("，".join(str(item) for item in value))
                elif value is not None:
                    editor.setText(str(value))

        def edited_game_state(self, question: str) -> GameState:
            values: dict[str, object] = {"user_question": question}
            list_fields = {"augments", "augment_choices", "item_components",
                           "completed_items", "emblems", "core_units"}
            for key, editor in self.state_inputs.items():
                text = editor.text().strip()
                if not text:
                    continue
                if key in list_fields:
                    values[key] = [item.strip() for item in text.replace("，", ",").split(",")
                                   if item.strip()]
                elif key in {"level", "gold", "hp"}:
                    values[key] = int(text)
                else:
                    values[key] = text
            return GameState.model_validate(values)

        def clear_game_state(self) -> None:
            for editor in self.state_inputs.values():
                editor.clear()
            asyncio.run(self.service.save_game_state(self.session_id, GameState()))
            self.context.setText("当前局面：尚未记录")

        def choose_screenshot(self) -> None:
            name, _ = QFileDialog.getOpenFileName(
                self, "选择截图", "", "Images (*.png *.jpg *.jpeg)"
            )
            if name:
                self.screenshot = Path(name)
                self.state.setText(f"已附加：{self.screenshot.name}")

        def paste_screenshot(self) -> None:
            image = QApplication.clipboard().image()
            if image.isNull():
                self.state.setText("剪贴板中没有图片")
                return
            temporary = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
            temporary.close()
            self.generated_capture = Path(temporary.name)
            image.save(str(self.generated_capture), b"PNG")
            self.screenshot = self.generated_capture
            self.state.setText("已附加剪贴板截图")

        def send(self) -> None:
            text = self.input.toPlainText().strip()
            if not text:
                return
            if self.busy:
                self.state.setText("上一个问题仍在处理；可先点“取消”。")
                return
            self.busy = True
            self.chat.append(f"<b>你：</b>{text}")
            self.input.clear()
            edited_state = None
            if self.pages.product_mode.value == "match_guidance":
                try:
                    edited_state = self.edited_game_state(text)
                except ValueError as exc:
                    self.busy = False
                    self.state.setText(f"GameState 输入无效：{exc}")
                    return
            request = ChatRequest(text=text, session_id=self.session_id,
                                  screenshot_path=self.screenshot,
                                  confirmed_game_state=edited_state,
                                  product_mode=self.pages.product_mode.value)
            self.screenshot = None
            self.worker_thread = QThread(self)
            self.worker = AskWorker(self.service, request)
            self.worker.moveToThread(self.worker_thread)
            self.worker_thread.started.connect(self.worker.run)
            self.worker.chat_event.connect(self.show_event)
            self.worker.finished.connect(self.worker_thread.quit)
            self.worker.finished.connect(self.worker.deleteLater)
            self.worker.finished.connect(self.cleanup_capture)
            self.worker_thread.finished.connect(self.worker_thread.deleteLater)
            self.worker_thread.finished.connect(self.request_finished)
            self.worker_thread.start()

        def request_finished(self) -> None:
            self.busy = False
            self.worker = None
            self.worker_thread = None

        def show_event(self, event: ChatEvent) -> None:
            if event.type == "answer":
                safe_message = html.escape(event.message).replace("\n", "<br>")
                self.chat.append(f"<b>Agent：</b>{safe_message}")
                self.state.setText("回答完成")
                if event.retrieval_plan is not None:
                    self.state.setToolTip(
                        f"Retrieval Plan: {event.retrieval_plan}\n"
                        f"Context gaps: {event.context_gaps}\n"
                        f"LLM calls: {event.llm_call_count}"
                    )
                try:
                    self.context.setText(asyncio.run(self.service.session_summary(self.session_id)))
                except Exception:
                    pass
                if len(event.advice_options) <= 1:
                    return
                for option in event.advice_options:
                    sources = "<br>".join(
                        f'<a href="{item.url}">{item.label}</a>' for item in option.evidence
                    )
                    self.chat.append(
                        "<div style='margin:8px;padding:10px;border:1px solid #888'>"
                        f"<b>{option.title}</b><br>{option.action}<br>"
                        f"收益：{option.expected_benefit}<br>风险：{option.risk}<br>{sources}</div>"
                    )
            elif event.type == "observation" and event.game_state is not None:
                state = event.game_state
                self.state.setText(
                    f"阶段 {state.stage or '?'}｜{state.level or '?'}级｜"
                    f"{state.gold or '?'}金币"
                )
            else:
                self.state.setText(event.message)

        def clear_history(self) -> None:
            asyncio.run(
                self.service.clear_product_session(self.session_id, self.pages.product_mode)
            )
            self.chat.clear()
            self.context.setText("当前局面：尚未记录")

        def cancel_request(self) -> None:
            self.service.cancel(self.session_id)
            if self.worker is not None:
                self.worker.cancel()

        def cleanup_capture(self) -> None:
            if self.generated_capture is not None:
                self.generated_capture.unlink(missing_ok=True)
                self.generated_capture = None

        def toggle_live(self) -> None:
            if not self.service.settings.experimental_live_enabled:
                QMessageBox.warning(self, "实验模式未启用", "请先在配置中显式启用实验实时模式。")
                return
            if self.timer.isActive():
                self.timer.stop()
                self.live.setText("启动实验监控")
            else:
                self.monitor = MSSScreenMonitor()
                self.timer.start()
                self.live.setText("暂停实验监控")
                self.banner.setText("实验实时模式：仅分析自己的画面；禁止公开发布、对手追踪和自动操作。")

        def poll_screen(self) -> None:
            if self.monitor is None or self.busy:
                return
            try:
                frame = self.monitor.capture()
                if not self.monitor.is_stable_change(frame):
                    return
                temporary = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
                temporary.write(frame.png_bytes)
                temporary.close()
                self.generated_capture = Path(temporary.name)
                self.screenshot = self.generated_capture
                self.input.setPlainText("根据当前己方局面给出三个有数据依据的选择。")
                self.send()
            except Exception as exc:
                self.timer.stop()
                self.state.setText(f"监控已停止：{exc}")

    app = QApplication([])
    window = MainWindow()
    window.show()
    app.exec()


if __name__ == "__main__":
    main()
