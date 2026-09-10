"""Shared asynchronous request lifecycle for both desktop workspaces."""

from __future__ import annotations

import asyncio

from PySide6.QtCore import QObject, QThread, Signal

from tft_agent.app.service import AgentService
from tft_agent.core.schemas import ChatEvent, ChatRequest


class AskWorker(QObject):
    chat_event = Signal(object)
    finished = Signal()

    def __init__(self, service: AgentService, request: ChatRequest) -> None:
        super().__init__()
        self.service = service
        self.request = request
        self.loop: asyncio.AbstractEventLoop | None = None
        self.task: asyncio.Task[None] | None = None

    def run(self) -> None:
        async def consume() -> None:
            async for event in self.service.ask(self.request):
                self.chat_event.emit(event)

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


class DesktopRequestController(QObject):
    chat_event = Signal(str, object)
    request_finished = Signal(str)

    def __init__(self, service: AgentService) -> None:
        super().__init__()
        self.service = service
        self.worker_thread: QThread | None = None
        self.worker: AskWorker | None = None
        self.workspace: str | None = None

    @property
    def busy(self) -> bool:
        return self.worker_thread is not None

    def submit(self, workspace: str, request: ChatRequest) -> bool:
        if self.busy:
            return False
        self.workspace = workspace
        self.worker_thread = QThread(self)
        self.worker = AskWorker(self.service, request)
        self.worker.moveToThread(self.worker_thread)
        self.worker_thread.started.connect(self.worker.run)
        self.worker.chat_event.connect(
            lambda event: self.chat_event.emit(workspace, event)
        )
        self.worker.finished.connect(self.worker_thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.worker_thread.finished.connect(self.worker_thread.deleteLater)
        self.worker_thread.finished.connect(self._finished)
        self.worker_thread.start()
        return True

    def cancel(self) -> None:
        if self.worker is not None:
            self.worker.cancel()

    def _finished(self) -> None:
        workspace = self.workspace or ""
        self.worker = None
        self.worker_thread = None
        self.workspace = None
        self.request_finished.emit(workspace)
