"""Desktop application entry point."""

from __future__ import annotations


def main() -> None:
    try:
        from PySide6.QtWidgets import QApplication

        from tft_agent.desktop.window import DesktopMainWindow
    except ImportError as exc:
        raise SystemExit("桌面界面需要 PySide6 和 mss，请运行：uv sync --all-extras") from exc

    app = QApplication([])
    window = DesktopMainWindow()
    window.show()
    app.exec()


if __name__ == "__main__":
    main()
