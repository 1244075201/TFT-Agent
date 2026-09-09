@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\pythonw.exe" (
  echo First-time setup is required. Run: uv sync --all-extras
  pause
  exit /b 1
)
start "" ".venv\Scripts\pythonw.exe" -m tft_agent.desktop.main
