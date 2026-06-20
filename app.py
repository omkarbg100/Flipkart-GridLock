from __future__ import annotations

from nicegui import ui

from dashboard.page import dashboard  # noqa: F401
from dashboard.theme import PAGE_TITLE


if __name__ == "__main__":
    ui.run(
        title=PAGE_TITLE,
        host="0.0.0.0",
        port=8501,
        dark=True,
        reload=False,
        show=False,
        storage_secret="gridlock-ui-secret",
    )
