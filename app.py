from __future__ import annotations

import asyncio
import sys

if sys.platform == "win32" and hasattr(asyncio, "WindowsSelectorEventLoopPolicy"):
    # Windows selector loop avoids the noisy proactor connection-reset logs here.
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

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
