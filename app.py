from __future__ import annotations

import asyncio
import os
import socket
import sys

if sys.platform == "win32" and hasattr(asyncio, "WindowsSelectorEventLoopPolicy"):
    # Windows selector loop avoids the noisy proactor connection-reset logs here.
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from nicegui import ui

from dashboard.page import dashboard  # noqa: F401
from dashboard.theme import PAGE_TITLE


def _pick_port(preferred_port: int) -> int:
    for port in range(preferred_port, preferred_port + 20):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.bind(("0.0.0.0", port))
            except OSError:
                continue
            else:
                return port
    return preferred_port


if __name__ == "__main__":
    preferred_port = int(os.getenv("PORT", "8501"))
    port = _pick_port(preferred_port)
    if port != preferred_port:
        print(f"[WARN] Port {preferred_port} is busy, falling back to {port}.")
    ui.run(
        title=PAGE_TITLE,
        host="0.0.0.0",
        port=port,
        dark=True,
        reload=False,
        show=False,
        storage_secret="gridlock-ui-secret",
    )
