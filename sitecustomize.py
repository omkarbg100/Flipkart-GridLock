from __future__ import annotations

import asyncio
import sys

if sys.platform == "win32" and hasattr(asyncio, "WindowsSelectorEventLoopPolicy"):
    # Load before app imports so Windows starts on the selector loop.
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
