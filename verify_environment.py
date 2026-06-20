from __future__ import annotations

import importlib.util
import os
import sys

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

REQUIRED_LIBS = ["pandas", "nicegui", "plotly", "requests"]
OPTIONAL_ANALYSIS_LIBS = ["cv2", "numpy", "ultralytics", "easyocr", "torch"]


def has_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


print("==========================================")
print("[INFO] Running GridLock environment check")
print("==========================================")

missing_required = [name for name in REQUIRED_LIBS if not has_module(name)]
if missing_required:
    print(f"[FAIL] Missing dashboard dependencies: {', '.join(missing_required)}")
    sys.exit(1)

print("[SUCCESS] NiceGUI dashboard dependencies are available.")

optional_missing = [name for name in OPTIONAL_ANALYSIS_LIBS if not has_module(name)]
if optional_missing:
    print(
        "[WARN] Optional analysis stack is missing: "
        + ", ".join(optional_missing)
        + ". Video analysis will stay disabled until these are installed."
    )
else:
    print("[SUCCESS] Optional analysis stack is available.")

try:
    from services.db_service import get_counts_summary, init_db

    init_db()
    summary = get_counts_summary()
    print("[SUCCESS] SQLite database connectivity and schema initialized.")
    print(f"   Current total violations: {summary['total']}")
except Exception as exc:
    print(f"[FAIL] SQLite initialization failed: {exc}")
    sys.exit(1)

try:
    from services.video_processor import VideoProcessor

    status = VideoProcessor().dependency_status()
    print("[INFO] Video processor status:")
    print(f"   OpenCV: {status['opencv']}")
    print(f"   Detector ready: {status['detector_available']}")
    print(f"   OCR ready: {status['ocr_available']}")
except Exception as exc:
    print(f"[WARN] Video processor could not be instantiated cleanly: {exc}")

print("\n==========================================")
print("[SUCCESS] Environment check completed.")
print("==========================================")
