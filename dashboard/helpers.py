from __future__ import annotations

import json
import re
import uuid
from datetime import date as date_type
from datetime import datetime, time as time_type
from pathlib import Path
from typing import Any

from config.settings import DEFAULT_CAMERAS, UPLOAD_DIR
from services.db_service import get_cameras_with_density
from services.job_manager import JobManager

UPLOAD_PATH = Path(UPLOAD_DIR)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".mpg", ".mpeg"}


def human_time(value: float | None) -> str:
    if not value:
        return "n/a"
    return datetime.fromtimestamp(value).strftime("%Y-%m-%d %H:%M:%S")


def human_size(num_bytes: int | float | None) -> str:
    if num_bytes is None:
        return "0 B"
    size = float(num_bytes)
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size < 1024.0 or unit == "TB":
            return f"{size:.1f} {unit}" if unit != "B" else f"{int(size)} {unit}"
        size /= 1024.0
    return f"{size:.1f} TB"


def safe_slug(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", value).strip("_")
    return slug.lower() or "upload"


def today_prefix() -> str:
    return datetime.now().date().isoformat()


def now_time_string() -> str:
    return datetime.now().strftime("%H:%M")


def parse_date(value: str | None) -> date_type:
    if not value:
        return datetime.now().date()
    try:
        return datetime.fromisoformat(value).date()
    except ValueError:
        return datetime.now().date()


def parse_time(value: str | None) -> time_type:
    if not value:
        return datetime.now().time().replace(microsecond=0)
    try:
        return datetime.strptime(value, "%H:%M").time()
    except ValueError:
        return datetime.now().time().replace(microsecond=0)


def safe_float(value: Any, fallback: float) -> float:
    try:
        return float(value)
    except Exception:
        return fallback


def safe_int(value: Any, fallback: int) -> int:
    try:
        return int(float(value))
    except Exception:
        return fallback


def camera_options() -> list[str]:
    cameras_df = get_cameras_with_density()
    ids = [cam["id"] for cam in DEFAULT_CAMERAS]
    if not cameras_df.empty and "camera_id" in cameras_df.columns:
        ids.extend(cameras_df["camera_id"].dropna().astype(str).tolist())
    return ["ALL"] + sorted(dict.fromkeys(ids))


def detect_media_kind(filename: str) -> str:
    suffix = Path(filename).suffix.lower()
    if suffix in IMAGE_EXTENSIONS:
        return "image"
    if suffix in VIDEO_EXTENSIONS:
        return "video"
    return "media"


def table_columns(df) -> list[dict[str, Any]]:
    return [
        {"name": str(col), "label": str(col), "field": str(col), "sortable": True, "align": "left"}
        for col in df.columns
    ]


def dataframe_to_rows(df) -> list[dict[str, Any]]:
    if df.empty:
        return []
    return df.fillna("").to_dict("records")


async def save_uploaded_media(upload_file) -> Path:
    UPLOAD_PATH.mkdir(parents=True, exist_ok=True)
    original_name = Path(getattr(upload_file, "name", "upload.mp4")).name
    stem = safe_slug(Path(original_name).stem)
    suffix = Path(original_name).suffix.lower() or ".mp4"
    destination = UPLOAD_PATH / f"{stem}_{datetime.now().strftime('%Y%m%d%H%M%S')}_{uuid.uuid4().hex[:8]}{suffix}"
    await upload_file.save(destination)
    return destination


async def save_uploaded_video(upload_file) -> Path:
    return await save_uploaded_media(upload_file)


def get_analysis_scope(storage: dict[str, Any]) -> str | None:
    value = str(storage.get("filter_date_prefix", "")).strip()
    return value or None


def build_options_from_controls(
    *args,
    source_label: str,
) -> Any:
    if len(args) == 3 and isinstance(args[0], JobManager):
        job_manager: JobManager = args[0]
        controls: dict[str, Any] = args[1]
        settings = args[2]
    elif len(args) == 2:
        job_manager = None
        controls = args[0]
        settings = args[1]
    else:
        raise TypeError("build_options_from_controls expects (controls, settings) or (job_manager, controls, settings)")

    parking_zones_text = str(controls["parking_zones_json"].value or "").strip()
    try:
        parking_zones = json.loads(parking_zones_text or "[]")
        if not isinstance(parking_zones, list):
            raise ValueError
    except Exception:
        parking_zones = settings.parking_zones

    if job_manager is None:
        from services.runtime import get_job_manager

        job_manager = get_job_manager()

    preprocess_profile = settings.preprocess_profile
    if "preprocess_profile" in controls:
        preprocess_profile = str(controls["preprocess_profile"].value or settings.preprocess_profile)

    return job_manager.build_options(
        settings,
        camera_id=str(controls["camera_id"].value or "CAM-01").strip() or "CAM-01",
        location_name=str(controls["location_name"].value or "Mall Road Intersection").strip() or "Mall Road Intersection",
        lat_coord=safe_float(controls["latitude"].value, 30.3218),
        lon_coord=safe_float(controls["longitude"].value, 78.0464),
        selected_date=parse_date(str(controls["selected_date"].value or today_prefix())),
        selected_time=parse_time(str(controls["selected_time"].value or now_time_string())),
        enable_parking_check=bool(controls["enable_parking_check"].value),
        enable_wrong_side=bool(controls["enable_wrong_side"].value),
        enable_signal_check=bool(controls["enable_signal_check"].value),
        road_type=str(controls["road_type"].value or "One-Way Road"),
        allowed_direction=str(controls["allowed_direction"].value or "down"),
        left_allowed_dir=str(controls["left_allowed_dir"].value or "up"),
        right_allowed_dir=str(controls["right_allowed_dir"].value or "down"),
        signal_state=str(controls["signal_state"].value or "RED"),
        stop_line_y=safe_int(controls["stop_line_y"].value, 300),
        preprocess_profile=preprocess_profile,
        frame_skip=safe_int(controls["frame_skip"].value, settings.frame_skip),
        confidence_threshold=safe_float(controls["confidence_threshold"].value, settings.confidence_threshold),
        parking_violation_seconds=safe_float(controls["parking_violation_seconds"].value, settings.parking_violation_seconds),
        parking_zones=parking_zones,
        source_label=source_label,
    )


def format_job_rows(jobs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for job in jobs:
        rows.append(
            {
                "id": job["id"],
                "kind": job["kind"].title(),
                "source": job["source_label"],
                "status": job["status"].title(),
                "progress": f"{round(float(job.get('progress', 0.0)) * 100, 1)}%",
                "frames": job.get("frames_processed", 0),
                "violations": job.get("violations_logged", 0),
                "message": job.get("message", ""),
                "updated": human_time(job.get("updated_at")),
            }
        )
    return rows


def format_default_rows(settings) -> list[dict[str, Any]]:
    return [
        {"setting": "Confidence threshold", "value": settings.confidence_threshold},
        {"setting": "Frame skip", "value": settings.frame_skip},
        {"setting": "Parking dwell seconds", "value": settings.parking_violation_seconds},
        {"setting": "Wrong-side min move", "value": settings.wrong_side_min_move},
        {"setting": "Triple overlap ratio", "value": settings.triple_overlap_ratio},
        {"setting": "Helmet skin ratio", "value": settings.helmet_skin_ratio},
        {"setting": "Frame width", "value": settings.frame_width},
        {"setting": "Frame height", "value": settings.frame_height},
        {"setting": "Signal state", "value": settings.signal_state},
        {"setting": "Road type", "value": settings.road_type},
        {"setting": "Allowed direction", "value": settings.allowed_direction},
        {"setting": "Left lane direction", "value": settings.left_allowed_dir},
        {"setting": "Right lane direction", "value": settings.right_allowed_dir},
        {"setting": "Stop line Y", "value": settings.stop_line_y},
        {"setting": "Preprocessing profile", "value": settings.preprocess_profile},
    ]


def seed_user_state(storage: dict[str, Any], settings) -> None:
    camera = DEFAULT_CAMERAS[0]
    defaults = {
        "uploaded_media_path": "",
        "uploaded_media_name": "",
        "uploaded_media_kind": "",
        "uploaded_video_path": "",
        "uploaded_video_name": "",
        "uploaded_video_size": 0,
        "benchmark_package_path": "",
        "benchmark_package_name": "",
        "selected_job_id": "",
        "selected_violation_id": "",
        "filter_query": "",
        "filter_violation_type": "ALL",
        "filter_camera_id": "ALL",
        "filter_date_prefix": today_prefix(),
        "camera_id": camera["id"],
        "location_name": camera["location"],
        "latitude": camera["latitude"],
        "longitude": camera["longitude"],
        "selected_date": today_prefix(),
        "selected_time": now_time_string(),
        "signal_state": settings.signal_state,
        "road_type": settings.road_type,
        "allowed_direction": settings.allowed_direction,
        "left_allowed_dir": settings.left_allowed_dir,
        "right_allowed_dir": settings.right_allowed_dir,
        "preprocess_profile": settings.preprocess_profile,
        "enable_parking_check": True,
        "enable_wrong_side": True,
        "enable_signal_check": True,
        "frame_skip": settings.frame_skip,
        "confidence_threshold": settings.confidence_threshold,
        "parking_violation_seconds": settings.parking_violation_seconds,
        "wrong_side_min_move": settings.wrong_side_min_move,
        "triple_overlap_ratio": settings.triple_overlap_ratio,
        "helmet_skin_ratio": settings.helmet_skin_ratio,
        "stop_line_y": settings.stop_line_y,
        "frame_width": settings.frame_width,
        "frame_height": settings.frame_height,
        "parking_zones_json": json.dumps(settings.parking_zones, indent=2),
    }
    for key, value in defaults.items():
        storage.setdefault(key, value)


def set_control_values_from_settings(controls: dict[str, Any], settings) -> None:
    controls["signal_state"].value = settings.signal_state
    controls["road_type"].value = settings.road_type
    controls["allowed_direction"].value = settings.allowed_direction
    controls["left_allowed_dir"].value = settings.left_allowed_dir
    controls["right_allowed_dir"].value = settings.right_allowed_dir
    if "preprocess_profile" in controls:
        controls["preprocess_profile"].value = settings.preprocess_profile
    controls["frame_skip"].value = settings.frame_skip
    controls["confidence_threshold"].value = settings.confidence_threshold
    controls["parking_violation_seconds"].value = settings.parking_violation_seconds
    controls["wrong_side_min_move"].value = settings.wrong_side_min_move
    controls["triple_overlap_ratio"].value = settings.triple_overlap_ratio
    controls["helmet_skin_ratio"].value = settings.helmet_skin_ratio
    controls["stop_line_y"].value = settings.stop_line_y
    controls["frame_width"].value = settings.frame_width
    controls["frame_height"].value = settings.frame_height
    controls["parking_zones_json"].value = json.dumps(settings.parking_zones, indent=2)
