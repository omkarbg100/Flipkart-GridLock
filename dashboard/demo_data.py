from __future__ import annotations

import base64
import hashlib
import html
import random
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from functools import lru_cache
from typing import Any

import pandas as pd

from config.settings import DEFAULT_CAMERAS, VIOLATION_TYPES

DEMO_PLATES = [
    "UK07AB1234",
    "DL01CT2048",
    "HR26DK8337",
    "UP14XY4123",
    "PB10MN3098",
    "RJ14KP7788",
    "KA05HZ6621",
    "MH12AZ7701",
]

DEMO_VIOLATION_TYPES = list(VIOLATION_TYPES.values())


@dataclass(frozen=True)
class DemoBundle:
    summary: dict[str, int]
    violations_by_type: pd.DataFrame
    violations_over_time: pd.DataFrame
    camera_wise: pd.DataFrame
    recent_violations: pd.DataFrame
    alerts: pd.DataFrame
    jobs: list[dict[str, Any]]
    live_snapshot: dict[str, Any]
    live_frame_url: str
    evidence_frame_url: str


def _scope_key(scope: str | None) -> str:
    return str(scope or "all").strip() or "all"


def _time_bucket() -> int:
    return int(datetime.now().timestamp() // 6)


def _seed(scope: str | None, bucket: int, salt: str) -> int:
    raw = f"{_scope_key(scope)}|{bucket}|{salt}".encode("utf-8")
    digest = hashlib.sha256(raw).hexdigest()
    return int(digest[:16], 16)


def _rng(scope: str | None, bucket: int, salt: str) -> random.Random:
    return random.Random(_seed(scope, bucket, salt))


def _to_data_url(svg: str) -> str:
    encoded = base64.b64encode(svg.encode("utf-8")).decode("ascii")
    return f"data:image/svg+xml;base64,{encoded}"


def _frame_svg(
    *,
    title: str,
    subtitle: str,
    scope: str | None,
    bucket: int,
    purpose: str,
) -> str:
    rng = _rng(scope, bucket, f"{purpose}-frame")
    stop_line_y = rng.randint(270, 350)
    road_top = 180
    road_bottom = 582
    stop_label = "RED LIGHT"
    signal_label = "SIGNAL: RED" if rng.random() > 0.35 else "SIGNAL: GREEN"
    traffic_way = "ONE WAY" if rng.random() > 0.5 else "TWO WAY"
    plate = rng.choice(DEMO_PLATES)

    vehicles = []
    base_x = [180, 500, 840]
    colors = ["#22c55e", "#f59e0b", "#ef4444"]
    labels = ["Vehicle", "Rider", "Plate OCR"]
    for idx, x in enumerate(base_x):
        box_w = rng.randint(120, 170)
        box_h = rng.randint(64, 96)
        y = rng.randint(220, 350)
        vehicles.append(
            {
                "x": x + rng.randint(-24, 24),
                "y": y,
                "w": box_w,
                "h": box_h,
                "color": colors[idx],
                "label": labels[idx],
            }
        )

    lane_lines = []
    for offset in range(4):
        x = 120 + offset * 240 + rng.randint(-18, 18)
        lane_lines.append(f'<line x1="{x}" y1="190" x2="{x + 120}" y2="570" stroke="#334155" stroke-width="6" stroke-dasharray="18 16"/>')

    vehicle_boxes = []
    for item in vehicles:
        vehicle_boxes.append(
            f'''
            <rect x="{item["x"]}" y="{item["y"]}" width="{item["w"]}" height="{item["h"]}" rx="16"
                  fill="none" stroke="{item["color"]}" stroke-width="5"/>
            <rect x="{item["x"] + 10}" y="{item["y"] - 30}" width="{item["w"] - 20}" height="24" rx="10"
                  fill="{item["color"]}" opacity="0.9"/>
            <text x="{item["x"] + 20}" y="{item["y"] - 13}" fill="#ffffff" font-size="14"
                  font-family="Arial, Helvetica, sans-serif">{html.escape(item["label"])}</text>
            '''
        )

    chip_lines = [
        ("Vehicles", "#22c55e"),
        (signal_label, "#ef4444" if "RED" in signal_label else "#22c55e"),
        (traffic_way, "#38bdf8"),
        (f"OCR: {plate}", "#f59e0b"),
        (stop_label, "#fb7185"),
    ]

    chip_svg = []
    for idx, (text, color) in enumerate(chip_lines):
        x = 52 + idx * 220
        chip_svg.append(
            f'''
            <rect x="{x}" y="92" width="190" height="42" rx="16" fill="{color}" opacity="0.18" stroke="{color}" stroke-width="1.5"/>
            <text x="{x + 16}" y="119" fill="#f8fafc" font-size="16" font-family="Arial, Helvetica, sans-serif">{html.escape(text)}</text>
            '''
        )

    spark_bars = []
    bar_rng = _rng(scope, bucket, f"{purpose}-bars")
    for idx in range(12):
        height = bar_rng.randint(18, 72)
        x = 910 + idx * 18
        y = 548 - height
        spark_bars.append(
            f'<rect x="{x}" y="{y}" width="10" height="{height}" rx="5" fill="#22c55e" opacity="{0.35 + idx * 0.04:.2f}"/>'
        )

    return f"""
    <svg xmlns="http://www.w3.org/2000/svg" width="1280" height="720" viewBox="0 0 1280 720">
      <defs>
        <linearGradient id="bg" x1="0" y1="0" x2="1" y2="1">
          <stop offset="0%" stop-color="#04060b"/>
          <stop offset="100%" stop-color="#111827"/>
        </linearGradient>
        <linearGradient id="road" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stop-color="#1f2937"/>
          <stop offset="100%" stop-color="#0f172a"/>
        </linearGradient>
        <linearGradient id="glow" x1="0" y1="0" x2="1" y2="0">
          <stop offset="0%" stop-color="#ef4444" stop-opacity="0.15"/>
          <stop offset="100%" stop-color="#22c55e" stop-opacity="0.15"/>
        </linearGradient>
      </defs>
      <rect width="1280" height="720" fill="url(#bg)"/>
      <circle cx="180" cy="130" r="180" fill="#ef4444" opacity="0.08"/>
      <circle cx="1110" cy="110" r="170" fill="#22c55e" opacity="0.08"/>
      <rect x="0" y="{road_top}" width="1280" height="{road_bottom - road_top}" fill="url(#road)"/>
      <rect x="0" y="{stop_line_y}" width="1280" height="8" fill="#ef4444" opacity="0.95"/>
      <rect x="0" y="{stop_line_y - 10}" width="1280" height="28" fill="url(#glow)"/>
      {"".join(lane_lines)}
      {"".join(vehicle_boxes)}
      <rect x="36" y="26" width="520" height="186" rx="28" fill="#020617" opacity="0.76" stroke="#334155" stroke-width="1.5"/>
      <text x="54" y="66" fill="#f8fafc" font-size="34" font-family="Arial, Helvetica, sans-serif">{html.escape(title)}</text>
      <text x="54" y="102" fill="#cbd5e1" font-size="19" font-family="Arial, Helvetica, sans-serif">{html.escape(subtitle)}</text>
      {"".join(chip_svg)}
      <text x="54" y="160" fill="#94a3b8" font-size="17" font-family="Arial, Helvetica, sans-serif">Scope: {html.escape(_scope_key(scope))}</text>
      <text x="54" y="188" fill="#94a3b8" font-size="17" font-family="Arial, Helvetica, sans-serif">Stop line Y: {stop_line_y}px</text>
      <rect x="904" y="496" width="314" height="148" rx="24" fill="#020617" opacity="0.72" stroke="#475569" stroke-width="1.5"/>
      <text x="926" y="532" fill="#f8fafc" font-size="18" font-family="Arial, Helvetica, sans-serif">Live overlay stats</text>
      <text x="926" y="566" fill="#cbd5e1" font-size="15" font-family="Arial, Helvetica, sans-serif">Signal: {html.escape(signal_label)}</text>
      <text x="926" y="592" fill="#cbd5e1" font-size="15" font-family="Arial, Helvetica, sans-serif">Road: {html.escape(traffic_way)}</text>
      <text x="926" y="618" fill="#cbd5e1" font-size="15" font-family="Arial, Helvetica, sans-serif">Plate: {html.escape(plate)}</text>
      {"".join(spark_bars)}
    </svg>
    """.strip()


def _build_recent_rows(scope: str | None, bucket: int) -> list[dict[str, Any]]:
    rng = _rng(scope, bucket, "recent")
    now = datetime.now().replace(microsecond=0)
    cameras = list(DEFAULT_CAMERAS)
    rows: list[dict[str, Any]] = []
    for index in range(12):
        camera = cameras[index % len(cameras)]
        violation_type = rng.choice(DEMO_VIOLATION_TYPES)
        plate = rng.choice(DEMO_PLATES)
        timestamp = now - timedelta(minutes=index * 6 + rng.randint(0, 3))
        rows.append(
            {
                "id": 9000 + index,
                "Time": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                "Camera ID": camera["id"],
                "Location": camera["location"],
                "Violation Type": violation_type,
                "Plate Number": plate,
                "Confidence": round(rng.uniform(0.66, 0.98), 2),
                "Evidence": "demo:frame",
            }
        )
    return rows


def _build_trend_rows(scope: str | None, bucket: int) -> pd.DataFrame:
    rng = _rng(scope, bucket, "trend")
    today = date.today()
    rows = []
    for offset in range(6, -1, -1):
        day = today - timedelta(days=offset)
        count = rng.randint(3, 16)
        rows.append({"Date": day.isoformat(), "Count": count})
    return pd.DataFrame(rows)


def _build_camera_rows(scope: str | None, bucket: int, recent_rows: list[dict[str, Any]]) -> pd.DataFrame:
    rng = _rng(scope, bucket, "camera")
    counts = pd.DataFrame(recent_rows).groupby(["Camera ID", "Location"]).size().reset_index(name="count")
    rows = []
    for cam in DEFAULT_CAMERAS:
        match = counts[counts["Camera ID"] == cam["id"]]
        count = int(match["count"].iloc[0]) if not match.empty else rng.randint(1, 6)
        rows.append(
            {
                "camera_id": cam["id"],
                "location": cam["location"],
                "latitude": cam["latitude"],
                "longitude": cam["longitude"],
                "count": count,
            }
        )
    return pd.DataFrame(rows).sort_values("count", ascending=False).reset_index(drop=True)


def _build_alert_rows(recent_rows: list[dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(recent_rows)
    if df.empty:
        return pd.DataFrame(columns=["Plate Number", "Offense Count", "Latest Violation", "Last Detected", "Offense History"])

    grouped = []
    for plate, plate_df in df.groupby("Plate Number"):
        if len(plate_df) < 2:
            continue
        latest = plate_df.iloc[0]
        history = "; ".join(
            f"{row['Violation Type']} at {row['Location']} ({row['Time']})"
            for _, row in plate_df.iterrows()
        )
        grouped.append(
            {
                "Plate Number": plate,
                "Offense Count": int(len(plate_df)),
                "Latest Violation": str(latest["Violation Type"]),
                "Last Detected": str(latest["Time"]),
                "Offense History": history,
            }
        )

    if not grouped:
        fallback_rows = df.head(3).copy()
        for idx, (_, row) in enumerate(fallback_rows.iterrows(), start=1):
            grouped.append(
                {
                    "Plate Number": row["Plate Number"],
                    "Offense Count": idx + 1,
                    "Latest Violation": row["Violation Type"],
                    "Last Detected": row["Time"],
                    "Offense History": f"{row['Violation Type']} at {row['Location']} ({row['Time']})",
                }
            )

    return pd.DataFrame(grouped).sort_values("Offense Count", ascending=False).reset_index(drop=True)


def _build_summary(recent_rows: list[dict[str, Any]], alerts_df: pd.DataFrame) -> dict[str, int]:
    df = pd.DataFrame(recent_rows)
    if df.empty:
        return {"total": 0, "helmet": 0, "triple": 0, "parking": 0, "repeat": 0, "cameras": len(DEFAULT_CAMERAS)}

    types = df["Violation Type"].astype(str).tolist()
    return {
        "total": int(len(df)),
        "helmet": int(sum("Helmet" in item for item in types)),
        "triple": int(sum("Triple" in item for item in types)),
        "parking": int(sum("Parking" in item for item in types)),
        "repeat": int(len(alerts_df)),
        "cameras": int(df["Camera ID"].nunique()),
    }


def _build_jobs(scope: str | None, bucket: int, summary: dict[str, int], recent_rows: list[dict[str, Any]], frame_url: str) -> list[dict[str, Any]]:
    now = datetime.now().timestamp()
    scope_label = _scope_key(scope)
    running = {
        "id": f"demo-{bucket}",
        "kind": "video",
        "source_label": f"Synthetic CCTV feed ({scope_label})" if scope_label != "all" else "Synthetic CCTV feed",
        "status": "running",
        "message": f"Processed {len(recent_rows)} frame(s) in demo mode",
        "progress": round(min(0.92, 0.42 + (summary["total"] / 40.0)), 2),
        "total_frames": max(120, summary["total"] * 20),
        "frames_processed": max(24, summary["total"] * 4),
        "violations_logged": summary["repeat"] + 2,
        "started_at": now - 74,
        "updated_at": now,
        "finished_at": None,
        "summary": summary,
        "recent_violations": recent_rows[:5],
        "preview_b64": frame_url.split(",", 1)[-1],
        "preview_caption": "Demo annotated frame with stop line, lanes, vehicles, and OCR overlays.",
        "preview_seq": bucket,
        "preview_updated_at": now,
        "error": "",
        "result": {"demo": True},
    }
    queued = {
        "id": f"demo-queued-{bucket}",
        "kind": "upload",
        "source_label": "Sample traffic archive",
        "status": "queued",
        "message": "Queued demo review",
        "progress": 0.18,
        "total_frames": 96,
        "frames_processed": 18,
        "violations_logged": max(1, summary["repeat"] - 1),
        "started_at": now - 33,
        "updated_at": now - 3,
        "finished_at": None,
        "summary": summary,
        "recent_violations": recent_rows[5:8],
        "preview_b64": frame_url.split(",", 1)[-1],
        "preview_caption": "Queued demo batch",
        "preview_seq": bucket - 1,
        "preview_updated_at": now - 3,
        "error": "",
        "result": {"demo": True},
    }
    return [running, queued]


@lru_cache(maxsize=64)
def _build_bundle(scope_key: str, bucket: int) -> DemoBundle:
    scope = scope_key or "all"
    recent_rows = _build_recent_rows(scope, bucket)
    alerts_df = _build_alert_rows(recent_rows)
    summary = _build_summary(recent_rows, alerts_df)
    trend_df = _build_trend_rows(scope, bucket)
    camera_df = _build_camera_rows(scope, bucket, recent_rows)

    recent_df = pd.DataFrame(recent_rows)
    type_df = (
        recent_df.groupby("Violation Type").size().reset_index(name="Count").sort_values("Count", ascending=False).reset_index(drop=True)
        if not recent_df.empty
        else pd.DataFrame(columns=["Violation Type", "Count"])
    )

    live_frame_url = _to_data_url(
        _frame_svg(
            title="GridLock Demo Feed",
            subtitle="Synthetic live scan output for a traffic violation dashboard",
            scope=scope,
            bucket=bucket,
            purpose="live",
        )
    )
    evidence_frame_url = _to_data_url(
        _frame_svg(
            title="GridLock Evidence Preview",
            subtitle="Annotated sample evidence frame",
            scope=scope,
            bucket=bucket,
            purpose="evidence",
        )
    )

    jobs = _build_jobs(scope, bucket, summary, recent_rows, live_frame_url)

    live_snapshot = jobs[0].copy()
    live_snapshot["id"] = f"demo-live-{bucket}"
    live_snapshot["kind"] = "demo"
    live_snapshot["status"] = "running"
    live_snapshot["message"] = f"Demo scan updated {bucket}"
    live_snapshot["preview_b64"] = live_frame_url.split(",", 1)[-1]
    live_snapshot["preview_caption"] = "Synthetic live frame with red line, parking zones, lane markers, and plate OCR."
    live_snapshot["preview_seq"] = bucket
    live_snapshot["preview_updated_at"] = datetime.now().timestamp()
    live_snapshot["updated_at"] = live_snapshot["preview_updated_at"]

    return DemoBundle(
        summary=summary,
        violations_by_type=type_df,
        violations_over_time=trend_df,
        camera_wise=camera_df,
        recent_violations=recent_df,
        alerts=alerts_df,
        jobs=jobs,
        live_snapshot=live_snapshot,
        live_frame_url=live_frame_url,
        evidence_frame_url=evidence_frame_url,
    )


def get_demo_bundle(scope: str | None = None) -> DemoBundle:
    return _build_bundle(_scope_key(scope), _time_bucket())
