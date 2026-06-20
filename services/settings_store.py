from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from config.settings import (
    APP_SETTINGS_PATH,
    CONFIDENCE_THRESHOLD,
    DEFAULT_FRAME_HEIGHT,
    DEFAULT_FRAME_SKIP,
    DEFAULT_FRAME_WIDTH,
    DEFAULT_HELMET_SKIN_RATIO,
    DEFAULT_PARKING_COORDS,
    DEFAULT_PARKING_VIOLATION_SECONDS,
    DEFAULT_TRIPLE_OVERLAP_RATIO,
    DEFAULT_WRONG_SIDE_MIN_MOVE,
)


@dataclass
class DashboardSettings:
    confidence_threshold: float = CONFIDENCE_THRESHOLD
    frame_skip: int = DEFAULT_FRAME_SKIP
    parking_violation_seconds: float = DEFAULT_PARKING_VIOLATION_SECONDS
    wrong_side_min_move: int = DEFAULT_WRONG_SIDE_MIN_MOVE
    triple_overlap_ratio: float = DEFAULT_TRIPLE_OVERLAP_RATIO
    helmet_skin_ratio: float = DEFAULT_HELMET_SKIN_RATIO
    frame_width: int = DEFAULT_FRAME_WIDTH
    frame_height: int = DEFAULT_FRAME_HEIGHT
    signal_state: str = "RED"
    road_type: str = "One-Way Road"
    allowed_direction: str = "down"
    left_allowed_dir: str = "up"
    right_allowed_dir: str = "down"
    stop_line_y: int = 300
    preprocess_profile: str = "auto"
    parking_zones: list = field(default_factory=lambda: [zone.copy() for zone in DEFAULT_PARKING_COORDS])


class SettingsStore:
    def __init__(self, path: str = APP_SETTINGS_PATH):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._settings = self._load()

    def _load(self) -> DashboardSettings:
        if not self.path.exists():
            return DashboardSettings()
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            return DashboardSettings(**raw)
        except Exception:
            return DashboardSettings()

    def get(self) -> DashboardSettings:
        return self._settings

    def update(self, **kwargs) -> DashboardSettings:
        data = asdict(self._settings)
        data.update(kwargs)
        self._settings = DashboardSettings(**data)
        self.save()
        return self._settings

    def save(self) -> None:
        self.path.write_text(json.dumps(asdict(self._settings), indent=2), encoding="utf-8")
