from __future__ import annotations

import base64
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

try:
    import cv2
except Exception:  # pragma: no cover - optional dependency
    cv2 = None

from services.settings_store import DashboardSettings
from services.video_processor import AnalysisOptions, VideoProcessor


@dataclass
class DashboardJob:
    id: str
    kind: str
    source_label: str
    options: AnalysisOptions | None = None
    status: str = "queued"
    message: str = "Queued"
    progress: float = 0.0
    total_frames: int = 0
    frames_processed: int = 0
    violations_logged: int = 0
    started_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    finished_at: Optional[float] = None
    summary: dict = field(default_factory=dict)
    recent_violations: list = field(default_factory=list)
    preview_b64: str = ""
    preview_caption: str = ""
    preview_seq: int = 0
    error: str = ""
    result: dict = field(default_factory=dict)
    cancel_event: threading.Event = field(default_factory=threading.Event, repr=False)
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def update_status(self, **kwargs) -> None:
        with self.lock:
            for key, value in kwargs.items():
                if value is not None:
                    setattr(self, key, value)
            self.updated_at = time.time()
            if self.status in {"completed", "cancelled", "failed"} and self.finished_at is None:
                self.finished_at = self.updated_at

    def set_preview_frame(self, frame, caption: str = "") -> None:
        if cv2 is None or frame is None:
            return

        ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
        if not ok:
            return

        with self.lock:
            self.preview_b64 = base64.b64encode(encoded.tobytes()).decode("utf-8")
            self.preview_caption = caption
            self.preview_seq += 1
            self.updated_at = time.time()

    def apply_options(self, updates: AnalysisOptions) -> bool:
        if self.options is None:
            return False

        with self.lock:
            for key, value in vars(updates).items():
                if key == "source_label":
                    continue
                setattr(self.options, key, value)
            self.updated_at = time.time()
        return True

    def snapshot(self) -> dict:
        with self.lock:
            return {
                "id": self.id,
                "kind": self.kind,
                "source_label": self.source_label,
                "status": self.status,
                "message": self.message,
                "progress": self.progress,
                "total_frames": self.total_frames,
                "frames_processed": self.frames_processed,
                "violations_logged": self.violations_logged,
                "started_at": self.started_at,
                "updated_at": self.updated_at,
                "finished_at": self.finished_at,
                "summary": self.summary,
                "error": self.error,
                "preview_b64": self.preview_b64,
                "preview_caption": self.preview_caption,
                "preview_seq": self.preview_seq,
            }


class JobManager:
    def __init__(self, processor: Optional[VideoProcessor] = None):
        self.processor = processor or VideoProcessor()
        self._jobs: dict[str, DashboardJob] = {}
        self._lock = threading.RLock()

    def _register(self, job: DashboardJob) -> DashboardJob:
        with self._lock:
            self._jobs[job.id] = job
        return job

    def _new_job(self, kind: str, source_label: str) -> DashboardJob:
        job = DashboardJob(
            id=uuid.uuid4().hex[:10],
            kind=kind,
            source_label=source_label,
        )
        return self._register(job)

    @staticmethod
    def _attach_options(job: DashboardJob, options: AnalysisOptions) -> None:
        job.options = options

    def list_jobs(self) -> list[dict]:
        with self._lock:
            jobs = list(self._jobs.values())
        jobs.sort(key=lambda job: job.updated_at, reverse=True)
        return [job.snapshot() for job in jobs]

    def get_job(self, job_id: str) -> Optional[DashboardJob]:
        with self._lock:
            return self._jobs.get(job_id)

    def get_active_job(self) -> Optional[DashboardJob]:
        with self._lock:
            running = [job for job in self._jobs.values() if job.status in {"queued", "running"}]
        if not running:
            return None
        running.sort(key=lambda job: job.started_at, reverse=True)
        return running[0]

    def latest_snapshot(self) -> Optional[dict]:
        active = self.get_active_job()
        if active:
            return active.snapshot()
        with self._lock:
            if not self._jobs:
                return None
            latest = sorted(self._jobs.values(), key=lambda job: job.updated_at, reverse=True)[0]
        return latest.snapshot()

    def cancel(self, job_id: str) -> bool:
        job = self.get_job(job_id)
        if not job:
            return False
        job.cancel_event.set()
        job.update_status(status="cancelled", message="Cancellation requested")
        return True

    def submit_upload(self, video_path: str, options: AnalysisOptions) -> DashboardJob:
        self.processor.ensure_ready()
        source_label = options.source_label or Path(video_path).name
        job = self._new_job("upload", source_label)
        self._attach_options(job, options)
        threading.Thread(target=self._run_job, args=(job, video_path, options, False), daemon=True).start()
        return job

    def submit_image(self, image_path: str, options: AnalysisOptions) -> DashboardJob:
        self.processor.ensure_ready()
        source_label = options.source_label or Path(image_path).name
        job = self._new_job("image", source_label)
        self._attach_options(job, options)
        threading.Thread(target=self._run_image_job, args=(job, image_path, options), daemon=True).start()
        return job

    def start_live_camera(self, options: AnalysisOptions, camera_index: int = 0) -> DashboardJob:
        self.processor.ensure_ready()
        source_label = options.source_label or f"Camera {camera_index}"
        job = self._new_job("live", source_label)
        self._attach_options(job, options)
        threading.Thread(target=self._run_job, args=(job, camera_index, options, True), daemon=True).start()
        return job

    def apply_live_options(self, job_id: str, updates: AnalysisOptions) -> bool:
        job = self.get_job(job_id)
        if not job:
            return False
        return job.apply_options(updates)

    def _run_job(self, job: DashboardJob, source, options: AnalysisOptions, live: bool) -> None:
        try:
            job.update_status(status="running", message="Starting analysis")
            result = self.processor.process(source=source, options=options, job=job, live=live)
            if job.status not in {"cancelled", "failed"}:
                job.update_status(
                    status="completed",
                    message="Analysis complete",
                    result=result,
                    summary=result.get("summary", {}),
                    recent_violations=result.get("recent_violations", []),
                    frames_processed=result.get("frames_processed", 0),
                    violations_logged=result.get("violations_logged", 0),
                    progress=1.0 if not live else job.progress,
                )
        except Exception as exc:
            job.update_status(status="failed", message=f"Analysis failed: {exc}", error=str(exc))

    def _run_image_job(self, job: DashboardJob, source: str, options: AnalysisOptions) -> None:
        try:
            job.update_status(status="running", message="Starting image analysis")
            result = self.processor.process_image(source=source, options=options, job=job)
            if job.status not in {"cancelled", "failed"}:
                job.update_status(
                    status="completed",
                    message="Analysis complete",
                    result=result,
                    summary=result.get("summary", {}),
                    recent_violations=result.get("recent_violations", []),
                    frames_processed=result.get("frames_processed", 1),
                    violations_logged=result.get("violations_logged", 0),
                    progress=1.0,
                    total_frames=1,
                )
        except Exception as exc:
            job.update_status(status="failed", message=f"Analysis failed: {exc}", error=str(exc))

    def build_options(
        self,
        settings: DashboardSettings,
        *,
        camera_id: str,
        location_name: str,
        lat_coord: float,
        lon_coord: float,
        selected_date,
        selected_time,
        enable_parking_check: bool,
        enable_wrong_side: bool,
        enable_signal_check: bool,
        road_type: str,
        allowed_direction: str,
        left_allowed_dir: str,
        right_allowed_dir: str,
        signal_state: str,
        stop_line_y: int,
        frame_skip: Optional[int] = None,
        confidence_threshold: Optional[float] = None,
        parking_violation_seconds: Optional[float] = None,
        parking_zones: Optional[list] = None,
        source_label: str = "",
        preprocess_profile: str = "auto",
    ) -> AnalysisOptions:
        return AnalysisOptions(
            camera_id=camera_id,
            location_name=location_name,
            lat_coord=lat_coord,
            lon_coord=lon_coord,
            selected_date=selected_date,
            selected_time=selected_time,
            enable_parking_check=enable_parking_check,
            enable_wrong_side=enable_wrong_side,
            enable_signal_check=enable_signal_check,
            road_type=road_type,
            allowed_direction=allowed_direction,
            left_allowed_dir=left_allowed_dir,
            right_allowed_dir=right_allowed_dir,
            signal_state=signal_state,
            stop_line_y=stop_line_y,
            frame_skip=frame_skip if frame_skip is not None else settings.frame_skip,
            confidence_threshold=confidence_threshold if confidence_threshold is not None else settings.confidence_threshold,
            frame_width=settings.frame_width,
            frame_height=settings.frame_height,
            parking_violation_seconds=parking_violation_seconds if parking_violation_seconds is not None else settings.parking_violation_seconds,
            wrong_side_min_move=settings.wrong_side_min_move,
            triple_overlap_ratio=settings.triple_overlap_ratio,
            helmet_skin_ratio=settings.helmet_skin_ratio,
            parking_zones=parking_zones if parking_zones is not None else settings.parking_zones,
            source_label=source_label,
            preprocess_profile=preprocess_profile if preprocess_profile else settings.preprocess_profile,
        )
