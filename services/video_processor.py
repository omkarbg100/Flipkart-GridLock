from __future__ import annotations

import base64
import os
import time
from dataclasses import dataclass, field
from datetime import date as date_type, datetime, time as time_type, timedelta
from pathlib import Path
from typing import Any, Iterable, Optional

try:
    import cv2
except Exception:  # pragma: no cover - optional dependency
    cv2 = None

from config.settings import DEFAULT_PARKING_COORDS, EVIDENCE_DIR, CONFIDENCE_THRESHOLD
from services.db_service import add_camera_if_not_exists, add_violation, get_counts_summary, get_recent_violations
from services.detector import TrafficDetector
from services.preprocessing import PreprocessingConfig, enhance_frame
from services.ocr_service import PlateOCRReader
from services.violation_engine import ViolationEngine


@dataclass
class AnalysisOptions:
    camera_id: str = "CAM-01"
    location_name: str = "Mall Road Intersection"
    lat_coord: float = 30.3218
    lon_coord: float = 78.0464
    selected_date: date_type = field(default_factory=date_type.today)
    selected_time: time_type = field(default_factory=lambda: datetime.now().time().replace(microsecond=0))
    enable_parking_check: bool = True
    enable_wrong_side: bool = True
    enable_signal_check: bool = True
    road_type: str = "One-Way Road"
    allowed_direction: str = "down"
    left_allowed_dir: str = "up"
    right_allowed_dir: str = "down"
    signal_state: str = "RED"
    stop_line_y: int = 300
    frame_skip: int = 6
    confidence_threshold: float = CONFIDENCE_THRESHOLD
    frame_width: int = 800
    frame_height: int = 500
    parking_violation_seconds: float = 5.0
    wrong_side_min_move: int = 15
    triple_overlap_ratio: float = 0.3
    helmet_skin_ratio: float = 0.15
    parking_zones: list | None = None
    max_duration_seconds: Optional[int] = None
    source_label: str = ""
    preprocess_profile: str = "auto"


class VideoProcessor:
    def __init__(
        self,
        detector: Optional[TrafficDetector] = None,
        ocr_reader: Optional[PlateOCRReader] = None,
        violation_engine: Optional[ViolationEngine] = None,
    ) -> None:
        self.detector = detector or TrafficDetector()
        self.ocr_reader = ocr_reader or PlateOCRReader.get_instance()
        self.violation_engine = violation_engine or ViolationEngine()

    def dependency_status(self) -> dict[str, Any]:
        detector_available = getattr(self.detector, "available", False)
        detector_missing = getattr(self.detector, "missing_dependencies", [])
        ocr_available = getattr(self.ocr_reader, "available", False)
        return {
            "opencv": cv2 is not None,
            "detector_available": detector_available,
            "detector_missing": detector_missing,
            "ocr_available": ocr_available,
        }

    def ensure_ready(self) -> None:
        status = self.dependency_status()
        missing: list[str] = []
        if not status["opencv"]:
            missing.append("opencv-python-headless")
        if not status["detector_available"]:
            detector_missing = status.get("detector_missing") or []
            if detector_missing:
                missing.extend(detector_missing)
            else:
                missing.append("ultralytics")
        if missing:
            raise RuntimeError(
                "Video analysis is unavailable until the optional CV dependencies are installed: "
                + ", ".join(dict.fromkeys(missing))
            )

    @staticmethod
    def _build_direction_config(options: AnalysisOptions) -> Optional[dict[str, str]]:
        if not options.enable_wrong_side:
            return None
        return {
            "road_type": "two-way" if options.road_type == "Two-Way Road (Split Left/Right)" else "one-way",
            "allowed_dir": options.allowed_direction,
            "left_allowed_dir": options.left_allowed_dir,
            "right_allowed_dir": options.right_allowed_dir,
        }

    @staticmethod
    def _encode_preview(frame) -> str:
        ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ok:
            return ""
        return base64.b64encode(encoded.tobytes()).decode("utf-8")

    def _apply_preprocessing(self, frame, options: AnalysisOptions):
        if cv2 is None or frame is None:
            return frame
        profile = (options.preprocess_profile or "off").strip().lower()
        if profile == "off":
            return frame
        return enhance_frame(
            frame,
            PreprocessingConfig(
                enabled=True,
                profile=profile,
            ),
        )

    def _process_frame(
        self,
        frame,
        options: AnalysisOptions,
        *,
        current_time_str: str,
        parking_zones,
        direction_config,
        stop_line_config,
    ) -> tuple[Any, int]:
        analysis_frame = self._apply_preprocessing(frame, options)
        if analysis_frame is None:
            analysis_frame = frame

        detections = self.detector.detect_frame(analysis_frame, conf_threshold=options.confidence_threshold)
        violations = self.violation_engine.process_violations(
            frame=analysis_frame,
            detections=detections,
            camera_id=options.camera_id,
            camera_location=options.location_name,
            parking_zones=parking_zones,
            direction_settings=direction_config,
            stop_line_settings=stop_line_config,
            signal_state=options.signal_state,
            custom_datetime_str=current_time_str,
        )

        annotated = analysis_frame.copy()
        if options.enable_signal_check:
            line_color = (0, 0, 255) if options.signal_state == "RED" else (0, 255, 0)
            cv2.line(annotated, (0, options.stop_line_y), (annotated.shape[1], options.stop_line_y), line_color, 2)
            cv2.putText(annotated, "STOP LINE", (10, options.stop_line_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, line_color, 1)

        if options.enable_parking_check:
            for idx_z, zone in enumerate(parking_zones):
                zx1, zy1, zx2, zy2 = zone
                cv2.rectangle(annotated, (zx1, zy1), (zx2, zy2), (0, 255, 0), 1)
                cv2.putText(annotated, f"Zone {idx_z+1}", (zx1, zy1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

        if options.enable_wrong_side:
            if options.road_type == "Two-Way Road (Split Left/Right)":
                w_half = annotated.shape[1] // 2
                cv2.line(annotated, (w_half, 0), (w_half, annotated.shape[0]), (255, 255, 0), 2)
                cv2.putText(annotated, f"LANE A ({options.left_allowed_dir.upper()})", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                cv2.putText(annotated, f"LANE B ({options.right_allowed_dir.upper()})", (w_half + 10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            else:
                cv2.putText(annotated, f"ONE-WAY ({options.allowed_direction.upper()})", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        plate_boxes = []
        for violation in violations:
            vx1, vy1, vx2, vy2 = violation["box"]
            vh = vy2 - vy1
            vw = vx2 - vx1
            plate_y1 = max(0, vy1 + int(vh * 0.6))
            plate_y2 = min(analysis_frame.shape[0], vy2)
            plate_x1 = max(0, vx1 + int(vw * 0.15))
            plate_x2 = min(analysis_frame.shape[1], vx2 - int(vw * 0.15))

            plate_text = "UNKNOWN"
            ocr_conf = 0.0

            if plate_y2 > plate_y1 and plate_x2 > plate_x1:
                plate_crop = analysis_frame[plate_y1:plate_y2, plate_x1:plate_x2]
                plate_text, ocr_conf = self.ocr_reader.read_plate_text(plate_crop)
                if not plate_text:
                    plate_text = f"UP07-{options.camera_id}-{violation['object_id']}"
                    ocr_conf = 0.50
                plate_boxes.append((plate_x1, plate_y1, plate_x2, plate_y2, plate_text))

            evidence_filename = f"evidence_{int(time.time())}_{options.camera_id}_{violation['type'].replace(' ', '_')}.jpg"
            evidence_path = os.path.join(EVIDENCE_DIR, evidence_filename)
            evidence_frame = analysis_frame.copy()
            cv2.rectangle(evidence_frame, (vx1, vy1), (vx2, vy2), (0, 0, 255), 2)
            cv2.putText(
                evidence_frame,
                f"{violation['type']}: {plate_text}",
                (vx1, max(20, vy1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2,
            )
            cv2.imwrite(evidence_path, evidence_frame)

            add_violation(
                camera_id=options.camera_id,
                location=options.location_name,
                violation_type=violation["type"],
                timestamp=current_time_str,
                plate_number=plate_text,
                confidence=violation["confidence"] if ocr_conf == 0.0 else ocr_conf,
                evidence_path=evidence_path,
            )

        annotated = self.detector.draw_detections(annotated, detections, violations, plate_boxes)
        return annotated, len(violations)

    def process(
        self,
        source: str | int,
        options: AnalysisOptions,
        job,
        live: bool = False,
    ) -> dict[str, Any]:
        self.ensure_ready()
        add_camera_if_not_exists(options.camera_id, options.location_name, options.lat_coord, options.lon_coord)
        self.violation_engine.reset()
        self.violation_engine.parking_violation_seconds = options.parking_violation_seconds
        self.violation_engine.wrong_side_min_move = options.wrong_side_min_move
        self.violation_engine.triple_overlap_ratio = options.triple_overlap_ratio
        self.violation_engine.helmet_skin_ratio = options.helmet_skin_ratio

        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise ValueError(f"Could not open source: {source}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if not live else 0
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps <= 0:
            fps = 25.0

        frame_idx = 0
        processed_frames = 0
        violations_logged = 0
        last_progress = 0.0

        job.update_status(
            status="running",
            message=f"Processing {options.source_label or 'video'}...",
            total_frames=total_frames,
            progress=0.0,
        )

        try:
            while cap.isOpened():
                runtime_options = options
                if runtime_options.max_duration_seconds and processed_frames >= runtime_options.max_duration_seconds:
                    break
                if getattr(job, "cancel_event", None) is not None and job.cancel_event.is_set():
                    job.update_status(status="cancelled", message="Cancelled by user")
                    break

                self.violation_engine.parking_violation_seconds = runtime_options.parking_violation_seconds
                self.violation_engine.wrong_side_min_move = runtime_options.wrong_side_min_move
                self.violation_engine.triple_overlap_ratio = runtime_options.triple_overlap_ratio
                self.violation_engine.helmet_skin_ratio = runtime_options.helmet_skin_ratio

                start_datetime = datetime.combine(runtime_options.selected_date, runtime_options.selected_time)
                direction_config = self._build_direction_config(runtime_options)
                stop_line_config = {"stop_line_y": runtime_options.stop_line_y} if runtime_options.enable_signal_check else None
                if runtime_options.enable_parking_check:
                    parking_zones = runtime_options.parking_zones or DEFAULT_PARKING_COORDS
                else:
                    parking_zones = []

                ret, frame = cap.read()
                if not ret:
                    break

                frame_idx += 1
                if frame_idx % max(runtime_options.frame_skip, 1) != 0:
                    continue

                frame = cv2.resize(frame, (runtime_options.frame_width, runtime_options.frame_height))
                elapsed_seconds = frame_idx / fps
                current_time_str = (start_datetime + timedelta(seconds=elapsed_seconds)).strftime("%Y-%m-%d %H:%M:%S")
                annotated, frame_violations = self._process_frame(
                    frame,
                    runtime_options,
                    current_time_str=current_time_str,
                    parking_zones=parking_zones,
                    direction_config=direction_config,
                    stop_line_config=stop_line_config,
                )
                violations_logged += frame_violations

                processed_frames += 1
                if total_frames > 0 and not live:
                    last_progress = min(1.0, frame_idx / total_frames)
                else:
                    last_progress = 0.0

                job.set_preview_frame(annotated, caption=f"{runtime_options.source_label or 'Video'} | {current_time_str}")
                job.update_status(
                    message=f"Processed {processed_frames} frame(s)",
                    frames_processed=processed_frames,
                    violations_logged=violations_logged,
                    progress=last_progress,
                    total_frames=total_frames,
                )

        except Exception as exc:
            job.update_status(status="failed", message=f"Processing failed: {exc}", error=str(exc))
            raise
        finally:
            cap.release()

        summary = get_counts_summary()
        recent_violations = get_recent_violations(limit=25).to_dict(orient="records")
        job.update_status(
            status="completed" if job.status not in {"cancelled", "failed"} else job.status,
            message="Analysis complete" if job.status not in {"cancelled", "failed"} else job.message,
            summary=summary,
            recent_violations=recent_violations,
            progress=1.0 if total_frames > 0 else job.progress,
            total_frames=total_frames,
        )
        return {
            "frames_processed": processed_frames,
            "violations_logged": violations_logged,
            "summary": summary,
            "recent_violations": recent_violations,
        }

    def process_image(
        self,
        source: str,
        options: AnalysisOptions,
        job,
    ) -> dict[str, Any]:
        self.ensure_ready()
        image_path = Path(source)
        frame = cv2.imread(str(image_path))
        if frame is None:
            raise ValueError(f"Could not open image: {source}")

        frame = cv2.resize(frame, (options.frame_width, options.frame_height))
        job.update_status(
            status="running",
            message=f"Processing {options.source_label or image_path.name}...",
            total_frames=1,
            progress=0.0,
        )

        runtime_options = options
        self.violation_engine.parking_violation_seconds = runtime_options.parking_violation_seconds
        self.violation_engine.wrong_side_min_move = runtime_options.wrong_side_min_move
        self.violation_engine.triple_overlap_ratio = runtime_options.triple_overlap_ratio
        self.violation_engine.helmet_skin_ratio = runtime_options.helmet_skin_ratio

        start_datetime = datetime.combine(runtime_options.selected_date, runtime_options.selected_time)
        current_time_str = start_datetime.strftime("%Y-%m-%d %H:%M:%S")
        direction_config = self._build_direction_config(runtime_options)
        stop_line_config = {"stop_line_y": runtime_options.stop_line_y} if runtime_options.enable_signal_check else None
        if runtime_options.enable_parking_check:
            parking_zones = runtime_options.parking_zones or DEFAULT_PARKING_COORDS
        else:
            parking_zones = []

        annotated, frame_violations = self._process_frame(
            frame,
            runtime_options,
            current_time_str=current_time_str,
            parking_zones=parking_zones,
            direction_config=direction_config,
            stop_line_config=stop_line_config,
        )
        job.set_preview_frame(annotated, caption=f"{runtime_options.source_label or image_path.name} | {current_time_str}")
        job.update_status(
            message="Processed 1 image",
            frames_processed=1,
            violations_logged=frame_violations,
            progress=1.0,
            total_frames=1,
        )

        summary = get_counts_summary()
        recent_violations = get_recent_violations(limit=25).to_dict(orient="records")
        job.update_status(
            status="completed",
            message="Analysis complete",
            summary=summary,
            recent_violations=recent_violations,
            progress=1.0,
            total_frames=1,
        )
        return {
            "frames_processed": 1,
            "violations_logged": frame_violations,
            "summary": summary,
            "recent_violations": recent_violations,
        }
