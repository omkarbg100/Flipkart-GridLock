from __future__ import annotations

from importlib import import_module, util
from pathlib import Path

try:
    import cv2
except Exception:  # pragma: no cover - optional dependency
    cv2 = None

try:
    import numpy as np
except Exception:  # pragma: no cover - optional dependency
    np = None

from config.settings import CONFIDENCE_THRESHOLD


class TrafficDetector:
    def __init__(self, model_name: str = "yolov8n.pt"):
        """Initialize the YOLOv8 detector when the optional CV stack is present."""
        self.model_name = model_name
        self.device = "cpu"
        self.model = None
        self.available = self._has_required_modules() and Path(model_name).exists()
        self.missing_dependencies = self._collect_missing_dependencies()
        if not Path(model_name).exists() and model_name not in self.missing_dependencies:
            self.missing_dependencies.append(model_name)

        # COCO class mapping we are interested in
        # 0: person, 2: car, 3: motorcycle, 5: bus, 7: truck
        self.target_classes = {0: "person", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

    @classmethod
    def _collect_missing_dependencies(cls) -> list[str]:
        missing: list[str] = []
        if cv2 is None:
            missing.append("opencv-python-headless")
        if np is None:
            missing.append("numpy")
        if util.find_spec("torch") is None:
            missing.append("torch")
        if util.find_spec("ultralytics") is None:
            missing.append("ultralytics")
        return missing

    @staticmethod
    def _has_required_modules() -> bool:
        return cv2 is not None and np is not None and util.find_spec("torch") is not None and util.find_spec("ultralytics") is not None

    def ensure_available(self) -> None:
        if not self.available:
            raise RuntimeError(
                "Traffic detection is unavailable until the optional CV dependencies are installed: "
                f"{', '.join(self.missing_dependencies)}"
            )

    def _ensure_model(self) -> None:
        if self.model is not None:
            return
        if not self._has_required_modules():
            self.available = False
            return
        if not Path(self.model_name).exists():
            self.available = False
            if self.model_name not in self.missing_dependencies:
                self.missing_dependencies.append(self.model_name)
            return

        torch = import_module("torch")
        YOLO = import_module("ultralytics").YOLO
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        try:
            self.model = YOLO(self.model_name)
            self.model.to(self.device)
            self.available = True
        except Exception:
            self.model = None
            self.available = False
            if self.model_name not in self.missing_dependencies:
                self.missing_dependencies.append(self.model_name)

    @staticmethod
    def _box_iou(box_a, box_b):
        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b

        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)

        if inter_x1 >= inter_x2 or inter_y1 >= inter_y2:
            return 0.0

        inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
        area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
        area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
        union_area = area_a + area_b - inter_area
        return inter_area / union_area if union_area > 0 else 0.0

    def detect_frame(self, frame, conf_threshold=CONFIDENCE_THRESHOLD):
        """Run inference on a single frame and return a list of detections."""
        self._ensure_model()
        self.ensure_available()
        if frame is None:
            return []

        results = self.model(frame, verbose=False, conf=conf_threshold)[0]

        detections = []
        for box in results.boxes:
            class_id = int(box.cls[0].item())

            if class_id not in self.target_classes:
                continue

            class_name = self.target_classes[class_id]
            conf = float(box.conf[0].item())
            coords = box.xyxy[0].cpu().numpy().astype(int)
            x1, y1, x2, y2 = coords

            detections.append(
                {
                    "class_id": class_id,
                    "class_name": class_name,
                    "confidence": conf,
                    "box": (x1, y1, x2, y2),
                }
            )

        return detections

    def draw_detections(self, frame, detections, violations=None, plate_boxes=None):
        """Annotate the frame with bounding boxes."""
        if cv2 is None:
            return frame

        annotated_frame = frame.copy()

        if violations is None:
            violations = []

        for d in detections:
            x1, y1, x2, y2 = d["box"]
            label = f"{d['class_name']} ({d['confidence']:.2f})"

            is_violating = False
            for v_obj in violations:
                vx1, vy1, vx2, vy2 = v_obj["box"]
                if self._box_iou((x1, y1, x2, y2), (vx1, vy1, vx2, vy2)) > 0.25:
                    is_violating = True
                    label += f" - {v_obj['type']}"
                    break

            color = (0, 0, 255) if is_violating else (0, 255, 0)
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)

            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(annotated_frame, (x1, y1 - 20), (x1 + w, y1), color, -1)
            cv2.putText(annotated_frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        if plate_boxes:
            for p_box in plate_boxes:
                px1, py1, px2, py2, p_text = p_box
                cv2.rectangle(annotated_frame, (px1, py1), (px2, py2), (0, 255, 255), 2)
                if p_text:
                    (w, h), _ = cv2.getTextSize(p_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                    cv2.rectangle(annotated_frame, (px1, py1 - 20), (px1 + w, py1), (0, 255, 255), -1)
                    cv2.putText(annotated_frame, p_text, (px1, py1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        return annotated_frame
