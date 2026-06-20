from __future__ import annotations

import re
from importlib import import_module, util

try:
    import cv2
except Exception:  # pragma: no cover - optional dependency
    cv2 = None


class PlateOCRReader:
    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        """Initialize EasyOCR when it is available, otherwise run in fallback mode."""
        self.available = util.find_spec("easyocr") is not None and cv2 is not None
        self.reader = None

    def _ensure_reader(self) -> None:
        if self.reader is not None or not self.available:
            return
        try:
            easyocr = import_module("easyocr")
            self.reader = easyocr.Reader(["en"], gpu=False)
        except Exception:
            self.available = False
            self.reader = None

    def read_plate_text(self, plate_crop):
        """Perform OCR on a cropped plate image."""
        if plate_crop is None or getattr(plate_crop, "size", 0) == 0:
            return "", 0.0
        if not self.available or self.reader is None or cv2 is None:
            self._ensure_reader()
        if not self.available or self.reader is None or cv2 is None:
            return "", 0.0

        gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape[:2]
        if w < 100:
            gray = cv2.resize(gray, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)

        try:
            results = self.reader.readtext(gray)
        except Exception:
            return "", 0.0

        if not results:
            return "", 0.0

        best_text = ""
        max_conf = 0.0

        texts = []
        for bbox, text, conf in results:
            cleaned_text = self.clean_plate_text(text)
            if len(cleaned_text) >= 4:
                texts.append((cleaned_text, conf))

        if texts:
            texts.sort(key=lambda x: x[1], reverse=True)
            best_text, max_conf = texts[0]

        return best_text, max_conf

    def clean_plate_text(self, text):
        """Remove spaces and non-alphanumeric characters."""
        text = text.upper()
        text = re.sub(r"[^A-Z0-9]", "", text)
        return text

    def is_valid_license_plate(self, text):
        """Heuristic to check if the plate format looks like a standard Indian plate."""
        cleaned = self.clean_plate_text(text)
        pattern = r"^[A-Z]{2}[0-9]{2}[A-Z]{0,2}[0-9]{3,4}$"
        return bool(re.match(pattern, cleaned))
