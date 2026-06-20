from __future__ import annotations

from dataclasses import dataclass
from typing import Any

try:
    import cv2
except Exception:  # pragma: no cover - optional dependency
    cv2 = None

try:
    import numpy as np
except Exception:  # pragma: no cover - optional dependency
    np = None


@dataclass
class PreprocessingConfig:
    enabled: bool = True
    profile: str = "auto"
    gamma: float = 1.15
    clahe_clip_limit: float = 2.0
    clahe_tile_grid_size: int = 8
    denoise_strength: int = 7
    sharpen_strength: float = 0.35
    blur_threshold: float = 85.0


def supports_preprocessing() -> bool:
    return cv2 is not None and np is not None


def estimate_brightness(frame) -> float:
    if frame is None or cv2 is None or np is None:
        return 0.0
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(np.mean(gray))


def estimate_blur_score(frame) -> float:
    if frame is None or cv2 is None or np is None:
        return 0.0
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _parse_config(config: PreprocessingConfig | dict[str, Any] | str | None) -> PreprocessingConfig:
    if isinstance(config, PreprocessingConfig):
        return config
    if isinstance(config, str):
        return PreprocessingConfig(profile=config)
    if isinstance(config, dict):
        base = PreprocessingConfig()
        for key, value in config.items():
            if hasattr(base, key):
                setattr(base, key, value)
        return base
    return PreprocessingConfig()


def apply_gamma_correction(frame, gamma: float) -> Any:
    if cv2 is None or np is None or frame is None:
        return frame
    gamma = max(0.2, float(gamma))
    lookup = np.array([(i / 255.0) ** (1.0 / gamma) * 255 for i in range(256)]).astype("uint8")
    return cv2.LUT(frame, lookup)


def apply_clahe(frame, clip_limit: float = 2.0, tile_grid_size: int = 8) -> Any:
    if cv2 is None or np is None or frame is None:
        return frame
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=float(clip_limit), tileGridSize=(int(tile_grid_size), int(tile_grid_size)))
    l_channel = clahe.apply(l_channel)
    merged = cv2.merge((l_channel, a_channel, b_channel))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


def apply_denoise(frame, strength: int = 7) -> Any:
    if cv2 is None or frame is None:
        return frame
    strength = max(1, int(strength))
    if len(frame.shape) == 2:
        return cv2.fastNlMeansDenoising(frame, None, strength, 7, 21)
    return cv2.fastNlMeansDenoisingColored(frame, None, strength, strength, 7, 21)


def apply_unsharp_mask(frame, amount: float = 0.35) -> Any:
    if cv2 is None or frame is None:
        return frame
    amount = max(0.0, float(amount))
    blurred = cv2.GaussianBlur(frame, (0, 0), 3)
    return cv2.addWeighted(frame, 1.0 + amount, blurred, -amount, 0)


def enhance_frame(frame, config: PreprocessingConfig | dict[str, Any] | str | None = None) -> Any:
    if frame is None or not supports_preprocessing():
        return frame

    cfg = _parse_config(config)
    if not cfg.enabled or cfg.profile.lower() == "off":
        return frame.copy()

    result = frame.copy()
    profile = cfg.profile.lower().strip()
    brightness = estimate_brightness(result)
    blur_score = estimate_blur_score(result)

    use_clahe = profile in {"auto", "low_light", "enhanced", "contrast", "night"}
    use_denoise = profile in {"auto", "low_light", "enhanced", "blur", "motion_blur", "night"}
    use_sharpen = profile in {"auto", "enhanced", "blur", "motion_blur"}

    if profile in {"auto", "low_light", "enhanced", "night"} and brightness < 145:
        result = apply_clahe(result, cfg.clahe_clip_limit, cfg.clahe_tile_grid_size)
        gamma = cfg.gamma + ((145.0 - brightness) / 220.0)
        result = apply_gamma_correction(result, min(1.8, max(1.0, gamma)))
    elif use_clahe and profile != "off":
        result = apply_clahe(result, cfg.clahe_clip_limit, cfg.clahe_tile_grid_size)

    if use_denoise:
        result = apply_denoise(result, cfg.denoise_strength)

    if use_sharpen and blur_score < cfg.blur_threshold * 1.2:
        result = apply_unsharp_mask(result, cfg.sharpen_strength)

    return result

