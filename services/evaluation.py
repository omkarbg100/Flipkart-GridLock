from __future__ import annotations

import ast
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass
class EvaluationSummary:
    samples: int = 0
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    map50: float = 0.0
    mean_ap: float = 0.0
    matched_pairs: int = 0
    false_positives: int = 0
    false_negatives: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _canonical_column(df: pd.DataFrame, candidates: list[str], fallback: str | None = None) -> str:
    lowered = {str(col).lower(): str(col) for col in df.columns}
    for candidate in candidates:
        if candidate.lower() in lowered:
            return lowered[candidate.lower()]
    if fallback is not None:
        return fallback
    raise KeyError(f"None of the columns are available: {', '.join(candidates)}")


def _parse_box(value: Any) -> tuple[float, float, float, float] | None:
    if value is None:
        return None
    if isinstance(value, (list, tuple)) and len(value) == 4:
        return tuple(float(v) for v in value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            parsed = json.loads(text)
        except Exception:
            try:
                parsed = ast.literal_eval(text)
            except Exception:
                parts = [part.strip() for part in text.strip("[]()").split(",")]
                if len(parts) != 4:
                    return None
                try:
                    return tuple(float(part) for part in parts)  # type: ignore[return-value]
                except Exception:
                    return None
        if isinstance(parsed, (list, tuple)) and len(parsed) == 4:
            return tuple(float(v) for v in parsed)
    return None


def _row_box(row: pd.Series) -> tuple[float, float, float, float] | None:
    for key in ("bbox", "box"):
        if key in row and pd.notna(row[key]):
            parsed = _parse_box(row[key])
            if parsed is not None:
                return parsed

    aliases = [
        ("x1", "y1", "x2", "y2"),
        ("xmin", "ymin", "xmax", "ymax"),
        ("left", "top", "right", "bottom"),
    ]
    lowered = {str(col).lower(): col for col in row.index}
    for a1, a2, a3, a4 in aliases:
        if all(name in lowered for name in (a1, a2, a3, a4)):
            try:
                return (
                    float(row[lowered[a1]]),
                    float(row[lowered[a2]]),
                    float(row[lowered[a3]]),
                    float(row[lowered[a4]]),
                )
            except Exception:
                return None
    return None


def _box_iou(box_a: tuple[float, float, float, float], box_b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    if inter_x1 >= inter_x2 or inter_y1 >= inter_y2:
        return 0.0
    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union_area = area_a + area_b - inter_area
    return inter_area / union_area if union_area > 0 else 0.0


def load_benchmark_package(path: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    package_path = Path(path)
    if package_path.suffix.lower() != ".json":
        raise ValueError("Benchmark package must be a JSON file")

    payload = json.loads(package_path.read_text(encoding="utf-8"))
    ground_truth = pd.DataFrame(payload.get("ground_truth") or payload.get("annotations") or [])
    predictions = pd.DataFrame(payload.get("predictions") or payload.get("results") or [])
    return ground_truth, predictions


def summarize_classification_metrics(ground_truth: pd.DataFrame, predictions: pd.DataFrame) -> dict[str, Any]:
    if ground_truth.empty:
        return EvaluationSummary().to_dict()
    if predictions.empty:
        total_gt = int(len(ground_truth))
        summary = EvaluationSummary(
            samples=total_gt,
            accuracy=0.0,
            precision=0.0,
            recall=0.0,
            f1=0.0,
            matched_pairs=0,
            false_positives=0,
            false_negatives=total_gt,
        )
        return summary.to_dict()

    gt_label_col = _canonical_column(ground_truth, ["label", "class", "violation_type", "true_label"])
    pred_label_col = _canonical_column(predictions, ["label", "class", "violation_type", "predicted_label"], fallback=gt_label_col)
    gt_image_col = _canonical_column(ground_truth, ["image_id", "frame_id", "sample_id"], fallback="__row__")
    pred_image_col = _canonical_column(predictions, ["image_id", "frame_id", "sample_id"], fallback="__row__")

    gt = ground_truth.copy()
    pred = predictions.copy()
    gt["__image__"] = gt[gt_image_col].astype(str) if gt_image_col != "__row__" else gt.index.astype(str)
    pred["__image__"] = pred[pred_image_col].astype(str) if pred_image_col != "__row__" else pred.index.astype(str)
    gt["__label__"] = gt[gt_label_col].astype(str).map(_normalize_text)
    pred["__label__"] = pred[pred_label_col].astype(str).map(_normalize_text)

    gt_counts = gt.groupby(["__image__", "__label__"]).size()
    pred_counts = pred.groupby(["__image__", "__label__"]).size()

    keys = set(gt_counts.index).union(pred_counts.index)
    matched = 0
    total_gt = int(gt_counts.sum())
    total_pred = int(pred_counts.sum())
    for key in keys:
        matched += min(int(gt_counts.get(key, 0)), int(pred_counts.get(key, 0)))

    false_positives = max(0, total_pred - matched)
    false_negatives = max(0, total_gt - matched)
    precision = matched / (matched + false_positives) if matched + false_positives else 0.0
    recall = matched / (matched + false_negatives) if matched + false_negatives else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if precision + recall else 0.0
    accuracy = matched / max(total_gt, total_pred, 1)

    summary = EvaluationSummary(
        samples=max(total_gt, total_pred),
        accuracy=round(accuracy, 4),
        precision=round(precision, 4),
        recall=round(recall, 4),
        f1=round(f1, 4),
        matched_pairs=matched,
        false_positives=false_positives,
        false_negatives=false_negatives,
    )
    return summary.to_dict()


def summarize_detection_map(
    ground_truth: pd.DataFrame,
    predictions: pd.DataFrame,
    *,
    iou_threshold: float = 0.5,
) -> dict[str, Any]:
    if ground_truth.empty or predictions.empty:
        return {
            "map50": 0.0,
            "mean_ap": 0.0,
            "ap_by_class": [],
            "samples": int(max(len(ground_truth), len(predictions))),
            "note": "Bounding boxes were not provided, or one side of the benchmark was empty.",
        }

    gt_label_col = _canonical_column(ground_truth, ["label", "class", "violation_type", "true_label"])
    pred_label_col = _canonical_column(predictions, ["label", "class", "violation_type", "predicted_label"], fallback=gt_label_col)
    gt_image_col = _canonical_column(ground_truth, ["image_id", "frame_id", "sample_id"], fallback="__row__")
    pred_image_col = _canonical_column(predictions, ["image_id", "frame_id", "sample_id"], fallback="__row__")
    pred_conf_col = _canonical_column(predictions, ["confidence", "score", "probability"], fallback="__conf__")

    gt = ground_truth.copy()
    pred = predictions.copy()
    gt["__image__"] = gt[gt_image_col].astype(str) if gt_image_col != "__row__" else gt.index.astype(str)
    pred["__image__"] = pred[pred_image_col].astype(str) if pred_image_col != "__row__" else pred.index.astype(str)
    gt["__label__"] = gt[gt_label_col].astype(str).map(_normalize_text)
    pred["__label__"] = pred[pred_label_col].astype(str).map(_normalize_text)
    if pred_conf_col in pred.columns:
        pred["__conf__"] = pd.to_numeric(pred[pred_conf_col], errors="coerce").fillna(0.0)
    else:
        pred["__conf__"] = 0.0

    gt["__box__"] = gt.apply(_row_box, axis=1)
    pred["__box__"] = pred.apply(_row_box, axis=1)
    has_boxes = gt["__box__"].notna().any() and pred["__box__"].notna().any()
    if not has_boxes:
        classification_only = summarize_classification_metrics(ground_truth, predictions)
        return {
            "map50": 0.0,
            "mean_ap": 0.0,
            "ap_by_class": [],
            "samples": classification_only.get("samples", 0),
            "note": "Bounding boxes were not provided, so mAP was skipped.",
        }

    classes = sorted(set(gt["__label__"].tolist()) | set(pred["__label__"].tolist()))
    ap_rows: list[dict[str, Any]] = []

    for class_name in classes:
        class_gt = gt[gt["__label__"] == class_name]
        class_pred = pred[pred["__label__"] == class_name].sort_values("__conf__", ascending=False)

        gt_by_image: dict[str, list[dict[str, Any]]] = {}
        for _, row in class_gt.iterrows():
            box = row["__box__"]
            if box is None:
                continue
            gt_by_image.setdefault(str(row["__image__"]), []).append({"box": box, "matched": False})

        tp: list[int] = []
        fp: list[int] = []
        for _, row in class_pred.iterrows():
            box = row["__box__"]
            if box is None:
                continue
            candidates = gt_by_image.get(str(row["__image__"]), [])
            best_iou = 0.0
            best_idx = -1
            for idx, candidate in enumerate(candidates):
                if candidate["matched"]:
                    continue
                iou = _box_iou(box, candidate["box"])
                if iou > best_iou:
                    best_iou = iou
                    best_idx = idx
            if best_iou >= iou_threshold and best_idx >= 0:
                candidates[best_idx]["matched"] = True
                tp.append(1)
                fp.append(0)
            else:
                tp.append(0)
                fp.append(1)

        if not tp:
            ap_rows.append({"class": class_name, "ap50": 0.0, "ground_truth": int(len(class_gt))})
            continue

        tp_cum = pd.Series(tp).cumsum()
        fp_cum = pd.Series(fp).cumsum()
        recalls = tp_cum / max(len(class_gt), 1)
        precisions = tp_cum / (tp_cum + fp_cum)

        mrec = pd.concat([pd.Series([0.0]), recalls, pd.Series([1.0])], ignore_index=True)
        mpre = pd.concat([pd.Series([0.0]), precisions, pd.Series([0.0])], ignore_index=True)
        for idx in range(len(mpre) - 2, -1, -1):
            mpre.iloc[idx] = max(mpre.iloc[idx], mpre.iloc[idx + 1])

        ap = 0.0
        for idx in range(1, len(mrec)):
            if mrec.iloc[idx] != mrec.iloc[idx - 1]:
                ap += (mrec.iloc[idx] - mrec.iloc[idx - 1]) * mpre.iloc[idx]

        ap_rows.append({"class": class_name, "ap50": round(float(ap), 4), "ground_truth": int(len(class_gt))})

    map50 = round(sum(row["ap50"] for row in ap_rows) / len(ap_rows), 4) if ap_rows else 0.0
    return {
        "map50": map50,
        "mean_ap": map50,
        "ap_by_class": ap_rows,
        "samples": int(max(len(gt), len(pred))),
        "note": f"mAP computed with IoU threshold {iou_threshold:.2f}.",
    }


def summarize_benchmark(ground_truth: pd.DataFrame, predictions: pd.DataFrame) -> dict[str, Any]:
    classification = summarize_classification_metrics(ground_truth, predictions)
    detection = summarize_detection_map(ground_truth, predictions)
    return {
        "classification": classification,
        "detection": detection,
    }
