#!/usr/bin/env python3
"""YOLO-style bbox metrics for LocAny pad/hole predictions.

Predictions are parsed from class-major LocAny text
(``<ref>SMD Pad</ref><box>…</box>+`` / ``Locating Hole``), see
``Embodied/data/data-detect-rule.md``.

Classes (display names match common pad_detect YOLO tables)::
  hole -> circle (id 0)
  pad  -> rect   (id 1)

Input records expect::
  pad_hole_label:  [{"type": "pad"|"hole", "bbox": [x1,y1,x2,y2]}, ...]
  pad_hole_result: same schema (predictions; conf optional, default 1.0)

Coords are treated as the same space as GT (LocAny norm1000).
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

IOU_THRESHOLDS = [round(0.5 + 0.05 * i, 2) for i in range(10)]
CLASS_NAMES = {0: "circle", 1: "rect"}  # hole, pad
TYPE_TO_ID = {"hole": 0, "pad": 1, "circle": 0, "rect": 1}


def box_area(box: list[float]) -> float:
    x1, y1, x2, y2 = box
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def box_iou(a: list[float], b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = box_area(a) + box_area(b) - inter
    return inter / union if union > 0 else 0.0


def _norm_box(bbox: list[Any]) -> list[float] | None:
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return None
    try:
        x1, y1, x2, y2 = (float(v) for v in bbox)
    except (TypeError, ValueError):
        return None
    return [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]


def items_to_instances(items: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for it in items or []:
        typ = str(it.get("type") or "").strip().lower()
        cls_id = TYPE_TO_ID.get(typ)
        box = _norm_box(it.get("bbox") or [])
        if cls_id is None or box is None:
            continue
        conf = float(it.get("conf", it.get("score", 1.0)) or 1.0)
        out.append({"class_id": cls_id, "bbox": box, "conf": conf})
    return out


def match_instances(
    preds: list[dict[str, Any]],
    gts: list[dict[str, Any]],
    iou_thr: float,
) -> tuple[list[bool], list[bool]]:
    pred_used = [False] * len(preds)
    gt_used = [False] * len(gts)
    candidates: list[tuple[float, int, int]] = []
    for pi, pred in enumerate(preds):
        for gi, gt in enumerate(gts):
            if pred["class_id"] != gt["class_id"]:
                continue
            iou = box_iou(pred["bbox"], gt["bbox"])
            if iou >= iou_thr:
                candidates.append((iou, pi, gi))
    for _iou, pi, gi in sorted(candidates, key=lambda x: x[0], reverse=True):
        if pred_used[pi] or gt_used[gi]:
            continue
        pred_used[pi] = True
        gt_used[gi] = True
    return pred_used, gt_used


def compute_ap(recalls: list[float], precisions: list[float]) -> float:
    if not recalls:
        return 0.0
    recalls = [0.0] + sorted(recalls) + [1.0]
    precisions = [0.0] + precisions + [0.0]
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = max(precisions[i], precisions[i + 1])
    ap = 0.0
    for i in range(1, len(recalls)):
        if recalls[i] != recalls[i - 1]:
            ap += (recalls[i] - recalls[i - 1]) * precisions[i]
    return ap


def evaluate_class_at_iou(
    records: list[dict[str, Any]],
    class_id: int,
    iou_thr: float,
) -> dict[str, Any]:
    scored_preds: list[tuple[float, float]] = []  # (conf, is_tp)
    total_gt = 0
    tp = fp = 0

    for rec in records:
        gts = [
            x
            for x in items_to_instances(rec.get("pad_hole_label"))
            if x["class_id"] == class_id
        ]
        preds = [
            x
            for x in items_to_instances(rec.get("pad_hole_result"))
            if x["class_id"] == class_id
        ]
        preds = sorted(preds, key=lambda x: x["conf"], reverse=True)
        total_gt += len(gts)
        pred_used, _gt_used = match_instances(preds, gts, iou_thr)
        for pi, pred in enumerate(preds):
            hit = bool(pred_used[pi])
            scored_preds.append((pred["conf"], 1.0 if hit else 0.0))
            if hit:
                tp += 1
            else:
                fp += 1

    scored_preds.sort(key=lambda x: x[0], reverse=True)
    tp_c = fp_c = 0
    precisions: list[float] = []
    recalls: list[float] = []
    for _conf, is_tp in scored_preds:
        if is_tp:
            tp_c += 1
        else:
            fp_c += 1
        precisions.append(tp_c / (tp_c + fp_c) if (tp_c + fp_c) else 0.0)
        recalls.append(tp_c / total_gt if total_gt else 0.0)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / total_gt if total_gt else 0.0
    return {
        "class_id": class_id,
        "class_name": CLASS_NAMES[class_id],
        "instances": total_gt,
        "tp": tp,
        "fp": fp,
        "fn": max(0, total_gt - tp),
        "precision": precision,
        "recall": recall,
        "ap": compute_ap(recalls, precisions),
    }


def collect_split_stats(records: list[dict[str, Any]]) -> dict[int, dict[str, int]]:
    stats: dict[int, dict[str, int]] = defaultdict(lambda: {"images": 0, "instances": 0})
    for rec in records:
        items = items_to_instances(rec.get("pad_hole_label"))
        seen: set[int] = set()
        for item in items:
            cid = item["class_id"]
            stats[cid]["instances"] += 1
            seen.add(cid)
        for cid in seen:
            stats[cid]["images"] += 1
    return dict(stats)


def build_pad_hole_summary(
    records: list[dict[str, Any]],
    iou_report: float = 0.5,
) -> dict[str, Any]:
    """Build YOLO-style summary (Box P/R at ``iou_report``, plus mAP50 / mAP50-95)."""
    split_stats = collect_split_stats(records)
    total_images = len(records)
    total_instances = sum(v["instances"] for v in split_stats.values())

    per_class: list[dict[str, Any]] = []
    map50_values: list[float] = []
    map5095_values: list[float] = []

    for class_id in sorted(CLASS_NAMES):
        main = evaluate_class_at_iou(records, class_id, iou_report)
        ap50 = evaluate_class_at_iou(records, class_id, 0.5)["ap"]
        ap_values = [
            evaluate_class_at_iou(records, class_id, thr)["ap"] for thr in IOU_THRESHOLDS
        ]
        ap5095 = sum(ap_values) / len(ap_values) if ap_values else 0.0
        class_stats = split_stats.get(class_id, {"images": 0, "instances": 0})
        per_class.append(
            {
                **main,
                "images": class_stats["images"],
                "map50": ap50,
                "map50_95": ap5095,
            }
        )
        map50_values.append(ap50)
        map5095_values.append(ap5095)

    valid = [c for c in per_class if c["instances"] > 0]
    macro_p = sum(c["precision"] for c in valid) / len(valid) if valid else 0.0
    macro_r = sum(c["recall"] for c in valid) / len(valid) if valid else 0.0

    return {
        "iou_report": iou_report,
        "metrics": {
            "images": total_images,
            "instances": total_instances,
            "precision": macro_p,
            "recall": macro_r,
            "map50": sum(map50_values) / len(map50_values) if map50_values else 0.0,
            "map50_95": sum(map5095_values) / len(map5095_values) if map5095_values else 0.0,
        },
        "per_class": per_class,
    }


def format_metric(value: float | None) -> str:
    if value is None:
        return " " * 10
    return f"{value:10.3f}"


def print_pad_hole_table(summary: dict[str, Any]) -> None:
    """Print Ultralytics-style table (Class / Images / Instances / Box(P) / R / mAP50 / mAP50-95)."""
    metrics = summary["metrics"]
    print(f"Class{'':>16}Images  Instances      Box(P          R      mAP50  mAP50-95)")
    print(
        f"all{'':>18}"
        f"{metrics['images']:>6}"
        f"{metrics['instances']:>11}"
        f"{format_metric(metrics['precision'])}"
        f"{format_metric(metrics['recall'])}"
        f"{format_metric(metrics['map50'])}"
        f"{format_metric(metrics['map50_95'])}"
    )
    for item in summary["per_class"]:
        print(
            f"{item['class_name']:>22}"
            f"{item['images']:>6}"
            f"{item['instances']:>11}"
            f"{format_metric(item['precision'])}"
            f"{format_metric(item['recall'])}"
            f"{format_metric(item['map50'])}"
            f"{format_metric(item['map50_95'])}"
        )
