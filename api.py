"""
Fingerprint API (CNN pattern classification + Poincaré core/delta + rules + quality gating)

Endpoints:
  - GET  /health
  - GET  /demo
  - POST /detect   (multipart/form-data file=...)

Key pipeline:
  1) Decode grayscale
  2) Structure gate (reject partial/broken/invalid region)
  3) ImageQualityAssessor gate (reject low ridge quality)
  4) CNN classification
  5) Poincaré core/delta detection + ridge counts
  6) Rule validation + rule-based rerank (final class)
  7) Pattern feasibility gate (final class must match delta count constraints)
"""

import os
import base64
import math
import tempfile
import subprocess
import threading
import time
import asyncio
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageDraw, ImageFont
from torchvision import models, transforms

torch.set_num_threads(1)
torch.set_grad_enabled(False)

# Ensure Matplotlib works on server contexts (no GUI backend, writable cache dir)
BASE_DIR = Path(__file__).resolve().parent
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(BASE_DIR / ".mplconfig"))

# -------------------------
# Config
# -------------------------
CLASS_NAMES = ["wpe", "ws", "wd", "we", "lu", "au", "at", "as"]
CHECKPOINT_PATH = (
    BASE_DIR / "results_finetune" / "seed_2025" / "finetune_best_model.pth"
)
JAVA_STREAM_LIMIT = 512_000  # allow long base64 lines from Java live capture

# Quality gate (this is your main "bad ridge / noisy / smudged" guard)
# RELAXED thresholds for zoomed/focused scanner images
QUALITY_THRESH = {
    "min_mean_quality": 0.45,  # Lowered from 0.55 for scanner images
    "min_mean_coherence": 0.60,  # Lowered from 0.66
}
COHERENCE_THRESH = {"reject": 0.40, "warn": 0.50}  # VERY RELAXED for scanner images (was 0.60/0.65)

# Structure gate thresholds (focus on "valid fingerprint region exists", not fine quality)
STRUCT_THRESH = {
    "min_fg_ratio": 0.35,
    # IMPORTANT:
    # Some valid inked/rolled prints can have very high FG after Otsu.
    # So we keep a *soft* max here and only hard-reject when extremely high
    # AND there is real smudge/noise evidence.
    "max_fg_ratio_soft": 0.985,
    "max_fg_ratio_hard": 0.995,
    # Fragmentation (large components)
    "max_components_large": 8,
    "min_main_component_share": 0.70,
    # Center coverage (centroid/ROI center)
    "warn_center_fg_ratio": 0.12,  # warn only (do not reject by itself)
    "hard_roi_center_fg_ratio": 0.10,
    # Edge density in FG: treat as warning unless extreme
    "min_edge_density_fg": 0.015,
    "warn_edge_density_fg": 0.22,  # Raised from 0.18
    "max_edge_density_fg": 0.35,  # RELAXED from 0.24 - scanner images are sharper/more focused
    # Span / aspect
    "min_span": 0.40,
    "max_aspect": 2.2,
    # Speckle evidence
    "max_small_components": 150,
    "max_small_area_ratio": 0.10,
    # Smudge-specific (used with quality/edge evidence)
    "smudge_fg_ratio": 0.970,  # Raised from 0.965
    "smudge_edge_density_fg": 0.35,  # Match max_edge_density_fg
}

# Relaxed thresholds for live scanner (scanner images naturally have higher edge density)
STRUCT_THRESH_LIVE_SCANNER = {
    "min_fg_ratio": 0.30,  # More lenient
    "max_fg_ratio_soft": 0.990,  # Higher tolerance
    "max_fg_ratio_hard": 0.998,  # Higher tolerance
    "max_components_large": 10,  # Allow more components
    "min_main_component_share": 0.65,  # More lenient
    "warn_center_fg_ratio": 0.10,  # More lenient
    "hard_roi_center_fg_ratio": 0.08,  # More lenient
    # RELAXED edge density for scanner (key change)
    "min_edge_density_fg": 0.010,  # More lenient
    "warn_edge_density_fg": 0.26,  # Higher threshold
    "max_edge_density_fg": 0.40,  # VERY RELAXED: scanner images have sharper edges and are more focused
    "min_span": 0.35,  # More lenient
    "max_aspect": 2.4,  # More lenient
    "max_small_components": 200,  # Allow more noise
    "max_small_area_ratio": 0.15,  # More lenient
    # Relaxed smudge detection for scanner
    "smudge_fg_ratio": 0.980,  # Higher threshold
    "smudge_edge_density_fg": 0.40,  # RELAXED to match max_edge_density_fg
}

# Pattern consistency rules (core/delta count range)
PATTERN_RULES = {
    "ws": {"cores": (1, 1), "deltas": (1, 2)},
    "we": {"cores": (1, 1), "deltas": (1, 2)},
    "wd": {"cores": (2, 2), "deltas": (1, 2)},
    "wpe": {"cores": (1, 1), "deltas": (1, 2)},
    # LU/AU structurally 1 core + 1 delta; ridge count decides between them
    "lu": {"cores": (1, 1), "deltas": (1, 1)},
    "au": {"cores": (1, 1), "deltas": (1, 1)},
    "as": {"cores": (0, 0), "deltas": (0, 0)},
    "at": {"cores": (0, 0), "deltas": (0, 0)},
}

WHORL = {"wpe", "ws", "wd", "we"}
LOOP = {"lu", "au"}
ARCH = {"as", "at"}
FINAL_POLICY = (
    "prefer_rule_consistent_if_conflict"  # or "reject_if_conflict" / "prefer_cnn"
)

WARNING_TEMPLATES = {
    "edge_high": {
        "possible_causes": [
            "Dirty or oily sensor",
            "Scratches on sensor surface",
            "Excessive finger pressure",
            "Over-sharpened or high-contrast capture",
            "Dry skin causing broken ridge edges",
        ],
        "suggestions": [
            "Clean the sensor or camera lens",
            "Reduce finger pressure slightly",
            "Re-capture with natural finger contact",
            "Avoid wiping finger right before capture",
        ],
    },
    "low_roi_center_coverage": {
        "possible_causes": [
            "Finger not centered",
            "Partial fingertip captured",
            "Camera misalignment",
        ],
        "suggestions": [
            "Center the fingertip in the capture area",
            "Ensure the full fingertip is visible",
            "Re-align finger before capture",
        ],
    },
    "fragmentation_suspect": {
        "possible_causes": [
            "Very dry skin causing ridge breaks",
            "Light pressure leading to disconnected ridges",
            "Low contrast capture",
            "Uneven illumination",
        ],
        "suggestions": [
            "Increase finger pressure slightly",
            "Moisturize finger lightly if permitted",
            "Ensure even lighting",
        ],
    },
    "speckle_noise": {
        "possible_causes": [
            "Dust particles",
            "Sensor scratches",
            "Background texture leaking into foreground",
            "JPEG artifacts from low-quality compression",
        ],
        "suggestions": [
            "Clean sensor",
            "Use lossless formats (BMP/PNG)",
            "Re-capture under stable lighting",
        ],
    },
    "low_edge_density": {
        "possible_causes": [
            "Very light pressure",
            "Over-exposure or washed-out ridges",
            "Motion blur",
            "Extremely dry or wet finger",
        ],
        "suggestions": [
            "Increase pressure slightly",
            "Hold finger still",
            "Adjust lighting or exposure",
        ],
    },
    "elongated_partial": {
        "possible_causes": [
            "Only side of fingertip captured",
            "Finger tilted",
            "Partial contact",
        ],
        "suggestions": [
            "Place fingertip flat",
            "Avoid rolling finger",
            "Capture full pad of fingertip",
        ],
    },
    "quality_borderline": {
        "possible_causes": [
            "Mixed noise and blur",
            "Slight motion",
            "Uneven ridge clarity",
        ],
        "suggestions": [
            "Re-capture for higher confidence",
            "Clean sensor and finger",
            "Improve lighting",
        ],
    },
    "cnn_rules_disagree": {
        "possible_causes": [
            "Ambiguous fingerprint pattern",
            "Extra core/delta detected due to noise",
            "Transitional pattern between classes",
        ],
        "suggestions": [
            "Trust rule-adjusted result for structural consistency",
            "Re-capture if classification is critical",
        ],
    },
    "coherence_low": {
        "possible_causes": [
            "Slight finger movement during capture",
            "Uneven finger pressure",
            "Dry or wet skin condition",
        ],
        "suggestions": [
            "Hold finger still during capture",
            "Apply even pressure",
            "Re-capture if higher accuracy is required",
        ],
    },
}

# -------------------------
# Imports from your poincare_index.py
# -------------------------
from poincare_index import (
    ImprovedFingerprintDetector,
    ImageQualityAssessor,
    visualize_detection,
)


# -------------------------
# Torch / Model
# -------------------------
def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class ApplyCLAHE:
    def __init__(
        self, clip_limit: float = 2.0, tile_grid_size: Tuple[int, int] = (8, 8)
    ):
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size

    def __call__(self, img: Image.Image) -> Image.Image:
        gray = img.convert("L")
        np_img = np.array(gray)
        clahe = cv2.createCLAHE(
            clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size
        )
        enhanced = clahe.apply(np_img)
        return Image.fromarray(enhanced)


def build_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            ApplyCLAHE(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ]
    )


def create_model(device: torch.device) -> nn.Module:
    model = models.resnet18(weights=None)

    # convert conv1 to 1-channel
    original_conv1 = model.conv1
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    with torch.no_grad():
        model.conv1.weight.copy_(original_conv1.weight.mean(dim=1, keepdim=True))

    model.fc = nn.Linear(model.fc.in_features, len(CLASS_NAMES))
    return model.to(device)


def decode_image_bytes(image_bytes: bytes) -> np.ndarray:
    np_data = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_data, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Invalid image data (cannot decode).")
    return img


def preprocess_for_model(
    image_bytes: bytes, transform: transforms.Compose, device: torch.device
) -> torch.Tensor:
    gray = decode_image_bytes(image_bytes)
    pil_img = Image.fromarray(gray)
    tensor = transform(pil_img).unsqueeze(0).to(device)
    return tensor


def classify_image(
    image_bytes: bytes,
    model: nn.Module,
    transform: transforms.Compose,
    device: torch.device,
) -> Dict[str, Any]:
    inputs = preprocess_for_model(image_bytes, transform, device)
    with torch.no_grad():
        outputs = model(inputs)
        probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
    pred_idx = int(np.argmax(probs))
    return {
        "predicted_class": CLASS_NAMES[pred_idx],
        "confidence": float(probs[pred_idx]),
        "probabilities": {CLASS_NAMES[i]: float(p) for i, p in enumerate(probs)},
    }


# -------------------------
# Serialization
# -------------------------
def serialize_points(points: List[List[float]]) -> List[Dict[str, Any]]:
    out = []
    for y, x, pi, conf in points:
        out.append(
            {"y": float(y), "x": float(x), "pi": float(pi), "confidence": float(conf)}
        )
    return out


def serialize_poincare_result(result: Dict[str, Any]) -> Dict[str, Any]:
    base = {"success": bool(result.get("success", False))}
    if not base["success"]:
        base["error"] = result.get("error", "detection failed")
        return base

    base.update(
        {
            "num_cores": int(result.get("num_cores", 0)),
            "num_deltas": int(result.get("num_deltas", 0)),
            "cores": serialize_points(result.get("cores", [])),
            "deltas": serialize_points(result.get("deltas", [])),
        }
    )

    if "ridge_counts" in result:
        base["ridge_counts"] = [int(rc) for rc in result.get("ridge_counts", [])]

    if "ridge_count_details" in result:
        details = []
        for d in result.get("ridge_count_details", []):
            details.append(
                {
                    "core": [float(d["core"][0]), float(d["core"][1])],
                    "delta": [float(d["delta"][0]), float(d["delta"][1])],
                    "ridge_count": int(d["ridge_count"]),
                }
            )
        base["ridge_count_details"] = details

    return base


# -------------------------
# Rules
# -------------------------
def _in_range(v: int, r: Tuple[int, int]) -> bool:
    return r[0] <= v <= r[1]


def get_min_ridge_count(poincare_serialized: Dict[str, Any]) -> int:
    details = poincare_serialized.get("ridge_count_details", [])
    if not details:
        return 999
    return min(int(d.get("ridge_count", 999)) for d in details)


def validate_by_rules(
    pred_class: str, poincare_serialized: Dict[str, Any]
) -> Dict[str, Any]:
    rule = PATTERN_RULES.get(pred_class)
    n_core = int(poincare_serialized.get("num_cores", 0))
    n_delta = int(poincare_serialized.get("num_deltas", 0))

    if rule is None:
        return {"rule_pass": True, "reason": "no_rule_defined"}

    core_ok = _in_range(n_core, rule["cores"])
    delta_ok = _in_range(n_delta, rule["deltas"])

    return {
        "rule_pass": bool(core_ok and delta_ok),
        "core_ok": bool(core_ok),
        "delta_ok": bool(delta_ok),
        "expected_cores": list(rule["cores"]),
        "expected_deltas": list(rule["deltas"]),
        "observed_cores": n_core,
        "observed_deltas": n_delta,
    }


def rule_fit_score(cls: str, n_core: int, n_delta: int) -> float:
    rule = PATTERN_RULES.get(cls)
    if rule is None:
        return 0.0

    score = 0.0
    cmin, cmax = rule["cores"]
    dmin, dmax = rule["deltas"]

    if n_core < cmin:
        score -= (cmin - n_core) * 2.0
    if n_core > cmax:
        score -= (n_core - cmax) * 2.0
    if n_delta < dmin:
        score -= (dmin - n_delta) * 2.0
    if n_delta > dmax:
        score -= (n_delta - dmax) * 2.0

    return score


def rerank_with_rules(
    classification: Dict[str, Any],
    poincare_serialized: Dict[str, Any],
    lam: float = 0.8,
) -> Dict[str, Any]:
    probs = classification.get("probabilities", {})
    n_core = int(poincare_serialized.get("num_cores", 0))
    n_delta = int(poincare_serialized.get("num_deltas", 0))
    min_ridge = get_min_ridge_count(poincare_serialized)  # 999 if unknown

    ranked = []
    for cls, p in probs.items():
        p = max(float(p), 1e-9)
        rscore = rule_fit_score(cls, n_core, n_delta)

        # LU/AU disambiguation by ridge count (only if ridge is known)
        if min_ridge != 999:
            if cls == "au":
                rscore += 3.0 if min_ridge <= 5 else -3.0
            if cls == "lu":
                rscore += 3.0 if min_ridge > 5 else -3.0

        final = math.log(p) + lam * rscore
        ranked.append((final, cls, p, rscore))

    ranked.sort(reverse=True, key=lambda x: x[0])
    best = ranked[0]
    return {
        "rule_adjusted_class": best[1],
        "rule_adjusted_confidence": best[2],
        "lambda": lam,
        "evidence": {
            "num_cores": n_core,
            "num_deltas": n_delta,
            "min_ridge_count": min_ridge,
        },
        "top5": [
            {"class": cls, "model_prob": p, "rule_score": rs, "final_score": sc}
            for (sc, cls, p, rs) in ranked[:5]
        ],
    }


def best_rule_consistent_class(
    classification: Dict[str, Any], poincare_serialized: Dict[str, Any], top_k: int = 5
) -> Optional[Dict[str, Any]]:
    probs = classification.get("probabilities", {})
    n_core = int(poincare_serialized.get("num_cores", 0))
    n_delta = int(poincare_serialized.get("num_deltas", 0))
    min_ridge = get_min_ridge_count(poincare_serialized)

    sorted_probs = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
    feasible = []
    for cls, p in sorted_probs:
        rule = PATTERN_RULES.get(cls)
        if not rule:
            continue
        core_ok = rule["cores"][0] <= n_core <= rule["cores"][1]
        delta_ok = rule["deltas"][0] <= n_delta <= rule["deltas"][1]
        if not (core_ok and delta_ok):
            continue
        if cls == "lu" and min_ridge != 999 and min_ridge <= 5:
            continue
        if cls == "au" and min_ridge != 999 and min_ridge > 5:
            continue
        feasible.append((cls, float(p)))

    if not feasible:
        return None
    feasible.sort(key=lambda x: x[1], reverse=True)
    top_cls, top_p = feasible[0]
    return {"class": top_cls, "prob": top_p, "top_k": top_k}


# -------------------------
# Structure gate (FIXED)
# -------------------------
def _safe_div(a: float, b: float) -> float:
    return float(a) / float(max(1e-9, b))


def _fg_ratio_in_window(
    mask: np.ndarray, cx: int, cy: int, win_w: int, win_h: int
) -> float:
    h, w = mask.shape
    x1 = max(0, cx - win_w // 2)
    y1 = max(0, cy - win_h // 2)
    x2 = min(w, cx + win_w // 2)
    y2 = min(h, cy + win_h // 2)
    roi = mask[y1:y2, x1:x2]
    if roi.size == 0:
        return 0.0
    return float((roi > 0).sum()) / float(roi.size)


def safe_float(val: Any) -> Optional[float]:
    try:
        v = float(val)
        if not np.isfinite(v):
            return None
        return v
    except Exception:
        return None


def build_warning(code: str, message: str) -> Dict[str, Any]:
    base = {"code": code, "message": message}
    tpl = WARNING_TEMPLATES.get(code)
    if tpl:
        base["possible_causes"] = tpl.get("possible_causes", [])
        base["suggestions"] = tpl.get("suggestions", [])
    return base


def structure_gate(
    gray: np.ndarray,
    use_scanner_thresholds: bool = False,
) -> Tuple[bool, List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    """
    Purpose:
      - Verify there's a valid fingerprint region (not empty / extremely partial / extremely fragmented).
    
    Args:
      - gray: Grayscale fingerprint image
      - use_scanner_thresholds: If True, use relaxed thresholds for live scanner images
      - Do NOT be overly strict on noise/smudge — that's what mean_quality is for.

    Key fixes:
      - Use centroid-based center-coverage instead of ROI geometric center only.
      - Don't reject just because component counts are large; require "noise evidence".
    """
    h, w = gray.shape
    total = max(1, h * w)

    # Otsu normal/invert and pick higher FG ratio mask
    _, mask_inv = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    _, mask_norm = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    fg_ratio_inv = float((mask_inv > 0).sum()) / total
    fg_ratio_norm = float((mask_norm > 0).sum()) / total

    mask = mask_inv if fg_ratio_inv >= fg_ratio_norm else mask_norm

    # Morphology (slightly stronger close to reconnect ridge breaks)
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8), iterations=1
    )
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1
    )

    fg_ratio = float((mask > 0).sum()) / total

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask, connectivity=8
    )
    reasons: List[Dict[str, Any]] = []
    warnings: List[Dict[str, Any]] = []

    if num_labels <= 1:
        reasons.append(
            {
                "code": "no_fingerprint_region",
                "message": "No foreground components found.",
            }
        )
        metrics = {"fg_ratio_gate": fg_ratio}
        return False, reasons, warnings, metrics

    # component areas (exclude background)
    areas = stats[1:, cv2.CC_STAT_AREA]
    num_components_all = int(len(areas))

    min_component_area = max(80, int(0.0003 * h * w))

    large_areas = (
        areas[areas >= min_component_area]
        if areas.size
        else np.array([], dtype=np.int32)
    )
    num_components_large = int(len(large_areas))

    # Share of largest among "large" (fallback to all)
    if large_areas.size:
        main_component_share = float(large_areas.max()) / max(
            1.0, float(large_areas.sum())
        )
    else:
        main_component_share = float(areas.max()) / max(1.0, float(areas.sum()))

    # Small fragments evidence
    small_areas = (
        areas[areas < min_component_area]
        if areas.size
        else np.array([], dtype=np.int32)
    )
    small_components = int(len(small_areas))
    fg_area = int((mask > 0).sum())
    small_area_sum = int(small_areas.sum()) if small_areas.size else 0
    small_area_ratio = _safe_div(small_area_sum, fg_area)

    # Main component bbox + centroid (by max area)
    largest_idx = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    x, y, bw, bh, _ = stats[largest_idx]
    cx, cy = centroids[largest_idx]
    cx_i, cy_i = int(round(cx)), int(round(cy))

    roi_bbox = {"x": int(x), "y": int(y), "w": int(bw), "h": int(bh)}

    # Center coverage:
    # - roi_center (geometric center of bbox) can fail when finger is off-center
    # - centroid_center is robust
    roi_center_fg_ratio = 0.0
    roi = mask[y : y + bh, x : x + bw]
    if roi.size > 0:
        rx1, rx2 = int(0.3 * bw), int(0.7 * bw)
        ry1, ry2 = int(0.3 * bh), int(0.7 * bh)
        roi_center = roi[ry1:ry2, rx1:rx2]
        if roi_center.size > 0:
            roi_center_fg_ratio = float((roi_center > 0).sum()) / float(roi_center.size)

    # centroid window size relative to main ROI
    win = int(0.35 * min(max(bw, 1), max(bh, 1)))
    win = max(40, min(win, int(0.8 * min(w, h))))
    centroid_fg_ratio = _fg_ratio_in_window(mask, cx_i, cy_i, win, win)

    center_fg_ratio_effective = max(roi_center_fg_ratio, centroid_fg_ratio)

    # span on full mask
    x_span = y_span = 0.0
    if (mask > 0).any():
        ys, xs = np.where(mask > 0)
        x_span = float(xs.max() - xs.min()) / max(1.0, float(w))
        y_span = float(ys.max() - ys.min()) / max(1.0, float(h))

    # aspect on main ROI
    aspect = 0.0
    if bw > 0 and bh > 0:
        aspect = max(bw / max(1, bh), bh / max(1, bw))

    edges = cv2.Canny(gray, 50, 150)

    fg = (mask > 0).astype(np.uint8)
    fg_area = int(fg.sum())
    edge_fg = int(((edges > 0) & (fg > 0)).sum())
    edge_density_fg = float(edge_fg) / max(1, fg_area)
    edge_density_all = float((edges > 0).sum()) / total

    # (optional) global center metric for debugging
    y0, y1 = int(0.3 * h), int(0.7 * h)
    x0, x1 = int(0.3 * w), int(0.7 * w)
    center_fg_ratio_global = float((mask[y0:y1, x0:x1] > 0).sum()) / max(
        1, (y1 - y0) * (x1 - x0)
    )

    # Use relaxed thresholds for live scanner if requested
    t = STRUCT_THRESH_LIVE_SCANNER if use_scanner_thresholds else STRUCT_THRESH

    # Basic region existence
    if fg_ratio < t["min_fg_ratio"]:
        reasons.append(
            {
                "code": "fg_too_small",
                "message": "Foreground too small (very partial fingerprint).",
            }
        )

    # High-FG is common for good inked/rolled prints; treat as warning unless combined with other evidence.
    if fg_ratio > t["max_fg_ratio_soft"]:
        warnings.append(
            build_warning(
                "fg_high",
                f"Foreground ratio is high (fg_ratio={fg_ratio:.3f}). Normal for inked prints; will only reject if combined with smudge/noise evidence.",
            )
        )

    if x_span < t["min_span"] or y_span < t["min_span"]:
        reasons.append(
            {
                "code": "incomplete_span",
                "message": f"Insufficient span (x={x_span:.2f}, y={y_span:.2f}).",
            }
        )
    if aspect > t["max_aspect"]:
        reasons.append(
            {
                "code": "elongated_partial",
                "message": f"Main ROI too elongated (aspect={aspect:.2f}).",
            }
        )

    # Edge density (FG): low -> blank/washed out; high -> noisy/smudged (warn unless extreme)
    if edge_density_fg < t["min_edge_density_fg"]:
        reasons.append(
            {
                "code": "low_edge_density",
                "message": "Edge density too low (weak ridges / blank image).",
            }
        )
    elif edge_density_fg > t["max_edge_density_fg"]:
        reasons.append(
            {
                "code": "edge_too_dense",
                "message": f"Edges extremely dense in FG (edge_density_fg={edge_density_fg:.3f}) -> likely noisy/smudged.",
            }
        )
    elif edge_density_fg > t["warn_edge_density_fg"]:
        warnings.append(
            build_warning(
                "edge_high",
                f"Edges dense in FG (edge_density_fg={edge_density_fg:.3f}) — may be noisy/smudged.",
            )
        )

    # Fragmentation: ONLY reject if large-components count is high AND we have noise evidence.
    fragmentation_suspect = num_components_large > t["max_components_large"]
    noise_evidence = (small_area_ratio > t["max_small_area_ratio"]) or (
        small_components > t["max_small_components"]
    )
    if fragmentation_suspect and noise_evidence:
        reasons.append(
            {
                "code": "fragmented_print",
                "message": (
                    f"Fragmented/broken: many large components (large={num_components_large}) "
                    f"+ noise evidence (small_area_ratio={small_area_ratio:.3f}, small_components={small_components})."
                ),
            }
        )
    elif fragmentation_suspect:
        warnings.append(
            build_warning(
                "fragmentation_suspect",
                "Many large components detected (possible fragmentation).",
            )
        )

    # Fragmentation plus very low center coverage => hard reject
    if fragmentation_suspect and center_fg_ratio_effective < 0.20:
        reasons.append(
            {
                "code": "missing_core_region",
                "message": "Fragmentation with very low core-region coverage (likely missing core/delta region).",
            }
        )
        reasons.append(
            {
                "code": "fragmented_print",
                "message": "Fragmented print with insufficient center coverage.",
            }
        )

    if (
        small_components > t["max_small_components"]
        or small_area_ratio > t["max_small_area_ratio"]
    ):
        warnings.append(
            build_warning(
                "speckle_noise",
                "High amount of tiny fragments (possible dust/scratches/compression).",
            )
        )

    # Main component share: still meaningful for true fragmentation/partial
    if main_component_share < t["min_main_component_share"]:
        reasons.append(
            {
                "code": "broken_or_partial",
                "message": f"Main component share too small (share={main_component_share:.3f}).",
            }
        )

    # Center coverage (centroid/ROI center) is informational/warn-only here
    if center_fg_ratio_effective < t["warn_center_fg_ratio"]:
        warnings.append(
            build_warning(
                "low_roi_center_coverage",
                f"Center coverage is low (effective={center_fg_ratio_effective:.3f}).",
            )
        )

    # Smudge evidence flags (used later with quality)
    smudge_flags = {
        "fg_high": fg_ratio > t.get("smudge_fg_ratio", 0.965),
        "edges_extreme": edge_density_fg
        > t.get("smudge_edge_density_fg", t["max_edge_density_fg"]),
        "speckle": (small_area_ratio > t["max_small_area_ratio"])
        or (small_components > t["max_small_components"]),
        "fragmented": fragmentation_suspect,
    }

    metrics: Dict[str, Any] = {
        "fg_ratio_gate": float(fg_ratio),
        "num_components_all": int(num_components_all),
        "num_components_large": int(num_components_large),
        "min_component_area": float(min_component_area),
        "main_component_share": float(main_component_share),
        "edge_density_all": float(edge_density_all),
        "edge_density_fg": float(edge_density_fg),
        "smudge_evidence": smudge_flags,
        "center_fg_ratio": float(center_fg_ratio_global),
        "roi_center_fg_ratio": float(roi_center_fg_ratio),
        "centroid_fg_ratio": float(centroid_fg_ratio),
        "center_fg_ratio_effective": float(center_fg_ratio_effective),
        "centroid": {"x": float(cx), "y": float(cy), "win": int(win)},
        "roi_bbox": roi_bbox,
        "x_span": float(x_span),
        "y_span": float(y_span),
        "main_component_aspect": float(aspect),
        "small_components": int(small_components),
        "small_area_ratio": float(small_area_ratio),
    }

    return (len(reasons) == 0), reasons, warnings, metrics


# -------------------------
# Live Capture Session Manager
# -------------------------
class LiveCaptureSession:
    """Manages a live fingerprint scanning session"""
    def __init__(self):
        self.process: Optional[subprocess.Popen] = None
        self.latest_frame: Optional[bytes] = None
        self.latest_quality: int = 0
        self.is_running: bool = False
        self.error: Optional[str] = None
        self.lock = threading.Lock()
        self.reader_thread: Optional[threading.Thread] = None
        
    def start(self) -> bool:
        """Start live capture process"""
        if self.is_running:
            return True
            
        try:
            # Build classpath (match test_java_capture.py)
            java_dir = BASE_DIR / "java_ capture"
            zkfp_jar = BASE_DIR / "ZKFinger Standard SDK 5.3.0.33" / "Java" / "lib" / "ZKFingerReader.jar"
            
            if not zkfp_jar.exists():
                self.error = "ZKFinger SDK not found"
                return False
            
            # Cross-platform PATH separator
            path_sep = ";" if os.name == "nt" else ":"
            cp = f".{path_sep}{zkfp_jar.absolute()}"
            
            # Set environment
            env = os.environ.copy()
            java_dir_abs = java_dir.absolute()
            env["PATH"] = f"{java_dir_abs}{path_sep}{env.get('PATH', '')}"
            
            # Start Java process
            self.process = subprocess.Popen(
                [
                    "java",
                    "--enable-native-access=ALL-UNNAMED",
                    "-cp",
                    cp,
                    "LiveCapture",
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=str(java_dir_abs),
                env=env,
                bufsize=1,
            )
            
            # Wait for READY signal
            ready_line = self.process.stdout.readline().strip()
            if not ready_line.startswith("READY:"):
                stderr = self.process.stderr.read()
                self.error = f"Failed to start: {stderr}"
                return False
                
            self.is_running = True
            
            # Start reader thread
            self.reader_thread = threading.Thread(target=self._read_frames, daemon=True)
            self.reader_thread.start()
            
            return True
            
        except Exception as e:
            self.error = str(e)
            return False
            
    def _read_frames(self):
        """Background thread to read frames from Java process"""
        try:
            while self.is_running and self.process:
                line = self.process.stdout.readline()
                if not line:
                    break
                    
                line = line.strip()
                
                if line.startswith("FRAME:"):
                    # Parse frame: FRAME:<base64>:<quality>
                    parts = line.split(":", 2)
                    if len(parts) >= 3:
                        try:
                            frame_data = base64.b64decode(parts[1])
                            quality = int(parts[2]) if len(parts) > 2 else 0
                            
                            with self.lock:
                                self.latest_frame = frame_data
                                self.latest_quality = quality
                        except Exception:
                            pass
                            
                elif line.startswith("IDLE"):
                    with self.lock:
                        self.latest_frame = None
                        self.latest_quality = 0
                        
                elif line.startswith("ERROR:"):
                    self.error = line[6:]
                    break
                    
        except Exception as e:
            self.error = str(e)
        finally:
            self.is_running = False
            
    def get_latest_frame(self) -> Tuple[Optional[bytes], int]:
        """Get the latest captured frame"""
        with self.lock:
            return self.latest_frame, self.latest_quality
            
    def stop(self):
        """Stop the live capture session"""
        self.is_running = False
        if self.process:
            try:
                self.process.terminate()
                self.process.wait(timeout=5)
            except Exception:
                try:
                    self.process.kill()
                except Exception:
                    pass
            self.process = None
        self.latest_frame = None
        self.latest_quality = 0

# Global live capture session
live_session: Optional[LiveCaptureSession] = None

# -------------------------
# App + Load artifacts
# -------------------------
app = FastAPI(title="Fingerprint Analysis API", version="2.0.0")

# Ensure results directory exists (important in Docker/Render)
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Mount static files for serving saved failure images
app.mount("/results", StaticFiles(directory=str(RESULTS_DIR)), name="results")

DEVICE = get_device()
TRANSFORM = build_transform()
MODEL: Optional[nn.Module] = None

def get_model() -> nn.Module:
    global MODEL
    if MODEL is None:
        if not CHECKPOINT_PATH.is_file():
            raise RuntimeError(f"Checkpoint not found at {CHECKPOINT_PATH}")
        model = create_model(DEVICE)
        state = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
        model.load_state_dict(state)
        model.eval()
        MODEL = model
    return MODEL

detector = ImprovedFingerprintDetector(block_size=10)
quality_assessor = ImageQualityAssessor(block_size=10)


# -------------------------
# Helpers
# -------------------------
def reject_422(
    *,
    code: str,
    message: str,
    filename: Optional[str],
    content_type: Optional[str],
    reject_stage: str,  # NEW: "structure" | "quality" | "poincare" | "feasibility"
    reject_detail: Dict[str, Any],  # NEW: any extra details
    structure_passed: bool,
    structure_reasons: List[Dict[str, Any]],
    structure_warnings: List[Dict[str, Any]],
    structure_metrics: Dict[str, Any],
    quality: Dict[str, Any],
    extra: Optional[Dict[str, Any]] = None,
) -> JSONResponse:
    reasons = (
        structure_reasons
        if structure_reasons
        else [
            {
                "code": code if code else "unspecified",
                "message": (
                    message if message else "No structural rejection reason (bug)."
                ),
            }
        ]
    )

    payload: Dict[str, Any] = {
        "success": False,
        "error": code,
        "message": message,
        "reject_stage": reject_stage,
        "reject_detail": reject_detail,
        "filename": filename,
        "content_type": content_type,
        "structure": {
            "passed": bool(structure_passed),
            "reasons": reasons,
            "warnings": structure_warnings,
            "metrics": structure_metrics,
        },
        "quality": quality,
        "action_hint": [
            "Place the full fingertip flat on the sensor/camera.",
            "Increase pressure slightly to capture full ridges.",
            "Clean the sensor/camera lens and recapture to avoid smudges.",
        ],
    }
    if extra:
        payload.update(extra)
    return JSONResponse(status_code=200, content=payload)


def maybe_save_failure(
    gray: np.ndarray, reason_code: str, original_name: str
) -> Dict[str, Any]:
    out_dir = BASE_DIR / "results" / "quality_fail" / reason_code
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    stem = Path(original_name or "upload").stem
    name = f"{stem}_{ts}.png"
    path = out_dir / name
    try:
        cv2.imwrite(str(path), gray)
    except Exception:
        pass
    return {
        "saved_path": str(path),
        "url": f"/results/quality_fail/{reason_code}/{name}",
    }


def generate_overlay_base64(result: Dict[str, Any], src_path: str, block_size: int = 10) -> Optional[str]:
    """Generate overlay using matplotlib visualize_detection (2-panel with orientation field)
    
    Falls back to OpenCV overlay if matplotlib fails (common in server environments)
    """
    tmp_path: Optional[str] = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp_path = tmp.name

        visualize_detection(
            result,
            src_path,
            tmp_path,
            block_size=block_size,
        )

        with open(tmp_path, "rb") as f:
            encoded = base64.b64encode(f.read()).decode("utf-8")

        os.remove(tmp_path)
        return encoded
    except Exception as e:
        # Matplotlib might fail in server environment, fall back to OpenCV overlay
        print(f"[WARNING] Matplotlib overlay failed: {e}. Using OpenCV fallback.")
        if tmp_path:
            try:
                os.remove(tmp_path)
            except OSError:
                pass
        
        # Try OpenCV overlay as fallback
        try:
            gray = decode_image_bytes(open(src_path, 'rb').read())
            return generate_overlay_base64_opencv(gray, result)
        except Exception as e2:
            print(f"[ERROR] OpenCV overlay also failed: {e2}")
        return None


def generate_overlay_base64_opencv(gray: np.ndarray, result: Dict[str, Any]) -> Optional[str]:
    """Lightweight overlay (no Matplotlib rendering) with Matplotlib-like label styling."""
    try:
        if gray is None or gray.ndim != 2 or not result.get("success"):
            return None

        img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        h, w = img.shape[:2]

        # Match Matplotlib overlay palette as closely as possible
        core_color = (0, 0, 255)     # red (BGR)
        delta_color = (255, 0, 0)    # blue (BGR)
        ridge_line_color = (0, 215, 255)  # yellow
        ridge_point_color = (0, 215, 255)
        ridge_linepoint_color = (0, 215, 255)

        # --- Text + bbox drawing (PIL, for consistent font/rounded boxes) ---
        # We'll draw geometry with OpenCV first, then render labels with PIL on top.
        def _get_font(size: int, bold: bool = True):
            """Use PIL default font to avoid heavy font lookups."""
            try:
                return ImageFont.load_default()
            except Exception:
                return ImageFont.load_default()

        labels: List[Dict[str, Any]] = []

        def _queue_label(
            x: int,
            y: int,
            text: str,
            text_rgb: tuple[int, int, int],
            *,
            font_size: int = 14,
            padding: int = 6,
            box_rgba: tuple[int, int, int, int] = (255, 255, 255, 204),
            anchor: str = "mm",
        ) -> None:
            labels.append(
                {
                    "x": x,
                    "y": y,
                    "text": text,
                    "text_rgb": text_rgb,
                    "font_size": font_size,
                    "padding": padding,
                    "box_rgba": box_rgba,
                    "anchor": anchor,
                }
            )

        def _draw_core(points: List[List[float]]) -> None:
            for y, x, _, conf in points:
                y_i, x_i = int(round(y)), int(round(x))
                if not (0 <= x_i < w and 0 <= y_i < h):
                    continue
                # More visible markers
                r = int(10 + float(conf) * 8)
                cv2.circle(img, (x_i, y_i), r, core_color, 2, lineType=cv2.LINE_AA)
                # Add small center dot for precise location
                cv2.circle(img, (x_i, y_i), 3, (0, 0, 255), -1, lineType=cv2.LINE_AA)
                _queue_label(
                    x_i,
                    y_i - r - 12,
                    f"CORE\n{float(conf):.2f}",
                    (255, 0, 0),
                    font_size=12,
                    padding=5,
                    anchor="mb",
                )

        def _draw_delta(points: List[List[float]]) -> None:
            for y, x, _, conf in points:
                y_i, x_i = int(round(y)), int(round(x))
                if not (0 <= x_i < w and 0 <= y_i < h):
                    continue
                # More visible triangle markers
                s = int(10 + float(conf) * 8)
                pts = np.array([[x_i, y_i - s], [x_i - s, y_i + s], [x_i + s, y_i + s]], np.int32)
                cv2.polylines(img, [pts], isClosed=True, color=delta_color, thickness=2, lineType=cv2.LINE_AA)
                # Add small center dot for precise location
                cv2.circle(img, (x_i, y_i), 3, (255, 0, 0), -1, lineType=cv2.LINE_AA)
                _queue_label(
                    x_i,
                    y_i - s - 12,
                    f"DELTA\n{float(conf):.2f}",
                    (0, 0, 255),
                    font_size=12,
                    padding=5,
                    anchor="mb",
                )

        _draw_core(result.get("cores", []))
        _draw_delta(result.get("deltas", []))

        for d in result.get("ridge_count_details", []):
            core = d.get("core")
            delta = d.get("delta")
            if not core or not delta:
                continue
            cy, cx = core
            dy, dx = delta
            cy, cx, dy, dx = int(round(cy)), int(round(cx)), int(round(dy)), int(round(dx))
            line_points = d.get("line_points", [])
            ridge_positions = d.get("ridge_positions", [])
            
            # Draw SOLID LINE first (Bresenham path) for visibility
            if line_points and len(line_points) > 1:
                # Convert line_points to numpy array for polyline
                pts = np.array([[int(round(pt[1])), int(round(pt[0]))] for pt in line_points], np.int32)
                cv2.polylines(img, [pts], isClosed=False, color=(0, 200, 0), thickness=2, lineType=cv2.LINE_AA)
                
                # Draw ridge detection points (yellow circles) on top
                for ridge_idx in ridge_positions:
                    if ridge_idx < len(line_points):
                        ly, lx = int(round(line_points[ridge_idx][0])), int(round(line_points[ridge_idx][1]))
                        # Draw larger, more visible markers for detected ridges
                        cv2.circle(img, (lx, ly), 4, (0, 255, 255), -1, lineType=cv2.LINE_AA)  # Yellow filled
                        cv2.circle(img, (lx, ly), 5, (0, 165, 255), 2, lineType=cv2.LINE_AA)  # Orange outline
            else:
                # Fallback: simple line
                cv2.line(img, (cx, cy), (dx, dy), ridge_line_color, 2, lineType=cv2.LINE_AA)

            rc = d.get("ridge_count")
            if rc is not None:
                # Calculate perpendicular offset for label positioning
                line_dx = dx - cx
                line_dy = dy - cy
                line_len = np.sqrt(line_dx**2 + line_dy**2) + 1e-6
                perp_x = -line_dy / line_len
                perp_y = line_dx / line_len
                
                # Position label to the side of the line
                mid_x = (cx + dx) // 2
                mid_y = (cy + dy) // 2
                offset = 25  # pixels to the side
                label_x = int(mid_x + perp_x * offset)
                label_y = int(mid_y + perp_y * offset)
                
                _queue_label(
                    label_x,
                    label_y,
                    f"RC: {int(rc)}",
                    (255, 255, 255),
                    font_size=13,
                    padding=6,
                    box_rgba=(0, 150, 0, 230),
                    anchor="mm",
                )

        # Render queued labels on top of the OpenCV geometry
        pil_rgba = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).convert("RGBA")
        draw = ImageDraw.Draw(pil_rgba)

        def _draw_label_now(item: Dict[str, Any]) -> None:
            x = int(item["x"])
            y = int(item["y"])
            text = str(item["text"])
            text_rgb = item["text_rgb"]
            font_size = int(item["font_size"])
            padding = int(item["padding"])
            box_rgba = item["box_rgba"]
            anchor = item["anchor"]

            font = _get_font(font_size, bold=True)
            bbox = draw.multiline_textbbox(
                (0, 0), text, font=font, spacing=2, align="center"
            )
            tw = bbox[2] - bbox[0]
            th = bbox[3] - bbox[1]

            if anchor == "mm":
                tx0 = int(x - tw / 2)
                ty0 = int(y - th / 2)
            elif anchor == "mb":
                tx0 = int(x - tw / 2)
                ty0 = int(y - th)
            else:
                tx0 = int(x)
                ty0 = int(y)

            rx0 = tx0 - padding
            ry0 = ty0 - padding
            rx1 = tx0 + tw + padding
            ry1 = ty0 + th + padding

            draw.rounded_rectangle((rx0, ry0, rx1, ry1), radius=6, fill=box_rgba)
            draw.multiline_text(
                (tx0, ty0),
                text,
                font=font,
                fill=(text_rgb[0], text_rgb[1], text_rgb[2], 255),
                spacing=2,
                align="center",
            )

        for item in labels:
            _draw_label_now(item)

        img = cv2.cvtColor(np.array(pil_rgba.convert("RGB")), cv2.COLOR_RGB2BGR)

        ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        if not ok:
            return None
        return base64.b64encode(buf.tobytes()).decode("utf-8")
    except Exception:
        try:
            if os.environ.get("OVERLAY_DEBUG") == "1":
                import traceback

                print("generate_overlay_base64_opencv failed:\n", traceback.format_exc())
        except Exception:
            pass
        return None


# def generate_overlay_base64_opencv(gray: np.ndarray, result: Dict[str, Any]) -> Optional[str]:
#     """Lightweight overlay (OpenCV + PIL) that mimics the Matplotlib-style demo image."""
#     print("[OVERLAY] using OPENCV overlay")

#     try:
#         if gray is None or gray.ndim != 2 or not result.get("success"):
#             return None

#         # --- base image ---
#         img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
#         h, w = img.shape[:2]

#         # ---------- Style constants (match screenshot) ----------
#         # BGR for OpenCV drawing
#         CORE_BGR  = (0, 0, 255)        # red circle
#         DELTA_BGR = (255, 0, 0)        # blue triangle
#         YELLOW_BGR = (0, 215, 255)     # dotted ridge points (gold/yellow)
#         GREEN_RGBA = (0, 128, 0, 230)  # RC box (PIL RGBA)
#         WHITE_BOX = (255, 255, 255, 210)

#         # PIL text colors (RGB)
#         CORE_TXT = (255, 0, 0)         # red
#         DELTA_TXT = (0, 0, 255)        # blue
#         BLACK_TXT = (0, 0, 0)
#         WHITE_TXT = (255, 255, 255)

#         # ---------- Font helper ----------
#         def _get_font(size: int, bold: bool = True):
#             try:
#                 from matplotlib.font_manager import FontProperties, findfont
#                 prop = FontProperties(family="DejaVu Sans", weight=("bold" if bold else "normal"))
#                 font_path = findfont(prop, fallback_to_default=True)
#                 return ImageFont.truetype(font_path, size=size)
#             except Exception:
#                 return ImageFont.load_default()

#         labels: List[Dict[str, Any]] = []

#         def _queue_label(
#             x: int,
#             y: int,
#             text: str,
#             text_rgb: tuple[int, int, int],
#             *,
#             font_size: int = 16,
#             padding: int = 8,
#             box_rgba: tuple[int, int, int, int] = WHITE_BOX,
#             anchor: str = "mb",
#             radius: int = 10,
#         ):
#             labels.append({
#                 "x": x, "y": y, "text": text, "text_rgb": text_rgb,
#                 "font_size": font_size, "padding": padding,
#                 "box_rgba": box_rgba, "anchor": anchor, "radius": radius
#             })

#         # ---------- Draw cores/deltas (geometry first) ----------
#         def _draw_core(points: List[List[float]]) -> None:
#             for y, x, _, conf in points:
#                 y_i, x_i = int(round(y)), int(round(x))
#                 if not (0 <= x_i < w and 0 <= y_i < h):
#                     continue
#                 r = int(16 + float(conf) * 8)  # bigger like screenshot
#                 cv2.circle(img, (x_i, y_i), r, CORE_BGR, 3, lineType=cv2.LINE_AA)
#                 _queue_label(
#                     x_i,
#                     max(0, y_i - r - 10),
#                     f"CORE\n{float(conf):.2f}",
#                     CORE_TXT,
#                     font_size=18,
#                     padding=10,
#                     box_rgba=WHITE_BOX,
#                     anchor="mb",
#                     radius=10,
#                 )

#         def _draw_delta(points: List[List[float]]) -> None:
#             for y, x, _, conf in points:
#                 y_i, x_i = int(round(y)), int(round(x))
#                 if not (0 <= x_i < w and 0 <= y_i < h):
#                     continue
#                 s = int(16 + float(conf) * 8)
#                 pts = np.array(
#                     [[x_i, y_i - s], [x_i - s, y_i + s], [x_i + s, y_i + s]],
#                     np.int32
#                 )
#                 cv2.polylines(img, [pts], True, DELTA_BGR, 3, lineType=cv2.LINE_AA)
#                 _queue_label(
#                     x_i,
#                     max(0, y_i - s - 10),
#                     f"DELTA\n{float(conf):.2f}",
#                     DELTA_TXT,
#                     font_size=18,
#                     padding=10,
#                     box_rgba=WHITE_BOX,
#                     anchor="mb",
#                     radius=10,
#                 )

#         _draw_core(result.get("cores", []))
#         _draw_delta(result.get("deltas", []))

#         # ---------- Ridge count dotted lines + RC labels ----------
#         for d in result.get("ridge_count_details", []):
#             core = d.get("core")
#             delta = d.get("delta")
#             if not core or not delta:
#                 continue
#             cy, cx = int(round(core[0])), int(round(core[1]))
#             dy, dx = int(round(delta[0])), int(round(delta[1]))

#             line_points = d.get("line_points", [])
#             if line_points:
#                     _draw_thick_ridge_band(img, line_points, thickness=18)
#                 # screenshot uses bigger yellow dots
#                 # for pt in line_points:
#                 #     ly, lx = int(round(pt[0])), int(round(pt[1]))
#                 #     if 0 <= lx < w and 0 <= ly < h:
#                 #         cv2.circle(img, (lx, ly), 5, YELLOW_BGR, -1, lineType=cv2.LINE_AA)
#                 #         # tiny darker outline for contrast
#                 #         cv2.circle(img, (lx, ly), 5, (0, 140, 200), 1, lineType=cv2.LINE_AA)
#             else:
#                 # fallback: draw a dotted approximation
#                 steps = 24
#                 for t in np.linspace(0, 1, steps):
#                     lx = int(round(cx + (dx - cx) * t))
#                     ly = int(round(cy + (dy - cy) * t))
#                     cv2.circle(img, (lx, ly), 5, YELLOW_BGR, -1, lineType=cv2.LINE_AA)
#                     cv2.circle(img, (lx, ly), 5, (0, 140, 200), 1, lineType=cv2.LINE_AA)

#             cv2.circle(img, (cx, cy), 6, (0, 215, 255), -1)
#             cv2.circle(img, (dx, dy), 6, (0, 215, 255), -1)

#             rc = d.get("ridge_count")
#             if rc is not None:
#                 mid_idx = len(line_points) // 2
#                 my, mx = int(line_points[mid_idx][0]), int(line_points[mid_idx][1])
#                 _queue_label(   
#                     mx,
#                     my,
#                 f"RC: {int(rc)}",
#                 (255, 255, 255), font_size=18, padding=10, box_rgba=(0, 128, 0, 230), anchor="mm",)

#                 # mid_x, mid_y = (cx + dx) // 2, (cy + dy) // 2
#                 # _queue_label(
#                 #     mid_x,
#                 #     mid_y,
#                 #     f"RC: {int(rc)}",
#                 #     WHITE_TXT,
#                 #     font_size=18,
#                 #     padding=10,
#                 #     box_rgba=GREEN_RGBA,
#                 #     anchor="mm",
#                 #     radius=10,
#                 # )

#         # ---------- PIL layer for text/boxes ----------
#         pil_rgba = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).convert("RGBA")
#         draw = ImageDraw.Draw(pil_rgba)

#         def _draw_label(item: Dict[str, Any]) -> None:
#             x = int(item["x"]); y = int(item["y"])
#             text = str(item["text"])
#             text_rgb = item["text_rgb"]
#             font_size = int(item["font_size"])
#             padding = int(item["padding"])
#             box_rgba = item["box_rgba"]
#             anchor = item["anchor"]
#             radius = int(item["radius"])

#             font = _get_font(font_size, bold=True)

#             bbox = draw.multiline_textbbox((0, 0), text, font=font, spacing=2, align="center")
#             tw = bbox[2] - bbox[0]
#             th = bbox[3] - bbox[1]

#             if anchor == "mm":
#                 tx0 = int(x - tw / 2); ty0 = int(y - th / 2)
#             elif anchor == "mb":
#                 tx0 = int(x - tw / 2); ty0 = int(y - th)
#             else:
#                 tx0 = int(x); ty0 = int(y)

#             rx0 = tx0 - padding
#             ry0 = ty0 - padding
#             rx1 = tx0 + tw + padding
#             ry1 = ty0 + th + padding

#             # Clamp to canvas
#             rx0 = max(0, rx0); ry0 = max(0, ry0)
#             rx1 = min(w - 1, rx1); ry1 = min(h - 1, ry1)

#             draw.rounded_rectangle((rx0, ry0, rx1, ry1), radius=radius, fill=box_rgba)

#             # small shadow to match screenshot readability
#             shadow = (0, 0, 0, 120)
#             draw.multiline_text((tx0 + 1, ty0 + 1), text, font=font, fill=shadow, spacing=2, align="center")
#             draw.multiline_text((tx0, ty0), text, font=font, fill=(text_rgb[0], text_rgb[1], text_rgb[2], 255),
#                                 spacing=2, align="center")

#         # ---------- Title (top center) ----------
#         title1 = "Structural Correction (Ridge-Aligned Cores)"
#         # If you have filename in result, use it; else omit
#         fname = result.get("filename", "")
#         title2 = str(fname) if fname else ""

#         title_font = _get_font(22, bold=True)
#         sub_font = _get_font(18, bold=False)

#         def _center_text(y: int, text: str, font):
#             if not text:
#                 return
#             bb = draw.textbbox((0, 0), text, font=font)
#             tw = bb[2] - bb[0]
#             x = (w - tw) // 2
#             # slight shadow
#             draw.text((x + 1, y + 1), text, font=font, fill=(0, 0, 0, 120))
#             draw.text((x, y), text, font=font, fill=(0, 0, 0, 255))

#         _center_text(18, title1, title_font)
#         _center_text(48, title2, sub_font)

#         # Draw queued labels
#         for item in labels:
#             _draw_label(item)

#         # ---------- Bottom summary banner ----------
#         cores_n = len(result.get("cores", []))
#         deltas_n = len(result.get("deltas", []))
#         avg_rc = result.get("avg_ridge_count")
#         if avg_rc is None:
#             # compute from ridge_count_details if available
#             rcs = [d.get("ridge_count") for d in result.get("ridge_count_details", []) if d.get("ridge_count") is not None]
#             avg_rc = (sum(rcs) / len(rcs)) if rcs else None

#         avg_str = f"{avg_rc:.1f}" if isinstance(avg_rc, (int, float)) else "N/A"
#         summary = f"Cores: {cores_n} | Deltas: {deltas_n} | Avg Ridge Count: {avg_str}"

#         banner_font = _get_font(22, bold=True)
#         bb = draw.textbbox((0, 0), summary, font=banner_font)
#         tw = bb[2] - bb[0]
#         th = bb[3] - bb[1]

#         pad_x = 18
#         pad_y = 10
#         bx0 = (w - tw) // 2 - pad_x
#         by1 = h - 18
#         by0 = by1 - th - 2 * pad_y
#         bx1 = (w + tw) // 2 + pad_x

#         # yellow with black border like screenshot
#         YELLOW_BOX = (255, 235, 100, 235)
#         draw.rounded_rectangle((bx0, by0, bx1, by1), radius=12, fill=YELLOW_BOX, outline=(0, 0, 0, 255), width=3)

#         tx = (w - tw) // 2
#         ty = by0 + pad_y
#         draw.text((tx + 1, ty + 1), summary, font=banner_font, fill=(0, 0, 0, 120))
#         draw.text((tx, ty), summary, font=banner_font, fill=(0, 0, 0, 255))

#         # ---------- Encode (JPG saves memory) ----------
#         out_bgr = cv2.cvtColor(np.array(pil_rgba.convert("RGB")), cv2.COLOR_RGB2BGR)
#         ok, buf = cv2.imencode(".jpg", out_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
#         if not ok:
#             return None
#         return base64.b64encode(buf.tobytes()).decode("utf-8")

#     except Exception:
#         if os.environ.get("OVERLAY_DEBUG") == "1":
#             import traceback
#             print("generate_overlay_base64_opencv failed:\n", traceback.format_exc())
#         return None

# def _draw_thick_ridge_band(img, pts, thickness=16):
#     """
#     Draw a thick filled yellow ridge-count band between core and delta.
#     pts: [(y, x), ...] ordered from core → delta
#     """
#     if len(pts) < 2:
#         return

#     band_color = (0, 215, 255)  # yellow (BGR)
#     outline = (0, 160, 210)     # darker yellow edge

#     for i in range(len(pts) - 1):
#         y1, x1 = int(pts[i][0]), int(pts[i][1])
#         y2, x2 = int(pts[i + 1][0]), int(pts[i + 1][1])

#         # main thick segment
#         cv2.line(
#             img,
#             (x1, y1),
#             (x2, y2),
#             band_color,
#             thickness,
#             lineType=cv2.LINE_AA,
#         )

#         # subtle darker outline
#         cv2.line(
#             img,
#             (x1, y1),
#             (x2, y2),
#             outline,
#             thickness // 2,
#             lineType=cv2.LINE_AA,
#         )


def pattern_expected_deltas(cls: str) -> Optional[int]:
    if cls in WHORL:
        return None  # handled by rule ranges (1-2)
    if cls in LOOP:
        return 1
    if cls in ARCH:
        return 0
    return None


def pattern_expected_cores(cls: str) -> Optional[int]:
    """Get expected core count for a pattern class"""
    if cls == "wd":  # Whorl Double Loop
        return 2
    if cls in {"ws", "we", "wpe"}:  # Other whorls
        return 1
    if cls in LOOP:  # LU, AU
        return 1
    if cls in ARCH:  # AS, AT
        return 0
    return None


def infer_class_from_rules(n_core: int, n_delta: int, min_ridge: int) -> Optional[str]:
    candidates = ["wd", "ws", "we", "wpe", "lu", "au", "at", "as"]
    for cls in candidates:
        rule = PATTERN_RULES.get(cls)
        if not rule:
            continue
        cmin, cmax = rule["cores"]
        dmin, dmax = rule["deltas"]
        if not (cmin <= n_core <= cmax and dmin <= n_delta <= dmax):
            continue
        if cls == "lu" and min_ridge != 999 and min_ridge <= 5:
            continue
        if cls == "au" and min_ridge != 999 and min_ridge > 5:
            continue
        return cls
    return None


# -------------------------
# Routes
# -------------------------
@app.get("/health")
async def health():
    return {
        "status": "ok",
        "device": str(DEVICE),
        "checkpoint": str(CHECKPOINT_PATH),
        "classes": CLASS_NAMES,
        "quality_gate": QUALITY_THRESH,
        "structure_gate": STRUCT_THRESH,
    }


@app.get("/")
async def root():
    return {"status": "ok", "message": "See /docs or /demo"}


@app.get("/demo")
async def demo():
    demo_file = BASE_DIR / "demo.html"
    if not demo_file.exists():
        raise HTTPException(status_code=404, detail="demo.html not found.")
    return FileResponse(demo_file)


@app.post("/live_scan/start")
async def start_live_scan():
    """Start a live scanning session"""
    global live_session
    
    # Stop any existing session
    if live_session:
        live_session.stop()
        
    live_session = LiveCaptureSession()
    
    if not live_session.start():
        error = live_session.error or "Failed to start scanner"
        live_session = None
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": error}
        )
        
    return JSONResponse(content={"success": True, "message": "Live scan started"})


@app.get("/live_scan/frame")
async def get_live_frame():
    """Get the latest frame from live scanning"""
    global live_session
    
    if not live_session or not live_session.is_running:
        return JSONResponse(
            status_code=404,
            content={"success": False, "error": "No active scanning session"}
        )
        
    if live_session.error:
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": live_session.error}
        )
        
    frame, quality = live_session.get_latest_frame()
    
    if frame is None:
        return JSONResponse(content={
            "success": True,
            "status": "waiting",
            "message": "No finger detected. Please place finger on scanner."
        })
        
    # Convert frame to base64
    frame_b64 = base64.b64encode(frame).decode('utf-8')
    
    return JSONResponse(content={
        "success": True,
        "status": "capturing",
        "frame": frame_b64,
        "quality": quality,
        "acceptable": quality >= 50  # Threshold for acceptable quality
    })


@app.post("/live_scan/capture")
async def capture_from_live_scan():
    """Capture and analyze the current frame from live scanning"""
    global live_session
    
    if not live_session or not live_session.is_running:
        return JSONResponse(
            status_code=404,
            content={"success": False, "error": "No active scanning session"}
        )
        
    frame, quality = live_session.get_latest_frame()
    
    if frame is None:
        return JSONResponse(
            status_code=400,
            content={"success": False, "error": "No frame available. Place finger on scanner."}
        )
        
    if quality < 30:
        return JSONResponse(
            status_code=400,
            content={
                "success": False,
                "error": "Image quality too low. Press firmer and hold still.",
                "quality": quality
            }
        )
        
    # Process the frame through the same pipeline as /detect
    try:
        # Decode the BMP frame
        gray = decode_image_bytes(frame)
    except ValueError as e:
        return JSONResponse(
            status_code=400,
            content={"success": False, "error": f"Invalid image: {str(e)}"}
        )
        
    # Save captured image permanently for reference
    capture_dir = RESULTS_DIR / "live_captures"
    capture_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    saved_filename = f"capture_{timestamp}.bmp"
    saved_path = capture_dir / saved_filename
    
    # Write to saved file
    with open(saved_path, 'wb') as f:
        f.write(frame)
    
    # Also write to temp file for processing
    with tempfile.NamedTemporaryFile(delete=False, suffix=".bmp") as tmp:
        tmp.write(frame)
        tmp_path = tmp.name
        
    try:
        # Run the same detection pipeline as /detect
        # (Reuse the existing pipeline logic)
        
        # Structure gate (use relaxed thresholds for live scanner)
        gate_passed, gate_reasons, gate_warnings, gate_metrics = structure_gate(gray, use_scanner_thresholds=True)
        warnings_all: List[Dict[str, Any]] = list(gate_warnings)
        
        # Quality assessment
        try:
            quality_result = quality_assessor.assess(tmp_path)
        except Exception as e:
            quality_result = {
                "status": "WARN",
                "reasons": [{"code": "quality_error", "message": str(e)}],
                "metrics": {},
            }
            
        if not gate_passed:
            reason_code = gate_reasons[0]["code"] if gate_reasons else "structure_invalid"
            return JSONResponse(content={
                "success": False,
                "error": "Invalid fingerprint structure",
                "stage": "structure_gate",
                "reasons": gate_reasons,
                "warnings": gate_warnings,
                "metrics": gate_metrics,
                "quality": quality_result,
                "captured_image_url": f"/results/live_captures/{saved_filename}",
            })
            
        # Quality gate
        qmetrics = quality_result.get("metrics", {}) or {}
        mean_quality = safe_float(qmetrics.get("mean_quality"))
        mean_coh = safe_float(qmetrics.get("mean_coherence"))
        
        min_q = QUALITY_THRESH["min_mean_quality"]
        if mean_quality is not None and mean_quality < min_q:
            return JSONResponse(content={
                "success": False,
                "error": f"Quality too low (mean_quality={mean_quality:.3f})",
                "stage": "quality_gate",
                "quality": quality_result,
                "captured_image_url": f"/results/live_captures/{saved_filename}",
            })
            
        # Coherence gate (added to match /detect endpoint)
        if mean_coh is not None and mean_coh < COHERENCE_THRESH["reject"]:
            return JSONResponse(content={
                "success": False,
                "error": f"Coherence too low (mean_coherence={mean_coh:.3f})",
                "stage": "quality_gate",
                "quality": quality_result,
                "captured_image_url": f"/results/live_captures/{saved_filename}",
            })
            
        # CNN classification
        try:
            model = get_model()
            classification = classify_image(frame, model, TRANSFORM, DEVICE)
        except Exception as e:
            return JSONResponse(
                status_code=500,
                content={"success": False, "error": f"Classification failed: {e}"}
            )
            
        # Poincaré detection
        result = detector.detect(tmp_path)
        poincare_serialized = serialize_poincare_result(result)
        
        overlay_base64: Optional[str] = None
        if poincare_serialized.get("success"):
            overlay_base64 = generate_overlay_base64(result, tmp_path, block_size=10)
            
        if not poincare_serialized.get("success"):
            return JSONResponse(content={
                "success": True,
                "warnings": [{"code": "poincare_failed", "message": "Core/delta detection failed"}],
                "classification": classification,
                "final": {
                    "class": classification["predicted_class"],
                    "confidence": classification["confidence"],
                    "source": "model_only",
                },
                "quality": quality_result,
                "structure": {
                    "passed": True,
                    "reasons": [],
                    "warnings": gate_warnings,
                    "metrics": gate_metrics,
                },
                "poincare": poincare_serialized,
                "overlay_base64": overlay_base64,
                "captured_image_url": f"/results/live_captures/{saved_filename}",
                "captured_image_path": str(saved_path),
            })
            
        # Rules + rerank
        pred_cls = classification["predicted_class"]
        rule_validation = validate_by_rules(pred_cls, poincare_serialized)
        rule_rerank = rerank_with_rules(classification, poincare_serialized, lam=0.8)
        
        cnn_prob = float(classification["probabilities"].get(pred_cls, classification["confidence"]))
        rule_pick = best_rule_consistent_class(classification, poincare_serialized, top_k=5)
        conflict = (not rule_validation.get("rule_pass", True) and rule_pick is not None and rule_pick["class"] != pred_cls)
        
        final_cls = pred_cls
        final_conf = cnn_prob
        final_source = "cnn"
        
        if conflict and FINAL_POLICY == "prefer_rule_consistent_if_conflict" and rule_pick is not None:
            final_cls = rule_pick["class"]
            final_conf = rule_pick["prob"]
            final_source = "rules"
            
        final = {"class": final_cls, "confidence": float(final_conf), "source": final_source}
        
        # Build response
        n_core = int(poincare_serialized.get("num_cores", 0))
        n_delta = int(poincare_serialized.get("num_deltas", 0))
        min_ridge = get_min_ridge_count(poincare_serialized)
        rule_suggested = infer_class_from_rules(n_core, n_delta, min_ridge)
        rule_match = rule_suggested == pred_cls if rule_suggested else True
        
        response_data = {
            "success": True,
            "classification": classification,
            "final": final,
            "quality": quality_result,
            "structure": {
                "passed": True,
                "reasons": [],
                "warnings": gate_warnings,
                "metrics": gate_metrics,
            },
            "rule_validation": rule_validation,
            "rule_rerank": rule_rerank,
            "cnn": {
                "class": pred_cls,
                "confidence": classification["confidence"],
                "probabilities": classification["probabilities"],
            },
            "rule_inference": {
                "suggested_class": rule_suggested,
                "evidence": {"cores": n_core, "deltas": n_delta, "min_ridge": min_ridge},
                "agree_with_cnn": rule_match,
            },
            "warnings": warnings_all,
            "overlay_base64": overlay_base64,
            # Add saved capture image URL
            "captured_image_url": f"/results/live_captures/{saved_filename}",
            "captured_image_path": str(saved_path),
        }
        response_data.update(poincare_serialized)
        
        return JSONResponse(content=response_data)
        
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


@app.post("/live_scan/stop")
async def stop_live_scan():
    """Stop the live scanning session"""
    global live_session
    
    if live_session:
        live_session.stop()
        live_session = None
        
    return JSONResponse(content={"success": True, "message": "Live scan stopped"})



@app.post("/detect")
async def detect_fingerprint(
    file: UploadFile = File(..., description="Fingerprint image (bmp/png/jpg/jpeg)"),
):

    print("DETECT called: model_loaded =", MODEL is not None)

    # Read upload ONCE (UploadFile stream is consumed after reading)
    content = await file.read()
    print("Upload:", file.filename, "bytes:", len(content))



    if not file:
        raise HTTPException(status_code=400, detail="No file uploaded.")

    if file.content_type not in {
        "image/bmp",
        "image/png",
        "image/jpeg",
    } and not file.filename.lower().endswith((".bmp", ".png", ".jpg", ".jpeg")):
        raise HTTPException(
            status_code=400, detail="Unsupported file type. Use bmp/png/jpg/jpeg."
        )

    if not content:
        raise HTTPException(status_code=400, detail="Empty file.")

    # Decode once (for structure checks and saving fails)
    try:
        gray = decode_image_bytes(content)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Write to temp for poincare + quality assessor
    suffix = Path(file.filename).suffix or ".png"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    try:
        # 1) Structure gate (use relaxed scanner thresholds for uploaded images too)
        gate_passed, gate_reasons, gate_warnings, gate_metrics = structure_gate(gray, use_scanner_thresholds=True)
        warnings_all: List[Dict[str, Any]] = list(gate_warnings)

        # 2) Quality assessor (always computed so you can debug in rejects)
        try:
            quality = quality_assessor.assess(tmp_path)
        except Exception as e:
            quality = {
                "status": "WARN",
                "reasons": [{"code": "quality_error", "message": str(e)}],
                "metrics": {},
            }

        if not gate_passed:
            reason_code = (
                gate_reasons[0]["code"] if gate_reasons else "structure_invalid"
            )
            artifacts = maybe_save_failure(gray, reason_code, file.filename or "upload")
            return reject_422(
                code="structure_invalid",
                message="Invalid fingerprint – please recapture (partial/broken/invalid region).",
                filename=file.filename,
                content_type=file.content_type,
                reject_stage="structure_gate",
                reject_detail={},
                structure_passed=False,
                structure_reasons=gate_reasons,
                structure_warnings=gate_warnings,
                structure_metrics=gate_metrics,
                quality=quality,
                extra={
                    "analysis_artifacts": artifacts,
                    "reject_stage": "structure_gate",
                },
            )

        # 3) Quality gate (main noise/smudge guard)
        qmetrics = quality.get("metrics", {}) or {}
        mean_quality = safe_float(qmetrics.get("mean_quality"))
        mean_coh = safe_float(qmetrics.get("mean_coherence"))
        quality_available = (mean_quality is not None) and (mean_coh is not None)

        min_q = QUALITY_THRESH["min_mean_quality"]
        if mean_quality is not None and mean_quality < min_q:
            artifacts = maybe_save_failure(
                gray, "quality_too_low", file.filename or "upload"
            )
            return reject_422(
                code="quality_too_low",
                message=f"Invalid fingerprint – please recapture (low ridge quality, mean_quality={mean_quality:.3f}, threshold={min_q:.2f}).",
                filename=file.filename,
                content_type=file.content_type,
                reject_stage="quality_gate",
                reject_detail={
                    "mean_quality": mean_quality,
                    "threshold": min_q,
                    "quality_metrics": qmetrics,
                },
                structure_passed=True,
                structure_reasons=[],
                structure_warnings=gate_warnings,
                structure_metrics=gate_metrics,
                quality=quality,
                extra={"analysis_artifacts": artifacts, "reject_stage": "quality_gate"},
            )

        if mean_coh is not None and mean_coh < COHERENCE_THRESH["reject"]:
            artifacts = maybe_save_failure(
                gray, "coherence_too_low", file.filename or "upload"
            )
            return reject_422(
                code="coherence_too_low",
                message=f"Invalid fingerprint – please recapture (low ridge coherence, mean_coherence={mean_coh:.3f}).",
                filename=file.filename,
                content_type=file.content_type,
                reject_stage="quality_gate",
                reject_detail={
                    "mean_coherence": mean_coh,
                    "threshold": COHERENCE_THRESH["reject"],
                    "quality_metrics": qmetrics,
                },
                structure_passed=True,
                structure_reasons=[],
                structure_warnings=gate_warnings,
                structure_metrics=gate_metrics,
                quality=quality,
                extra={"analysis_artifacts": artifacts, "reject_stage": "quality_gate"},
            )
        elif mean_coh is not None and mean_coh < COHERENCE_THRESH["warn"]:
            warnings_all.append(
                build_warning(
                    "coherence_low",
                    f"Ridge flow is less consistent (mean_coherence={mean_coh:.3f}).",
                )
            )
        elif mean_coh is None or mean_quality is None:
            warnings_all.append(
                build_warning(
                    "quality_missing",
                    "Quality metrics unavailable; skipping quality-based smudge check.",
                )
            )

        # High-foreground smudge composite (after quality/coherence known)
        t_struct = STRUCT_THRESH
        fg = float(gate_metrics.get("fg_ratio_gate", 0.0))
        ed_fg = float(gate_metrics.get("edge_density_fg", 0.0))
        small_area_ratio = float(gate_metrics.get("small_area_ratio", 0.0))
        small_components = int(gate_metrics.get("small_components", 0))
        num_large = int(gate_metrics.get("num_components_large", 0))

        fg_high = fg >= t_struct["smudge_fg_ratio"]
        edges_extreme = ed_fg >= t_struct["smudge_edge_density_fg"]

        speckle = (
            small_area_ratio >= t_struct["max_small_area_ratio"]
            or small_components >= t_struct["max_small_components"]
        )

        fragmented_suspect = num_large >= t_struct["max_components_large"]
        fragmented = fragmented_suspect and speckle

        ridges_bad = False
        ridges_mediocre = False
        if quality_available:
            ridges_bad = (mean_quality < QUALITY_THRESH["min_mean_quality"]) or (
                mean_coh < COHERENCE_THRESH["reject"]
            )
            ridges_mediocre = (
                COHERENCE_THRESH["reject"] <= mean_coh < COHERENCE_THRESH["warn"]
            )

        gate_metrics["smudge_evidence"] = {
            "fg_high": fg_high,
            "edges_extreme": edges_extreme,
            "speckle": speckle,
            "fragmented_suspect": fragmented_suspect,
            "fragmented": fragmented,
            "ridges_bad": ridges_bad,
            "ridges_mediocre": ridges_mediocre,
        }

        if fg_high:
            strong_evidences = [edges_extreme, speckle, fragmented, ridges_bad]
            evidence_count = sum(1 for v in strong_evidences if v)
            if edges_extreme and evidence_count >= 2:
                artifacts = maybe_save_failure(
                    gray, "fg_smudged", file.filename or "upload"
                )
                return reject_422(
                    code="fg_smudged",
                    message="Foreground dominates image with smudge/noise evidence. Please recapture.",
                    filename=file.filename,
                    content_type=file.content_type,
                    reject_stage="structure_gate",
                    reject_detail={
                        "fg_ratio_gate": fg,
                        "edge_density_fg": ed_fg,
                        "mean_quality": mean_quality,
                        "mean_coherence": mean_coh,
                    },
                    structure_passed=False,
                    structure_reasons=[
                        {
                            "code": "fg_smudged",
                            "message": "High foreground with smudge/noise evidence.",
                        }
                    ],
                    structure_warnings=gate_warnings,
                    structure_metrics=gate_metrics,
                    quality=quality,
                    extra={
                        "analysis_artifacts": artifacts,
                        "reject_stage": "structure_gate",
                    },
                )
            else:
                if edges_extreme:
                    warnings_all.append(
                        build_warning(
                            "edge_high_fg",
                            f"Edges are dense in FG (edge_density_fg={ed_fg:.3f}); smudge evidence insufficient to reject.",
                        )
                    )
                if ridges_mediocre:
                    warnings_all.append(
                        build_warning(
                            "coherence_low",
                            f"Ridge coherence is low (mean_coherence={mean_coh:.3f}); may affect core/delta reliability.",
                        )
                    )
                warnings_all.append(
                    build_warning(
                        "fg_high",
                        f"Foreground ratio is very high (fg_ratio_gate={fg:.3f}) — smudge evidence insufficient to reject.",
                    )
                )

        center_low = gate_metrics.get("roi_center_fg_ratio", 1.0) < STRUCT_THRESH.get(
            "hard_roi_center_fg_ratio", 0.10
        )
        quality_lowish = mean_quality < (QUALITY_THRESH["min_mean_quality"] + 0.03)
        if center_low and quality_lowish:
            artifacts = maybe_save_failure(
                gray, "center_coverage_low", file.filename or "upload"
            )
            return reject_422(
                code="center_coverage_low",
                message="Invalid fingerprint – core region not visible enough.",
                filename=file.filename,
                content_type=file.content_type,
                reject_stage="structure_gate",
                reject_detail={
                    "roi_center_fg_ratio": gate_metrics.get("roi_center_fg_ratio", 0),
                    "mean_quality": mean_quality,
                },
                structure_passed=False,
                structure_reasons=[
                    {
                        "code": "center_coverage_low",
                        "message": "Core region coverage too low.",
                    }
                ],
                structure_warnings=gate_warnings,
                structure_metrics=gate_metrics,
                quality=quality,
                extra={
                    "analysis_artifacts": artifacts,
                    "reject_stage": "structure_gate",
                },
            )

        # 4) CNN classification
        try:
            model = get_model()
            classification = classify_image(content, model, TRANSFORM, DEVICE)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Classification failed: {e}")

        # 5) Poincaré detection
        result = detector.detect(tmp_path)
        poincare_serialized = serialize_poincare_result(result)

        overlay_base64: Optional[str] = None
        if poincare_serialized.get("success"):
            overlay_base64 = generate_overlay_base64(result, tmp_path, block_size=10)

        if not poincare_serialized.get("success"):
            return JSONResponse(
                status_code=200,
                content={
                    "success": True,
                    "warnings": [
                        {
                            "code": "poincare_failed",
                            "message": "Core/delta detection failed; returning CNN only.",
                        }
                    ],
                    "classification": classification,
                    "final": {
                        "class": classification["predicted_class"],
                        "confidence": classification["confidence"],
                        "source": "model_only",
                    },
                    "quality": quality,
                    "structure": {
                        "passed": True,
                        "reasons": [],
                        "warnings": gate_warnings,
                        "metrics": gate_metrics,
                    },
                    "poincare": poincare_serialized,
                    "overlay_base64": overlay_base64,
                },
            )

        # 6) Rules + rerank => FINAL
        pred_cls = classification["predicted_class"]
        rule_validation = validate_by_rules(pred_cls, poincare_serialized)
        rule_rerank = rerank_with_rules(classification, poincare_serialized, lam=0.8)

        cnn_prob = float(
            classification["probabilities"].get(pred_cls, classification["confidence"])
        )
        rule_pick = best_rule_consistent_class(
            classification, poincare_serialized, top_k=5
        )
        conflict = (
            (not rule_validation.get("rule_pass", True))
            and rule_pick is not None
            and rule_pick["class"] != pred_cls
        )

        explanation = []
        if conflict:
            explanation = [
                f"CNN predicted {pred_cls.upper()} with probability {cnn_prob:.2%}.",
                f"Poincaré detected {poincare_serialized.get('num_cores', 0)} cores and {poincare_serialized.get('num_deltas', 0)} deltas.",
                f"Rule check failed for {pred_cls.upper()} (expected cores {rule_validation.get('expected_cores')}, deltas {rule_validation.get('expected_deltas')}).",
                f"Among top-{rule_pick['top_k']} CNN classes, {rule_pick['class'].upper()} is rule-consistent (prob {rule_pick['prob']:.2%}).",
                "Mismatch can happen due to false core/delta detection or confusion between similar patterns.",
            ]
        else:
            explanation = [
                "CNN prediction is consistent with detected core/delta structure."
            ]

        final_cls = pred_cls
        final_conf = cnn_prob
        final_source = "cnn"
        if (
            conflict
            and FINAL_POLICY == "prefer_rule_consistent_if_conflict"
            and rule_pick is not None
        ):
            final_cls = rule_pick["class"]
            final_conf = rule_pick["prob"]
            final_source = "rules"
        elif conflict and FINAL_POLICY == "reject_if_conflict":
            return reject_422(
                code="ambiguous_pattern",
                message="CNN prediction conflicts with core/delta rules. Please recapture.",
                filename=file.filename,
                content_type=file.content_type,
                reject_stage="feasibility",
                reject_detail={"cnn_pred": pred_cls, "rule_pick": rule_pick},
                structure_passed=True,
                structure_reasons=[],
                structure_warnings=gate_warnings,
                structure_metrics=gate_metrics,
                quality=quality,
                extra={
                    "decision": {
                        "cnn_pred": {"class": pred_cls, "prob": cnn_prob},
                        "rule_validation": rule_validation,
                        "rule_suggestion": rule_pick,
                        "conflict": conflict,
                        "policy": FINAL_POLICY,
                        "explanation": explanation,
                    }
                },
            )

        final = {
            "class": final_cls,
            "confidence": float(final_conf),
            "source": final_source,
        }

        decision = {
            "cnn_pred": {"class": pred_cls, "prob": cnn_prob},
            "rule_validation": rule_validation,
            "rule_suggestion": rule_pick,
            "conflict": conflict,
            "policy": FINAL_POLICY,
            "explanation": explanation,
        }

        # Rule inference summary
        n_core = int(poincare_serialized.get("num_cores", 0))
        n_delta = int(poincare_serialized.get("num_deltas", 0))
        min_ridge = get_min_ridge_count(poincare_serialized)
        rule_suggested = infer_class_from_rules(n_core, n_delta, min_ridge)
        rule_match = rule_suggested == pred_cls if rule_suggested else True
        if n_delta == 0:
            warnings_all.append(
                {
                    "code": "delta_missing",
                    "message": "Structure evidence suggests unknown (no delta detected). Please recapture for higher confidence.",
                }
            )

        # 7) Pattern feasibility gate (USE FINAL CLASS) - Check BOTH cores and deltas
        expected_deltas = pattern_expected_deltas(final_cls)
        expected_cores = pattern_expected_cores(final_cls)
        observed_deltas = int(poincare_serialized.get("num_deltas", 0))
        observed_cores = int(poincare_serialized.get("num_cores", 0))
        ridge_counts = poincare_serialized.get("ridge_counts", [])
        min_ridge = min(ridge_counts) if ridge_counts else None

        violations = []
        # Check delta count
        if expected_deltas is not None and observed_deltas != expected_deltas:
            violations.append(
                {
                    "code": "pattern_feasibility_failed",
                    "message": f"Final class {final_cls} expects {expected_deltas} deltas but detected {observed_deltas}.",
                }
            )
        
        # Check core count
        if expected_cores is not None and observed_cores != expected_cores:
            violations.append(
                {
                    "code": "pattern_feasibility_failed_cores",
                    "message": f"Final class {final_cls} expects {expected_cores} cores but detected {observed_cores}. Pattern classification may be incorrect.",
                }
            )

        if violations:
            response_data: Dict[str, Any] = {
                "success": True,
                "warnings": warnings_all
                + [
                    {
                        "code": "pattern_feasibility_failed",
                        "message": violations[0]["message"],
                    }
                ],
                "classification": classification,
                "final": final,
                "poincare_summary": {
                    "num_cores": int(poincare_serialized.get("num_cores", 0)),
                    "num_deltas": observed_deltas,
                    "ridge_counts_min": min_ridge,
                },
                "rule_validation": rule_validation,
                "rule_rerank": rule_rerank,
                "decision": decision,
                "quality": quality,
                "structure": {
                    "passed": True,
                    "reasons": [],
                    "warnings": gate_warnings,
                    "metrics": gate_metrics,
                },
                "cnn": {
                    "class": pred_cls,
                    "confidence": classification["confidence"],
                    "probabilities": classification["probabilities"],
                },
                "rule_inference": {
                    "suggested_class": rule_suggested,
                    "evidence": {
                        "cores": n_core,
                        "deltas": observed_deltas,
                        "min_ridge": min_ridge,
                    },
                    "agree_with_cnn": rule_match,
                },
                "overlay_base64": overlay_base64,
            }
            response_data.update(poincare_serialized)

            return JSONResponse(status_code=200, content=response_data)

        # Success response
        response_data: Dict[str, Any] = {
            "success": True,
            "classification": classification,
            "final": final,
            "quality": quality,
            "structure": {
                "passed": True,
                "reasons": [],
                "warnings": gate_warnings,
                "metrics": gate_metrics,
            },
            "rule_validation": rule_validation,
            "rule_rerank": rule_rerank,
            "decision": decision,
            "cnn": {
                "class": pred_cls,
                "confidence": classification["confidence"],
                "probabilities": classification["probabilities"],
            },
            "rule_inference": {
                "suggested_class": rule_suggested,
                "evidence": {
                    "cores": n_core,
                    "deltas": observed_deltas,
                    "min_ridge": min_ridge,
                },
                "agree_with_cnn": rule_match,
            },
            "warnings": warnings_all,
            "overlay_base64": overlay_base64,
        }
        response_data.update(poincare_serialized)

        return JSONResponse(response_data)

    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


# Global storage for live capture
_live_capture_data = {"base64": None, "timestamp": None}

@app.get("/scanner/live")
async def live_scanner_stream():
    """
    Start live fingerprint scanner that continuously captures fingerprints.
    Returns Server-Sent Events (SSE) stream with capture notifications.
    """
    async def event_generator():
        # Path to LiveCapture.java
        java_dir = BASE_DIR / "java_ capture"
        jar_path = BASE_DIR / "ZKFinger Standard SDK 5.3.0.33" / "Java" / "lib" / "ZKFingerReader.jar"
        
        # Check if files exist
        if not (java_dir / "LiveCapture.class").exists():
            yield f"data: ERROR:LiveCapture.class not found. Please compile it first.\n\n"
            return
        
        if not jar_path.exists():
            yield f"data: ERROR:ZKFingerReader.jar not found\n\n"
            return
        
        # Check for existing Java processes that might be using the scanner
        # Cross-platform process check
        try:
            import platform
            if platform.system() == "Windows":
                result = subprocess.run(
                    ["tasklist", "/FI", "IMAGENAME eq java.exe"],
                    capture_output=True,
                    text=True,
                    timeout=2
                )
                java_processes = [line for line in result.stdout.split('\n') if 'java.exe' in line.lower()]
            else:
                # Linux/Mac: use ps command
                result = subprocess.run(
                    ["ps", "aux"],
                    capture_output=True,
                    text=True,
                    timeout=2
                )
                java_processes = [line for line in result.stdout.split('\n') if 'java' in line.lower() and 'LiveCapture' not in line]
            
            if len(java_processes) > 1:  # More than just this one
                print(f"WARNING: Found {len(java_processes)} Java processes running. Scanner might be in use.")
        except:
            pass
        
        # Start the Java live capture process
        process = None
        try:
            # Setup environment - add java_capture dir to PATH for native DLL
            env = os.environ.copy()
            java_dir_abs = java_dir.absolute()
            jar_path_abs = jar_path.absolute()
            
            # Cross-platform PATH separator
            path_sep = ";" if os.name == "nt" else ":"
            env["PATH"] = f"{java_dir_abs}{path_sep}{env.get('PATH', '')}"
            
            # Build classpath with absolute paths (cross-platform separator)
            cp = f".{path_sep}{jar_path_abs}"
            
            print(f"Starting LiveCapture with:")
            print(f"  Working dir: {java_dir_abs}")
            print(f"  Classpath: {cp}")
            print(f"  PATH includes: {java_dir_abs}")
            
            # Use asyncio.create_subprocess_exec for async I/O
            # Add --enable-native-access flag to avoid Java warnings and allow native library access
            process = await asyncio.create_subprocess_exec(
                "java",
                "--enable-native-access=ALL-UNNAMED",
                "-cp",
                cp,
                "LiveCapture",
                cwd=str(java_dir_abs),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
                limit=JAVA_STREAM_LIMIT  # match test_java_capture: permit long base64 line
            )
            
            print(f"Started LiveCapture process, PID: {process.pid}")
            
            # Create a task to read stderr in background
            async def log_stderr():
                while True:
                    try:
                        err_line = await process.stderr.readline()
                        if not err_line:
                            break
                        err_msg = err_line.decode('utf-8').strip()
                        if err_msg:
                            print(f"LiveCapture [stderr]: {err_msg}")
                    except:
                        break
            
            # Start stderr logging task
            stderr_task = asyncio.create_task(log_stderr())
            
            # Read output line by line asynchronously
            while True:
                try:
                    # Read line with timeout to avoid hanging
                    line = await asyncio.wait_for(
                        process.stdout.readline(),
                        timeout=30.0  # 30 second timeout
                    )
                    
                    if not line:
                        # EOF reached
                        print("LiveCapture process ended (EOF)")
                        break
                    
                    line = line.decode('utf-8').strip()
                    
                    if line == "INIT":
                        print("LiveCapture: Initializing...")
                        yield f"data: INIT\n\n"
                    elif line == "READY":
                        print("LiveCapture: Ready")
                        yield f"data: READY\n\n"
                    elif line.startswith("OK:"):
                        # Got a fingerprint capture - store it and send notification
                        base64_data = line[3:]  # Remove "OK:" prefix
                        timestamp = datetime.now().isoformat()
                        
                        # Store the capture data globally
                        _live_capture_data["base64"] = base64_data
                        _live_capture_data["timestamp"] = timestamp
                        
                        print(f"LiveCapture: Fingerprint captured, size: {len(base64_data)} chars")
                        
                        # Send just a notification (not the full image)
                        yield f"data: CAPTURED:{timestamp}\n\n"
                        
                    elif line.startswith("ERROR:"):
                        print(f"LiveCapture error: {line}")
                        
                        # Provide helpful error message
                        if "Failed to open fingerprint scanner" in line:
                            error_detail = (
                                "ERROR:Scanner in use or not connected. "
                                "Please:\n"
                                "1. Close any other apps using the scanner\n"
                                "2. Unplug and replug the scanner\n"
                                "3. Wait 5 seconds and try again"
                            )
                            yield f"data: {error_detail}\n\n"
                        elif "No fingerprint scanner detected" in line:
                            yield f"data: ERROR:Scanner not detected. Please connect your ZKTeco scanner and try again.\n\n"
                        else:
                            yield f"data: {line}\n\n"
                        break
                    # Ignore other lines (debug output goes to stderr)
                    
                    # Check if process ended
                    if process.returncode is not None:
                        print(f"LiveCapture process exited with code: {process.returncode}")
                        break
                        
                except asyncio.TimeoutError:
                    # Check if process is still running
                    if process.returncode is not None:
                        print(f"LiveCapture process died (code: {process.returncode})")
                        break
                    # Continue waiting
                    continue
                except asyncio.LimitOverrunError as e:
                    # Line exceeded reader limit; log and retry (should not happen with raised limit)
                    print(f"LiveCapture warning: line too long ({e.consumed} bytes); consider increasing JAVA_STREAM_LIMIT")
                    continue
                    
        except Exception as e:
            error_msg = f"ERROR:Exception - {str(e)}"
            print(f"LiveCapture error: {error_msg}")
            yield f"data: {error_msg}\n\n"
            
        finally:
            # Cancel stderr task
            if 'stderr_task' in locals():
                stderr_task.cancel()
                try:
                    await stderr_task
                except asyncio.CancelledError:
                    pass
            
            # Cleanup process
            if process and process.returncode is None:
                print("Terminating LiveCapture process...")
                try:
                    process.terminate()
                    await asyncio.wait_for(process.wait(), timeout=2.0)
                except asyncio.TimeoutError:
                    print("Force killing LiveCapture process...")
                    process.kill()
                    await process.wait()
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


@app.get("/scanner/latest")
async def get_latest_capture():
    """
    Get the latest captured fingerprint image.
    Returns the base64-encoded BMP image.
    """
    if not _live_capture_data["base64"]:
        raise HTTPException(status_code=404, detail="No capture available")
    
    return JSONResponse({
        "success": True,
        "base64": _live_capture_data["base64"],
        "timestamp": _live_capture_data["timestamp"]
    })


if __name__ == "__main__":
    import uvicorn
    import os

    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
