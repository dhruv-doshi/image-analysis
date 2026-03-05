from __future__ import annotations

import logging
import math
from itertools import permutations

import cv2
import numpy as np
from PIL import Image

from src.models import CompositionScores

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level face cascade (lazy-safe)
# ---------------------------------------------------------------------------

try:
    _face_cascade: cv2.CascadeClassifier | None = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    if _face_cascade.empty():
        _face_cascade = None
except Exception as _exc:
    logger.warning("Could not load face cascade: %s", _exc)
    _face_cascade = None


# ---------------------------------------------------------------------------
# Saliency map via rembg U²-Net
# ---------------------------------------------------------------------------


def _saliency_map(pil_image: Image.Image) -> np.ndarray:
    """Return float32 saliency map in [0, 1] from rembg alpha channel."""
    import time
    t = time.perf_counter()
    try:
        from rembg import remove  # lazy import — heavy dependency

        result = remove(pil_image)
        alpha = np.array(result)[:, :, 3].astype(np.float32) / 255.0
        logger.debug("rembg saliency complete  %.2fs", time.perf_counter() - t)
        return alpha
    except Exception as exc:
        logger.warning("rembg saliency failed, using uniform fallback: %s", exc)
        arr = np.array(pil_image)
        h, w = arr.shape[:2]
        return np.ones((h, w), dtype=np.float32)


# ---------------------------------------------------------------------------
# Centroid
# ---------------------------------------------------------------------------


def _centroid(saliency: np.ndarray) -> tuple[float, float]:
    """Weighted centre of mass normalised to [0, 1]. Returns (0.5, 0.5) if blank."""
    total = saliency.sum()
    if total == 0:
        return 0.5, 0.5
    h, w = saliency.shape
    ys, xs = np.mgrid[0:h, 0:w]
    cx = float((xs * saliency).sum() / total) / w
    cy = float((ys * saliency).sum() / total) / h
    return cx, cy


# ---------------------------------------------------------------------------
# Alignment scores
# ---------------------------------------------------------------------------


def _rot_alignment(cx: float, cy: float) -> float:
    """Min distance to 4 Rule-of-Thirds power points, normalised to [0, 1]."""
    rot_points = [(1 / 3, 1 / 3), (1 / 3, 2 / 3), (2 / 3, 1 / 3), (2 / 3, 2 / 3)]
    min_dist = min(math.hypot(cx - px, cy - py) for px, py in rot_points)
    # Max possible min-distance: corner (0,0) to nearest point (1/3, 1/3)
    max_dist = math.hypot(1 / 3, 1 / 3)
    return min(min_dist / max_dist, 1.0)


def _golden_ratio_alignment(cx: float, cy: float) -> float:
    """Min distance to 4 Golden Ratio power points, normalised to [0, 1]."""
    phi = 0.618
    gr_points = [(1 - phi, 1 - phi), (1 - phi, phi), (phi, 1 - phi), (phi, phi)]
    min_dist = min(math.hypot(cx - px, cy - py) for px, py in gr_points)
    # Max possible min-distance: corner (0,0) to nearest point (0.382, 0.382)
    max_dist = math.hypot(1 - phi, 1 - phi)
    return min(min_dist / max_dist, 1.0)


# ---------------------------------------------------------------------------
# Negative space
# ---------------------------------------------------------------------------


def _negative_space(saliency: np.ndarray, threshold: float = 0.5) -> float:
    """Fraction of pixels below 50% of max saliency (non-salient area)."""
    max_val = saliency.max()
    if max_val == 0:
        return 1.0
    return float((saliency < threshold * max_val).sum() / saliency.size)


# ---------------------------------------------------------------------------
# Visual weight
# ---------------------------------------------------------------------------


def _visual_weight(saliency: np.ndarray) -> tuple[dict, float]:
    """Per-quadrant normalised saliency sums + balance ratio (max/min, 1.0=balanced)."""
    h, w = saliency.shape
    mh, mw = h // 2, w // 2
    quadrants = {
        "top_left": saliency[:mh, :mw],
        "top_right": saliency[:mh, mw:],
        "bottom_left": saliency[mh:, :mw],
        "bottom_right": saliency[mh:, mw:],
    }
    total = saliency.sum()
    if total == 0:
        return dict.fromkeys(quadrants, 0.25), 1.0
    weights = {k: float(q.sum() / total) for k, q in quadrants.items()}
    vals = list(weights.values())
    min_w, max_w = min(vals), max(vals)
    # Cap at 99 when a quadrant is completely empty (avoids float("inf") in JSON)
    balance = max_w / min_w if min_w > 0 else 99.0
    return weights, balance


# ---------------------------------------------------------------------------
# Symmetry
# ---------------------------------------------------------------------------


def _ncc(a: np.ndarray, b: np.ndarray) -> float:
    """Normalised cross-correlation in [-1, 1].

    Returns 1.0 when both arrays are constant (identical flat signals are
    perfectly correlated). Returns 0.0 when only one side is constant.
    """
    a_std, b_std = float(a.std()), float(b.std())
    if a_std == 0 and b_std == 0:
        return 1.0
    if a_std == 0 or b_std == 0:
        return 0.0
    a_norm = (a - a.mean()) / a_std
    b_norm = (b - b.mean()) / b_std
    return float(np.clip(np.mean(a_norm * b_norm), -1.0, 1.0))


def _symmetry(saliency: np.ndarray) -> tuple[float, float]:
    """NCC of left↔right and top↔bottom halves. Returns (h_sym, v_sym) in [-1, 1]."""
    h, w = saliency.shape
    mh, mw = h // 2, w // 2

    # Horizontal symmetry: first mw cols vs last mw cols (mirrored)
    left = saliency[:, :mw].astype(np.float64)
    right = np.fliplr(saliency[:, w - mw :]).astype(np.float64)
    h_sym = _ncc(left, right)

    # Vertical symmetry: first mh rows vs last mh rows (mirrored)
    top = saliency[:mh, :].astype(np.float64)
    bottom = np.flipud(saliency[h - mh :, :]).astype(np.float64)
    v_sym = _ncc(top, bottom)

    return h_sym, v_sym


# ---------------------------------------------------------------------------
# Leading lines — detection + horizon tilt
# ---------------------------------------------------------------------------


def _detect_lines(bgr_array: np.ndarray) -> list[float]:
    """Run Canny + HoughLinesP; return raw angle list in degrees [0, 180)."""
    gray = cv2.cvtColor(bgr_array, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=80,
        minLineLength=100,
        maxLineGap=10,
    )
    if lines is None:
        return []
    return [
        math.degrees(math.atan2(y2 - y1, x2 - x1)) % 180
        for x1, y1, x2, y2 in (line[0] for line in lines)
    ]


def _horizon_tilt(angles: list[float]) -> float | None:
    """Return median signed tilt (degrees) from near-horizontal lines, or None if < 2 found.

    Positive = clockwise tilt, negative = counterclockwise.
    """
    horizontal: list[float] = []
    for a in angles:
        if a < 20:
            horizontal.append(a)         # slight clockwise → positive
        elif a > 160:
            horizontal.append(a - 180)   # slight counterclockwise → negative
    if len(horizontal) < 2:
        return None
    return float(np.median(horizontal))


def _classify_pattern(angles: list[float]) -> str:
    """Classify dominant line pattern from angles in degrees [0, 180)."""
    if not angles:
        return "none"
    total = len(angles)
    horizontal = sum(1 for a in angles if a < 20 or a > 160)
    vertical = sum(1 for a in angles if 70 < a < 110)
    diagonal = total - horizontal - vertical
    h_frac = horizontal / total
    v_frac = vertical / total
    d_frac = diagonal / total
    if h_frac > 0.5:
        return "horizontal"
    if v_frac > 0.5:
        return "vertical"
    if d_frac > 0.5:
        return "diagonal"
    return "mixed"


def _leading_lines(bgr_array: np.ndarray, cx: float, cy: float) -> tuple[list[float], bool, str]:
    """Canny + HoughLinesP → (angles_deg, converges_to_subject, line_pattern)."""
    gray = cv2.cvtColor(bgr_array, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=80,
        minLineLength=100,
        maxLineGap=10,
    )
    if lines is None:
        return [], False, "none"

    h, w = bgr_array.shape[:2]
    subject_x = cx * w
    subject_y = cy * h
    angles: list[float] = []
    converging = 0

    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = math.degrees(math.atan2(y2 - y1, x2 - x1)) % 180
        angles.append(angle)
        # Point-to-infinite-line distance from subject to this line
        dx, dy = x2 - x1, y2 - y1
        length = math.hypot(dx, dy)
        if length > 0:
            dist = abs(dy * subject_x - dx * subject_y + x2 * y1 - y2 * x1) / length
            if dist < 0.1 * max(h, w):
                converging += 1

    converges = converging >= max(2, int(len(angles) * 0.3))
    pattern = _classify_pattern(angles)

    # Cluster into 15-degree bins; return median of each populated bin (≤12 total)
    _bin = 15
    bins: dict[int, list[float]] = {}
    for a in angles:
        bins.setdefault(int(a / _bin), []).append(a)
    dominant_angles = sorted(
        cluster[len(cluster) // 2]
        for cluster in sorted(bins.values(), key=len, reverse=True)
    )

    return dominant_angles, converges, pattern


# ---------------------------------------------------------------------------
# Scene classification
# ---------------------------------------------------------------------------


def _classify_scene(
    bgr: np.ndarray,
    pil_image: Image.Image,
    raw_angles: list[float],
    exif_data=None,
) -> tuple[str, list]:
    """Detect image genre: portrait|landscape|architecture|macro|general."""
    h, w = bgr.shape[:2]
    image_area = h * w

    # 1. Portrait: face bounding box > 5% of image area
    face_bboxes: list = []
    if _face_cascade is not None:
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        detections = _face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        if len(detections) > 0:
            face_bboxes = [tuple(int(v) for v in d) for d in detections]

    for _fx, _fy, fw, fh in face_bboxes:
        if (fw * fh) / image_area > 0.05:
            return "portrait", face_bboxes

    # 2. Landscape: brighter sky (top 20%) + wide format + no faces
    gray_f = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    top_rows = gray_f[: max(1, int(h * 0.2)), :]
    bottom_rows = gray_f[int(h * 0.8) :, :]
    top_mean = float(top_rows.mean())
    bottom_mean = float(bottom_rows.mean()) if bottom_rows.size > 0 else top_mean
    if not face_bboxes and top_mean >= bottom_mean + 30 and w > h * 1.2:
        return "landscape", []

    # 3. Architecture: many vertical lines (70°–110°)
    if len(raw_angles) > 8:
        vertical = sum(1 for a in raw_angles if 70 < a < 110)
        if vertical / len(raw_angles) >= 0.5:
            return "architecture", []

    # 4. Macro: no face + long focal length OR small dim + high sharpness variance
    focal_length = None
    if exif_data is not None:
        focal_length = getattr(exif_data, "focal_length", None)
    if not face_bboxes:
        if focal_length is not None and focal_length > 90:
            return "macro", []
        min_dim = min(h, w)
        if min_dim < 600:
            mh, mw = h // 2, w // 2
            quads = [
                gray_f[:mh, :mw], gray_f[:mh, mw:],
                gray_f[mh:, :mw], gray_f[mh:, mw:],
            ]
            sharpness = [
                float(cv2.Laplacian(q, cv2.CV_32F).var()) if q.size > 0 else 0.0
                for q in quads
            ]
            mean_s = sum(sharpness) / max(1, len(sharpness))
            max_s = max(sharpness) if sharpness else 0.0
            if mean_s > 0 and max_s > 2 * mean_s:
                return "macro", []

    return "general", face_bboxes


# ---------------------------------------------------------------------------
# Color harmony
# ---------------------------------------------------------------------------


def _cluster_hues(hues: list[int], threshold: int = 30) -> list[int]:
    """Cluster circular hue values (0–360); return list of cluster medians."""
    if not hues:
        return []
    sorted_h = sorted(hues)
    clusters: list[list[int]] = [[sorted_h[0]]]
    for h in sorted_h[1:]:
        center = int(sum(clusters[-1]) / len(clusters[-1]))
        diff = abs(h - center) % 360
        diff = min(diff, 360 - diff)
        if diff <= threshold:
            clusters[-1].append(h)
        else:
            clusters.append([h])
    # Wrap-around: merge first and last if close
    if len(clusters) > 1:
        first_c = int(sum(clusters[0]) / len(clusters[0]))
        last_c = int(sum(clusters[-1]) / len(clusters[-1]))
        d = abs(first_c - last_c) % 360
        if min(d, 360 - d) <= threshold:
            clusters[0].extend(clusters[-1])
            clusters.pop()
    return [int(sum(c) / len(c)) for c in clusters]


def _hue_diff(a: int, b: int) -> float:
    d = abs(a - b) % 360
    return float(min(d, 360 - d))


def _color_harmony(bgr: np.ndarray) -> tuple[list[list[int]], str, float]:
    """K-means colour analysis → (dominant_lab_colors, harmony_type, harmony_score)."""
    small = cv2.resize(bgr, (100, 100))
    lab = cv2.cvtColor(small, cv2.COLOR_BGR2Lab)
    pixels = lab.reshape(-1, 3).astype(np.float32)

    k = 5
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    try:
        _, labels, centers = cv2.kmeans(
            pixels, k, None, criteria, 10, cv2.KMEANS_PP_CENTERS
        )
    except Exception as exc:
        logger.warning("kmeans color harmony failed: %s", exc)
        return [], "complex", 0.0

    counts = np.bincount(labels.flatten(), minlength=k)
    order = np.argsort(-counts)
    centers = centers[order]

    dominant_colors = [[int(c[0]), int(c[1]), int(c[2])] for c in centers]

    # Convert each Lab center → HSV for hue/saturation
    hues: list[int] = []
    sats: list[int] = []
    for c in centers:
        lab_u8 = np.uint8([[np.clip(c, 0, 255)]])
        bgr_px = cv2.cvtColor(lab_u8, cv2.COLOR_Lab2BGR)
        hsv_px = cv2.cvtColor(bgr_px, cv2.COLOR_BGR2HSV)
        hues.append(int(hsv_px[0, 0, 0]) * 2)   # [0,179] → [0,358]
        sats.append(int(hsv_px[0, 0, 1]))

    # 1. Monochromatic
    if all(s < 20 for s in sats):
        return dominant_colors, "monochromatic", 1.0

    active_hues = [h for h, s in zip(hues, sats, strict=False) if s >= 20]
    if not active_hues:
        return dominant_colors, "monochromatic", 1.0

    # 2. Analogous: all pairwise hue diffs < 30°
    if len(active_hues) >= 2:
        diffs = [
            _hue_diff(a, b)
            for i, a in enumerate(active_hues)
            for j, b in enumerate(active_hues)
            if i < j
        ]
        max_diff = max(diffs)
        if max_diff < 30:
            return dominant_colors, "analogous", round(1.0 - max_diff / 30, 3)

    clusters = _cluster_hues(active_hues, threshold=30)

    # 3. Complementary: 2 clusters, diff ~180°
    if len(clusters) == 2:
        diff = _hue_diff(clusters[0], clusters[1])
        if abs(diff - 180) <= 30:
            return dominant_colors, "complementary", round(1.0 - abs(diff - 180) / 30, 3)

    # 4. Triadic: 3 clusters ~120° apart
    if len(clusters) >= 3:
        c3 = clusters[:3]
        best_tri: float | None = None
        for perm in permutations(c3):
            d1, d2 = _hue_diff(perm[0], perm[1]), _hue_diff(perm[1], perm[2])
            dev = max(abs(d1 - 120), abs(d2 - 120))
            if dev <= 20:
                score = 1.0 - dev / 20
                if best_tri is None or score > best_tri:
                    best_tri = score
        if best_tri is not None:
            return dominant_colors, "triadic", round(best_tri, 3)

        # 5. Split-complementary: one hue + two others ~150° away
        best_sc: float | None = None
        for perm in permutations(c3):
            d1, d2 = _hue_diff(perm[0], perm[1]), _hue_diff(perm[0], perm[2])
            dev = max(abs(d1 - 150), abs(d2 - 150))
            if dev <= 20:
                score = 1.0 - dev / 20
                if best_sc is None or score > best_sc:
                    best_sc = score
        if best_sc is not None:
            return dominant_colors, "split-complementary", round(best_sc, 3)

    return dominant_colors, "complex", 0.0


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def analyse(
    bgr_array: np.ndarray,
    pil_image: Image.Image,
    exif_data=None,
) -> CompositionScores:
    """
    Run all Layer-2 composition metrics on a single image.

    Args:
        bgr_array: uint8 numpy array, shape (H, W, 3), BGR channel order.
        pil_image: PIL Image (RGB).
        exif_data: Optional ExifData instance for scene classification.

    Returns:
        CompositionScores Pydantic instance.
    """
    import time

    saliency = _saliency_map(pil_image)
    cx, cy = _centroid(saliency)
    rot_score = _rot_alignment(cx, cy)
    gr_score = _golden_ratio_alignment(cx, cy)
    best_alignment = "rule_of_thirds" if rot_score <= gr_score else "golden_ratio"
    neg_space = _negative_space(saliency)
    weights, balance = _visual_weight(saliency)
    h_sym, v_sym = _symmetry(saliency)

    _t_lines = time.perf_counter()
    angles, converges, pattern = _leading_lines(bgr_array, cx, cy)
    logger.debug("leading_lines complete  %.2fs", time.perf_counter() - _t_lines)

    raw_angles = _detect_lines(bgr_array)
    horizon_tilt = _horizon_tilt(raw_angles)
    scene_type, _ = _classify_scene(bgr_array, pil_image, raw_angles, exif_data)
    dom_colors, harmony_type, harmony_score = _color_harmony(bgr_array)

    return CompositionScores(
        saliency_centroid_x=cx,
        saliency_centroid_y=cy,
        rot_alignment_score=rot_score,
        golden_ratio_alignment_score=gr_score,
        best_alignment=best_alignment,
        negative_space_ratio=neg_space,
        visual_weight_quadrants=weights,
        visual_weight_balance=balance,
        symmetry_horizontal=h_sym,
        symmetry_vertical=v_sym,
        dominant_line_angles=angles,
        leading_lines_converge_to_subject=converges,
        line_pattern=pattern,
        horizon_tilt_degrees=horizon_tilt,
        scene_type=scene_type,
        dominant_colors=dom_colors,
        color_harmony_type=harmony_type,
        color_harmony_score=harmony_score,
        saliency_map=saliency,
    )
