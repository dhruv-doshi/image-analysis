from __future__ import annotations

import math

import cv2
import numpy as np
from PIL import Image, ImageDraw


# ---------------------------------------------------------------------------
# Cohen-Sutherland line clipping
# ---------------------------------------------------------------------------

_INSIDE = 0
_LEFT = 1
_RIGHT = 2
_BOTTOM = 4
_TOP = 8


def _region_code(x: float, y: float, xmin: int, ymin: int, xmax: int, ymax: int) -> int:
    code = _INSIDE
    if x < xmin:
        code |= _LEFT
    elif x > xmax:
        code |= _RIGHT
    if y < ymin:
        code |= _TOP
    elif y > ymax:
        code |= _BOTTOM
    return code


def _clip_line_to_rect(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    xmin: int,
    ymin: int,
    xmax: int,
    ymax: int,
) -> tuple[float, float, float, float] | None:
    """Cohen-Sutherland line clipping. Returns clipped (x1, y1, x2, y2) or None if outside."""
    code1 = _region_code(x1, y1, xmin, ymin, xmax, ymax)
    code2 = _region_code(x2, y2, xmin, ymin, xmax, ymax)

    while True:
        if not (code1 | code2):
            return x1, y1, x2, y2
        if code1 & code2:
            return None

        code_out = code1 if code1 else code2
        if code_out & _BOTTOM:
            x = x1 + (x2 - x1) * (ymax - y1) / (y2 - y1)
            y = float(ymax)
        elif code_out & _TOP:
            x = x1 + (x2 - x1) * (ymin - y1) / (y2 - y1)
            y = float(ymin)
        elif code_out & _RIGHT:
            y = y1 + (y2 - y1) * (xmax - x1) / (x2 - x1)
            x = float(xmax)
        else:
            y = y1 + (y2 - y1) * (xmin - x1) / (x2 - x1)
            x = float(xmin)

        if code_out is code1:
            x1, y1 = x, y
            code1 = _region_code(x1, y1, xmin, ymin, xmax, ymax)
        else:
            x2, y2 = x, y
            code2 = _region_code(x2, y2, xmin, ymin, xmax, ymax)


# ---------------------------------------------------------------------------
# Centroid colour helper
# ---------------------------------------------------------------------------


def _centroid_colour(alignment_score: float) -> tuple[int, int, int, int]:
    if alignment_score < 0.25:
        return (0, 220, 0, 230)   # green — perfect
    if alignment_score <= 0.5:
        return (255, 200, 0, 230)  # yellow — fair
    return (220, 40, 40, 230)      # red — off power points


def _draw_crosshair(
    draw: ImageDraw.ImageDraw,
    px: int,
    py: int,
    colour: tuple[int, int, int, int],
    arm: int = 15,
    radius: int = 8,
    width: int = 2,
) -> None:
    draw.line([(px - arm, py), (px + arm, py)], fill=colour, width=width)
    draw.line([(px, py - arm), (px, py + arm)], fill=colour, width=width)
    draw.ellipse(
        [(px - radius, py - radius), (px + radius, py + radius)],
        outline=colour,
        width=width,
    )


# ---------------------------------------------------------------------------
# A. Rule of Thirds
# ---------------------------------------------------------------------------


def draw_rot_grid(
    pil_image: Image.Image,
    cx: float,
    cy: float,
    alignment_score: float,
) -> Image.Image:
    """Overlay Rule-of-Thirds grid + subject centroid."""
    img = pil_image.convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    w, h = img.size
    line_colour = (0, 255, 255, 153)
    circle_colour = (0, 255, 255, 200)

    # Grid lines
    for frac in (1 / 3, 2 / 3):
        x = int(w * frac)
        y = int(h * frac)
        draw.line([(x, 0), (x, h)], fill=line_colour, width=2)
        draw.line([(0, y), (w, y)], fill=line_colour, width=2)

    # Intersection circles
    for fx in (1 / 3, 2 / 3):
        for fy in (1 / 3, 2 / 3):
            ix, iy = int(w * fx), int(h * fy)
            r = 8
            draw.ellipse([(ix - r, iy - r), (ix + r, iy + r)], outline=circle_colour, width=2)

    # Subject centroid
    px, py = int(cx * w), int(cy * h)
    _draw_crosshair(draw, px, py, _centroid_colour(alignment_score))

    return Image.alpha_composite(img, overlay).convert("RGB")


# ---------------------------------------------------------------------------
# B. Golden Ratio
# ---------------------------------------------------------------------------


def draw_golden_ratio_grid(
    pil_image: Image.Image,
    cx: float,
    cy: float,
    alignment_score: float,
) -> Image.Image:
    """Overlay Golden Ratio grid + subject centroid."""
    img = pil_image.convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    w, h = img.size
    phi = 0.618
    line_colour = (255, 0, 255, 153)
    circle_colour = (255, 0, 255, 200)

    # Grid lines
    for frac in (1 - phi, phi):
        x = int(w * frac)
        y = int(h * frac)
        draw.line([(x, 0), (x, h)], fill=line_colour, width=2)
        draw.line([(0, y), (w, y)], fill=line_colour, width=2)

    # Intersection circles
    for fx in (1 - phi, phi):
        for fy in (1 - phi, phi):
            ix, iy = int(w * fx), int(h * fy)
            r = 8
            draw.ellipse([(ix - r, iy - r), (ix + r, iy + r)], outline=circle_colour, width=2)

    # Subject centroid
    px, py = int(cx * w), int(cy * h)
    _draw_crosshair(draw, px, py, _centroid_colour(alignment_score))

    return Image.alpha_composite(img, overlay).convert("RGB")


# ---------------------------------------------------------------------------
# C. Subject Dynamics
# ---------------------------------------------------------------------------


def draw_subject_dynamics(
    pil_image: Image.Image,
    saliency: np.ndarray | None,
    cx: float,
    cy: float,
    visual_weight_quadrants: dict,
) -> Image.Image:
    """Saliency heatmap + quadrant weight boxes + centroid crosshair."""
    img = pil_image.convert("RGBA")
    w, h = img.size

    if saliency is not None and saliency.shape[:2] == (h, w):
        sal = saliency
    elif saliency is not None:
        sal_resized = cv2.resize(saliency, (w, h), interpolation=cv2.INTER_LINEAR)
        sal = sal_resized
    else:
        sal = np.ones((h, w), dtype=np.float32)

    # JET heatmap at 50% alpha
    sal_uint8 = (np.clip(sal, 0, 1) * 255).astype(np.uint8)
    jet_bgr = cv2.applyColorMap(sal_uint8, cv2.COLORMAP_JET)
    jet_rgb = cv2.cvtColor(jet_bgr, cv2.COLOR_BGR2RGB)
    heat_rgba = np.dstack([jet_rgb, np.full((h, w), 128, dtype=np.uint8)])
    heat_pil = Image.fromarray(heat_rgba, "RGBA")
    img = Image.alpha_composite(img, heat_pil)

    # Quadrant rectangles
    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    mw, mh = w // 2, h // 2
    quad_regions = {
        "top_left": (0, 0, mw, mh),
        "top_right": (mw, 0, w, mh),
        "bottom_left": (0, mh, mw, h),
        "bottom_right": (mw, mh, w, h),
    }
    weights = visual_weight_quadrants or {}
    vals = list(weights.values())
    wmin, wmax = (min(vals), max(vals)) if vals else (0.0, 1.0)
    wrange = wmax - wmin if wmax != wmin else 1.0

    for key, (x0, y0, x1, y1) in quad_regions.items():
        wval = weights.get(key, 0.25)
        t = (wval - wmin) / wrange  # 0=lightest, 1=heaviest
        r = int(t * 255)
        b = int((1 - t) * 255)
        fill = (r, 0, b, 77)   # 30% alpha
        draw.rectangle([(x0, y0), (x1 - 1, y1 - 1)], fill=fill, outline=(255, 255, 255, 180), width=1)

    img = Image.alpha_composite(img, overlay)

    # Centroid crosshair
    ch_overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    ch_draw = ImageDraw.Draw(ch_overlay)
    px, py = int(cx * w), int(cy * h)
    _draw_crosshair(ch_draw, px, py, (255, 255, 255, 230), arm=15, radius=12, width=3)
    img = Image.alpha_composite(img, ch_overlay)

    return img.convert("RGB")


# ---------------------------------------------------------------------------
# D. Leading Lines
# ---------------------------------------------------------------------------


def draw_leading_lines(
    pil_image: Image.Image,
    dominant_angles: list[float],
    cx: float,
    cy: float,
    converges_to_subject: bool,
) -> Image.Image:
    """Synthesise leading lines through subject centroid from dominant angles."""
    img = pil_image.convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    w, h = img.size
    colour = (255, 0, 255, 220) if converges_to_subject else (0, 255, 255, 220)
    px, py = cx * w, cy * h

    for angle_deg in dominant_angles:
        rad = math.radians(angle_deg)
        dx, dy = math.cos(rad), math.sin(rad)
        # Extend far in both directions, then clip to image bounds
        far = max(w, h) * 2
        x1, y1 = px - dx * far, py - dy * far
        x2, y2 = px + dx * far, py + dy * far
        clipped = _clip_line_to_rect(x1, y1, x2, y2, 0, 0, w - 1, h - 1)
        if clipped:
            cx1, cy1, cx2, cy2 = clipped
            draw.line([(cx1, cy1), (cx2, cy2)], fill=colour, width=3)

    return Image.alpha_composite(img, overlay).convert("RGB")
