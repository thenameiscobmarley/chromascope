"""Colour detection: HSV threshold inside the FOV, merge nearby pixels, return boxes + preview image."""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class Target:
    x: int; y: int; w: int; h: int          # full-frame pixels
    area: int
    dist: float                              # distance of the box centre from the FOV centre


def hsv_of(rgb):
    """(r, g, b) 0-255 → OpenCV HSV (h 0-179)."""
    px = np.uint8([[[rgb[2], rgb[1], rgb[0]]]])
    return cv2.cvtColor(px, cv2.COLOR_BGR2HSV)[0, 0].tolist()


def fov_rect(cfg, w, h):
    s = min(cfg["fov"], w, h)
    cx, cy = w // 2 + cfg.get("fov_dx", 0), h // 2 + cfg.get("fov_dy", 0)
    x0, y0 = max(0, min(w - s, cx - s // 2)), max(0, min(h - s, cy - s // 2))
    return x0, y0, s, s


def detect(frame, cfg, ignore=()):
    """frame: (h, w, 4) BGRx. Returns (targets, mask, fov_rect). `ignore` = frame rects (x, y, w, h, band) whose
    outline band is drawn by our own overlay and must not be detected (stops the overlay feeding back)."""
    H, W = frame.shape[:2]
    fx, fy, fs, _ = fov_rect(cfg, W, H)
    roi = cv2.cvtColor(frame[fy:fy + fs, fx:fx + fs], cv2.COLOR_BGRA2BGR)   # contiguous copy: ~8x faster HSV
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    h, tol = cfg["hue"], cfg["tolerance"]
    lo_s, lo_v = cfg["min_sat"], cfg["min_val"]
    lo, hi = h - tol, h + tol
    mask = cv2.inRange(hsv, (max(lo, 0), lo_s, lo_v), (min(hi, 179), 255, 255))
    if lo < 0: mask |= cv2.inRange(hsv, (180 + lo, lo_s, lo_v), (179, 255, 255))
    if hi > 179: mask |= cv2.inRange(hsv, (0, lo_s, lo_v), (hi - 180, 255, 255))
    if cfg["fov_shape"] == "circle":
        circ = np.zeros_like(mask); cv2.circle(circ, (fs // 2, fs // 2), fs // 2, 255, -1); mask &= circ
    for (x, y, w, hh, band) in ignore:      # our own overlay outlines
        x, y = x - fx, y - fy
        cv2.rectangle(mask, (x, y), (x + w, y + hh), 0, band)

    merge = cfg["merge"]
    grown = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_RECT, (merge * 2 + 1,) * 2)) if merge > 0 else mask
    n, _lab, stats, _ = cv2.connectedComponentsWithStats(grown, connectivity=8)
    c = fs / 2
    out = []
    for i in range(1, n):
        x, y, w, hh, _ = stats[i]
        sub = mask[y:y + hh, x:x + w]
        real = int(cv2.countNonZero(sub))
        if real < cfg["min_area"]: continue
        bx, by, bw, bh = cv2.boundingRect(sub)      # shrink back to the real pixels (dilation made it fatter)
        x0, y0 = x + bx, y + by
        d = float(np.hypot(x0 + bw / 2 - c, y0 + bh / 2 - c))
        out.append(Target(int(x0 + fx), int(y0 + fy), int(bw), int(bh), real, d))
    out.sort(key=(lambda t: t.dist) if cfg["sort"] == "closest" else (lambda t: -t.area))
    return out[:cfg["max_targets"]], mask, (fx, fy, fs, fs)


def tint(img, mask, strength=0.65, glow=10):
    """Turn masked pixels a red shade and add a soft red halo around them (in place on a BGR(x) image)."""
    m = mask > 0
    if not m.any(): return img
    red = np.array([40, 30, 255], np.float32)
    px = img[..., :3]
    if glow > 0:
        k = glow * 2 + 1
        halo = cv2.GaussianBlur(cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=max(1, glow // 4)), (k, k), 0)
        a = (halo.astype(np.float32) / 255.0 * 0.75)[..., None]
        px[:] = (px * (1 - a) + red * a).astype(np.uint8)
    px[m] = (px[m] * (1 - strength) + red * strength).astype(np.uint8)
    return img
