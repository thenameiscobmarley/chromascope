"""Colour detection: HSV thresholds inside the FOV for each enabled colour, merge nearby pixels, return boxes.
Also: sampling a colour range from a dragged area, the red glow tint, and the colour-blind assist layer."""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class Target:
    x: int; y: int; w: int; h: int          # full-frame pixels
    area: int
    dist: float                              # distance of the box centre from the FOV centre
    cls: int = 0                             # index into cfg["colors"]


def hsv_of(rgb):
    """(r, g, b) 0-255 → OpenCV HSV (h 0-179)."""
    px = np.uint8([[[rgb[2], rgb[1], rgb[0]]]])
    return cv2.cvtColor(px, cv2.COLOR_BGR2HSV)[0, 0].tolist()


def fov_rect(cfg, w, h):
    """(x, y, width, height) of the scanned area in frame pixels."""
    if cfg["fov_shape"] == "full": return 0, 0, w, h
    s = min(cfg["fov"], w, h)
    cx, cy = w // 2 + cfg.get("fov_dx", 0), h // 2 + cfg.get("fov_dy", 0)
    x0, y0 = max(0, min(w - s, cx - s // 2)), max(0, min(h - s, cy - s // 2))
    return x0, y0, s, s


def color_mask(hsv, c):
    h, tol, lo_s, lo_v = c["hue"], c["tolerance"], c["min_sat"], c["min_val"]
    lo, hi = h - tol, h + tol
    mask = cv2.inRange(hsv, (max(lo, 0), lo_s, lo_v), (min(hi, 179), 255, 255))
    if lo < 0: mask |= cv2.inRange(hsv, (180 + lo, lo_s, lo_v), (179, 255, 255))
    if hi > 179: mask |= cv2.inRange(hsv, (0, lo_s, lo_v), (hi - 180, 255, 255))
    return mask


def detect(frame, cfg, ignore=()):
    """frame: (h, w, 4) BGRx. Returns (targets, combined mask, fov_rect); the mask may be half-size for big areas. `ignore` = frame rects (x, y, w, h, band)
    whose outline band our own overlay drew and must not be detected (fallback if KWin can't hide the overlay); band < 0 = the whole rect."""
    H, W = frame.shape[:2]
    fx, fy, fw, fh = fov_rect(cfg, W, H)
    k = 2 if fw * fh > 500_000 else 1         # big areas (full screen): work at half resolution, ~4x faster
    roi = frame[fy:fy + fh, fx:fx + fw]
    if k > 1: roi = cv2.resize(roi, (fw // k, fh // k), interpolation=cv2.INTER_NEAREST)
    hsv = cv2.cvtColor(cv2.cvtColor(roi, cv2.COLOR_BGRA2BGR), cv2.COLOR_BGR2HSV)   # contiguous copy: ~8x faster HSV
    fx0, fy0 = fx, fy
    fw, fh = hsv.shape[1], hsv.shape[0]
    keep = None
    if cfg["fov_shape"] == "circle":
        keep = np.zeros((fh, fw), np.uint8); cv2.circle(keep, (fw // 2, fh // 2), fw // 2, 255, -1)
    if ignore:
        keep = np.full((fh, fw), 255, np.uint8) if keep is None else keep
        for (x, y, w, hh, band) in ignore:
            cv2.rectangle(keep, ((x - fx) // k, (y - fy) // k), ((x - fx + w) // k, (y - fy + hh) // k), 0,
                          -1 if band < 0 else max(1, band // k))      # band < 0: the whole rect (e.g. our own window)

    merge = cfg["merge"] // k
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (merge * 2 + 1,) * 2) if merge > 0 else None
    total = np.zeros((fh, fw), np.uint8)
    cx, cy = fw / 2, fh / 2
    out = []
    for ci, c in enumerate(cfg["colors"]):
        if not c.get("on", True): continue
        mask = color_mask(hsv, c)
        if keep is not None: mask &= keep
        total |= mask
        grown = cv2.dilate(mask, kernel) if kernel is not None else mask
        n, _lab, stats, _ = cv2.connectedComponentsWithStats(grown, connectivity=8)
        for i in range(1, n):
            x, y, w, hh, _ = stats[i]
            sub = mask[y:y + hh, x:x + w]
            real = int(cv2.countNonZero(sub)) * k * k
            if real < cfg["min_area"]: continue
            bx, by, bw, bh = cv2.boundingRect(sub)      # shrink back to the real pixels (dilation made it fatter)
            x0, y0 = x + bx, y + by
            d = float(np.hypot(x0 + bw / 2 - cx, y0 + bh / 2 - cy)) * k
            out.append(Target(int(x0 * k + fx0), int(y0 * k + fy0), int(bw * k), int(bh * k), real, d, ci))
    out.sort(key=(lambda t: t.dist) if cfg["sort"] == "closest" else (lambda t: -t.area))
    return out[:cfg["max_targets"]], total, (fx0, fy0, fw * k, fh * k)


def sample(frame, x, y, w, h):
    """Work out a colour range from an area: the dominant hue among its colourful pixels, with tolerance,
    saturation and brightness floors that cover ~95% of them. Returns a dict or None."""
    H, W = frame.shape[:2]
    x, y = max(0, min(W - 1, x)), max(0, min(H - 1, y))
    w, h = max(1, min(W - x, w)), max(1, min(H - y, h))
    px = np.ascontiguousarray(frame[y:y + h, x:x + w, :3]).reshape(-1, 1, 3)
    hsv = cv2.cvtColor(px, cv2.COLOR_BGR2HSV).reshape(-1, 3).astype(np.int32)
    colourful = (hsv[:, 1] >= 40) & (hsv[:, 2] >= 40)
    if colourful.sum() < 3: colourful[:] = True
    idx = np.nonzero(colourful)[0]
    hh = hsv[idx]
    hist = np.bincount(hh[:, 0], minlength=180).astype(np.float32)
    hist = sum(np.roll(hist, k) for k in range(-3, 4))        # circular smoothing so reds near 0/179 join up
    peak = int(np.argmax(hist))
    dev = (hh[:, 0] - peak + 90) % 180 - 90
    near = np.abs(dev) <= 20
    if near.sum() == 0: return None
    sel, d = hh[near], dev[near]
    centre = int((peak + round(float(np.median(d)))) % 180)
    tol = int(np.clip(np.percentile(np.abs(d - (centre - peak)), 95) + 3, 5, 40))
    b, g, r = np.median(px.reshape(-1, 3)[idx[near]], axis=0).astype(int)
    return {"color": "#%02x%02x%02x" % (r, g, b), "hue": centre, "tolerance": tol,
            "min_sat": int(max(0, np.percentile(sel[:, 1], 5) - 20)), "min_val": int(max(0, np.percentile(sel[:, 2], 5) - 20))}


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


_stripes = {}


def assist_layer(mask, style, rgb, max_w=960):
    """Colour-blind assist: a premultiplied BGRA image (downscaled) that recolours, stripes or outlines the
    matching pixels. Drawn over the preview and the on-screen overlay."""
    h, w = mask.shape
    s = min(1.0, max_w / w)
    m = cv2.resize(mask, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_NEAREST) if s < 1 else mask
    h, w = m.shape
    if style == "outline":
        m = cv2.morphologyEx(m, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)); a = 255
    elif style == "stripes":
        pat = _stripes.get((h, w))
        if pat is None:
            yy, xx = np.indices((h, w)); pat = _stripes[(h, w)] = np.where(((xx + yy) // 3) % 2 == 0, 255, 0).astype(np.uint8)
        m = m & pat; a = 255
    else:
        a = 205
    out = np.zeros((h, w, 4), np.uint8)
    r, g, b = rgb
    out[m > 0] = (b * a // 255, g * a // 255, r * a // 255, a)
    return out


# ── Self-test: can the screen capture see our own overlay / window? ──────────
def find_probes(frame):
    """Look for the self-test marks (two 8×8 patches side by side, pure green and pure magenta). The overlay draws
    green|magenta, the window magenta|green. Returns (overlay mark position, window mark position) in frame px of the
    left patch's centre, or None where not found."""
    f = frame[::2, ::2, :3].astype(np.int16)                 # half resolution is plenty for 8×8 patches
    B, G, R = f[..., 0], f[..., 1], f[..., 2]
    green = ((G > 170) & (R < 100) & (B < 100)).astype(np.uint8)
    magenta = ((R > 170) & (B > 150) & (G < 100)).astype(np.uint8)

    def blobs(m):
        n, _lab, st, cen = cv2.connectedComponentsWithStats(m, connectivity=8)
        return [(cen[i], st[i, 4]) for i in range(1, n) if 6 <= st[i, 4] <= 400]

    over = win = None
    for gc, ga in blobs(green):
        for mc, ma in blobs(magenta):
            dx, dy = mc[0] - gc[0], abs(mc[1] - gc[1])
            if dy < 3 and 2 <= abs(dx) <= 20 and 0.4 < ga / ma < 2.5:
                if dx > 0: over = (float(gc[0] * 2), float(gc[1] * 2))   # green left: the overlay's mark
                else: win = (float(mc[0] * 2), float(mc[1] * 2))         # magenta left: the window's mark
    return over, win
