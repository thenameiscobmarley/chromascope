"""Colour tools: pixel info + names, WCAG contrast, dominant-colour palette, profile share codes."""
from __future__ import annotations

import base64
import json
import zlib

import cv2
import numpy as np

# plain colour names from hue bands + lightness (predictable, unlike nearest-match tables)
HUES = [(12, "red"), (40, "orange"), (65, "yellow"), (95, "lime"), (160, "green"), (195, "cyan"), (250, "blue"),
        (290, "purple"), (330, "magenta"), (350, "pink"), (360, "red")]


def name_of(rgb):
    r, g, b = (int(v) for v in rgb)
    h, l, s = (int(x) for x in cv2.cvtColor(np.uint8([[[r, g, b]]]), cv2.COLOR_RGB2HLS_FULL)[0, 0])
    h, l, s = h * 360 / 256, l / 2.55, s / 2.55
    if s < 15 or l < 6 or l > 96:
        return "black" if l < 12 else "dark grey" if l < 35 else "grey" if l < 65 else "silver" if l < 88 else "white"
    hue = next(n for top, n in HUES if h < top)
    if hue == "orange" and l < 35: return "brown"
    if hue == "yellow" and l < 35: return "olive"
    if hue == "red" and l > 75: return "pink"
    if s < 35 and 20 < l < 80: return f"greyish {hue}"
    return f"dark {hue}" if l < 28 else f"light {hue}" if l > 75 else hue


def info(rgb):
    """Readable descriptions of one colour."""
    r, g, b = (int(v) for v in rgb)
    h, s, v = (int(x) for x in cv2.cvtColor(np.uint8([[[r, g, b]]]), cv2.COLOR_RGB2HSV_FULL)[0, 0])
    return {"hex": f"#{r:02X}{g:02X}{b:02X}", "rgb": f"rgb({r}, {g}, {b})",
            "hsv": f"hsv({round(h * 360 / 256)}°, {round(s / 2.55)}%, {round(v / 2.55)}%)", "name": name_of((r, g, b))}


def luminance(rgb):
    c = np.array(rgb, np.float64) / 255
    c = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    return float(0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2])


def contrast(a, b):
    """WCAG 2 contrast ratio (1–21) and which levels it passes."""
    la, lb = sorted((luminance(a), luminance(b)), reverse=True)
    r = (la + 0.05) / (lb + 0.05)
    return r, {"AA": r >= 4.5, "AA large": r >= 3, "AAA": r >= 7, "AAA large": r >= 4.5}


def palette(region, k=8):
    """Dominant colours of a BGR(x) image: [(rgb, share 0-1, hsv, hue spread)] biggest first."""
    h, w = region.shape[:2]
    s = min(1.0, 96 / max(h, w))
    small = cv2.resize(region[..., :3], (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA)
    px = small.reshape(-1, 3).astype(np.float32)
    k = max(1, min(k, len(px)))
    _, labels, centers = cv2.kmeans(px, k, None, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0), 3, cv2.KMEANS_PP_CENTERS)
    labels = labels.ravel()
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV).reshape(-1, 3).astype(np.int32)
    out = []
    for i in range(k):
        sel = labels == i
        if not sel.any(): continue
        b, g, r = centers[i].astype(int)
        hh = hsv[sel]
        c_hsv = cv2.cvtColor(np.uint8([[[b, g, r]]]), cv2.COLOR_BGR2HSV)[0, 0].astype(int)
        dev = (hh[:, 0] - c_hsv[0] + 90) % 180 - 90
        out.append(((int(r), int(g), int(b)), float(sel.mean()), c_hsv.tolist(),
                    {"tolerance": int(np.clip(np.percentile(np.abs(dev), 90) + 3, 5, 40)),
                     "min_sat": int(max(0, np.percentile(hh[:, 1], 10) - 20)), "min_val": int(max(0, np.percentile(hh[:, 2], 10) - 20))}))
    out.sort(key=lambda t: -t[1])
    return out


CODE_PREFIX = "CSP1:"


def to_code(profile: dict) -> str:
    return CODE_PREFIX + base64.urlsafe_b64encode(zlib.compress(json.dumps(profile, separators=(",", ":")).encode(), 9)).decode()


def from_code(code: str) -> dict:
    code = "".join(code.split())
    if not code.startswith(CODE_PREFIX): raise ValueError("Not a ChromaScope profile code (they start with CSP1:)")
    return json.loads(zlib.decompress(base64.urlsafe_b64decode(code[len(CODE_PREFIX):])))
