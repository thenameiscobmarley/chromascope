"""ChromaScope — live colour detection on your screen (Wayland / KDE Plasma), built with UiVault3.

Run:  ./run.sh        Settings are saved in ~/.config/chromascope/ (settings.json, profiles/).
Keys: F8 start/stop capture · F9 overlay on/off · F10 preview mode
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, QThread, QTimer, QUrl, Signal
from PySide6.QtGui import QColor, QDesktopServices, QFont, QImage, QKeySequence, QPainter, QPainterPath, QPen, QShortcut
from PySide6.QtWidgets import (QApplication, QColorDialog, QComboBox, QFileDialog, QGridLayout, QHBoxLayout, QInputDialog,
                               QLabel, QScrollArea, QSizePolicy, QStackedWidget, QVBoxLayout, QWidget)

import uivault3 as uv
import tools
from capture import MONITOR, WINDOW, Capture
from detect import assist_layer, detect, fov_rect, hsv_of, sample, tint
import nnio
import nnmm
from inputwatch import RightButton
from extras import EXAMPLES, CursorWatcher, FocusWatcher, LockWatcher, Tour, UndoStack

CONFIG = Path.home() / ".config" / "chromascope"
MODELS = CONFIG / "models"                                    # imported models
MODEL_DIRS = [MODELS, Path(__file__).resolve().parent / "models"]   # + the bundled ones (aim)
PROFILES = CONFIG / "profiles"
PROFILES.mkdir(parents=True, exist_ok=True)
SETTINGS = CONFIG / "settings.json"
OVERLAY_TITLE = "ChromaScope Overlay"

DEFAULTS = {
    "colors": None, "sel": 0, "tolerance": 10, "min_sat": 120, "min_val": 110,     # tolerance/sat/val mirror the selected colour
    "min_area": 30, "merge": 6, "max_targets": 8, "sort": "largest",
    "fov": 420, "fov_shape": "circle", "fov_dx": 0, "fov_dy": 0, "show_fov": True, "crosshair": True, "regions": {},
    "mode": "box", "thickness": 2, "glow": 10, "labels": True,
    "assist": False, "assist_style": "stripes", "assist_color": "#2a6bff",
    "view": "full", "hud": True, "max_fps": 60,
    "overlay": False, "overlay_fov": True, "overlay_fps": True, "overlay_screen": 0,
    "source": MONITOR, "cursor": False, "minimize": False, "restore_token": None,
    "pause_locked": True, "pause_static": False, "pause_static_s": 30, "pause_focus": False, "focus_match": "",
    "history": [], "contrast_a": "#1a1917", "contrast_b": "#f4f2ee",
    "kit": "win98", "theme": "paper", "profile": "", "toured": False,
    "nn_model": "aim", "nn_hold": "both",
}
# what a profile stores: everything about what to find and how to draw it, not window/capture/theme settings
PROFILE_KEYS = ["colors", "sel", "min_area", "merge", "max_targets", "sort", "fov", "fov_shape", "fov_dx", "fov_dy",
                "show_fov", "crosshair", "mode", "thickness", "glow", "labels", "assist", "assist_style", "assist_color", "max_fps"]
UNDO_KEYS = PROFILE_KEYS + ["regions"]
# what "Reset this tab" puts back, per tab
RESET_KEYS = {"Detect": ["colors", "sel", "tolerance", "min_sat", "min_val", "min_area", "merge", "max_targets", "sort"],
              "FOV": ["fov", "fov_shape", "fov_dx", "fov_dy", "show_fov", "crosshair", "max_fps"],
              "Draw": ["mode", "thickness", "glow", "labels", "hud", "assist", "assist_style", "assist_color"],
              "Setup": ["overlay_fov", "overlay_fps", "overlay_screen", "source", "cursor", "minimize",
                        "pause_locked", "pause_static", "pause_static_s", "pause_focus", "focus_match"],
              "NNMM": ["nn_model", "nn_hold"]}
PRESETS = [("Red", "#ff2a2a"), ("Orange", "#ff8a00"), ("Yellow", "#ffe100"), ("Green", "#22e35a"),
           ("Cyan", "#00e5ff"), ("Blue", "#2a6bff"), ("Purple", "#b13cff"), ("Magenta", "#ff2ad4")]
BOX_COLORS = [("Red", "#ff3b3b"), ("Lime", "#7CFF4F"), ("White", "#ffffff"), ("Cyan", "#2ae6ff"), ("Gold", "#ffc83a"), ("Pink", "#ff4fd8")]
MAX_COLORS = 6


def new_color(name="Red", color="#ff2a2a", box="#ff3b3b", tolerance=10, min_sat=120, min_val=110):
    return {"name": name, "color": color, "hue": hsv_of(QColor(color).getRgb()[:3])[0], "box": box,
            "tolerance": tolerance, "min_sat": min_sat, "min_val": min_val, "on": True}


def fix_colors(cfg, raw=None):
    """Make sure cfg has a valid colour list (migrates the single-colour settings of v1)."""
    raw = raw or {}
    if not cfg.get("colors"):
        cfg["colors"] = [new_color("Target", raw.get("color", "#ff2a2a"), raw.get("box_color", "#ff3b3b"),
                                   cfg["tolerance"], cfg["min_sat"], cfg["min_val"])]
    for c in cfg["colors"]:
        c.setdefault("on", True); c.setdefault("box", "#ff3b3b")
        c["hue"] = c.get("hue", hsv_of(QColor(c["color"]).getRgb()[:3])[0])
    cfg["sel"] = max(0, min(cfg["sel"], len(cfg["colors"]) - 1))
    s = cfg["colors"][cfg["sel"]]
    cfg.update(tolerance=s["tolerance"], min_sat=s["min_sat"], min_val=s["min_val"])


def load_settings():
    try: raw = json.loads(SETTINGS.read_text())
    except Exception: raw = {}
    cfg = {**DEFAULTS, **{k: v for k, v in raw.items() if k in DEFAULTS}}
    fix_colors(cfg, raw)
    if cfg["nn_model"] in ("sandbox", "teacher"): cfg["nn_model"] = "aim"     # the old sandbox choices are gone
    return cfg


def save_settings(cfg):
    SETTINGS.write_text(json.dumps({k: cfg[k] for k in DEFAULTS}, indent=1))


def profile_names(): return sorted(p.stem for p in PROFILES.glob("*.json"))
def profile_path(name): return PROFILES / (re.sub(r"[^\w\- ]", "_", name).strip() + ".json")


def tok(name): return uv.style().color(name)


def label(text, size=13, muted=False, bold=False, mono=False):
    l = QLabel(text)
    f = QFont("Geist Mono") if mono else l.font()
    f.setPixelSize(size); f.setBold(bold); l.setFont(f)
    l.setProperty("muted", muted); l.setWordWrap(muted)
    if muted: l.setStyleSheet(f"color: {tok('muted').name()}")
    return l


# ── Engine: capture → detect → preview, off the UI thread ────────────────────
class Engine(QThread):
    frameReady = Signal(object)
    failed = Signal(str)
    started_ok = Signal()
    pausedChanged = Signal(str)           # "" = running, otherwise why it's paused

    def __init__(self, cfg):
        super().__init__()
        self.cfg, self.cap, self.last_frame, self.reason = cfg, Capture(), None, ""
        self.overlay_scale = 1.0          # overlay logical px per frame px (set by the UI)

    def start_capture(self):
        c = self.cfg
        self.cap.start(c["source"], c["cursor"], c["restore_token"], on_ready=lambda _c: self.started_ok.emit())
        self.start()

    def stop_capture(self):
        self.requestInterruption(); self.wait(1500); self.cap.stop()

    def _ignore(self, targets, cfg):
        """Outline bands our overlay drew last frame, in frame px (fallback for when KWin can't hide the overlay)."""
        if not cfg["overlay"]: return ()
        s = max(self.overlay_scale, 1e-3)
        pad = (4 + cfg["thickness"] + (cfg["glow"] if cfg["mode"] in ("glow", "both") else 0)) / s
        band = int(2 * pad + 6)
        return [(int(t.x - pad), int(t.y - pad), int(t.w + 2 * pad), int(t.h + 2 * pad), band) for t in targets]

    def run(self):
        last, prev, n, t0, fps, next_t = 0, [], 0, time.monotonic(), 0.0, 0.0
        sig, changed_at = None, time.monotonic()
        while not self.isInterruptionRequested():
            if self.cap.error: self.failed.emit(self.cap.error); return
            frame, last = self.cap.wait(last, 0.3)
            if frame is None: continue
            now = time.monotonic()
            cfg = dict(self.cfg); cfg["colors"] = [dict(c) for c in cfg["colors"]]
            self.last_frame = frame
            # auto-pause: reasons from the UI (lock, focus) or a screen that hasn't changed for a while
            s = frame[::24, ::24, :3]
            if sig is None or s.shape != sig.shape or not np.array_equal(s, sig): sig, changed_at = s.copy(), now
            reason = cfg.get("_pause") or ("screen hasn't changed" if cfg["pause_static"] and now - changed_at > cfg["pause_static_s"] else "")
            if reason != self.reason: self.reason = reason; self.pausedChanged.emit(reason)
            if reason: continue
            if now < next_t: time.sleep(next_t - now)
            next_t = time.monotonic() + 1.0 / max(1, cfg["max_fps"])
            t = time.perf_counter()
            targets, mask, fov = detect(frame, cfg, self._ignore(prev, cfg))
            img, src = self._preview(frame, mask, fov, cfg)
            assist = None
            if cfg["assist"]:
                a = assist_layer(mask, cfg["assist_style"], QColor(cfg["assist_color"]).getRgb()[:3])
                assist = QImage(a.data, a.shape[1], a.shape[0], a.strides[0], QImage.Format_ARGB32_Premultiplied).copy()
            ms = (time.perf_counter() - t) * 1000
            n += 1
            if time.monotonic() - t0 >= 0.5: fps, n, t0 = n / (time.monotonic() - t0), 0, time.monotonic()
            prev = targets
            self.frameReady.emit({"image": img, "src": src, "targets": targets, "fov": fov, "size": frame.shape[1::-1],
                                  "assist": assist, "colors": cfg["colors"], "ms": ms, "fps": fps, "cap_fps": self.cap.fps})

    def _preview(self, frame, mask, fov, cfg):
        fx, fy, fw, fh = fov
        glow = cfg["mode"] in ("glow", "both")
        if cfg["view"] == "mask":
            m = np.ascontiguousarray(mask)
            return QImage(m.data, m.shape[1], m.shape[0], m.strides[0], QImage.Format_Grayscale8).copy(), fov
        if cfg["view"] == "fov":
            img = np.ascontiguousarray(frame[fy:fy + fh, fx:fx + fw])
            if glow:
                m = mask if mask.shape == img.shape[:2] else cv2.resize(mask, (fw, fh), interpolation=cv2.INTER_NEAREST)
                tint(img, m, glow=cfg["glow"])
            src = fov
        else:
            H, W = frame.shape[:2]
            s = min(1.0, 1100 / W)
            img = cv2.resize(frame, (int(W * s), int(H * s)), interpolation=cv2.INTER_LINEAR) if s < 1 else frame.copy()
            if glow:
                x0, y0 = int(fx * s), int(fy * s)
                region = img[y0:y0 + max(1, int(fh * s)), x0:x0 + max(1, int(fw * s))]
                m = cv2.resize(mask, (region.shape[1], region.shape[0]), interpolation=cv2.INTER_NEAREST)
                tint(region, m, glow=max(2, int(cfg["glow"] * s)))
            src = (0, 0, W, H)
        h, w = img.shape[:2]
        return QImage(img.data, w, h, img.strides[0], QImage.Format_RGB32).copy(), src


# ── Drawing shared by the preview and the on-screen overlay ──────────────────
def draw_targets(p, targets, to_px, cfg, colors, overlay=False):
    mode, th = cfg["mode"], cfg["thickness"]
    if mode == "none": return
    f = QFont("Geist Mono"); f.setPixelSize(11); f.setBold(True); p.setFont(f)
    for t in targets:
        c = colors[t.cls] if t.cls < len(colors) else colors[0]
        col = QColor(c["box"])
        r = to_px(QRectF(t.x, t.y, t.w, t.h)).adjusted(-3, -3, 3, 3)
        if mode in ("glow", "both") and overlay:
            g = cfg["glow"]
            for k in range(g, 0, -2):          # soft red halo outside the box
                hc = QColor(255, 40, 40, int(150 * (1 - k / (g + 1)) ** 2))
                p.setPen(QPen(hc, 2)); p.setBrush(Qt.NoBrush)
                p.drawRoundedRect(r.adjusted(-k, -k, k, k), 4 + k, 4 + k)
        if mode in ("box", "both"):
            p.setPen(QPen(col, th)); p.setBrush(Qt.NoBrush); p.drawRoundedRect(r, 3, 3)
        elif mode == "corners":
            L = max(6.0, min(r.width(), r.height()) * 0.28)
            p.setPen(QPen(col, th + 1, Qt.SolidLine, Qt.SquareCap))
            for (x, y, dx, dy) in ((r.left(), r.top(), 1, 1), (r.right(), r.top(), -1, 1),
                                   (r.left(), r.bottom(), 1, -1), (r.right(), r.bottom(), -1, -1)):
                p.drawLine(QPointF(x, y), QPointF(x + dx * L, y)); p.drawLine(QPointF(x, y), QPointF(x, y + dy * L))
        if cfg["labels"] and not overlay:
            txt = f"{c['name']}  {t.w}×{t.h}"
            tw = p.fontMetrics().horizontalAdvance(txt) + 10
            chip = QRectF(r.left(), r.top() - 18, tw, 16)
            if chip.top() < 0: chip.moveTop(r.bottom() + 2)
            p.setPen(Qt.NoPen); p.setBrush(QColor(col.red(), col.green(), col.blue(), 220)); p.drawRoundedRect(chip, 4, 4)
            p.setPen(QColor("#0b0b0b") if col.lightness() > 140 else QColor("#ffffff"))
            p.drawText(chip, Qt.AlignCenter, txt)


def draw_fov(p, fov, to_px, cfg, ring_pad=0, active=False):
    if cfg["fov_shape"] == "full": return
    fx, fy, fw, fh = fov
    r = to_px(QRectF(fx, fy, fw, fh)).adjusted(-ring_pad, -ring_pad, ring_pad, ring_pad)
    p.setBrush(Qt.NoBrush)
    p.setPen(QPen(QColor(255, 220, 90, 230) if active else QColor(255, 255, 255, 150), 2 if active else 1.4, Qt.DashLine))
    p.drawEllipse(r) if cfg["fov_shape"] == "circle" else p.drawRect(r)
    if cfg["crosshair"]:
        c = r.center(); p.setPen(QPen(QColor(255, 255, 255, 200), 1.4))
        p.drawLine(QPointF(c.x() - 7, c.y()), QPointF(c.x() + 7, c.y())); p.drawLine(QPointF(c.x(), c.y() - 7), QPointF(c.x(), c.y() + 7))


class Preview(QWidget):
    sampled = Signal(int, int, int, int)     # frame px rect dragged (or clicked) while sampling
    fovMoved = Signal()                      # the FOV was dragged / resized (cfg already updated)
    picked = Signal(int, int)                # eyedropper click, frame px

    def __init__(self, cfg):
        super().__init__()
        self.cfg, self.data, self.sampling, self.eyedrop = cfg, None, False, False
        self.pixel = None                    # callable (x, y) -> (rgb, 11x11 RGB crop) | None, set by the window
        self.hover, self.paused = None, ""
        self.drag = None                     # ("sample", start QPointF, cur QPointF) | ("move"/"size", …)
        self.setMinimumSize(480, 300); self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMouseTracking(True)

    def show_frame(self, d): self.data = d; self.update()

    def _geom(self):
        img = self.data["image"]
        area = QRectF(self.rect()).adjusted(1, 1, -1, -1)
        s = min(area.width() / img.width(), area.height() / img.height())
        w, h = img.width() * s, img.height() * s
        return QRectF(area.center().x() - w / 2, area.center().y() - h / 2, w, h)

    def _mapper(self):
        g = self._geom(); sx, sy, sw, sh = self.data["src"]
        kx, ky = g.width() / sw, g.height() / sh
        return (lambda r: QRectF(g.x() + (r.x() - sx) * kx, g.y() + (r.y() - sy) * ky, r.width() * kx, r.height() * ky)), g

    def _to_frame(self, pt):
        g = self._geom(); sx, sy, sw, sh = self.data["src"]
        return sx + (pt.x() - g.x()) / g.width() * sw, sy + (pt.y() - g.y()) / g.height() * sh

    def _fov(self):
        """FOV from the live settings (not the last frame) so dragging feels instant."""
        return fov_rect(self.cfg, *self.data["size"])

    def _hit(self, pt):
        """'size' near the FOV edge, 'move' inside it, None elsewhere (only in the Screen view)."""
        if not self.data or self.cfg["view"] != "full" or self.cfg["fov_shape"] == "full": return None
        to_px, _ = self._mapper(); r = to_px(QRectF(*self._fov()))
        c = r.center(); d = QPointF(pt - c)
        if self.cfg["fov_shape"] == "circle":
            dist = (d.x() ** 2 + d.y() ** 2) ** 0.5
            return "size" if abs(dist - r.width() / 2) <= 8 else "move" if dist < r.width() / 2 else None
        edge = max(abs(d.x()), abs(d.y()))
        return "size" if abs(edge - r.width() / 2) <= 8 else "move" if edge < r.width() / 2 else None

    def paintEvent(self, _):
        p = QPainter(self); p.setRenderHint(QPainter.Antialiasing)
        bg = QPainterPath(); bg.addRoundedRect(QRectF(self.rect()), 14, 14); p.setClipPath(bg)
        p.fillRect(self.rect(), QColor("#0a0a0a"))
        if not self.data:
            p.setPen(QColor(255, 255, 255, 110)); f = p.font(); f.setPixelSize(15); p.setFont(f)
            p.drawText(self.rect(), Qt.AlignCenter, "Press  Start capture  (F8)\nKDE asks which screen to share the first time.\n\n"
                       "Drag the FOV ring to move it, drag its edge to resize.\nUse  Sample  then drag over a colour to learn it.")
            p.end(); return
        d, cfg = self.data, self.cfg
        to_px, g = self._mapper()
        p.setRenderHint(QPainter.SmoothPixmapTransform, cfg["view"] != "mask")
        p.drawImage(g, d["image"])
        if d.get("assist") is not None and cfg["view"] != "mask":
            p.drawImage(to_px(QRectF(*d["fov"])), d["assist"])
        if cfg["view"] == "full":
            if cfg["show_fov"] or self.drag: draw_fov(p, self._fov(), to_px, cfg, active=bool(self.drag and self.drag[0] != "sample"))
        elif cfg["crosshair"] and cfg["view"] != "mask":
            c = g.center(); p.setPen(QPen(QColor(255, 255, 255, 170), 1.2))
            p.drawLine(QPointF(c.x() - 7, c.y()), QPointF(c.x() + 7, c.y())); p.drawLine(QPointF(c.x(), c.y() - 7), QPointF(c.x(), c.y() + 7))
        draw_targets(p, d["targets"], to_px, cfg, d["colors"])
        if cfg["hud"]:
            f = QFont("Geist Mono"); f.setPixelSize(12); f.setBold(True); p.setFont(f)
            txt = f"{d['fps']:5.1f} FPS   {d['ms']:4.1f} ms   {len(d['targets'])} hit{'s' if len(d['targets']) != 1 else ''}"
            r = QRectF(g.x() + 10, g.y() + 10, p.fontMetrics().horizontalAdvance(txt) + 18, 24)
            p.setPen(Qt.NoPen); p.setBrush(QColor(0, 0, 0, 170)); p.drawRoundedRect(r, 7, 7)
            p.setPen(QColor("#7CFF9B") if d["fps"] >= 30 else QColor("#FFD25A")); p.drawText(r, Qt.AlignCenter, txt)
        if self.paused:
            f = QFont("Geist Mono"); f.setPixelSize(13); f.setBold(True); p.setFont(f)
            txt = f"PAUSED: {self.paused}"
            r = QRectF(g.center().x() - 200, g.center().y() - 16, 400, 32)
            p.setPen(Qt.NoPen); p.setBrush(QColor(0, 0, 0, 200)); p.drawRoundedRect(r, 8, 8)
            p.setPen(QColor("#FFD25A")); p.drawText(r, Qt.AlignCenter, txt)
        if self.eyedrop and self.hover is not None and self.pixel and g.contains(self.hover):
            self._loupe(p, self.hover)
        if self.sampling or self.eyedrop:
            p.setPen(QPen(tok("accent"), 2)); p.setBrush(Qt.NoBrush); p.drawRoundedRect(QRectF(self.rect()).adjusted(1, 1, -1, -1), 14, 14)
        if self.sampling:
            if self.drag and self.drag[0] == "sample":
                sel = QRectF(self.drag[1], self.drag[2]).normalized()
                p.setPen(QPen(QColor("#ffffff"), 1.5, Qt.DashLine)); p.setBrush(QColor(255, 255, 255, 40)); p.drawRect(sel)
            else:
                f = QFont("Geist Mono"); f.setPixelSize(12); f.setBold(True); p.setFont(f)
                txt = "Drag over a colour (or click it) to sample · Esc to cancel"
                r = QRectF(g.center().x() - 210, g.bottom() - 36, 420, 24)
                p.setPen(Qt.NoPen); p.setBrush(QColor(0, 0, 0, 190)); p.drawRoundedRect(r, 7, 7)
                p.setPen(QColor("#ffffff")); p.drawText(r, Qt.AlignCenter, txt)
        p.end()

    def _loupe(self, p, pt):
        """Magnified pixels round the cursor plus the colour's HEX / RGB / HSV / name."""
        got = self.pixel(*(int(v) for v in self._to_frame(pt)))
        if not got: return
        rgb, crop = got
        z, n = 8, crop.shape[0]
        img = QImage(np.ascontiguousarray(crop).data, crop.shape[1], n, crop.strides[0], QImage.Format_RGB888).copy()
        x = pt.x() + 18 if pt.x() + 18 + n * z + 190 < self.width() else pt.x() - 18 - n * z - 190
        y = max(6, min(self.height() - n * z - 6, pt.y() - n * z / 2))
        box = QRectF(x, y, n * z, n * z)
        p.setRenderHint(QPainter.SmoothPixmapTransform, False); p.drawImage(box, img)
        p.setPen(QPen(QColor("#ffffff"), 1.5)); p.setBrush(Qt.NoBrush); p.drawRect(box)
        c = QRectF(box.x() + (n // 2) * z, box.y() + (n // 2) * z, z, z)
        p.setPen(QPen(QColor("#000000"), 2)); p.drawRect(c.adjusted(-1, -1, 1, 1)); p.setPen(QPen(QColor("#ffffff"), 1)); p.drawRect(c)
        i = tools.info(rgb)
        f = QFont("Geist Mono"); f.setPixelSize(12); f.setBold(True); p.setFont(f)
        card = QRectF(box.right() + 6, box.y(), 184, 86)
        p.setPen(Qt.NoPen); p.setBrush(QColor(0, 0, 0, 215)); p.drawRoundedRect(card, 7, 7)
        p.setBrush(QColor(*rgb)); p.drawRoundedRect(QRectF(card.x() + 8, card.y() + 8, 22, 22), 4, 4)
        p.setPen(QColor("#ffffff")); p.drawText(QRectF(card.x() + 38, card.y() + 8, 140, 22), Qt.AlignVCenter, i["hex"])
        f.setBold(False); f.setPixelSize(11); p.setFont(f); p.setPen(QColor("#d8d8d8"))
        for k, key in enumerate(("rgb", "hsv", "name")):
            p.drawText(QRectF(card.x() + 8, card.y() + 36 + k * 15, 172, 15), Qt.AlignVCenter, i[key])

    def leaveEvent(self, _):
        if self.hover is not None: self.hover = None; self.update()

    def mousePressEvent(self, e):
        if not self.data or e.button() != Qt.LeftButton: return
        pt = e.position()
        if self.eyedrop:
            if self._geom().contains(pt) and self.cfg["view"] != "mask": self.picked.emit(*(int(v) for v in self._to_frame(pt)))
            return
        if self.sampling and self.cfg["view"] != "mask":
            if self._geom().contains(pt): self.drag = ("sample", pt, pt)
            return
        hit = self._hit(pt)
        if hit:
            fx, fy = self._to_frame(pt)
            self.drag = (hit, fx, fy, self.cfg["fov_dx"], self.cfg["fov_dy"]); self.update()

    def mouseMoveEvent(self, e):
        pt = e.position()
        if self.eyedrop:
            self.hover = pt; self.setCursor(Qt.CrossCursor); self.update(); return
        if not self.drag:
            hit = None if self.sampling else self._hit(pt)
            self.setCursor(Qt.CrossCursor if self.sampling else Qt.SizeFDiagCursor if hit == "size"
                           else Qt.SizeAllCursor if hit == "move" else Qt.ArrowCursor)
            return
        kind = self.drag[0]
        if kind == "sample":
            self.drag = ("sample", self.drag[1], pt); self.update(); return
        fx, fy = self._to_frame(pt)
        W, H = self.data["size"]
        if kind == "move":
            _, x0, y0, dx0, dy0 = self.drag
            self.cfg["fov_dx"] = int(max(-W // 2, min(W // 2, dx0 + fx - x0)))
            self.cfg["fov_dy"] = int(max(-H // 2, min(H // 2, dy0 + fy - y0)))
        else:
            x, y, w, h = self._fov(); cx, cy = x + w / 2, y + h / 2
            r = ((fx - cx) ** 2 + (fy - cy) ** 2) ** 0.5 if self.cfg["fov_shape"] == "circle" else max(abs(fx - cx), abs(fy - cy))
            self.cfg["fov"] = int(max(60, min(2000, 2 * r)))
        self.update()

    def mouseReleaseEvent(self, e):
        if not self.drag: return
        kind = self.drag[0]
        if kind == "sample":
            (x0, y0), (x1, y1) = self._to_frame(self.drag[1]), self._to_frame(self.drag[2])
            if abs(x1 - x0) < 4 and abs(y1 - y0) < 4: x0, y0, x1, y1 = x1 - 3, y1 - 3, x1 + 4, y1 + 4
            self.drag = None
            self.sampled.emit(int(min(x0, x1)), int(min(y0, y1)), int(abs(x1 - x0)) + 1, int(abs(y1 - y0)) + 1)
        else:
            self.drag = None; self.fovMoved.emit()
        self.update()


# ── Click-through overlay on the real screen ─────────────────────────────────
class Overlay(QWidget):
    def __init__(self, cfg):
        super().__init__(None, Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.WindowTransparentForInput
                         | Qt.WindowDoesNotAcceptFocus | Qt.Tool)
        self.cfg, self.data = cfg, None
        self.setWindowTitle(OVERLAY_TITLE)
        self.setAttribute(Qt.WA_TranslucentBackground); self.setAttribute(Qt.WA_ShowWithoutActivating)

    def show_on(self, index):  # fullscreen on the chosen monitor; the KWin rule keeps it above and out of the capture
        screens = QApplication.screens()
        scr = screens[index] if index < len(screens) else QApplication.primaryScreen()
        self.setScreen(scr); self.setGeometry(scr.geometry()); self.showFullScreen()

    def show_frame(self, d): self.data = d; self.update()

    def scale(self):
        if not self.data: return 1.0
        return self.width() / max(1, self.data["size"][0])

    def paintEvent(self, _):
        p = QPainter(self); p.setRenderHint(QPainter.Antialiasing)
        p.setCompositionMode(QPainter.CompositionMode_Source); p.fillRect(self.rect(), Qt.transparent)
        p.setCompositionMode(QPainter.CompositionMode_SourceOver)
        d, cfg = self.data, self.cfg
        if d:
            kx, ky = self.width() / d["size"][0], self.height() / d["size"][1]
            to_px = lambda r: QRectF(r.x() * kx, r.y() * ky, r.width() * kx, r.height() * ky)  # noqa: E731
            if d.get("assist") is not None:
                p.setRenderHint(QPainter.SmoothPixmapTransform, False); p.drawImage(to_px(QRectF(*d["fov"])), d["assist"])
            if cfg["overlay_fov"]: draw_fov(p, d["fov"], to_px, cfg, ring_pad=4)
            draw_targets(p, d["targets"], to_px, cfg, d["colors"], overlay=True)
            if cfg["overlay_fps"]:
                f = QFont("Geist Mono"); f.setPixelSize(13); f.setBold(True); p.setFont(f)
                txt = f"ChromaScope  {d['fps']:.0f} FPS  {len(d['targets'])} hits"
                r = QRectF(14, 14, p.fontMetrics().horizontalAdvance(txt) + 20, 26)
                p.setPen(Qt.NoPen); p.setBrush(QColor(0, 0, 0, 150)); p.drawRoundedRect(r, 7, 7)
                p.setPen(QColor("#e8e8e8")); p.drawText(r, Qt.AlignCenter, txt)
        p.end()


KWIN_RULES = {
    # Wayland apps can't keep themselves on top or hide from screen capture; KWin window rules can.
    "chromascope-overlay": {"Description": "ChromaScope overlay (keep above, click-through, hidden from capture)",
                            "title": OVERLAY_TITLE, "titlematch": "1", "above": "true", "aboverule": "2",
                            "acceptfocus": "false", "acceptfocusrule": "2", "skiptaskbar": "true", "skiptaskbarrule": "2",
                            "skipswitcher": "true", "skipswitcherrule": "2", "skippager": "true", "skippagerrule": "2",
                            "noborder": "true", "noborderrule": "2", "excludefromcapture": "true", "excludefromcapturerule": "2"},
    "chromascope-window": {"Description": "ChromaScope window (hidden from capture, so it never detects itself)",
                           "title": "ChromaScope", "titlematch": "1", "excludefromcapture": "true", "excludefromcapturerule": "2"},
}


def ensure_kwin_rules():
    """Write the KWin rules above (idempotent) and tell KWin to reload them."""
    rc = ["--file", "kwinrulesrc"]
    try:
        rules = subprocess.run(["kreadconfig6", *rc, "--group", "General", "--key", "rules"],
                               capture_output=True, text=True, timeout=3).stdout.strip()
        ids = [r for r in rules.split(",") if r]
        for group, keys in KWIN_RULES.items():
            for k, v in keys.items():
                subprocess.run(["kwriteconfig6", *rc, "--group", group, "--key", k, v], timeout=3)
            if group not in ids: ids.append(group)
        subprocess.run(["kwriteconfig6", *rc, "--group", "General", "--key", "rules", ",".join(ids)], timeout=3)
        subprocess.run(["kwriteconfig6", *rc, "--group", "General", "--key", "count", str(len(ids))], timeout=3)
        subprocess.run(["dbus-send", "--session", "--type=signal", "/KWin", "org.kde.KWin.reloadConfig"], timeout=3)
        return True
    except Exception:
        return False


# ── Settings widgets ─────────────────────────────────────────────────────────
class Panel(QWidget):
    """A page of settings rows bound to cfg keys. Controls register in `reg` so they can be re-synced."""

    def __init__(self, cfg, changed, reg):
        super().__init__()
        self.cfg, self.changed, self.reg = cfg, changed, reg
        self.lay = QVBoxLayout(self); self.lay.setContentsMargins(2, 2, 6, 2); self.lay.setSpacing(10)

    def section(self, title):
        self.lay.addSpacing(4); self.lay.addWidget(label(title.upper(), 11, muted=True, bold=True))

    def slider(self, title, key, lo, hi, fmt="{}", hint=None):
        box = QWidget(); g = QGridLayout(box); g.setContentsMargins(0, 0, 0, 0); g.setVerticalSpacing(0)
        val = label(fmt.format(self.cfg[key]), 12, mono=True)
        s = uv.Slider(self.cfg[key], lo, hi)
        g.addWidget(label(title, 13), 0, 0); g.addWidget(val, 0, 1, Qt.AlignRight); g.addWidget(s, 1, 0, 1, 2)
        if hint: g.addWidget(label(hint, 11, muted=True), 2, 0, 1, 2)
        s.valueChanged.connect(lambda v: (val.setText(fmt.format(v)), self.set(key, v)))
        self.reg[key] = ("slider", s); self.lay.addWidget(box); return s

    def choice(self, title, key, options):
        row = QVBoxLayout(); row.setSpacing(4)
        row.addWidget(label(title, 13))
        tb = uv.TabBar([o[0] for o in options]); tb.setUsesScrollButtons(False)
        vals = [o[1] for o in options]
        tb.setCurrentIndex(vals.index(self.cfg[key]) if self.cfg[key] in vals else 0)
        tb.currentChanged.connect(lambda i: self.set(key, vals[i]))
        self.reg[key] = ("choice", tb, vals)
        row.addWidget(tb, 0, Qt.AlignLeft); self.lay.addLayout(row); return tb

    def switch(self, text, key):
        s = uv.Switch(text, bool(self.cfg[key])); s.toggled.connect(lambda v: self.set(key, v))
        self.reg[key] = ("switch", s); self.lay.addWidget(s); return s

    def swatches(self, items, current, on_pick):
        row = QHBoxLayout(); row.setSpacing(2); sw = []
        for n, c in items:
            s = Swatch(n, c); s.on = c.lower() == (current or "").lower()
            s.clicked.connect(lambda col: (on_pick(col), [(setattr(w, "on", w.color == col), w.update()) for w in sw]))
            row.addWidget(s); sw.append(s)
        row.addStretch(1); self.lay.addLayout(row); return sw

    def set(self, key, value):
        self.cfg[key] = value; self.changed(key)

    def end(self, reset=None):
        self.lay.addStretch(1)
        if reset:
            b = uv.Button("Reset this tab", size="sm"); b.setToolTip("Put this tab's settings back to defaults (undo with Ctrl+Z)")
            b.clicked.connect(reset); self.lay.addWidget(b, 0, Qt.AlignLeft)


class Swatch(QWidget):
    clicked = Signal(str)

    def __init__(self, name, color, size=30):
        super().__init__(); self.name, self.color, self.on, self.px = name, color, False, size
        self.setFixedSize(size, size); self.setCursor(Qt.PointingHandCursor); self.setToolTip(name)

    def paintEvent(self, _):
        p = QPainter(self); p.setRenderHint(QPainter.Antialiasing); s = self.px
        if self.on: p.setPen(QPen(tok("text"), 2)); p.setBrush(Qt.NoBrush); p.drawEllipse(QRectF(1, 1, s - 2, s - 2))
        p.setPen(QPen(QColor(0, 0, 0, 60), 1)); p.setBrush(QColor(self.color)); p.drawEllipse(QRectF(s / 6, s / 6, s * 2 / 3, s * 2 / 3)); p.end()

    def mousePressEvent(self, _): self.clicked.emit(self.color)


class ColorRow(QWidget):
    """One entry in the colour list: on/off, target swatch, name, box colour, remove."""
    picked, toggled, removed, renamed = Signal(), Signal(bool), Signal(), Signal()

    def __init__(self, c, selected, can_remove):
        super().__init__(); self.selected = selected
        self.setCursor(Qt.PointingHandCursor); self.setFixedHeight(40)
        h = QHBoxLayout(self); h.setContentsMargins(8, 2, 6, 2); h.setSpacing(8)
        on = uv.CheckBox("", c.get("on", True)); on.toggled.connect(self.toggled.emit); on.setToolTip("Detect this colour")
        sw = Swatch(c["name"], c["color"], 24); sw.clicked.connect(lambda _c: self.picked.emit())
        name = label(c["name"], 13, bold=selected)
        info = label(f"±{c['tolerance']}  ", 11, mono=True, muted=True); info.setWordWrap(False)
        box = Swatch("Box colour", c["box"], 16); box.setToolTip("Box colour"); box.clicked.connect(lambda _c: self.picked.emit())
        h.addWidget(on); h.addWidget(sw); h.addWidget(name); h.addStretch(1); h.addWidget(info); h.addWidget(box)
        if can_remove:
            rm = uv.Button("✕", size="sm"); rm.setToolTip("Remove"); rm.clicked.connect(self.removed.emit); h.addWidget(rm)

    def paintEvent(self, _):
        if not self.selected: return
        p = QPainter(self); p.setRenderHint(QPainter.Antialiasing)
        a = QColor(tok("accent")); a.setAlpha(45); p.setBrush(a); p.setPen(QPen(tok("accent"), 1.2))
        p.drawRoundedRect(QRectF(self.rect()).adjusted(1, 1, -1, -1), 6, 6); p.end()

    def mousePressEvent(self, _): self.picked.emit()
    def mouseDoubleClickEvent(self, _): self.renamed.emit()


# ── Tools widgets ────────────────────────────────────────────────────────────
class ContrastSample(QWidget):
    """Shows text colour A on background B, big and small, like the WCAG checks."""

    def __init__(self):
        super().__init__(); self.a, self.b = "#000000", "#ffffff"; self.setFixedHeight(64)

    def paintEvent(self, _):
        p = QPainter(self); p.setRenderHint(QPainter.Antialiasing)
        p.setPen(QPen(QColor(0, 0, 0, 60), 1)); p.setBrush(QColor(self.b)); p.drawRoundedRect(QRectF(self.rect()).adjusted(1, 1, -1, -1), 6, 6)
        p.setPen(QColor(self.a)); f = p.font(); f.setPixelSize(26); f.setBold(True); p.setFont(f)
        p.drawText(QRectF(14, 0, 60, self.height()), Qt.AlignVCenter, "Aa")
        f.setPixelSize(13); f.setBold(False); p.setFont(f)
        p.drawText(QRectF(80, 0, self.width() - 90, self.height()), Qt.AlignVCenter | Qt.TextWordWrap, "The quick brown fox jumps over the lazy dog.")
        p.end()


# ── Main window ──────────────────────────────────────────────────────────────
TABS = ["Detect", "FOV", "Draw", "Tools", "Setup", "NNMM"]


class Window(uv.Surface):
    def __init__(self, app):
        super().__init__()
        self.app, self.cfg = app, load_settings()
        self.engine, self.overlay, self.running = None, Overlay(self.cfg), False
        self.reg, self.restoring, self.pick_for, self.eye_rgb = {}, False, None, None
        self.setWindowTitle("ChromaScope"); self.resize(1240, 720); self.setMinimumSize(980, 600)
        self.restyle_app()

        self.history = UndoStack()
        self.undo_timer = QTimer(self); self.undo_timer.setSingleShot(True); self.undo_timer.setInterval(500)
        self.undo_timer.timeout.connect(self.commit_undo)
        self.lock = LockWatcher(); self.lock.changed.connect(lambda _l: self.update_pause())
        self.focus = FocusWatcher(CONFIG); self.focus.changed.connect(lambda *_: self.update_pause())
        self.cursor = CursorWatcher(CONFIG)
        self.live = nnmm.Live(self.cfg, MODEL_DIRS, self.cursor, self.screen_map, self); self.live.changed.connect(self.show_live)
        self.rbutton = RightButton(self.cfg["nn_hold"]); self.rbutton.held.connect(self.on_right)

        root = QVBoxLayout(self); root.setContentsMargins(22, 14, 22, 16); root.setSpacing(12)
        head = QHBoxLayout(); root.addLayout(head)
        head.addWidget(uv.Heading("ChromaScope", 34))
        self.status = uv.Badge("Idle", "warn"); head.addWidget(self.status, 0, Qt.AlignVCenter)
        self.undo_box = QWidget(); ub = QHBoxLayout(self.undo_box); ub.setContentsMargins(8, 0, 0, 0); ub.setSpacing(4)
        self.undo_btn = uv.Button("↶", size="sm"); self.undo_btn.setToolTip("Undo (Ctrl+Z)"); self.undo_btn.clicked.connect(self.undo)
        self.redo_btn = uv.Button("↷", size="sm"); self.redo_btn.setToolTip("Redo (Ctrl+Shift+Z)"); self.redo_btn.clicked.connect(self.redo)
        ub.addWidget(self.undo_btn); ub.addWidget(self.redo_btn); head.addWidget(self.undo_box, 0, Qt.AlignVCenter)
        head.addStretch(1)
        self.view_tabs = uv.TabBar(["Screen", "FOV zoom", "Mask"]); self.view_tabs.setUsesScrollButtons(False)
        self.view_tabs.setCurrentIndex(["full", "fov", "mask"].index(self.cfg["view"]))
        self.view_tabs.currentChanged.connect(lambda i: self.changed("view", ["full", "fov", "mask"][i]))
        head.addWidget(self.view_tabs); head.addSpacing(14)
        self.ov_switch = uv.Switch("Show on main screen", self.cfg["overlay"]); self.ov_switch.toggled.connect(self.set_overlay)
        head.addWidget(self.ov_switch)
        self.go = uv.Button("Start capture", primary=True); self.go.clicked.connect(self.toggle); head.addWidget(self.go)

        body = QHBoxLayout(); body.setSpacing(16); root.addLayout(body, 1)
        left = QVBoxLayout(); left.setSpacing(10); body.addLayout(left, 1)
        self.preview = Preview(self.cfg); self.preview.pixel = self.pixel_at
        self.preview.sampled.connect(self.sample_at); self.preview.fovMoved.connect(self.fov_moved); self.preview.picked.connect(self.eye_picked)
        left.addWidget(self.preview, 1)
        stats = QHBoxLayout(); stats.setSpacing(6); left.addLayout(stats)
        self.b_cap, self.b_proc, self.b_ms = uv.Badge("Capture —", "info"), uv.Badge("Process —", "accent"), uv.Badge("— ms", "accent")
        self.b_hits, self.b_res = uv.Badge("0 hits", "success"), uv.Badge("—", "info")
        for b in (self.b_cap, self.b_proc, self.b_ms, self.b_hits, self.b_res): stats.addWidget(b)
        stats.addStretch(1)
        self.mouse_sw = uv.Switch("NN mouse (F7)", False); self.mouse_sw.toggled.connect(self.set_live)
        self.mouse_sw.setToolTip("NNMM live mouse: steer the mouse onto the nearest detected box (settings on the NNMM tab)")
        stats.addWidget(self.mouse_sw)

        side = uv.Card(padding=16); side.setFixedWidth(470); body.addWidget(side)
        self.nav = uv.NavBar(TABS); side.layout().addWidget(self.nav)
        self.pages = QStackedWidget(); side.layout().addWidget(self.pages, 1)
        self.build_pages()
        self.nav.currentChanged.connect(self.pages.setCurrentIndex)

        QShortcut(QKeySequence("F8"), self, self.toggle)
        QShortcut(QKeySequence("F9"), self, lambda: self.ov_switch.setChecked(not self.ov_switch.isChecked()))
        QShortcut(QKeySequence("F7"), self, lambda: self.set_live(not self.live.on))
        QShortcut(QKeySequence("F10"), self, lambda: self.view_tabs.setCurrentIndex((self.view_tabs.currentIndex() + 1) % 3))
        QShortcut(QKeySequence("Escape"), self, self.cancel_modes)
        QShortcut(QKeySequence(QKeySequence.StandardKey.Undo), self, self.undo)
        QShortcut(QKeySequence("Ctrl+Shift+Z"), self, self.redo); QShortcut(QKeySequence("Ctrl+Y"), self, self.redo)
        uv.style().changed.connect(self.refresh_labels)
        self.commit_undo(); self.update_pause()

    def go_tab(self, i):
        self.nav.current = i; self.nav.update(); self.pages.setCurrentIndex(i)

    def build_pages(self):
        """(Re)build every settings page from cfg — at start, after loading a profile, undo/redo and resets."""
        cur = self.pages.currentIndex()
        while self.pages.count():
            w = self.pages.widget(0); self.pages.removeWidget(w); w.deleteLater()
        self.reg = {}
        for page in (self.page_detect(), self.page_fov(), self.page_highlight(), self.page_tools(), self.page_setup(), self.page_nnmm()):
            sc = QScrollArea(); sc.setWidgetResizable(True); sc.setFrameShape(QScrollArea.NoFrame)
            sc.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
            sc.setStyleSheet("QScrollArea, QScrollArea > QWidget > QWidget { background: transparent; }"
                             "QScrollBar:vertical { background: transparent; width: 6px; margin: 0; }"
                             "QScrollBar::handle:vertical { background: rgba(128,128,128,110); border-radius: 3px; min-height: 30px; }"
                             "QScrollBar::add-line, QScrollBar::sub-line { height: 0; }"
                             "QScrollBar::add-page, QScrollBar::sub-page { background: none; }")
            sc.setWidget(page); self.pages.addWidget(sc)
        if cur >= 0: self.pages.setCurrentIndex(cur)

    # ── pages ────────────────────────────────────────────────────────────────
    def page_detect(self):
        p = Panel(self.cfg, self.changed, self.reg)
        p.section("Profile")
        self.prof = QComboBox(); self.refresh_profiles()
        self.prof.activated.connect(lambda _i: self.load_profile(self.prof.currentText()))
        p.lay.addWidget(self.prof)
        row = QHBoxLayout()
        for text, fn in (("Save", self.save_profile), ("Save as…", lambda: self.save_profile(ask=True)), ("Delete", self.delete_profile)):
            b = uv.Button(text, size="sm"); b.clicked.connect(fn); row.addWidget(b)
        row.addStretch(1); p.lay.addLayout(row)

        p.section("Colours to find")
        self.color_list = QWidget(); self.color_box = QVBoxLayout(self.color_list)
        self.color_box.setContentsMargins(0, 0, 0, 0); self.color_box.setSpacing(2); p.lay.addWidget(self.color_list)
        add = uv.Button("+ Add colour", size="sm"); add.clicked.connect(self.add_color); self.add_btn = add
        p.lay.addWidget(add, 0, Qt.AlignLeft)
        self.render_colors()

        p.section("Selected colour")
        self.color_sw = p.swatches(PRESETS, self.sel()["color"], self.set_color)
        self.cur = label("", 12, mono=True); p.lay.addWidget(self.cur)
        row2 = QHBoxLayout()
        b = uv.Button("Custom…", size="sm"); b.clicked.connect(self.custom_color); row2.addWidget(b)
        self.sample_btn = uv.Button("Sample from preview", size="sm"); self.sample_btn.clicked.connect(self.toggle_sample); row2.addWidget(self.sample_btn)
        row2.addStretch(1); p.lay.addLayout(row2)
        p.lay.addWidget(label("Sample: drag a box over the thing you want (or click it). The range is worked out from its pixels.", 11, muted=True))
        p.slider("Hue tolerance", "tolerance", 1, 40, "±{}", "How far from the target hue still counts.")
        p.slider("Min saturation", "min_sat", 0, 255, "{}", "Raise to ignore greyish / washed-out pixels.")
        p.slider("Min brightness", "min_val", 0, 255, "{}")
        p.lay.addWidget(label("Box colour", 13))
        self.box_sw = p.swatches(BOX_COLORS, self.sel()["box"], self.set_box)
        self.show_color()

        p.section("Objects")
        p.slider("Min size", "min_area", 1, 2000, "{} px")
        p.slider("Merge distance", "merge", 0, 30, "{} px", "Joins nearby pieces (outlines) into one box.")
        p.slider("Max boxes", "max_targets", 1, 64, "{}")
        p.choice("Order", "sort", [("Largest first", "largest"), ("Closest to centre", "closest")])
        p.end(lambda: self.reset_tab("Detect")); return p

    def page_fov(self):
        p = Panel(self.cfg, self.changed, self.reg)
        p.section("Field of view")
        p.lay.addWidget(label("Tip: drag the ring in the preview to move it, drag its edge to resize.", 11, muted=True))
        p.choice("Shape", "fov_shape", [("Circle", "circle"), ("Square", "square"), ("Full screen", "full")])
        p.slider("Size", "fov", 60, 2000, "{} px", "Only this area is scanned. Smaller = faster.")
        p.slider("Offset X", "fov_dx", -2000, 2000, "{} px")
        p.slider("Offset Y", "fov_dy", -2000, 2000, "{} px")
        b = uv.Button("Back to centre", size="sm"); b.clicked.connect(lambda: self.apply_values(fov_dx=0, fov_dy=0))
        p.lay.addWidget(b, 0, Qt.AlignLeft)
        p.switch("Show FOV ring", "show_fov")
        p.switch("Centre crosshair", "crosshair")
        p.section("Saved regions")
        self.region_box = QComboBox(); self.fill_regions()
        self.region_box.activated.connect(lambda _i: self.load_region(self.region_box.currentText()))
        p.lay.addWidget(self.region_box)
        row = QHBoxLayout()
        for text, fn in (("Save region…", self.save_region), ("Delete", self.delete_region)):
            b = uv.Button(text, size="sm"); b.clicked.connect(fn); row.addWidget(b)
        row.addStretch(1); p.lay.addLayout(row)
        p.section("Performance")
        p.slider("Max FPS", "max_fps", 5, 240, "{}", "Caps how often frames are processed.")
        p.end(lambda: self.reset_tab("FOV")); return p

    def page_highlight(self):
        p = Panel(self.cfg, self.changed, self.reg)
        p.section("Highlight mode")
        p.choice("Style", "mode", [("Box", "box"), ("Glow", "glow"), ("Both", "both"), ("Corners", "corners"), ("None", "none")])
        p.lay.addWidget(label("Glow turns the detected pixels a red shade with a soft halo on the display. Box colours are set per colour on the Detect tab.", 11, muted=True))
        p.slider("Glow radius", "glow", 0, 30, "{} px")
        p.slider("Line thickness", "thickness", 1, 6, "{} px")
        p.section("Colour-blind assist")
        self.assist_sw = p.switch("Recolour the detected colours", "assist")
        p.choice("Style", "assist_style", [("Stripes", "stripes"), ("Solid", "solid"), ("Outline", "outline")])
        p.lay.addWidget(label("Show as", 13))
        p.swatches([("Blue", "#2a6bff"), ("Yellow", "#ffe100"), ("White", "#ffffff"), ("Magenta", "#ff2ad4"), ("Cyan", "#00e5ff"), ("Black", "#111111")],
                   self.cfg["assist_color"], lambda c: self.changed("assist_color", c))
        p.lay.addWidget(label("Makes a hue you find hard to tell apart stand out, e.g. reds as blue stripes. Set the FOV shape to "
                              "Full screen and turn on Show on main screen to see it over every app.", 11, muted=True))
        p.section("Preview")
        p.switch("Name labels on boxes", "labels")
        p.switch("FPS / timing HUD", "hud")
        p.end(lambda: self.reset_tab("Draw")); return p

    def page_tools(self):
        p = Panel(self.cfg, self.changed, self.reg)
        p.section("Eyedropper")
        row = QHBoxLayout()
        self.eye_btn = uv.Button("Eyedropper", size="sm"); self.eye_btn.clicked.connect(lambda: self.start_eyedrop("info"))
        self.eye_btn.setToolTip("Hover the preview for a magnified view; click to copy the colour"); row.addWidget(self.eye_btn)
        self.copy_btn = uv.Button("Copy HEX", size="sm"); self.copy_btn.clicked.connect(lambda: self.copy(tools.info(self.eye_rgb)["hex"]))
        self.target_btn = uv.Button("Add as target", size="sm"); self.target_btn.clicked.connect(self.eye_to_target)
        row.addWidget(self.copy_btn); row.addWidget(self.target_btn); row.addStretch(1); p.lay.addLayout(row)
        info = QHBoxLayout(); info.setSpacing(12)
        self.eye_sw = Swatch("Picked colour", "#808080", 46); info.addWidget(self.eye_sw, 0, Qt.AlignTop)
        self.eye_info = label("Start the eyedropper, then click a pixel in the preview.", 12, mono=True); self.eye_info.setWordWrap(True)
        info.addWidget(self.eye_info, 1); p.lay.addLayout(info)
        self.copied = label("", 11, muted=True); p.lay.addWidget(self.copied)
        p.lay.addWidget(label("History", 13))
        self.hist_row = QHBoxLayout(); self.hist_row.setSpacing(2); p.lay.addLayout(self.hist_row)
        self.render_history(); self.show_eye()

        p.section("Contrast checker")
        cr = QHBoxLayout(); cr.setSpacing(6)
        self.ca_sw, self.cb_sw = Swatch("Text", self.cfg["contrast_a"], 34), Swatch("Background", self.cfg["contrast_b"], 34)
        self.ca_sw.clicked.connect(lambda _c: self.start_eyedrop("A")); self.cb_sw.clicked.connect(lambda _c: self.start_eyedrop("B"))
        cr.addWidget(self.ca_sw); cr.addWidget(label("Text", 12)); cr.addSpacing(8); cr.addWidget(self.cb_sw); cr.addWidget(label("Background", 12)); cr.addStretch(1)
        p.lay.addLayout(cr)
        row = QHBoxLayout()
        for text, fn in (("Pick text", lambda: self.start_eyedrop("A")), ("Pick background", lambda: self.start_eyedrop("B")), ("Swap", self.swap_contrast)):
            b = uv.Button(text, size="sm"); b.clicked.connect(fn); row.addWidget(b)
        row.addStretch(1); p.lay.addLayout(row)
        self.ratio = uv.Heading("", 30); p.lay.addWidget(self.ratio)
        badges = QHBoxLayout(); badges.setSpacing(4); self.wcag = {}
        for k in ("AA", "AA large", "AAA", "AAA large"):
            b = uv.Badge(k, "success"); self.wcag[k] = b; badges.addWidget(b)
        badges.addStretch(1); p.lay.addLayout(badges)
        self.csample = ContrastSample(); p.lay.addWidget(self.csample)
        p.lay.addWidget(label("WCAG 2: AA needs 4.5:1 for normal text and 3:1 for large text; AAA needs 7:1 and 4.5:1.", 11, muted=True))
        self.show_contrast()

        p.section("Palette from the FOV")
        b = uv.Button("Extract colours", size="sm"); b.clicked.connect(self.extract_palette); p.lay.addWidget(b, 0, Qt.AlignLeft)
        self.pal_box = QVBoxLayout(); self.pal_box.setSpacing(2); p.lay.addLayout(self.pal_box)
        p.lay.addWidget(label("Finds the 8 main colours in the FOV right now. Add any of them as a colour to find.", 11, muted=True))
        p.end(); return p

    def page_setup(self):
        p = Panel(self.cfg, self.changed, self.reg)
        p.section("Overlay on your screen")
        p.lay.addWidget(label("Draws the FOV and highlights on top of everything, click-through. Toggle with the switch at the top or F9.", 11, muted=True))
        p.switch("Draw FOV ring on screen", "overlay_fov")
        p.switch("FPS counter on screen", "overlay_fps")
        row = QHBoxLayout(); row.addWidget(label("Monitor", 13)); row.addStretch(1)
        self.screen_box = QComboBox(); self.screen_box.addItems([f"{i + 1}: {s.name()}  {s.size().width()}×{s.size().height()}" for i, s in enumerate(QApplication.screens())])
        self.screen_box.setCurrentIndex(min(self.cfg["overlay_screen"], self.screen_box.count() - 1))
        self.screen_box.currentIndexChanged.connect(lambda i: self.changed("overlay_screen", i))
        row.addWidget(self.screen_box); p.lay.addLayout(row)

        p.section("Capture")
        p.choice("Source", "source", [("Monitor", MONITOR), ("Window", WINDOW)])
        p.lay.addWidget(label("Window capture doesn't line up with the overlay (Wayland hides window positions); use Monitor for that.", 11, muted=True))
        p.switch("Include mouse cursor", "cursor")
        p.switch("Minimize ChromaScope while capturing", "minimize")
        b = uv.Button("Choose a different screen", size="sm"); b.clicked.connect(self.forget_screen); p.lay.addWidget(b, 0, Qt.AlignLeft)

        p.section("Auto-pause")
        p.switch("Pause while the screen is locked", "pause_locked")
        p.switch("Pause when the screen stops changing", "pause_static")
        p.slider("…after", "pause_static_s", 5, 300, "{} s")
        self.focus_sw = p.switch("Only run while a chosen app is focused", "pause_focus")
        self.focus_edit = uv.LineEdit(self.cfg["focus_match"], "App or window title, e.g. firefox")
        self.focus_edit.textChanged.connect(lambda t: self.changed("focus_match", t)); p.lay.addWidget(self.focus_edit)
        self.focus_seen = QComboBox(); self.focus_seen.addItem("(recently focused apps)"); self.focus_seen.addItems(self.focus.seen)
        self.focus_seen.activated.connect(lambda i: i and self.focus_edit.setText(self.focus_seen.currentText()))
        p.lay.addWidget(self.focus_seen)
        p.lay.addWidget(label("Paused capture skips all processing, so it costs almost no CPU. The focused app is reported by a tiny KWin script.", 11, muted=True))

        p.section("Share profiles")
        g = QGridLayout(); g.setSpacing(4)
        for i, (text, fn) in enumerate((("Export file…", self.export_profile), ("Import file…", self.import_profile_file),
                                        ("Copy share code", self.copy_code), ("Paste share code", self.paste_code),
                                        ("Add example profiles", self.add_examples))):
            b = uv.Button(text, size="sm"); b.clicked.connect(fn); g.addWidget(b, i // 2, i % 2)
        p.lay.addLayout(g)
        self.share_msg = label(getattr(self, "share_note", "Share codes are short text (CSP1:…) you can paste in a chat."), 11, muted=True)
        p.lay.addWidget(self.share_msg)

        p.section("Look (UI Vault kits)")
        grid = QGridLayout(); grid.setSpacing(4)
        for i, k in enumerate(uv.kits()):
            b = uv.Button(k.replace("-", " ").title(), size="sm")
            b.clicked.connect(lambda _=False, k=k: (self.cfg.update(kit=k, theme=None), self.restyle_app(), save_settings(self.cfg)))
            grid.addWidget(b, i // 3, i % 3)
        p.lay.addLayout(grid)
        row = QHBoxLayout(); row.addWidget(label("Theme", 13)); row.addStretch(1)
        th = QComboBox(); th.addItems(["(kit's theme)"] + uv.themes()); th.setMinimumWidth(170)
        if self.cfg["theme"] in uv.themes(): th.setCurrentText(self.cfg["theme"])
        th.currentTextChanged.connect(lambda t: (self.cfg.update(theme=None if t.startswith("(") else t), self.restyle_app(), save_settings(self.cfg)))
        row.addWidget(th); p.lay.addLayout(row)

        p.section("Help")
        b = uv.Button("Take the tour again", size="sm"); b.clicked.connect(self.start_tour); p.lay.addWidget(b, 0, Qt.AlignLeft)
        p.end(lambda: self.reset_tab("Setup")); return p

    def page_nnmm(self):
        p = Panel(self.cfg, self.changed, self.reg)
        p.section("Neural-network mouse")
        p.lay.addWidget(label("Load a model, start capture, turn this on, then hold right click: the model moves the mouse onto the "
                              "nearest detected box. Let go to stop.", 11, muted=True))
        row = QHBoxLayout(); row.addWidget(label("Model", 13)); row.addStretch(1)
        self.model_box = QComboBox(); self.model_box.setMinimumWidth(200); self.fill_models()
        self.model_box.activated.connect(lambda _i: self.pick_model(self.model_box.currentData()))
        row.addWidget(self.model_box); p.lay.addLayout(row)
        row = QHBoxLayout(); row.setSpacing(4)
        for text, fn in (("Load model (.npz)…", self.import_model), ("Open models folder", self.open_models)):
            b = uv.Button(text, size="sm"); b.clicked.connect(fn); row.addWidget(b)
        row.addStretch(1); p.lay.addLayout(row)
        self.model_msg = label(getattr(self, "model_note_text", "A model takes 6 inputs (target offset x/y, target speed x/y, current speed "
                                                                "x/y) and gives 2 outputs (speed x/y). OpenCV DNN runs it."), 11, muted=True)
        p.lay.addWidget(self.model_msg)
        p.lay.addSpacing(6)
        p.choice("Hold right click on", "nn_hold", [("Mouse or touchpad", "both"), ("Mouse", "mouse"), ("Touchpad", "touchpad")])
        self.live_sw = uv.Switch("Enable (F7)", self.live.on); self.live_sw.toggled.connect(self.set_live); p.lay.addWidget(self.live_sw)
        self.live_msg = label(self.live.status, 12, mono=True); self.live_msg.setWordWrap(True); p.lay.addWidget(self.live_msg)
        self.input_msg = label(self.input_text(), 11, muted=True); p.lay.addWidget(self.input_msg)
        p.end(lambda: (self.reset_tab("NNMM"), setattr(self.rbutton, "mode", self.cfg["nn_hold"]))); return p

    # ── NNMM ─────────────────────────────────────────────────────────────────
    def fill_models(self):
        b = self.model_box; b.blockSignals(True); b.clear()
        for n in sorted({n for f in MODEL_DIRS for n in nnio.names(f)}): b.addItem(n, n)
        i = b.findData(self.cfg["nn_model"]); b.setCurrentIndex(max(0, i)); b.blockSignals(False)

    def model_note(self, text):
        self.model_note_text = text
        if hasattr(self, "model_msg"): self.model_msg.setText(text)

    def pick_model(self, name):
        self.cfg["nn_model"] = name; save_settings(self.cfg)
        if self.live.on and not self.live.load_model(): self.set_live(False)
        self.show_live()

    def import_model(self):
        path, _ = QFileDialog.getOpenFileName(self, "Import model", str(Path.home()), "Models (*.npz *.onnx)")
        if not path: return
        try:
            m = nnio.CvModel(path); bad = m.check()
            if bad: raise ValueError(bad)
        except Exception as e:  # noqa: BLE001 - show why (format, shape, OpenCV, doesn't steer)
            self.model_note(f"Can't use that model: {e}"[:240]); return
        MODELS.mkdir(parents=True, exist_ok=True); dst = MODELS / Path(path).name
        dst.write_bytes(Path(path).read_bytes())
        self.fill_models(); self.model_box.setCurrentIndex(self.model_box.findData(dst.stem)); self.pick_model(dst.stem)
        self.model_note(f"Imported {dst.name} ({m.layers or 'ONNX'}); OpenCV runs it and it steers towards targets.")

    def open_models(self):
        MODELS.mkdir(parents=True, exist_ok=True); QDesktopServices.openUrl(QUrl.fromLocalFile(str(MODELS)))

    def input_text(self):
        mode = self.cfg["nn_hold"]
        how = {"mouse": "Hold the mouse's right button", "touchpad": "Press and hold the touchpad's bottom-right corner (or press with two fingers)",
               "both": "Hold the mouse's right button, or press and hold the touchpad's bottom-right corner"}[mode]
        if not self.rbutton.thread: return f"{how} in any window to move, let go to stop. Turn on Enable to start listening."
        if self.rbutton.denied and not self.rbutton.ok:
            return "Can't read your mouse buttons (no access to /dev/input). Run ./setup.sh once (it asks for your password)."
        kinds = ["mouse", "touchpad"] if mode == "both" else [mode]
        found = [n for k in kinds for n in self.rbutton.names(k)]
        if not found:
            return f"No {' or '.join(kinds)} found yet. Plug one in; it's picked up within 2 s."
        return f"{how}. Listening on: {', '.join(sorted(set(found)))}."

    def on_right(self, down):
        self.live.set_held(down)

    def set_live(self, on):
        if on == self.live.on: return
        if on:
            self.rbutton.start()
            if not self.live.start(): self.rbutton.stop()
            elif not self.running: self.live.status = "Enabled. Start capture (F8) so it can see targets."
        else: self.live.stop(); self.rbutton.stop()
        self.update_pause(); self.show_live()

    def screen_map(self, w, h):
        """Where the captured monitor sits on the desktop: (x, y, frame px per desktop px, height in desktop px)."""
        screens = QApplication.screens()
        g = screens[min(self.cfg["overlay_screen"], len(screens) - 1)].geometry()
        return g.x(), g.y(), w / max(1, g.width()), g.height()

    def show_live(self):
        if not hasattr(self, "live_msg") or not hasattr(self, "mouse_sw"): return
        l = self.live
        txt = l.status + (f"\nmodel {l.model.name} · {l.targets} boxes detected · right click {'held' if l.held else 'up'}"
                          f" · sent {l.counts} counts" if l.on and l.model else "")
        if self.live_msg.text() != txt: self.live_msg.setText(txt)
        t = self.input_text()
        if self.input_msg.text() != t: self.input_msg.setText(t)
        for sw in (self.live_sw, self.mouse_sw):
            if sw.isChecked() != l.on: sw.blockSignals(True); sw.setChecked(l.on); sw.blockSignals(False)

    # ── colours ──────────────────────────────────────────────────────────────
    def sel(self): return self.cfg["colors"][self.cfg["sel"]]

    def render_colors(self):
        while self.color_box.count():
            w = self.color_box.takeAt(0).widget()
            if w: w.hide(); w.deleteLater()
        many = len(self.cfg["colors"]) > 1
        for i, c in enumerate(self.cfg["colors"]):
            r = ColorRow(c, i == self.cfg["sel"], many)
            r.picked.connect(lambda i=i: self.select_color(i))
            r.toggled.connect(lambda on, i=i: (self.cfg["colors"][i].update(on=on), self.changed("colors")))
            r.removed.connect(lambda i=i: QTimer.singleShot(0, lambda: self.remove_color(i)))
            r.renamed.connect(lambda i=i: self.rename_color(i))
            self.color_box.addWidget(r)
        self.add_btn.setEnabled(len(self.cfg["colors"]) < MAX_COLORS)

    def select_color(self, i):
        if i == self.cfg["sel"]: return
        self.cfg["sel"] = i; s = self.sel()
        QTimer.singleShot(0, lambda: (self.apply_values(tolerance=s["tolerance"], min_sat=s["min_sat"], min_val=s["min_val"]),
                                      self.render_colors(), self.show_color()))

    def add_color(self, entry=None):
        if len(self.cfg["colors"]) >= MAX_COLORS: return False
        if entry is None:
            used = {c["color"].lower() for c in self.cfg["colors"]}
            name, col = next(((n, c) for n, c in PRESETS if c.lower() not in used), PRESETS[0])
            entry = new_color(name, col, BOX_COLORS[len(self.cfg["colors"]) % len(BOX_COLORS)][1])
        self.cfg["colors"].append(entry)
        self.cfg["sel"] = len(self.cfg["colors"]) - 2       # select_color() needs a change to act
        self.select_color(len(self.cfg["colors"]) - 1); self.changed("colors")
        return True

    def remove_color(self, i):
        if len(self.cfg["colors"]) <= 1: return
        self.cfg["colors"].pop(i)
        self.cfg["sel"] = -1; self.select_color(max(0, min(i, len(self.cfg["colors"]) - 1))); self.changed("colors")

    def rename_color(self, i):
        c = self.cfg["colors"][i]
        name, ok = QInputDialog.getText(self, "Rename colour", "Name:", text=c["name"])
        if ok and name.strip(): c["name"] = name.strip()[:24]; self.render_colors(); self.changed("colors")

    def set_color(self, col, name=None):
        s = self.sel()
        preset = next((n for n, c in PRESETS if c.lower() == col.lower()), None)
        if name or preset: s["name"] = name or preset
        s["color"] = col; s["hue"] = hsv_of(QColor(col).getRgb()[:3])[0]
        self.show_color(); self.render_colors(); self.changed("colors")

    def set_box(self, col):
        self.sel()["box"] = col; self.render_colors(); self.changed("colors")

    def show_color(self):
        s = self.sel()
        self.cur.setText(f"{s['name']}: {s['color'].upper()}  hue {s['hue'] * 2}°")
        for w in self.color_sw: w.on = w.color.lower() == s["color"].lower(); w.update()
        for w in self.box_sw: w.on = w.color.lower() == s["box"].lower(); w.update()

    def custom_color(self):
        c = QColorDialog.getColor(QColor(self.sel()["color"]), self, "Target colour")
        if c.isValid(): self.set_color(c.name(), "Custom")

    # ── preview modes: sample (drag) and eyedropper (click) ──────────────────
    def cancel_modes(self):
        if self.preview.sampling: self.toggle_sample()
        if self.preview.eyedrop: self.stop_eyedrop()

    def toggle_sample(self):
        if self.preview.eyedrop: self.stop_eyedrop()
        self.preview.sampling = not self.preview.sampling; self.preview.drag = None
        if self.preview.sampling and self.cfg["view"] == "mask": self.view_tabs.setCurrentIndex(0)
        self.sample_btn.setText("Drag over the preview…" if self.preview.sampling else "Sample from preview"); self.sample_btn.updateGeometry()
        self.preview.setCursor(Qt.CrossCursor if self.preview.sampling else Qt.ArrowCursor); self.preview.update()

    def sample_at(self, x, y, w, h):
        f = self.engine.last_frame if self.engine else None
        if f is not None:
            r = sample(f, x, y, w, h)
            if r:
                s = self.sel(); s.update(r); s["name"] = s["name"] if s["name"] not in dict(PRESETS) else "Sampled"
                self.apply_values(tolerance=r["tolerance"], min_sat=r["min_sat"], min_val=r["min_val"])
                self.show_color(); self.render_colors()
        if self.preview.sampling: self.toggle_sample()

    def pixel_at(self, x, y):
        f = self.engine.last_frame if self.engine else None
        if f is None: return None
        H, W = f.shape[:2]
        x, y = max(5, min(W - 6, x)), max(5, min(H - 6, y))
        crop = np.ascontiguousarray(f[y - 5:y + 6, x - 5:x + 6, 2::-1])        # BGRx → RGB
        return tuple(int(v) for v in crop[5, 5]), crop

    def start_eyedrop(self, purpose):
        if self.preview.sampling: self.toggle_sample()
        if self.preview.eyedrop and self.pick_for == purpose: return self.stop_eyedrop()
        if not self.running:
            self.eye_info.setText("Start capture first, then click a pixel in the preview."); self.go_tab(TABS.index("Tools")); return
        if self.cfg["view"] == "mask": self.view_tabs.setCurrentIndex(0)
        self.pick_for, self.preview.eyedrop = purpose, True
        self.eye_btn.setText("Picking… (Esc)" if purpose == "info" else "Eyedropper"); self.eye_btn.updateGeometry()
        self.preview.update()

    def stop_eyedrop(self):
        self.preview.eyedrop, self.preview.hover, self.pick_for = False, None, None
        self.eye_btn.setText("Eyedropper"); self.eye_btn.updateGeometry(); self.preview.unsetCursor(); self.preview.update()

    def eye_picked(self, x, y):
        got = self.pixel_at(x, y)
        if not got: return
        hexc = tools.info(got[0])["hex"]
        if self.pick_for in ("A", "B"):
            self.cfg["contrast_a" if self.pick_for == "A" else "contrast_b"] = hexc; save_settings(self.cfg)
            self.show_contrast(); self.stop_eyedrop(); return
        self.eye_rgb = got[0]; self.copy(hexc)
        self.cfg["history"] = ([hexc] + [h for h in self.cfg["history"] if h != hexc])[:14]; save_settings(self.cfg)
        self.render_history(); self.show_eye()

    def show_eye(self):
        on = self.eye_rgb is not None
        self.copy_btn.setEnabled(on); self.target_btn.setEnabled(on)
        if not on: return
        i = tools.info(self.eye_rgb)
        self.eye_sw.color = i["hex"]; self.eye_sw.update()
        self.eye_info.setText(f"{i['hex']}   ({i['name']})\n{i['rgb']}\n{i['hsv']}")

    def render_history(self):
        while self.hist_row.count():
            w = self.hist_row.takeAt(0).widget()
            if w: w.hide(); w.deleteLater()
        for h in self.cfg["history"]:
            s = Swatch(h, h, 26); s.clicked.connect(lambda c: (setattr(self, "eye_rgb", QColor(c).getRgb()[:3]), self.show_eye(), self.copy(c.upper())))
            self.hist_row.addWidget(s)
        if not self.cfg["history"]:
            l = label("Colours you pick appear here.", 11, muted=True); l.setWordWrap(False); self.hist_row.addWidget(l)
        self.hist_row.addStretch(1)

    def eye_to_target(self):
        if self.eye_rgb is None: return
        i = tools.info(self.eye_rgb); h, s, v = hsv_of(self.eye_rgb)
        entry = new_color(i["name"].title(), i["hex"].lower(), BOX_COLORS[len(self.cfg["colors"]) % len(BOX_COLORS)][1],
                          10, max(30, s - 50), max(30, v - 50))
        if self.add_color(entry): self.eye_info.setText(self.eye_info.text().split("\n\n")[0] + "\n\nAdded to the Detect tab.")

    def copy(self, text):
        QApplication.clipboard().setText(text)
        self.preview.setToolTip(""); self.copied.setText(f"Copied {text} to the clipboard")
        QTimer.singleShot(2500, lambda: self.copied.setText(""))

    def swap_contrast(self):
        self.cfg["contrast_a"], self.cfg["contrast_b"] = self.cfg["contrast_b"], self.cfg["contrast_a"]
        save_settings(self.cfg); self.show_contrast()

    def show_contrast(self):
        a, b = self.cfg["contrast_a"], self.cfg["contrast_b"]
        r, ok = tools.contrast(QColor(a).getRgb()[:3], QColor(b).getRgb()[:3])
        self.ca_sw.color, self.cb_sw.color = a, b; self.ca_sw.update(); self.cb_sw.update()
        self.ratio.text = f"{r:.2f} : 1"; self.ratio.updateGeometry(); self.ratio.update()
        for k, badge in self.wcag.items():
            badge.text, badge.kind = f"{k} {'✓' if ok[k] else '✗'}", "success" if ok[k] else "danger"
            badge._restyle() if hasattr(badge, "_restyle") else None; badge.updateGeometry(); badge.update()
        self.csample.a, self.csample.b = a, b; self.csample.update()

    def extract_palette(self):
        while self.pal_box.count():
            w = self.pal_box.takeAt(0).widget()
            if w: w.hide(); w.deleteLater()
        f = self.engine.last_frame if self.engine else None
        if f is None: self.pal_box.addWidget(label("Start capture first.", 12, muted=True)); return
        x, y, w, h = fov_rect(self.cfg, f.shape[1], f.shape[0])
        for rgb, share, _hsv, rng in tools.palette(f[y:y + h, x:x + w]):
            i = tools.info(rgb)
            row = QWidget(); hl = QHBoxLayout(row); hl.setContentsMargins(0, 0, 0, 0); hl.setSpacing(8)
            sw = Swatch(i["hex"], i["hex"], 26); sw.clicked.connect(self.copy); hl.addWidget(sw)
            hl.addWidget(label(f"{i['hex']}  {share * 100:4.1f}%", 12, mono=True)); hl.addWidget(label(i["name"], 12, muted=True)); hl.addStretch(1)
            b = uv.Button("+ Add", size="sm")
            b.clicked.connect(lambda _=False, i=i, rng=rng: self.add_color(new_color(i["name"].title(), i["hex"].lower(),
                              BOX_COLORS[len(self.cfg["colors"]) % len(BOX_COLORS)][1], **rng)))
            hl.addWidget(b); self.pal_box.addWidget(row)

    # ── profiles, sharing, regions ───────────────────────────────────────────
    def profile_data(self): return {k: self.cfg[k] for k in PROFILE_KEYS}

    def refresh_profiles(self):
        self.prof.clear(); self.prof.addItems(["(unsaved)"] + profile_names())
        if self.cfg["profile"] in profile_names(): self.prof.setCurrentText(self.cfg["profile"])

    def save_profile(self, ask=False):
        name = self.cfg["profile"]
        if ask or not name:
            name, ok = QInputDialog.getText(self, "Save profile", "Profile name:", text=name or "My profile")
            if not ok or not name.strip(): return
            name = name.strip()
        path = profile_path(name)
        path.write_text(json.dumps(self.profile_data(), indent=1))
        self.cfg["profile"] = path.stem; save_settings(self.cfg); self.refresh_profiles()

    def load_profile(self, name):
        try: data = json.loads(profile_path(name).read_text())
        except Exception: return
        self.cfg.update({k: v for k, v in data.items() if k in PROFILE_KEYS}); self.cfg["profile"] = name
        fix_colors(self.cfg); save_settings(self.cfg)
        QTimer.singleShot(0, lambda: (self.build_pages(), self.commit_undo()))

    def delete_profile(self):
        name = self.prof.currentText()
        if name.startswith("("): return
        profile_path(name).unlink(missing_ok=True); self.cfg["profile"] = ""; save_settings(self.cfg); self.refresh_profiles()

    def import_profile(self, data, name):
        data = {k: v for k, v in data.items() if k in PROFILE_KEYS}
        if not data.get("colors"): raise ValueError("That profile has no colours in it")
        base, n = (name or "Imported").strip()[:40], 2
        name = base
        while profile_path(name).exists(): name = f"{base} {n}"; n += 1
        profile_path(name).write_text(json.dumps(data, indent=1))
        self.load_profile(profile_path(name).stem)
        return profile_path(name).stem

    def note(self, text):
        """Message under Share profiles; kept across page rebuilds (importing a profile rebuilds the pages)."""
        self.share_note = text; self.share_msg.setText(text)

    def export_profile(self):
        name = self.cfg["profile"] or "ChromaScope profile"
        path, _ = QFileDialog.getSaveFileName(self, "Export profile", str(Path.home() / f"{name}.chromascope.json"), "ChromaScope profile (*.json)")
        if not path: return
        Path(path).write_text(json.dumps({"name": name, **self.profile_data()}, indent=1))
        self.note(f"Exported to {path}")

    def import_profile_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Import profile", str(Path.home()), "ChromaScope profile (*.json)")
        if not path: return
        try:
            data = json.loads(Path(path).read_text())
            got = self.import_profile(data, data.get("name") or Path(path).name.split(".")[0])
            self.note(f"Imported and loaded “{got}”.")
        except Exception as e:
            self.note(f"Couldn't import: {e}")

    def copy_code(self):
        QApplication.clipboard().setText(tools.to_code({"name": self.cfg["profile"] or "Shared profile", **self.profile_data()}))
        self.note("Share code copied to the clipboard.")

    def paste_code(self):
        text = QApplication.clipboard().text().strip()
        if not text.startswith(tools.CODE_PREFIX):
            text, ok = QInputDialog.getMultiLineText(self, "Paste share code", "Share code (starts with CSP1:):")
            if not ok: return
        try:
            data = tools.from_code(text)
            got = self.import_profile(data, data.get("name"))
            self.note(f"Imported and loaded “{got}”.")
        except Exception as e:
            self.note(f"Couldn't read that code: {e}")

    def add_examples(self):
        added = []
        for n, data in EXAMPLES.items():
            if profile_path(n).exists(): continue
            profile_path(n).write_text(json.dumps(data, indent=1)); added.append(n)
        self.refresh_profiles()
        self.note(f"Added {len(added)} example profile{'s' if len(added) != 1 else ''} — pick one in the Profile list on the Detect tab."
                               if added else "The example profiles are already there.")

    def fill_regions(self):
        self.region_box.clear(); self.region_box.addItems(["(choose a region)"] + sorted(self.cfg["regions"]))

    def save_region(self):
        name, ok = QInputDialog.getText(self, "Save region", "Region name:", text="Region")
        if not ok or not name.strip(): return
        self.cfg["regions"][name.strip()] = {k: self.cfg[k] for k in ("fov", "fov_shape", "fov_dx", "fov_dy")}
        self.changed("regions"); self.fill_regions(); self.region_box.setCurrentText(name.strip())

    def load_region(self, name):
        if name in self.cfg["regions"]: self.apply_values(**self.cfg["regions"][name])

    def delete_region(self):
        self.cfg["regions"].pop(self.region_box.currentText(), None); self.changed("regions"); self.fill_regions()

    # ── undo / reset ─────────────────────────────────────────────────────────
    def commit_undo(self):
        self.history.commit(json.loads(json.dumps({k: self.cfg[k] for k in UNDO_KEYS})))
        self.undo_btn.setEnabled(self.history.can_undo()); self.redo_btn.setEnabled(self.history.can_redo())

    def restore(self, snap):
        if snap is None: return
        self.restoring = True
        self.cfg.update(snap); fix_colors(self.cfg); save_settings(self.cfg); self.build_pages()
        self.restoring = False
        self.undo_btn.setEnabled(self.history.can_undo()); self.redo_btn.setEnabled(self.history.can_redo())
        self.preview.update(); self.overlay.update()

    def undo(self):
        if self.undo_timer.isActive(): self.undo_timer.stop(); self.commit_undo()   # keep the change in flight
        self.restore(self.history.undo())

    def redo(self): self.restore(self.history.redo())

    def reset_tab(self, tab):
        for k in RESET_KEYS[tab]: self.cfg[k] = json.loads(json.dumps(DEFAULTS[k]))
        fix_colors(self.cfg); save_settings(self.cfg)
        QTimer.singleShot(0, lambda: (self.build_pages(), self.commit_undo(), self.update_pause(), self.preview.update()))

    # ── behaviour ────────────────────────────────────────────────────────────
    def apply_values(self, **kv):
        """Set cfg values and move their controls to match."""
        self.cfg.update(kv)
        for k, v in kv.items():
            w = self.reg.get(k)
            if not w: continue
            if w[0] == "slider": w[1].setValue(int(v))
            elif w[0] == "choice" and v in w[2]: w[1].setCurrentIndex(w[2].index(v))
            elif w[0] == "switch": w[1].setChecked(bool(v))
        self.changed("fov")

    def fov_moved(self):
        self.apply_values(fov=self.cfg["fov"], fov_dx=self.cfg["fov_dx"], fov_dy=self.cfg["fov_dy"])

    def restyle_app(self):
        uv.apply(self.app, theme=self.cfg["theme"], kit=self.cfg["kit"])

    def refresh_labels(self):
        for l in self.findChildren(QLabel):
            if l.property("muted"): l.setStyleSheet(f"color: {tok('muted').name()}")
        self.update()

    def changed(self, key, value=None):
        if value is not None: self.cfg[key] = value
        if key in ("tolerance", "min_sat", "min_val"):
            self.sel()[key] = self.cfg[key]
            if key == "tolerance": self.render_colors()
        if key == "overlay_screen" and self.overlay.isVisible(): self.overlay.show_on(self.cfg["overlay_screen"])
        if key.startswith("pause_") or key == "focus_match": self.update_pause()
        if key == "nn_hold": self.rbutton.mode = self.cfg["nn_hold"]; self.show_live()
        save_settings(self.cfg); self.preview.update(); self.overlay.update()
        if not self.restoring and key in UNDO_KEYS + ["tolerance", "min_sat", "min_val", "fov"]: self.undo_timer.start()

    def update_pause(self):
        """Work out whether capture should be paused for lock / focus; the engine adds 'screen hasn't changed'."""
        if self.cfg["pause_focus"]: self.focus.start()
        else: self.focus.stop()
        if hasattr(self, "focus_seen") and self.focus_seen.count() - 1 != len(self.focus.seen):
            self.focus_seen.clear(); self.focus_seen.addItem("(recently focused apps)"); self.focus_seen.addItems(self.focus.seen)
        reason = ""
        if self.cfg["pause_locked"] and self.lock.locked: reason = "screen locked"
        elif self.cfg["pause_focus"] and self.cfg["focus_match"].strip() and self.focus.app and not self.focus.matches(self.cfg["focus_match"]):
            reason = f"“{self.cfg['focus_match'].strip()}” isn't focused"
        self.cfg["_pause"] = reason

    def on_paused(self, reason):
        self.preview.paused = reason; self.preview.update()
        if reason: self.live.feed(None, f"capture is paused ({reason})")
        if reason:
            self.set_status("Paused", "warn"); self.status.setToolTip(reason)
            self.overlay.data = None; self.overlay.update()
        else:
            self.set_status("Live", "success"); self.status.setToolTip("")

    def forget_screen(self):
        self.cfg["restore_token"] = None; save_settings(self.cfg)
        if self.running: self.toggle(); self.toggle()

    def set_overlay(self, on):
        self.cfg["overlay"] = on; save_settings(self.cfg)
        if on: self.overlay.show_on(self.cfg["overlay_screen"])
        else: self.overlay.hide()

    def toggle(self):
        if self.running:
            self.engine.stop_capture(); self.engine = None; self.running = False; self.live.feed(None)
            self.go.setText("Start capture"); self.set_status("Idle", "warn")
            self.preview.data = None; self.preview.paused = ""; self.preview.update(); self.overlay.data = None; self.overlay.update()
            if self.preview.eyedrop: self.stop_eyedrop()
        else:
            self.engine = Engine(self.cfg)
            self.engine.frameReady.connect(self.on_frame); self.engine.failed.connect(self.on_fail)
            self.engine.started_ok.connect(lambda: self.set_status("Live", "success")); self.engine.pausedChanged.connect(self.on_paused)
            self.engine.start_capture(); self.running = True
            self.go.setText("Stop capture"); self.set_status("Waiting for screen…", "info")
            if self.cfg["minimize"]: QTimer.singleShot(400, self.showMinimized)
        self.go.updateGeometry()
        if self.cfg["overlay"] and not self.overlay.isVisible(): self.set_overlay(True)

    def set_status(self, text, kind):
        self.status.text, self.status.kind = text, kind
        self.status._restyle() if hasattr(self.status, "_restyle") else None
        self.status.updateGeometry(); self.status.update()

    def restore_status(self):
        if not self.running: self.set_status("Idle", "warn")
        elif self.preview.paused: self.set_status("Paused", "warn")
        else: self.set_status("Live", "success")

    def on_fail(self, msg):
        if self.running: self.toggle()
        self.set_status(msg[:60], "danger")

    def on_frame(self, d):
        if not self.running: return
        if self.engine.cap.restore_token and self.engine.cap.restore_token != self.cfg["restore_token"]:
            self.cfg["restore_token"] = self.engine.cap.restore_token; save_settings(self.cfg)
        self.preview.show_frame(d)
        self.live.feed(d)
        if self.overlay.isVisible():
            self.overlay.show_frame(d); self.engine.overlay_scale = self.overlay.scale()
        w, h = d["size"]
        for b, t in ((self.b_cap, f"Capture {d['cap_fps']:.0f} fps"), (self.b_proc, f"Process {d['fps']:.0f} fps"),
                     (self.b_ms, f"{d['ms']:.1f} ms"), (self.b_hits, f"{len(d['targets'])} hits"), (self.b_res, f"{w}×{h}")):
            if b.text != t: b.text = t; b.updateGeometry(); b.update()

    # ── tour ─────────────────────────────────────────────────────────────────
    def start_tour(self):
        T = TABS.index
        steps = [
            (None, "Welcome to ChromaScope", "It finds colours on your screen and highlights them, live. This quick tour shows where everything is.", None),
            (lambda: self.go, "Start here", "Start capture (or F8). The first time, KDE asks which screen to share; it remembers after that.", None),
            (lambda: self.preview, "Live preview", "Your screen with every match boxed. Drag the dashed FOV ring to move it, drag its edge to resize. "
                                                   "Screen / FOV zoom / Mask above switch what you see.", None),
            (lambda: self.color_list, "Colours to find", "Up to 6 colours, each with its own box colour. Click one to edit it, double-click to rename.",
             lambda: self.go_tab(T("Detect"))),
            (lambda: self.sample_btn, "Teach it a colour", "Sample from preview, then drag over something on screen. The colour range is worked out for you.",
             lambda: self.go_tab(T("Detect"))),
            (lambda: self.assist_sw, "Highlights and assist", "The Draw tab picks Box, Glow or Corners, and the colour-blind assist that recolours a hue you find hard to see.",
             lambda: self.go_tab(T("Draw"))),
            (lambda: self.ov_switch, "On your real screen", "Draws the highlights over every app, click-through. It's hidden from the capture, so it never detects itself.", None),
            (lambda: self.eye_btn, "Colour tools", "Eyedropper with a magnifier, a WCAG contrast checker and a palette extractor.",
             lambda: self.go_tab(T("Tools"))),
            (lambda: self.undo_box, "Undo anything", "Every settings change can be undone (Ctrl+Z) and redone (Ctrl+Shift+Z). Each tab also has Reset this tab.", None),
            (lambda: self.focus_sw, "Setup", "Auto-pause, sharing profiles as files or codes, example profiles and the app's look all live on the Setup tab. "
                                             "You can replay this tour from there.", lambda: self.go_tab(T("Setup"))),
        ]
        self.tour = Tour(self, steps)
        self.tour.finished.connect(lambda: (self.cfg.update(toured=True), save_settings(self.cfg), self.go_tab(T("Detect"))))
        self.tour.start()

    def resizeEvent(self, e):
        super().resizeEvent(e)
        t = getattr(self, "tour", None)
        if t is not None and t.isVisible(): t.place()

    def closeEvent(self, e):
        if self.engine: self.engine.stop_capture()
        self.live.stop(); self.rbutton.stop(); self.focus.stop(); self.overlay.close(); super().closeEvent(e)


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("ChromaScope"); app.setDesktopFileName("chromascope")
    ensure_kwin_rules()
    w = Window(app); w.show()
    if not w.cfg["toured"]: QTimer.singleShot(600, w.start_tour)
    if "--start" in sys.argv: QTimer.singleShot(300, w.toggle)
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
