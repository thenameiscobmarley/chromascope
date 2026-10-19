"""ChromaScope — live colour detection on your screen (Wayland / KDE Plasma), built with UiVault3.

Run:  ./run.sh        Settings are saved in ~/.config/chromascope/settings.json.
Keys: F8 start/stop capture · F9 overlay on/off · F10 preview mode
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, QThread, QTimer, Signal
from PySide6.QtGui import QColor, QFont, QImage, QKeySequence, QPainter, QPainterPath, QPen, QShortcut
from PySide6.QtWidgets import (QApplication, QColorDialog, QComboBox, QGridLayout, QHBoxLayout, QLabel, QScrollArea,
                               QSizePolicy, QStackedWidget, QVBoxLayout, QWidget)

import uivault3 as uv
from capture import MONITOR, WINDOW, Capture
from detect import detect, fov_rect, hsv_of, tint

CONFIG = Path.home() / ".config" / "chromascope"
CONFIG.mkdir(parents=True, exist_ok=True)
SETTINGS = CONFIG / "settings.json"
OVERLAY_TITLE = "ChromaScope Overlay"

DEFAULTS = {
    "color": "#ff2a2a", "tolerance": 10, "min_sat": 120, "min_val": 110, "min_area": 30, "merge": 6,
    "max_targets": 8, "sort": "largest",
    "fov": 420, "fov_shape": "circle", "fov_dx": 0, "fov_dy": 0, "show_fov": True, "crosshair": True,
    "mode": "box", "box_color": "#ff3b3b", "thickness": 2, "glow": 10, "labels": True,
    "view": "full", "hud": True, "max_fps": 60,
    "overlay": False, "overlay_fov": True, "overlay_fps": True, "overlay_screen": 0,
    "source": MONITOR, "cursor": False, "minimize": False, "restore_token": None,
    "kit": "win98", "theme": "paper",
}
PRESETS = [("Red", "#ff2a2a"), ("Orange", "#ff8a00"), ("Yellow", "#ffe100"), ("Green", "#22e35a"),
           ("Cyan", "#00e5ff"), ("Blue", "#2a6bff"), ("Purple", "#b13cff"), ("Magenta", "#ff2ad4")]


def load_settings():
    try: cfg = {**DEFAULTS, **json.loads(SETTINGS.read_text())}
    except Exception: cfg = dict(DEFAULTS)
    cfg["hue"] = hsv_of(QColor(cfg["color"]).getRgb()[:3])[0]
    return cfg


def save_settings(cfg):
    SETTINGS.write_text(json.dumps({k: cfg[k] for k in DEFAULTS}, indent=1))


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

    def __init__(self, cfg):
        super().__init__()
        self.cfg, self.cap, self.last_frame = cfg, Capture(), None
        self.overlay_scale = 1.0          # overlay logical px per frame px (set by the UI)

    def start_capture(self):
        c = self.cfg
        self.cap.start(c["source"], c["cursor"], c["restore_token"], on_ready=lambda _c: self.started_ok.emit())
        self.start()

    def stop_capture(self):
        self.requestInterruption(); self.wait(1500); self.cap.stop()

    def _ignore(self, targets, cfg):
        """Outline bands our overlay drew last frame, in frame px, so they aren't detected as targets."""
        if not cfg["overlay"]: return ()
        s = max(self.overlay_scale, 1e-3)
        pad = (4 + cfg["thickness"] + (cfg["glow"] if cfg["mode"] in ("glow", "both") else 0)) / s
        band = int(2 * pad + 6)
        return [(int(t.x - pad), int(t.y - pad), int(t.w + 2 * pad), int(t.h + 2 * pad), band) for t in targets]

    def run(self):
        last, prev, n, t0, fps, next_t = 0, [], 0, time.monotonic(), 0.0, 0.0
        while not self.isInterruptionRequested():
            if self.cap.error: self.failed.emit(self.cap.error); return
            frame, last = self.cap.wait(last, 0.3)
            if frame is None: continue
            now = time.monotonic()
            cfg = dict(self.cfg)
            if now < next_t: time.sleep(next_t - now)
            next_t = time.monotonic() + 1.0 / max(1, cfg["max_fps"])
            t = time.perf_counter()
            self.last_frame = frame
            targets, mask, fov = detect(frame, cfg, self._ignore(prev, cfg))
            img, src = self._preview(frame, mask, fov, cfg)
            ms = (time.perf_counter() - t) * 1000
            n += 1
            if time.monotonic() - t0 >= 0.5: fps, n, t0 = n / (time.monotonic() - t0), 0, time.monotonic()
            prev = targets
            self.frameReady.emit({"image": img, "src": src, "targets": targets, "fov": fov, "size": frame.shape[1::-1],
                                  "ms": ms, "fps": fps, "cap_fps": self.cap.fps})

    def _preview(self, frame, mask, fov, cfg):
        fx, fy, fs, _ = fov
        glow = cfg["mode"] in ("glow", "both")
        if cfg["view"] == "mask":
            m = np.ascontiguousarray(mask)
            return QImage(m.data, fs, fs, fs, QImage.Format_Grayscale8).copy(), (fx, fy, fs, fs)
        if cfg["view"] == "fov":
            img = np.ascontiguousarray(frame[fy:fy + fs, fx:fx + fs])
            if glow: tint(img, mask, glow=cfg["glow"])
            src = (fx, fy, fs, fs)
        else:
            H, W = frame.shape[:2]
            s = min(1.0, 1100 / W)
            img = cv2.resize(frame, (int(W * s), int(H * s)), interpolation=cv2.INTER_LINEAR) if s < 1 else frame.copy()
            if glow:
                x0, y0, ss = int(fx * s), int(fy * s), max(1, int(fs * s))
                region = img[y0:y0 + ss, x0:x0 + ss]
                m = cv2.resize(mask, (region.shape[1], region.shape[0]), interpolation=cv2.INTER_NEAREST)
                tint(region, m, glow=max(2, int(cfg["glow"] * s)))
            src = (0, 0, W, H)
        h, w = img.shape[:2]
        return QImage(img.data, w, h, img.strides[0], QImage.Format_RGB32).copy(), src


# ── Drawing shared by the preview and the on-screen overlay ──────────────────
def draw_targets(p, targets, to_px, scale, cfg, overlay=False):
    col = QColor(cfg["box_color"]); th = cfg["thickness"]
    mode = cfg["mode"]
    f = QFont("Geist Mono"); f.setPixelSize(11); f.setBold(True); p.setFont(f)
    for i, t in enumerate(targets):
        r = to_px(QRectF(t.x, t.y, t.w, t.h)).adjusted(-3, -3, 3, 3)
        if mode in ("glow", "both") and overlay:
            g = cfg["glow"]
            for k in range(g, 0, -2):          # soft red halo outside the box
                c = QColor(255, 40, 40, int(150 * (1 - k / (g + 1)) ** 2))
                p.setPen(QPen(c, 2)); p.setBrush(Qt.NoBrush)
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
            txt = f"{i + 1}  {t.w}×{t.h}"
            tw = p.fontMetrics().horizontalAdvance(txt) + 10
            chip = QRectF(r.left(), r.top() - 18, tw, 16)
            if chip.top() < 0: chip.moveTop(r.bottom() + 2)
            p.setPen(Qt.NoPen); p.setBrush(QColor(col.red(), col.green(), col.blue(), 220)); p.drawRoundedRect(chip, 4, 4)
            p.setPen(QColor("#0b0b0b") if col.lightness() > 140 else QColor("#ffffff"))
            p.drawText(chip, Qt.AlignCenter, txt)


def draw_fov(p, fov, to_px, cfg, ring_pad=0):
    fx, fy, fs, _ = fov
    r = to_px(QRectF(fx, fy, fs, fs)).adjusted(-ring_pad, -ring_pad, ring_pad, ring_pad)
    p.setBrush(Qt.NoBrush)
    p.setPen(QPen(QColor(255, 255, 255, 150), 1.4, Qt.DashLine))
    p.drawEllipse(r) if cfg["fov_shape"] == "circle" else p.drawRect(r)
    if cfg["crosshair"]:
        c = r.center(); p.setPen(QPen(QColor(255, 255, 255, 200), 1.4))
        p.drawLine(QPointF(c.x() - 7, c.y()), QPointF(c.x() + 7, c.y())); p.drawLine(QPointF(c.x(), c.y() - 7), QPointF(c.x(), c.y() + 7))


class Preview(QWidget):
    picked = Signal(int, int)       # frame px

    def __init__(self, cfg):
        super().__init__()
        self.cfg, self.data, self.picking = cfg, None, False
        self.setMinimumSize(480, 300); self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMouseTracking(True)

    def show_frame(self, d): self.data = d; self.update()

    def _geom(self):
        d = self.data; img = d["image"]
        area = QRectF(self.rect()).adjusted(1, 1, -1, -1)
        s = min(area.width() / img.width(), area.height() / img.height())
        w, h = img.width() * s, img.height() * s
        return QRectF(area.center().x() - w / 2, area.center().y() - h / 2, w, h)

    def _mapper(self):
        g = self._geom(); sx, sy, sw, sh = self.data["src"]
        kx, ky = g.width() / sw, g.height() / sh
        return (lambda r: QRectF(g.x() + (r.x() - sx) * kx, g.y() + (r.y() - sy) * ky, r.width() * kx, r.height() * ky)), kx, g

    def paintEvent(self, _):
        p = QPainter(self); p.setRenderHint(QPainter.Antialiasing)
        bg = QPainterPath(); bg.addRoundedRect(QRectF(self.rect()), 14, 14); p.setClipPath(bg)
        p.fillRect(self.rect(), QColor("#0a0a0a"))
        if not self.data:
            p.setPen(QColor(255, 255, 255, 110)); f = p.font(); f.setPixelSize(15); p.setFont(f)
            p.drawText(self.rect(), Qt.AlignCenter, "Press  Start capture  (F8)\nKDE asks which screen to share the first time.\n\nTip: if this window sits inside the FOV it will see itself —\nmove it to another monitor or turn on the overlay and minimize it.")
            p.end(); return
        d, cfg = self.data, self.cfg
        to_px, k, g = self._mapper()
        p.setRenderHint(QPainter.SmoothPixmapTransform, cfg["view"] != "mask")
        p.drawImage(g, d["image"])
        if cfg["show_fov"] and cfg["view"] == "full": draw_fov(p, d["fov"], to_px, cfg)
        elif cfg["crosshair"] and cfg["view"] != "full":
            c = g.center(); p.setPen(QPen(QColor(255, 255, 255, 170), 1.2))
            p.drawLine(QPointF(c.x() - 7, c.y()), QPointF(c.x() + 7, c.y())); p.drawLine(QPointF(c.x(), c.y() - 7), QPointF(c.x(), c.y() + 7))
        draw_targets(p, d["targets"], to_px, k, cfg)
        if cfg["hud"]:
            f = QFont("Geist Mono"); f.setPixelSize(12); f.setBold(True); p.setFont(f)
            txt = f"{d['fps']:5.1f} FPS   {d['ms']:4.1f} ms   {len(d['targets'])} hit{'s' if len(d['targets']) != 1 else ''}"
            r = QRectF(g.x() + 10, g.y() + 10, p.fontMetrics().horizontalAdvance(txt) + 18, 24)
            p.setPen(Qt.NoPen); p.setBrush(QColor(0, 0, 0, 170)); p.drawRoundedRect(r, 7, 7)
            p.setPen(QColor("#7CFF9B") if d["fps"] >= 30 else QColor("#FFD25A")); p.drawText(r, Qt.AlignCenter, txt)
        if self.picking:
            p.setPen(QPen(tok("accent"), 2)); p.setBrush(Qt.NoBrush); p.drawRoundedRect(QRectF(self.rect()).adjusted(1, 1, -1, -1), 14, 14)
        p.end()

    def mousePressEvent(self, e):
        if not (self.picking and self.data) or self.cfg["view"] == "mask": return
        g = self._geom(); sx, sy, sw, sh = self.data["src"]
        pt = e.position()
        if not g.contains(pt): return
        self.picked.emit(int(sx + (pt.x() - g.x()) / g.width() * sw), int(sy + (pt.y() - g.y()) / g.height() * sh))

    def enterEvent(self, _): self.setCursor(Qt.CrossCursor if self.picking else Qt.ArrowCursor)


# ── Click-through overlay on the real screen ─────────────────────────────────
class Overlay(QWidget):
    def __init__(self, cfg):
        super().__init__(None, Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.WindowTransparentForInput
                         | Qt.WindowDoesNotAcceptFocus | Qt.Tool)
        self.cfg, self.data = cfg, None
        self.setWindowTitle(OVERLAY_TITLE)
        self.setAttribute(Qt.WA_TranslucentBackground); self.setAttribute(Qt.WA_ShowWithoutActivating)

    def show_on(self, index):  # noqa: D401 — fullscreen on the chosen monitor; KWin rule keeps it above
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
            if cfg["overlay_fov"]: draw_fov(p, d["fov"], to_px, cfg, ring_pad=4)
            draw_targets(p, d["targets"], to_px, kx, cfg, overlay=True)
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
    """A page of settings rows bound to cfg keys."""

    def __init__(self, cfg, changed):
        super().__init__()
        self.cfg, self.changed = cfg, changed
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
        self.lay.addWidget(box); return s

    def choice(self, title, key, options):
        row = QVBoxLayout(); row.setSpacing(4)
        row.addWidget(label(title, 13))
        tb = uv.TabBar([o[0] for o in options]); tb.setUsesScrollButtons(False)
        vals = [o[1] for o in options]
        tb.setCurrentIndex(vals.index(self.cfg[key]) if self.cfg[key] in vals else 0)
        tb.currentChanged.connect(lambda i: self.set(key, vals[i]))
        row.addWidget(tb, 0, Qt.AlignLeft); self.lay.addLayout(row); return tb

    def switch(self, text, key):
        s = uv.Switch(text, bool(self.cfg[key])); s.toggled.connect(lambda v: self.set(key, v))
        self.lay.addWidget(s); return s

    def set(self, key, value):
        self.cfg[key] = value; self.changed(key)

    def end(self): self.lay.addStretch(1)


class Swatch(QWidget):
    clicked = Signal(str)

    def __init__(self, name, color):
        super().__init__(); self.name, self.color, self.on = name, color, False
        self.setFixedSize(30, 30); self.setCursor(Qt.PointingHandCursor); self.setToolTip(name)

    def paintEvent(self, _):
        p = QPainter(self); p.setRenderHint(QPainter.Antialiasing)
        if self.on: p.setPen(QPen(tok("text"), 2)); p.setBrush(Qt.NoBrush); p.drawEllipse(QRectF(1, 1, 28, 28))
        p.setPen(Qt.NoPen); p.setBrush(QColor(self.color)); p.drawEllipse(QRectF(5, 5, 20, 20)); p.end()

    def mousePressEvent(self, _): self.clicked.emit(self.color)


# ── Main window ──────────────────────────────────────────────────────────────
class Window(uv.Surface):
    def __init__(self, app):
        super().__init__()
        self.app, self.cfg = app, load_settings()
        self.engine, self.overlay, self.running = None, Overlay(self.cfg), False
        self.setWindowTitle("ChromaScope"); self.resize(1240, 720); self.setMinimumSize(980, 600)
        self.restyle_app()

        root = QVBoxLayout(self); root.setContentsMargins(22, 14, 22, 16); root.setSpacing(12)
        head = QHBoxLayout(); root.addLayout(head)
        head.addWidget(uv.Heading("ChromaScope", 34))
        self.status = uv.Badge("Idle", "warn"); head.addWidget(self.status, 0, Qt.AlignVCenter)
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
        self.preview = Preview(self.cfg); self.preview.picked.connect(self.pick_at)
        left.addWidget(self.preview, 1)
        stats = QHBoxLayout(); stats.setSpacing(6); left.addLayout(stats)
        self.b_cap, self.b_proc, self.b_ms = uv.Badge("Capture —", "info"), uv.Badge("Process —", "accent"), uv.Badge("— ms", "accent")
        self.b_hits, self.b_res = uv.Badge("0 hits", "success"), uv.Badge("—", "info")
        for b in (self.b_cap, self.b_proc, self.b_ms, self.b_hits, self.b_res): stats.addWidget(b)
        stats.addStretch(1)

        side = uv.Card(padding=16); side.setFixedWidth(470); body.addWidget(side)
        self.nav = uv.NavBar(["Detect", "FOV", "Draw", "Screen", "NNMM", "Look"])
        side.layout().addWidget(self.nav)
        self.pages = QStackedWidget()
        for page in (self.page_detect(), self.page_fov(), self.page_highlight(), self.page_screen(), self.page_nnmm(), self.page_style()):
            sc = QScrollArea(); sc.setWidgetResizable(True); sc.setFrameShape(QScrollArea.NoFrame)
            sc.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
            sc.setStyleSheet("QScrollArea, QScrollArea > QWidget > QWidget { background: transparent; }"
                             "QScrollBar:vertical { background: transparent; width: 6px; margin: 0; }"
                             "QScrollBar::handle:vertical { background: rgba(128,128,128,110); border-radius: 3px; min-height: 30px; }"
                             "QScrollBar::add-line, QScrollBar::sub-line { height: 0; }"
                             "QScrollBar::add-page, QScrollBar::sub-page { background: none; }")
            sc.setWidget(page); self.pages.addWidget(sc)
        side.layout().addWidget(self.pages, 1)
        self.nav.currentChanged.connect(self.pages.setCurrentIndex)

        QShortcut(QKeySequence("F8"), self, self.toggle)
        QShortcut(QKeySequence("F9"), self, lambda: self.ov_switch.setChecked(not self.ov_switch.isChecked()))
        QShortcut(QKeySequence("F10"), self, lambda: self.view_tabs.setCurrentIndex((self.view_tabs.currentIndex() + 1) % 3))
        uv.style().changed.connect(self.refresh_labels)

    # pages
    def page_detect(self):
        p = Panel(self.cfg, self.changed)
        p.section("Target colour")
        row = QHBoxLayout(); row.setSpacing(2); self.swatches = []
        for n, c in PRESETS:
            s = Swatch(n, c); s.clicked.connect(self.set_color); row.addWidget(s); self.swatches.append(s)
        row.addStretch(1); p.lay.addLayout(row)
        self.cur = label("", 12, mono=True); p.lay.addWidget(self.cur)
        row2 = QHBoxLayout()
        b = uv.Button("Custom…", size="sm"); b.clicked.connect(self.custom_color); row2.addWidget(b)
        self.pick_btn = uv.Button("Pick from preview", size="sm"); self.pick_btn.clicked.connect(self.toggle_pick); row2.addWidget(self.pick_btn); row2.addStretch(1)
        p.lay.addLayout(row2)
        self.show_color()
        p.section("Matching")
        p.slider("Hue tolerance", "tolerance", 1, 40, "±{}", "How far from the target hue still counts.")
        p.slider("Min saturation", "min_sat", 0, 255, "{}", "Raise to ignore greyish / washed-out pixels.")
        p.slider("Min brightness", "min_val", 0, 255, "{}")
        p.section("Objects")
        p.slider("Min size", "min_area", 1, 2000, "{} px")
        p.slider("Merge distance", "merge", 0, 30, "{} px", "Joins nearby pieces (outlines) into one box.")
        p.slider("Max targets", "max_targets", 1, 32, "{}")
        p.choice("Order", "sort", [("Largest first", "largest"), ("Closest to centre", "closest")])
        p.end(); return p

    def page_fov(self):
        p = Panel(self.cfg, self.changed)
        p.section("Field of view")
        p.slider("Size", "fov", 60, 2000, "{} px", "Only this area around the centre is scanned. Smaller = faster.")
        p.choice("Shape", "fov_shape", [("Circle", "circle"), ("Square", "square")])
        p.slider("Offset X", "fov_dx", -600, 600, "{} px")
        p.slider("Offset Y", "fov_dy", -600, 600, "{} px")
        p.switch("Show FOV ring", "show_fov")
        p.switch("Centre crosshair", "crosshair")
        p.section("Performance")
        p.slider("Max FPS", "max_fps", 5, 240, "{}", "Caps how often frames are processed.")
        p.end(); return p

    def page_highlight(self):
        p = Panel(self.cfg, self.changed)
        p.section("Highlight mode")
        p.choice("Style", "mode", [("Box", "box"), ("Glow", "glow"), ("Both", "both"), ("Corners", "corners")])
        p.lay.addWidget(label("Glow turns the detected pixels a red shade with a soft halo on the display.", 11, muted=True))
        p.slider("Glow radius", "glow", 0, 30, "{} px")
        p.slider("Line thickness", "thickness", 1, 6, "{} px")
        p.section("Box colour")
        row = QHBoxLayout(); row.setSpacing(2); self.box_sw = []
        for n, c in [("Red", "#ff3b3b"), ("Lime", "#7CFF4F"), ("White", "#ffffff"), ("Cyan", "#2ae6ff"), ("Gold", "#ffc83a"), ("Pink", "#ff4fd8")]:
            s = Swatch(n, c); s.on = c == self.cfg["box_color"]
            s.clicked.connect(lambda col: (self.changed("box_color", col), [(setattr(w, "on", w.color == col), w.update()) for w in self.box_sw]))
            row.addWidget(s); self.box_sw.append(s)
        row.addStretch(1); p.lay.addLayout(row)
        p.section("Preview")
        p.switch("Size labels on boxes", "labels")
        p.switch("FPS / timing HUD", "hud")
        p.end(); return p

    def page_screen(self):
        p = Panel(self.cfg, self.changed)
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
        p.lay.addWidget(label("Keeps this window out of the shot on a single monitor (use the overlay to watch).", 11, muted=True))
        b = uv.Button("Choose a different screen", size="sm"); b.clicked.connect(self.forget_screen); p.lay.addWidget(b, 0, Qt.AlignLeft)
        p.lay.addWidget(label("Changes here apply the next time capture starts.", 11, muted=True))
        p.end(); return p

    def page_nnmm(self):
        p = Panel(self.cfg, self.changed)
        p.section("NNMM")
        p.lay.addWidget(uv.Alert("Placeholder: nothing here yet.", "info"))
        p.end(); return p

    def page_style(self):
        p = Panel(self.cfg, self.changed)
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
        p.lay.addWidget(uv.Alert("Every control here is painted by UiVault3 from the UI Vault website's designs.", "info"))
        p.end(); return p

    # behaviour
    def restyle_app(self):
        uv.apply(self.app, theme=self.cfg["theme"], kit=self.cfg["kit"])

    def refresh_labels(self):
        for l in self.findChildren(QLabel):
            if l.property("muted"): l.setStyleSheet(f"color: {tok('muted').name()}")
        self.update()

    def changed(self, key, value=None):
        if value is not None: self.cfg[key] = value
        if key == "overlay_screen" and self.overlay.isVisible(): self.overlay.show_on(self.cfg["overlay_screen"])
        save_settings(self.cfg); self.preview.update(); self.overlay.update()

    def set_color(self, c):
        self.cfg["color"] = c; self.cfg["hue"] = hsv_of(QColor(c).getRgb()[:3])[0]
        self.show_color(); save_settings(self.cfg)

    def show_color(self):
        c = self.cfg["color"]
        self.cur.setText(f"{c.upper()}  hue {self.cfg['hue'] * 2}°")
        for s in self.swatches: s.on = s.color.lower() == c.lower(); s.update()

    def custom_color(self):
        c = QColorDialog.getColor(QColor(self.cfg["color"]), self, "Target colour")
        if c.isValid(): self.set_color(c.name())

    def toggle_pick(self):
        self.preview.picking = not self.preview.picking
        self.pick_btn.setText("Click the preview…" if self.preview.picking else "Pick from preview"); self.pick_btn.updateGeometry()
        self.preview.update()

    def pick_at(self, x, y):
        f = self.engine.last_frame if self.engine else None
        if f is None: return
        y0, x0 = max(0, y - 1), max(0, x - 1)
        b, g, r = f[y0:y + 2, x0:x + 2, :3].reshape(-1, 3).mean(0).astype(int)
        hsv = hsv_of((r, g, b))
        self.set_color(QColor(r, g, b).name())
        if hsv[1] < self.cfg["min_sat"]: self.cfg["min_sat"] = max(30, hsv[1] - 40)
        if hsv[2] < self.cfg["min_val"]: self.cfg["min_val"] = max(30, hsv[2] - 40)
        self.toggle_pick()

    def forget_screen(self):
        self.cfg["restore_token"] = None; save_settings(self.cfg)
        if self.running: self.toggle(); self.toggle()

    def set_overlay(self, on):
        self.cfg["overlay"] = on; save_settings(self.cfg)
        if on:
            self.overlay.show_on(self.cfg["overlay_screen"])
        else:
            self.overlay.hide()

    def toggle(self):
        if self.running:
            self.engine.stop_capture(); self.engine = None; self.running = False
            self.go.setText("Start capture"); self.set_status("Idle", "warn")
            self.preview.data = None; self.preview.update(); self.overlay.data = None; self.overlay.update()
        else:
            self.engine = Engine(self.cfg)
            self.engine.frameReady.connect(self.on_frame); self.engine.failed.connect(self.on_fail)
            self.engine.started_ok.connect(lambda: self.set_status("Live", "success"))
            self.engine.start_capture(); self.running = True
            self.go.setText("Stop capture"); self.set_status("Waiting for screen…", "info")
            if self.cfg["minimize"]: QTimer.singleShot(400, self.showMinimized)
        self.go.updateGeometry()
        if self.cfg["overlay"] and not self.overlay.isVisible(): self.set_overlay(True)

    def set_status(self, text, kind):
        self.status.text, self.status.kind = text, kind
        self.status._restyle() if hasattr(self.status, "_restyle") else None
        self.status.updateGeometry(); self.status.update()

    def on_fail(self, msg):
        if self.running: self.toggle()
        self.set_status(msg[:60], "danger")

    def on_frame(self, d):
        if not self.running: return
        if self.engine.cap.restore_token and self.engine.cap.restore_token != self.cfg["restore_token"]:
            self.cfg["restore_token"] = self.engine.cap.restore_token; save_settings(self.cfg)
        self.preview.show_frame(d)
        if self.overlay.isVisible():
            self.overlay.show_frame(d); self.engine.overlay_scale = self.overlay.scale()
        w, h = d["size"]
        for b, t in ((self.b_cap, f"Capture {d['cap_fps']:.0f} fps"), (self.b_proc, f"Process {d['fps']:.0f} fps"),
                     (self.b_ms, f"{d['ms']:.1f} ms"), (self.b_hits, f"{len(d['targets'])} hits"), (self.b_res, f"{w}×{h}")):
            if b.text != t: b.text = t; b.updateGeometry(); b.update()

    def closeEvent(self, e):
        if self.engine: self.engine.stop_capture()
        self.overlay.close(); super().closeEvent(e)


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("ChromaScope"); app.setDesktopFileName("chromascope")
    ensure_kwin_rules()
    w = Window(app); w.show()
    if "--start" in sys.argv: QTimer.singleShot(300, w.toggle)
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
