"""Undo history, auto-pause watchers (screen lock, focused app via a KWin script), the first-run tour and example profiles."""
from __future__ import annotations

import json
from pathlib import Path

from PySide6.QtCore import ClassInfo, QObject, QPoint, QRect, QRectF, QSize, Qt, QTimer, Signal, Slot
from PySide6.QtDBus import QDBusConnection, QDBusInterface
from PySide6.QtGui import QColor, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import QHBoxLayout, QLabel, QWidget

import uivault3 as uv


# ── Undo / redo ──────────────────────────────────────────────────────────────
class UndoStack:
    """Snapshots of settings. Callers commit after a burst of changes settles (sliders send many)."""

    def __init__(self, limit=100):
        self.items, self.pos, self.limit = [], -1, limit

    def commit(self, snap: dict) -> bool:
        s = json.dumps(snap, sort_keys=True)
        if self.pos >= 0 and self.items[self.pos] == s: return False
        del self.items[self.pos + 1:]
        self.items.append(s)
        if len(self.items) > self.limit: self.items.pop(0)
        self.pos = len(self.items) - 1
        return True

    def can_undo(self): return self.pos > 0
    def can_redo(self): return self.pos < len(self.items) - 1

    def undo(self):
        if not self.can_undo(): return None
        self.pos -= 1; return json.loads(self.items[self.pos])

    def redo(self):
        if not self.can_redo(): return None
        self.pos += 1; return json.loads(self.items[self.pos])


# ── Auto-pause watchers ──────────────────────────────────────────────────────
KWIN_SCRIPT = """function report(w) { callDBus("io.github.chromascope", "/Focus", "io.github.chromascope.Focus", "activeWindow",
    w ? (w.resourceClass + "\\n" + w.caption) : ""); }
workspace.windowActivated.connect(report);
report(workspace.activeWindow);
"""


@ClassInfo({"D-Bus Interface": "io.github.chromascope.Focus"})
class FocusWatcher(QObject):
    """Learns which window is focused from a tiny KWin script (Wayland apps can't ask directly)."""
    changed = Signal(str, str)               # app class, window title

    def __init__(self, folder: Path):
        super().__init__()
        self.app, self.title, self.seen, self.running = "", "", [], False
        self.script = folder / "focus.js"
        self.bus = QDBusConnection.sessionBus()
        self.ok = self.bus.registerService("io.github.chromascope") and self.bus.registerObject("/Focus", self, QDBusConnection.ExportAllSlots)

    @Slot(str)
    def activeWindow(self, s):
        app, _, title = s.partition("\n")
        if app == "chromascope": return       # ignore ourselves, so clicking the app doesn't pause it
        self.app, self.title = app, title
        if app and app not in self.seen: self.seen = ([app] + self.seen)[:12]
        self.changed.emit(app, title)

    def _kwin(self): return QDBusInterface("org.kde.KWin", "/Scripting", "org.kde.kwin.Scripting", self.bus)

    def start(self):
        if self.running or not self.ok: return self.ok
        self.script.write_text(KWIN_SCRIPT)
        k = self._kwin(); k.call("unloadScript", "chromascope-focus")
        k.call("loadScript", str(self.script), "chromascope-focus"); k.call("start")
        self.running = True; return True

    def stop(self):
        if self.running: self._kwin().call("unloadScript", "chromascope-focus"); self.running = False

    def matches(self, text):
        t = text.strip().lower()
        return not t or t in self.app.lower() or t in self.title.lower()


CURSOR_SCRIPT = """function report() { var p = workspace.cursorPos;
    callDBus("io.github.chromascope", "/Cursor", "io.github.chromascope.Cursor", "pos", p.x, p.y); }
workspace.cursorPosChanged.connect(report);
report();
"""


@ClassInfo({"D-Bus Interface": "io.github.chromascope.Cursor"})
class CursorWatcher(QObject):
    """The pointer's position in desktop (logical) px, reported by a KWin script on every move. Needs FocusWatcher's D-Bus name."""

    def __init__(self, folder: Path):
        super().__init__()
        self.at, self.running = None, False
        self.script = folder / "cursor.js"
        self.bus = QDBusConnection.sessionBus()
        self.ok = self.bus.registerObject("/Cursor", self, QDBusConnection.ExportAllSlots)

    @Slot(int, int)
    def pos(self, x, y): self.at = (x, y)

    def _kwin(self): return QDBusInterface("org.kde.KWin", "/Scripting", "org.kde.kwin.Scripting", self.bus)

    def start(self):
        if self.running or not self.ok: return self.ok
        self.script.write_text(CURSOR_SCRIPT)
        k = self._kwin(); k.call("unloadScript", "chromascope-cursor")
        k.call("loadScript", str(self.script), "chromascope-cursor"); k.call("start")
        self.running = True; return True

    def stop(self):
        if self.running: self._kwin().call("unloadScript", "chromascope-cursor"); self.running = False; self.at = None


class LockWatcher(QObject):
    """Polls whether the screen is locked (org.freedesktop.ScreenSaver.GetActive)."""
    changed = Signal(bool)

    def __init__(self):
        super().__init__()
        self.locked = False
        self.iface = QDBusInterface("org.freedesktop.ScreenSaver", "/ScreenSaver", "org.freedesktop.ScreenSaver", QDBusConnection.sessionBus())
        self.timer = QTimer(self); self.timer.setInterval(1500); self.timer.timeout.connect(self.poll); self.timer.start()

    def poll(self):
        r = self.iface.call("GetActive").arguments()
        locked = bool(r[0]) if r else False
        if locked != self.locked: self.locked = locked; self.changed.emit(locked)


# ── First-run tour ───────────────────────────────────────────────────────────
class Tour(QWidget):
    """Dims the window, cuts a hole round one control at a time and explains it."""
    finished = Signal()

    def __init__(self, win, steps):
        super().__init__(win)
        self.win, self.steps, self.i = win, steps, 0
        self.setAttribute(Qt.WA_NoSystemBackground)
        self.card = uv.Card(self, padding=16); self.card.setFixedWidth(360)
        lay = self.card.layout(); lay.setSpacing(8)
        self.count = QLabel(); self.title = uv.Heading("", 24); self.text = QLabel(); self.text.setWordWrap(True)
        f = self.text.font(); f.setPixelSize(14); self.text.setFont(f)
        f = self.count.font(); f.setPixelSize(11); f.setBold(True); self.count.setFont(f)
        lay.addWidget(self.count); lay.addWidget(self.title); lay.addWidget(self.text)
        row = QHBoxLayout(); lay.addLayout(row)
        self.skip = uv.Button("Skip tour", size="sm"); self.skip.clicked.connect(self.done)
        self.back = uv.Button("Back", size="sm"); self.back.clicked.connect(lambda: self.go(self.i - 1))
        self.next = uv.Button("Next", primary=True, size="sm"); self.next.clicked.connect(lambda: self.go(self.i + 1))
        row.addWidget(self.skip); row.addStretch(1); row.addWidget(self.back); row.addWidget(self.next)

    def start(self):
        self.setGeometry(self.win.rect()); self.show(); self.raise_(); self.go(0)

    def go(self, i):
        if i >= len(self.steps): return self.done()
        self.i = max(0, i)
        target, title, text, before = self.steps[self.i]
        if before: before()
        self.count.setText(f"STEP {self.i + 1} OF {len(self.steps)}")
        self.count.setStyleSheet(f"color: {uv.style().color('muted').name()}")
        self.title.text = title; self.title.updateGeometry(); self.title.update()
        self.text.setText(text)
        self.back.setEnabled(self.i > 0)
        self.next.setText("Finish" if self.i == len(self.steps) - 1 else "Next"); self.next.updateGeometry()
        QTimer.singleShot(30, self.place)

    def hole(self):
        target = self.steps[self.i][0]
        w = target() if target else None
        if not w or not w.isVisible(): return None
        return QRect(w.mapTo(self.win, QPoint(0, 0)), QSize(w.width(), w.height())).adjusted(-6, -6, 6, 6)   # uv.Button shadows size()

    def place(self):
        self.setGeometry(self.win.rect()); self.card.adjustSize()
        h, cw, ch = self.hole(), self.card.width(), self.card.height()
        W, H = self.width(), self.height()
        if h is None: x, y = (W - cw) // 2, (H - ch) // 2
        else:
            # beside the hole if it fits, otherwise below / above it
            if h.right() + 16 + cw < W: x, y = h.right() + 16, h.top()
            elif h.left() - 16 - cw > 0: x, y = h.left() - 16 - cw, h.top()
            elif h.bottom() + 16 + ch < H: x, y = h.left(), h.bottom() + 16
            else: x, y = h.left(), h.top() - 16 - ch
            x, y = max(12, min(W - cw - 12, x)), max(12, min(H - ch - 12, y))
        self.card.move(x, y); self.update()

    def paintEvent(self, _):
        p = QPainter(self); p.setRenderHint(QPainter.Antialiasing)
        dim = QPainterPath(); dim.addRect(QRectF(self.rect()))
        h = self.hole()
        if h is not None:
            cut = QPainterPath(); cut.addRoundedRect(QRectF(h), 8, 8); dim = dim.subtracted(cut)
        p.fillPath(dim, QColor(0, 0, 0, 150))
        if h is not None:
            p.setPen(QPen(uv.style().color("accent"), 2.5)); p.setBrush(Qt.NoBrush); p.drawRoundedRect(QRectF(h), 8, 8)
        p.end()

    def mousePressEvent(self, e): e.accept()   # the tour is modal: clicks outside the card do nothing

    def done(self):
        self.hide(); self.deleteLater(); self.finished.emit()


# ── Example profiles ─────────────────────────────────────────────────────────
def _c(name, color, box, tol=10, sat=120, val=110):
    return {"name": name, "color": color, "box": box, "tolerance": tol, "min_sat": sat, "min_val": val, "on": True}


_BASE = {"sel": 0, "min_area": 30, "merge": 6, "max_targets": 16, "sort": "largest", "fov": 600, "fov_shape": "circle",
         "fov_dx": 0, "fov_dy": 0, "show_fov": True, "crosshair": False, "mode": "box", "thickness": 2, "glow": 10,
         "labels": True, "assist": False, "assist_style": "stripes", "assist_color": "#2a6bff", "max_fps": 30}

EXAMPLES = {
    "Example - Traffic lights": {**_BASE, "colors": [_c("Red", "#ff2a2a", "#ff3b3b"), _c("Amber", "#ffb000", "#ffc83a", 8),
                                                      _c("Green", "#22e35a", "#7CFF4F", 12)]},
    "Example - Status LEDs": {**_BASE, "min_area": 6, "merge": 2, "mode": "corners",
                              "colors": [_c("OK", "#22e35a", "#7CFF4F", 12, 150, 150), _c("Error", "#ff2a2a", "#ff3b3b", 10, 150, 150)]},
    "Example - Red for colour-blind eyes": {**_BASE, "fov_shape": "full", "mode": "none", "assist": True, "assist_style": "stripes",
                                            "assist_color": "#2a6bff", "show_fov": False, "colors": [_c("Red", "#ff2a2a", "#ff3b3b", 9, 110, 90)]},
    "Example - Highlighter yellow": {**_BASE, "fov_shape": "full", "mode": "glow", "merge": 10, "show_fov": False,
                                     "colors": [_c("Highlight", "#fff200", "#ffc83a", 7, 140, 180)]},
}
