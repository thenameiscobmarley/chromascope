"""NNMM: neural-network mouse movement. OpenCV finds the targets on the captured screen, a .npz model run by OpenCV DNN
decides how to move, and a virtual uinput mouse moves the pointer, while right click is held.

The nearest box becomes the model's 6 inputs (where the target is from the pointer, how the target moves, how the
pointer moves; in screen heights) and the model outputs the pointer's next velocity (1.0 = top speed). The pointer's
real position comes from KWin, 120 times a second. The bundled model models/aim.npz is made by train.py.
"""
from __future__ import annotations

import time

import numpy as np
from PySide6.QtCore import QObject, QTimer, Signal

import nnio
from vmouse import VirtualMouse

VMAX = 1.2                      # top speed the model was trained with, screen heights / s
SPEED = 2.9                     # pointer top speed, screen heights / s (a pointer can go faster than the training cursor)
DEAD = 2                        # stop when this many px from the target
LEAD = 0.15                     # how far ahead (s) the training teacher aims at a moving target
INPUTS = ["dx", "dy", "target vx", "target vy", "cursor vx", "cursor vy"]
OUTPUTS = ["vx", "vy"]


def clip_norm(v, n=1.0):
    m = float(np.hypot(*v))
    return v * (n / m) if m > n else v


class Live(QObject):
    """Enabled = model loaded + virtual mouse created. While right click is held: steer the pointer to the nearest box.

    Frames (≈30/s) give the target; a 120 Hz timer reads the pointer from KWin, runs the model and moves. KDE accelerates
    the pointer, so how far it moves per count is learned on the fly.
    """
    changed = Signal()

    def __init__(self, cfg, folders, cursor, screen_map, parent=None):
        super().__init__(parent)
        self.cfg, self.folders = cfg, folders                     # model folders, searched in order
        self.cursor, self.screen_map = cursor, screen_map          # screen_map(frame w, h) -> (screen x, y, frame px per px, height)
        self.timer = QTimer(self); self.timer.setInterval(8); self.timer.timeout.connect(self.tick)
        self.mouse, self.model, self.on, self.held, self.ratio = None, None, False, False, 1.0
        self.status, self.counts, self.frames, self.targets = "Off", 0, 0, 0
        self.lose()

    def lose(self):
        self.goal, self.goal_v, self.t_goal, self.aim, self.cmd, self.cur_prev = None, np.zeros(2), None, None, np.zeros(2), None
        self.sent, self.win_moved, self.win_sent, self.win_n, self.why = np.zeros(2), np.zeros(2), np.zeros(2), 0, ""

    def start(self):
        if not self.load_model(): self.changed.emit(); return False
        try: self.mouse = self.mouse or VirtualMouse()
        except OSError as e:
            self.status = f"Can't create the virtual mouse ({e.strerror}). Run ./setup.sh once."; self.changed.emit(); return False
        self.on = True; self.lose(); self.cursor.start(); self.timer.start()
        self.status = "Ready: hold right click to move the mouse"; self.changed.emit(); return True

    def stop(self, why="Off"):
        self.on = False; self.lose(); self.cursor.stop(); self.timer.stop(); self.status = why
        if self.mouse: self.mouse.close(); self.mouse = None
        self.changed.emit()

    def set_held(self, down):
        self.held = down; self.lose()
        if self.on and not down: self.status = "Ready: hold right click to move the mouse"; self.changed.emit()

    def load_model(self):
        name = self.cfg["nn_model"]
        path = nnio.find(self.folders, name)
        try:
            if path is None: raise FileNotFoundError("file not found")
            model = nnio.CvModel(path)
            bad = model.check()
            if bad: raise ValueError(bad)
        except Exception as e:  # noqa: BLE001 - show why (OpenCV's message, wrong shape, doesn't steer)
            self.status = f"Can't use model {name}: {e}"[:160]; return False
        self.model = model; return True

    def feed(self, d, why=""):
        """Per frame (d = Engine frame dict, None when capture stopped / paused): the nearest box becomes the goal, in desktop px."""
        self.frames += d is not None
        if d is not None and len(d["targets"]) != self.targets: self.targets = len(d["targets"]); self.changed.emit()
        if not (self.on and self.held): return
        if d is None: self.goal, self.why = None, why or "capture isn't running"; return
        w, h = d["size"]
        sx, sy, k, _sh = self.screen_map(w, h)
        if self.cursor.at is None: self.goal, self.why = None, "waiting for the pointer position from KWin"; return
        cur = np.array([(self.cursor.at[0] - sx) * k, (self.cursor.at[1] - sy) * k])
        pts = [np.array([t.x + t.w / 2, t.y + t.h / 2]) for t in d["targets"]]
        if not pts: self.goal, self.aim, self.why = None, None, "nothing detected"; return
        ref = self.aim if self.aim is not None else cur              # keep following the same box while it's there
        p = min(pts, key=lambda q: np.hypot(*(q - ref)))
        if self.aim is not None and np.hypot(*(p - ref)) > h * 0.12: p = min(pts, key=lambda q: np.hypot(*(q - cur))); self.goal = None
        self.aim, self.why = p, ""
        goal, now = np.array([sx + p[0] / k, sy + p[1] / k]), time.monotonic()
        if self.goal is not None and now > self.t_goal:
            self.goal_v = 0.6 * self.goal_v + 0.4 * np.clip((goal - self.goal) / (now - self.t_goal), -3000, 3000)
        self.goal, self.t_goal = goal, now

    def tick(self):
        if not (self.on and self.held) or self.mouse is None: return
        now = time.monotonic()
        if self.cursor.at is None: self.say("Holding right click · waiting for the pointer position from KWin"); return
        cur = np.array(self.cursor.at, float)
        if self.cur_prev is not None:                              # learn px per count from our own moves (~0.1 s windows)
            self.win_moved += cur - self.cur_prev; self.win_sent += self.sent; self.win_n += 1
            if self.win_n >= 12:
                if np.hypot(*self.win_sent) >= 8:
                    r = float(np.dot(self.win_moved, self.win_sent) / np.dot(self.win_sent, self.win_sent))
                    self.ratio = float(np.clip(0.7 * self.ratio + 0.3 * r, 0.3, 4.0))
                self.win_moved, self.win_sent, self.win_n = np.zeros(2), np.zeros(2), 0
        self.cur_prev, self.sent = cur, np.zeros(2)
        if self.goal is None or now - self.t_goal > 0.4:
            self.cmd = np.zeros(2); self.say(f"Holding right click · {self.why or 'nothing detected'}"); return
        sh = self.screen_map(1, 1)[3]
        e = (self.goal - cur) / sh                                 # error in screen heights, like the training world
        x = np.clip(np.concatenate([e / 0.8, self.goal_v / sh / 0.8, self.cmd]), -3, 3)
        self.cmd = cmd = clip_norm(np.asarray(self.model(x), float)[:2])
        dist = float(np.hypot(*(e * sh)))
        step = np.zeros(2) if dist < DEAD else cmd * SPEED * sh * 0.008   # desktop px this tick
        if np.hypot(*step) > dist: step = step * dist / max(np.hypot(*step), 1e-6)   # never past the target
        ix, iy = self.mouse.move(*clip_norm(step / self.ratio, 30))       # small steps keep KDE's acceleration mild
        self.sent = np.array([ix, iy], float); self.counts += abs(ix) + abs(iy)
        self.say(f"Moving ({self.model.name}) · {dist:.0f} px from target")

    def say(self, text):
        if text != self.status: self.status = text; self.changed.emit()
