"""NNMM: neural-network mouse movement. OpenCV finds the targets on the captured screen, a .npz model run by OpenCV DNN
decides how to move, and a virtual uinput mouse moves the pointer, while right click is held.

The nearest box becomes the model's 6 inputs (where the target is from the pointer, how the target moves, how the
pointer moves; in screen heights) and the model outputs the pointer's next velocity (1.0 = top speed). The pointer's
real position comes from KWin, 120 times a second. The model is always a file the user chose; nothing is built in.
"""
from __future__ import annotations

import time
from collections import deque
from pathlib import Path

import numpy as np
from PySide6.QtCore import QObject, QTimer, Signal

import nnio
from aimer.runtime import Aimer
from aimkit import kit
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
    lockChanged = Signal()                                       # the locked-on box appeared, moved or went away (redraw)

    def __init__(self, cfg, cursor, screen_map, parent=None):
        super().__init__(parent)
        self.cfg = cfg                                            # cfg["nn_model_path"]: the model file the user chose
        self.cursor, self.screen_map = cursor, screen_map          # screen_map(frame w, h) -> (screen x, y, frame px per px, height)
        self.timer = QTimer(self); self.timer.setInterval(8); self.timer.timeout.connect(self.tick)
        self.mouse, self.model, self.on, self.held, self.ratio = None, None, False, False, 1.0
        self.aimer = None                                         # aim-trainer runtime, when the model is one of those
        self.status, self.counts, self.frames, self.targets = "Off", 0, 0, 0
        self.map, self.box, self.colour_note = None, None, ""      # last frame's screen map; locked box (frame px)
        self.cur_frame = None                                      # where the "crosshair" was last frame (frame px)
        # Crosshair: the aimer was trained for 0-4 frames between a move and seeing it. Capture latency (game → screen →
        # capture → detection) easily exceeds that at 60 steps/s, and then it overshoots wildly. So the latency is measured
        # from our own moves, and the aimer is stepped every latency/3 s: it always sees about 3 frames of delay.
        self.step_iv, self.lat = cfg["nn_step_ms"] / 1000, deque(maxlen=7)   # saved from last time; 67 ms (15/s) at first
        self.t_step, self.lat_probe, self.step_next = None, None, None
        self.sent_log = deque(maxlen=64)                           # (time, counts) of recent moves, for watch_latency
        self.reached, self.slowed, self.dmin, self.unseen = False, 0, None, 0   # overshoot guard (see aim_step)
        self.since_best = np.zeros(2)                              # counts sent since the closest approach (to undo)
        self.lose()

    def lose(self):
        had = getattr(self, "box", None) is not None
        self.box = None
        if had: self.lockChanged.emit()
        self.goal, self.goal_v, self.t_goal, self.aim, self.cmd, self.cur_prev = None, np.zeros(2), None, None, np.zeros(2), None
        self.sent, self.win_moved, self.win_sent, self.win_n, self.why = np.zeros(2), np.zeros(2), np.zeros(2), 0, ""

    def start(self):
        if not self.load_model(): self.changed.emit(); return False
        try: self.mouse = self.mouse or VirtualMouse()
        except OSError as e:
            self.status = f"Can't create the virtual mouse ({e.strerror}). Run ./setup.sh once."; self.changed.emit(); return False
        self.on = True; self.lose(); self.mode_changed()
        self.status = "Ready: hold right click to move the mouse"; self.changed.emit(); return True

    @property
    def crosshair(self): return self.cfg["nn_aim_mode"] == "crosshair"

    def mode_changed(self):
        """Pointer mode needs the pointer position (KWin) and, for plain models, the 120 Hz steering timer."""
        self.lose()
        if self.aimer is not None: self.aimer.reset()
        if self.on and not self.crosshair: self.cursor.start()
        else: self.cursor.stop()
        if self.on and not self.crosshair and self.aimer is None: self.timer.start()
        else: self.timer.stop()

    def stop(self, why="Off"):
        self.on = False; self.mode_changed(); self.status = why
        if self.mouse: self.mouse.close(); self.mouse = None
        self.changed.emit()

    def set_held(self, down):
        self.held = down; self.lose(); self.t_step = self.lat_probe = None; self.reached, self.dmin, self.unseen = False, None, 0
        self.since_best = np.zeros(2)
        if down and self.step_next: self.step_iv, self.step_next = self.step_next, None
        self.cfg["nn_step_ms"] = round(self.step_iv * 1000, 1)     # remembered for next time (app saves settings)
        if self.aimer is not None and down: self.aimer.reset()   # every hold starts fresh: nothing learned earlier carries over
        if self.on and not down: self.status = "Ready: hold right click to move the mouse"; self.changed.emit()

    def load_model(self):
        path = self.cfg["nn_model_path"]
        name = Path(path).name if path else ""
        self.aimer = None
        if not path: self.status = "No model chosen. Use “Choose model file…” first."; return False
        try:
            if not Path(path).is_file(): raise FileNotFoundError(f"{path} doesn't exist")
            if kit.is_kit(path):
                # Kit network (aim-trainer kit/: aim4 … aim6): brain2 + the v2 setup + deadzone, on top of its base model.
                base = (self.cfg["nn_base_path"] or kit.find_base(path)) if kit.needs_base(path) else ""
                if kit.needs_base(path) and (not base or not Path(base).is_file()):
                    raise FileNotFoundError("it's a kit network and needs its base model.npz (Choose model file… asks for it)")
                self.aimer = kit.KitAimer(path, base, nnio.to_onnx, whole_counts=True, fire="hold")
                self.model = self.aimer
                return True
            if nnio.is_aimer(path):
                # Aim-trainer model: its own estimator + snap/track brain, the network (run by OpenCV) corrects the plan.
                # whole_counts: uinput only takes whole counts, the runtime carries the rest itself. fire="hold": its
                # hold-to-fire training (stay on a target that doesn't go away), which is what ChromaScope does; in the
                # default click mode it expects the target to vanish after each of its own clicks, and jitters more.
                self.aimer = Aimer(None, whole_counts=True, fire="hold")
                self.model = self.aimer.policy = nnio.CvPolicy(path); self.aimer.use_model = True
                return True
            model = nnio.CvModel(path)
            bad = model.check()
            if bad: raise ValueError(bad)
        except Exception as e:  # noqa: BLE001 - show why (OpenCV's message, wrong shape, doesn't steer)
            self.status = f"Can't use model {name}: {e}"[:160]; return False
        self.model = model; return True

    def wanted(self, d):
        """The frame's boxes in the colour chosen on the NNMM tab ("" = any colour)."""
        want, colors = self.cfg["nn_color"], d.get("colors") or []
        if not want: self.colour_note = ""; return d["targets"]
        if not any(c["name"] == want for c in colors):
            self.colour_note = f"no colour called “{want}” on the Detect tab, going for any"; return d["targets"]
        self.colour_note = ""
        return [t for t in d["targets"] if t.cls < len(colors) and colors[t.cls]["name"] == want]

    def lock(self):
        """What it's locked onto, in frame px, for drawing: {"box", "aim", "cur", "dist"} or None."""
        if self.box is None or self.map is None or self.cur_frame is None: return None
        k, cur = self.map[2], self.cur_frame
        if not self.crosshair and self.cursor.at is not None:     # pointer: draw from where it is right now
            sx, sy, k = self.map; cur = ((self.cursor.at[0] - sx) * k, (self.cursor.at[1] - sy) * k)
        return {"box": self.box, "aim": tuple(self.aim), "cur": tuple(cur), "dist": float(np.hypot(self.aim[0] - cur[0], self.aim[1] - cur[1])) / k}

    def feed(self, d, why=""):
        """Per frame (d = Engine frame dict, None when capture stopped / paused): the nearest box becomes the goal, in desktop px."""
        self.frames += d is not None
        targets = self.wanted(d) if d is not None else []
        if d is not None and len(targets) != self.targets: self.targets = len(targets); self.changed.emit()
        if not (self.on and self.held): return
        if d is None: self.lose(); self.why = why or "capture isn't running"; return
        w, h = d["size"]
        if self.crosshair:                                         # 3D: the crosshair is the screen centre, the camera turns
            if self.aimer is None:
                self.lose(); self.say("Crosshair (3D) needs an aim-trainer model (expert-….npz)"); return
            self.map, cur = (0, 0, 1.0), np.array([w / 2, h / 2])
        else:                                                      # 2D: the pointer is the crosshair
            sx, sy, k, sh = self.screen_map(w, h)
            if abs(w / h - (w / k) / sh) > 0.03:
                self.lose(); self.say("Holding: the captured screen isn't the one picked in Setup → Monitor (sizes differ)"); return
            self.map = (sx, sy, k)
            if self.cursor.at is None: self.goal, self.why = None, "waiting for the pointer position from KWin"; return
            cur = np.array([(self.cursor.at[0] - sx) * k, (self.cursor.at[1] - sy) * k])
        self.cur_frame = cur
        boxes = [(np.array([t.x + t.w / 2, t.y + t.h / 2]), t) for t in targets]
        if self.crosshair: self.watch_latency(d, boxes)
        if self.crosshair and self.t_step is not None and d.get("t", time.monotonic()) - self.t_step < self.step_iv * 0.9:
            return                                                 # between aimer steps (see step_iv)
        if not boxes:
            if self.aimer is not None: self.aim_step(d, None, cur)   # it was trained for targets that drop out for a while
            self.lose(); self.why = "nothing detected" + (f" in {self.cfg['nn_color']}" if self.cfg["nn_color"] and not self.colour_note else "")
            if self.aimer is not None: self.say(f"Holding right click · {self.why}")
            return
        ref = self.aim if self.aim is not None else cur              # keep following the same box while it's there
        p, t = min(boxes, key=lambda q: np.hypot(*(q[0] - ref)))
        if self.aim is not None and np.hypot(*(p - ref)) > h * 0.12: p, t = min(boxes, key=lambda q: np.hypot(*(q[0] - cur))); self.goal = None
        self.aim, self.why, self.box = p, "", (t.x, t.y, t.w, t.h)
        self.lockChanged.emit()
        if self.aimer is not None: self.aim_step(d, t, cur); return
        goal, now = np.array([sx + p[0] / k, sy + p[1] / k]), time.monotonic()
        if self.goal is not None and now > self.t_goal:
            self.goal_v = 0.6 * self.goal_v + 0.4 * np.clip((goal - self.goal) / (now - self.t_goal), -3000, 3000)
        self.goal, self.t_goal = goal, now

    def overshoot(self):
        """The aimer lost it badly (more delay than it was trained for, whatever the cause): crosshair steps it less
        often from now on, and either way it starts over."""
        # Turn back by everything sent since the closest approach (a camera turns linearly with counts), so the target
        # is in view again instead of somewhere behind us.
        back = -self.since_best
        while np.hypot(*back) >= 1:
            c = np.clip(back, -400, 400); ix, iy = self.mouse.move(*c); back -= (ix, iy)
            if not (ix or iy): break
        self.since_best = np.zeros(2)
        self.reached, self.dmin, self.unseen = False, None, 0; self.slowed += 1; self.aimer.reset(); self.t_step = None
        if self.crosshair: self.step_iv, self.step_next = min(0.2, max(self.step_iv * 2, 1 / 15)), None

    def watch_latency(self, d, boxes):
        """After a clear move, time until the target has shifted at least half the expected way (crosshair mode only)."""
        lp, now = self.lat_probe, d.get("t", time.monotonic())
        if lp is None or not boxes: return
        t0, p0, c, expect = lp
        if now - t0 > 0.6: self.lat_probe = None; return          # never saw it: try again on a later move
        p = min(boxes, key=lambda q: np.hypot(*(q[0] - p0)))[0]
        along = float(np.dot(p - p0, -c / np.hypot(*c)))           # the view turns the other way from the move
        if along < 0.5 * expect: return
        self.lat.append(now - t0); self.lat_probe = None
        iv = float(np.clip(np.median(self.lat) / 3, 1 / 60, 0.2))
        if len(self.lat) >= 3 and iv > 1.3 * self.step_iv:         # too fast for this latency: slow down now, start over
            self.step_iv = iv; self.aimer.reset(); self.t_step = None
        elif len(self.lat) >= 3: self.step_next = iv              # faster is fine: from the next hold on

    def start_probe_timer(self, now, t, counts, W):
        """Time this move if it should visibly shift the target (≥ 10 px by the aimer's own sensitivity estimate) and
        the view was quiet before it: with earlier moves still in flight, their shift shows first and the lag would be
        measured far too short (an aimer that sends a steady stream of small moves, like the kit networks)."""
        b = self.aimer.brain
        if not hasattr(b, "G") or self.aimer.status().get("confidence", 0) < 1: return
        recent = sum(np.hypot(*c) for tc, c in self.sent_log if now - tc < 0.25)
        expect = float(b.G[0] * np.hypot(*counts) * W / 2)
        if recent * b.G[0] * W / 2 > 0.15 * expect: return       # earlier moves would still be landing
        if expect >= 10: self.lat_probe = (now, np.array([t.x + t.w / 2, t.y + t.h / 2]), np.asarray(counts, float), expect)

    def aim_step(self, d, t, cur):
        """Aim-trainer runtime, once per frame; t = the chosen box, or None when nothing is visible this frame.
        Crosshair (3D): the target as seen, the crosshair is the screen centre, exactly how the runtime was trained.
        Pointer (2D): the target relative to the pointer, around the screen centre, so moving the pointer shifts it like a
        camera turn would. Either way the runtime works out how far a count moves things (sensitivity, acceleration)."""
        w, h = d["size"]
        sx, sy, k = self.map
        W, H = w / k, h / k                                      # crosshair: frame px; pointer: desktop px
        now = d.get("t", time.monotonic())
        dt = None                                                  # this step's length vs the usual one (uneven frames)
        if self.crosshair and self.t_step is not None: dt = float(np.clip((now - self.t_step) / self.step_iv, 0.5, 3))
        self.t_step = now
        if t is None:
            out = self.aimer.step_target((W / 2, H / 2), False, None, None, W, H, dt=dt)
            xy = None
        else:
            xy = ((t.x + t.w / 2 - cur[0]) / k + W / 2, (t.y + t.h / 2 - cur[1]) / k + H / 2)
            bw, bh = t.w / k, t.h / k
            out = self.aimer.step_target(xy, True, max(1.0, float(np.sqrt(bw * bh / np.pi))), (bw, bh), W, H, dt=dt)
        ix, iy = self.mouse.move(*np.clip(out.counts, -400, 400))
        self.aimer.applied((ix, iy)); self.counts += abs(ix) + abs(iy); self.since_best += (ix, iy)
        if self.crosshair and t is not None and self.lat_probe is None and (ix or iy): self.start_probe_timer(now, t, (ix, iy), W)
        if ix or iy: self.sent_log.append((now, (ix, iy)))
        if xy is None:
            self.unseen += 1
            if self.crosshair and self.dmin is not None and self.unseen * self.step_iv > 1.0: self.overshoot()   # spun away
            return
        self.unseen = 0
        st = self.aimer.status()
        dist = float(np.hypot(xy[0] - W / 2, xy[1] - H / 2))
        # Overshoot guard: it was on the target and now it's far off. That's the aimer seeing more delay than it was
        # trained for (whatever the cause), so step it less often and start it over.
        size = max(t.w, t.h) / k
        if st.get("mode") != "probing" and (self.dmin is None or dist <= self.dmin):
            self.dmin = dist; self.since_best = np.array([ix, iy], float)   # this frame shows the view before this move
        if dist < max(20.0, size): self.reached = True
        elif (self.reached and dist > max(250.0, 8 * size)) or (self.dmin is not None and dist > max(400.0, 3 * self.dmin)):
            self.overshoot()
        lat = f" · latency {np.median(self.lat) * 1000:.0f} ms, {1 / self.step_iv:.0f} steps/s" if self.crosshair and self.lat else ""
        if self.slowed: lat += f" · overshoot caught {self.slowed}×"
        self.say(f"Moving ({self.model.name}, {st.get('mode', '…')}) · {dist:.0f} px from target{lat}"
                 + (f" · {self.colour_note}" if self.colour_note else ""))

    def tick(self):
        if not (self.on and self.held) or self.mouse is None or self.aimer is not None: return
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
        self.say(f"Moving ({self.model.name}) · {dist:.0f} px from target" + (f" · {self.colour_note}" if self.colour_note else ""))

    def say(self, text):
        if text != self.status: self.status = text; self.changed.emit()
