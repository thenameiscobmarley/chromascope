"""The drop-in aimer for any 3D app.

    from aimer import Aimer
    aimer = Aimer("model.npz", colors=[(255, 40, 40)], aim="head")

    every frame:
        out = aimer.step_frame(frame)          # frame: (height, width, 3) RGB uint8
        # or, if your app already knows where the target is on screen:
        # out = aimer.step_target((x, y), True, radius_px, (box_w, box_h), width, height)
        dx, dy = out.counts                    # turn the camera by this, like mouse counts
        if out.click: fire()
        aimer.applied((dx + player_dx, dy + player_dy))   # everything that really moved the camera

The app must turn its camera from these counts exactly as it would from real
mouse input (through its own sensitivity, acceleration and so on) and report
everything that moved the camera back with applied(). The aimer is never told
the sensitivity: it works it out.

What the app knows can help (all optional):
    Aimer(..., whole_counts=True)    # the app only takes whole mouse counts: counts are rounded,
                                     # and the rest is carried over to the next frame
    Aimer(..., fire="hold")          # hold to fire: out.click stays on while on target
    aimer.step_target(..., dt=1.4)   # this frame took 1.4x the normal frame time
    aimer.step_target(..., zoomed=True)   # the player is scoped in
Recoil and the player's own movement don't need to be reported: the aimer was
trained to handle them (report them in applied() if the app knows them exactly).
"""
import os

import numpy as np

from .brain import Brain
from .config import ACT_DIM, DEG_PER_COUNT
from .detector import detect
from .policy import Policy


class AimOutput:
    def __init__(self, counts, click, found, aim_px):
        self.counts, self.click, self.found, self.aim_px = counts, click, found, aim_px


class Aimer:
    def __init__(self, model_path=None, colors=((255, 40, 40),), tolerance=40.0, aim="center",
                 head_pct=15.0, downscale=2, whole_counts=False, fire="click"):
        self.policy = None
        self.model_error = None
        if model_path and os.path.exists(model_path):
            try:
                self.policy = Policy.load(model_path)
            except Exception as e:      # keep working with the classic aimer
                self.model_error = str(e)
        self.use_model = self.policy is not None
        self.colors = [list(c) for c in colors]
        self.tolerance, self.aim, self.head_pct, self.downscale = tolerance, aim, head_pct, downscale
        self.brain = Brain(1)
        self.whole_counts = bool(whole_counts)
        self.hold = fire == "hold"
        self.last_switch = -1e9
        self.frame = 0
        self.size = None
        self.detection = None

    def reset(self):
        """Forget the learned sensitivity, delay and target track."""
        self.brain.reset(np.ones(1, bool))

    # ---- seeing ------------------------------------------------------------
    def detect(self, frame):
        """Find the target in an RGB frame. Returns a Detection in full-resolution pixels."""
        h, w = frame.shape[:2]
        ds = self.downscale
        prefer = None
        b = self.brain
        if self.size == (w, h) and hasattr(b, "lost") and not b.lost[0]:
            half = w / 2
            prefer = ((w / 2 + b.p_u[0, 0] * half) / ds, (h / 2 + b.p_u[0, 1] * half) / ds)
        d = detect(frame[::ds, ::ds], self.colors, self.tolerance, self.aim, self.head_pct, prefer)
        d.x, d.y, d.radius = d.x * ds, d.y * ds, d.radius * ds
        d.boxes = [tuple(v * ds for v in box) for box in d.boxes]
        d.box_w, d.box_h = d.box_w * ds, d.box_h * ds
        self.detection = d
        return d

    # ---- deciding ----------------------------------------------------------
    def step_frame(self, frame, fresh=True, dt=None, zoomed=None):
        h, w = frame.shape[:2]
        d = self.detect(frame)
        return self.step_target((d.x, d.y), d.found, d.radius, (d.box_w, d.box_h), w, h, fresh, dt, zoomed)

    def step_position(self, xy, visible, width, height, fresh=True, dt=None, zoomed=None):
        """For detectors that only give the target's position (no size): the aimer assumes a
        default size and was trained for this case too."""
        return self.step_target(xy, visible, None, None, width, height, fresh, dt, zoomed)

    def step_target(self, xy, visible, radius_px, box_px, width, height, fresh=True, dt=None, zoomed=None):
        """xy: aim point in pixels; radius_px: size of the aim zone; box_px: full target
        width/height. radius_px/box_px may be None if the detector only gives a position.
        dt: this frame's time compared with a normal frame (None = normal); zoomed: scoped in."""
        self.size = (width, height)
        self.frame += 1
        half = width / 2
        obs = np.array([[(xy[0] - width / 2) / half, (xy[1] - height / 2) / half]])
        box = None if box_px is None else np.array([[box_px[0] / 2 / half, box_px[1] / 2 / half]])
        radius = np.array([np.nan if radius_px is None else radius_px / half])
        b = self.brain
        if not visible and hasattr(b, "p"):
            obs = b.est.o_last.copy()
        feat = b.observe(obs, np.array([bool(visible)]), np.array([bool(fresh)]), radius, box,
                         dt=1.0 if dt is None else float(dt), whole=self.whole_counts, hold=self.hold,
                         zoomed=bool(zoomed))
        raw = self.policy.mean(feat) if (self.use_model and self.policy is not None) else np.zeros((1, ACT_DIM))
        counts, click = b.act(raw)
        if b.est.switched[0]:
            self.last_switch = self.frame
        aim_px = (width / 2 + b.p_u[0, 0] * half, height / 2 + b.p_u[0, 1] * half)
        return AimOutput(counts[0].copy(), bool(click[0]), bool(visible), aim_px)

    def applied(self, counts):
        """Report the mouse counts that actually moved the camera this frame."""
        self.brain.commit(np.asarray(counts, float).reshape(1, 2))

    # ---- debug -------------------------------------------------------------
    def status(self, hfov_deg=90.0):
        """What the aimer currently believes, for overlays. Pixel values use the last frame size.
        hfov_deg is the app's real FOV, used only to express the gain as a sensitivity number."""
        b = self.brain
        if not hasattr(b, "p") or self.size is None:
            return {}
        width, height = self.size
        half = width / 2
        th = np.tan(np.radians(hfov_deg / 2))
        k = float(np.exp(b.est.logk[0]))
        w = b.info.get("w", np.array([0.0]))[0]
        mode = "probing" if b.probing[0] else ("lost" if b.lost[0] else ("snap" if w > 0.5 else "track"))
        return dict(
            est_sens=float(b.G[0] * th / (np.pi / 180) / DEG_PER_COUNT),
            est_fov=float(2 * np.degrees(np.arctan(k))),
            confidence=float(b.conf[0]), delay=int(b.d[0]), mode=mode, blend=float(w),
            wobble=float(b.osc[0]), switched_recently=self.frame - self.last_switch < 90,
            speed_px=float(np.linalg.norm(b.v[0]) * half),
        )
