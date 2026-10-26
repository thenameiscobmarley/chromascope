"""The aim-trainer's kit/runtime2.Aimer2 for ChromaScope: a kit network (e.g. aim6_recoil.npz, a vector of trained
weights on top of a base model.npz) driving brain2 with the v2 setup, the two-zone deadzone always on (as in the
trainer's range). Same calls as aimer.Aimer: step_target / applied / reset / status.

Two kinds of kit file: a full export (x_full.npz, aimnet.export_full: every weight, loads on its own) and a
trained-weights vector (x.npz, "p"), which is rebuilt on top of the exact model.npz it was trained on (the trainer's
own model.npz, 57 -> 128 -> 128 -> 5). The network itself runs in OpenCV DNN (the same maths as aimnet.forward, as one
MatMul/Add/Tanh graph).
"""
from pathlib import Path

import cv2
import numpy as np

from aimer.runtime import AimOutput, Aimer

from . import aimnet
from .brain2 import KitBrain

SL = slice(0, 1)


def is_kit(path):
    """a kit network file: a full export, or one flat vector "p" of trained weights"""
    if Path(path).suffix != ".npz": return False
    with np.load(path, allow_pickle=False) as z: return is_full_z(z) or ("p" in z.files and "pi0" not in z.files)


def is_full_z(z): return "format" in z.files and str(z["format"]).startswith("kit-aimnet-full")


def needs_base(path):
    """a trained-weights vector needs its base model.npz; a full export doesn't"""
    with np.load(path, allow_pickle=False) as z: return not is_full_z(z)


def find_base(net_path):
    """the trainer's model.npz near a kit file (kit/models/x.npz -> ../../model.npz), or None"""
    p = Path(net_path).resolve().parent
    for d in (p, p.parent, p.parent.parent):
        if (d / "model.npz").is_file(): return d / "model.npz"
    return None


def check_base(base_path):
    """'' if base_path fits as a kit network's base (57 inputs or fewer -> 128 -> 128 -> 5), else why not"""
    with np.load(base_path, allow_pickle=False) as z:
        if "pi0" not in z.files: return "not an aim-trainer model (no pi0)"
        h = z["pi0"].shape[1]
        if h != aimnet.OLD_H: return f"its layers have {h} units; kit networks are built on a {aimnet.OLD_H}-unit model.npz"
    return ""


class CvRow(aimnet.NetRow):
    """aimnet.NetRow with the network's forward pass in OpenCV DNN."""

    def __init__(self, x, onnx):
        super().__init__(x)
        n = self.net
        W1 = np.concatenate([np.concatenate([n["W1o"], n["W1e"]], 0), n["W1n"]], 1)
        W2 = np.concatenate([np.concatenate([n["W2oo"], n["W2no"]], 0), n["W2n"]], 1)
        W3 = np.concatenate([np.concatenate([n["W3oo"], n["W3no"]], 0), n["W3k"]], 1)
        self.cv = cv2.dnn.readNetFromONNX(np.frombuffer(onnx([W1, W2, W3], [n["b1"], n["b2"], n["b3"]]), np.uint8))

    def forward_cv(self, z):
        self.cv.setInput(np.asarray(z, np.float32))
        return np.asarray(self.cv.forward(), np.float32).reshape(len(z), -1)

    def act(self, brain, feat, sl):
        z = np.concatenate([feat, brain.extra[sl]], 1)
        b = self.setup[0]
        if b > 0: z[:, aimnet.BLIND] = (1 - b) * z[:, aimnet.BLIND] + b * self.blind[aimnet.BLIND]
        o = self.forward_cv(z)
        aimnet.apply_knobs(brain, sl, o)
        raw = o[:, :5].copy()
        a = brain.k_smooth[sl][:, None]
        if self.prev is not None and (a < 1).any():
            fresh = (brain.est.since_jump[sl] < 2)[:, None]
            raw[:, 3:5] = np.where(fresh, raw[:, 3:5], a * raw[:, 3:5] + (1 - a) * self.prev)
        self.prev = raw[:, 3:5].copy()
        return raw


class KitAimer(Aimer):
    """runtime2.Aimer2: brain2 + a kit network + the deadzone, for one aimer."""

    def __init__(self, net_path, base_path, onnx, **kw):
        if needs_base(net_path):
            if not base_path: raise FileNotFoundError("this kit file holds only trained weights and needs its base model.npz")
            bad = check_base(base_path)
            if bad: raise ValueError(f"base model {Path(base_path).name}: {bad}")
            aimnet.MODEL, aimnet._W0 = str(base_path), None        # the weights the kit file was trained on top of
        super().__init__(None, **kw)
        net = aimnet.load_file(str(net_path))                       # full export as is, or the vector on its base
        self.row = CvRow(net, onnx)
        self.use_model = True
        self.brain = KitBrain(1)
        self.name, self.base = Path(net_path).stem, (Path(base_path) if base_path and needs_base(net_path) else None)

    def reset(self):
        self.brain.reset(np.ones(1, bool))
        self.row.prev = None

    def step_target(self, xy, visible, radius_px, box_px, width, height, fresh=True, dt=None, zoomed=None):
        self.size = (width, height)
        self.frame += 1
        half = width / 2
        obs = np.array([[(xy[0] - width / 2) / half, (xy[1] - height / 2) / half]])
        box = None if box_px is None else np.array([[box_px[0] / 2 / half, box_px[1] / 2 / half]])
        radius = np.array([np.nan if radius_px is None else radius_px / half])
        b = self.brain
        if not visible and hasattr(b, "p"): obs = b.est.o_last.copy()
        self.row.before(b, SL)
        b.dz_in[0] = max(b.dz_in[0], aimnet.V2["dz_in"])          # the deadzone always on, as in the trainer's range
        b.dz_out[0] = max(b.dz_out[0], aimnet.V2["dz_out"])
        feat = b.observe(obs, np.array([bool(visible)]), np.array([bool(fresh)]), radius, box,
                         dt=1.0 if dt is None else float(dt), whole=self.whole_counts, hold=self.hold, zoomed=bool(zoomed))
        raw = self.row.act(b, feat, SL)
        counts, click = b.act(raw)
        send, _ = self.row.after(b, SL, counts, np.zeros((1, 2)))
        if b.est.switched[0]: self.last_switch = self.frame
        aim_px = (width / 2 + b.p_u[0, 0] * half, height / 2 + b.p_u[0, 1] * half)
        return AimOutput(send[0].copy(), bool(click[0]), bool(visible), aim_px)

    def applied(self, counts):
        total = np.asarray(counts, float).reshape(1, 2)
        _, effect = aimnet.unmap(self.brain, SL, np.zeros((1, 2)), total)
        self.brain.commit(effect, sent=total)

    def status(self, hfov_deg=90.0):
        st = super().status(hfov_deg)
        if st: st["zone"] = ("normal", "careful", "deadzone")[int(self.brain.zone[0])]
        return st
