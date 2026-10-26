"""The new aim network: 77 inputs -> 256 -> 256 -> 15 outputs (about 89,000 weights; the current model has
57 -> 128 -> 128 -> 5, about 24,000).

Warm start: the current model.npz is copied into the first 128 units of each layer, the new 128 units get
small random incoming weights and ZERO outgoing weights, the 20 new inputs get zero weights into the old
units, and the 10 new outputs start at zero. So at the start the network IS the current code, exactly.

Weights are W = W0 + scale(group) * x, where W0 is that warm start and x is what training moves (x = 0 at
the start). Run 1 (2026-10-25) moved all 89k weights with tiny steps and learned nothing (luck dominated);
now the hidden layers are frozen (scale 0: the copied model + fixed random new units act as 256 ready-made
features) and only the 15 output heads train (3,855 weights), the knobs with 10x bigger steps. Outputs and their hard limits:
   0-4  gain fix, aim offset x/y, snap-vs-track, click  (the old five; the brain clips them to +-3)
   5    learn the sensitivity faster/slower   x0.25 .. x4
   6    learn the target's speed faster/slower x0.33 .. x3
   7    how much of the unseen-travel guess to use     0.05 .. 1 (never more than the code)
   8    ignore far-off detections (glitch gate)        off, or 202 .. 2 target sizes, at most 3 frames in a row
   9    largest move (anti-fling cap)                  off, or 201.5 .. 1.5 x (distance + size)
   10   accept a clear sensitivity change at once      off / on
   11-13 axis fix: flip x, flip y, swap               off / on (sticky: on above +0.5, off below -0.5)
   14   y sensitivity ratio fix                        x0.5 .. x2
   15   steady: on a calm target, ignore wobbles under  0 .. 0.4 target sizes (deadband)
   16   smooth snap/click: share of the new value used  1 .. 0.3 (the rest is last frame's)
Its own outputs are never fed back into its inputs.
"""
import os

import numpy as np  # bugcheck: ignore[PY040]

from .brain2 import EXTRA_DIM  # bugcheck: ignore[PY040]

OLD_IN, OLD_H = 57, 128
IN, H, OUT = OLD_IN + EXTRA_DIM, 256, 17
OUT_NAMES = ["gain fix", "aim x", "aim y", "snap", "click", "learn sens", "learn speed", "extrapolate",
             "glitch gate", "move cap", "fast switch", "flip x", "flip y", "swap", "y ratio", "steady", "smooth snap"]
MODEL = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "model.npz")

# layout of x: (name, shape, group name, scale)
_LAYOUT = []


def _add(name, shape, group, scale):
    _LAYOUT.append((name, shape, group, scale))


_add("W1_old", (IN, OLD_H), "layer 1 (old units)", 0.0)
_add("W1_new", (IN, H - OLD_H), "layer 1 (new units)", 0.03)
_add("b1", (H,), "layer 1 (old units)", 0.0)
_add("W2_old", (H, OLD_H), "layer 2 (old units)", 0.0)
_add("W2_new", (H, H - OLD_H), "layer 2 (new units)", 0.0)
_add("b2", (H,), "layer 2 (old units)", 0.0)
for o, nm in enumerate(OUT_NAMES):
    _add("W3_%d" % o, (H + 1,), "out: " + nm, 0.01 if o < 5 else 0.3)
# the v2 setup as 5 learned dials, each 0..1, all 0 at the start = exactly the current code (user 2026-10-25:
# "make the model start at the current code so it never starts below it")
SETUP_NAMES = ["blind own moves", "prediction off", "unseen-travel cut", "deadzone", "scoped fov"]
_add("setup", (len(SETUP_NAMES),), "v2 setup", 0.3)
SIZE = sum(int(np.prod(s)) for _, s, _, _ in _LAYOUT)
GROUP_NAMES = list(dict.fromkeys(g for _, _, g, _ in _LAYOUT))


def groups():
    ids, scales = [], []
    for _, shape, g, sc in _LAYOUT:
        k = int(np.prod(shape))
        ids.append(np.full(k, GROUP_NAMES.index(g)))
        scales.append(np.full(k, sc))
    return np.concatenate(ids), np.concatenate(scales)


_W0 = None
_SCALE = None


def base():
    """W0 as a flat vector in the layout's order (the warm start), built once"""
    global _W0, _SCALE
    if _W0 is not None:
        return _W0
    d = np.load(MODEL)
    pi = [d["pi%d" % i].astype(float) for i in range(6)]
    if pi[0].shape[0] < OLD_IN:
        pi[0] = np.concatenate([pi[0], np.zeros((OLD_IN - pi[0].shape[0], OLD_H))], 0)
    rng = np.random.default_rng(12345)                  # fixed: the new units' starting weights never change
    W1 = np.zeros((IN, H))
    W1[:OLD_IN, :OLD_H] = pi[0]
    W1[:, OLD_H:] = rng.normal(0, 0.5 / np.sqrt(IN), (IN, H - OLD_H))
    b1 = np.zeros(H)
    b1[:OLD_H] = pi[1]
    W2 = np.zeros((H, H))
    W2[:OLD_H, :OLD_H] = pi[2]
    W2[:, OLD_H:] = rng.normal(0, 0.5 / np.sqrt(H), (H, H - OLD_H))
    b2 = np.zeros(H)
    b2[:OLD_H] = pi[3]
    W3 = np.zeros((H + 1, OUT))                          # last row = bias
    W3[:OLD_H, :5] = pi[4]
    W3[H, :5] = pi[5]
    parts = {"W1_old": W1[:, :OLD_H], "W1_new": W1[:, OLD_H:], "b1": b1, "W2_old": W2[:, :OLD_H],
             "W2_new": W2[:, OLD_H:], "b2": b2}
    for o in range(OUT):
        parts["W3_%d" % o] = W3[:, o]
    parts["setup"] = np.zeros(len(SETUP_NAMES))
    _W0 = np.concatenate([parts[nm].ravel() for nm, _, _, _ in _LAYOUT])
    _SCALE = groups()[1]
    return _W0


def trainable():
    """indexes of the weights training moves (scale > 0)"""
    return np.flatnonzero(groups()[1] > 0)


TRAIN_SIZE = int((groups()[1] > 0).sum())


def heads():
    """indexes of the 17 output heads' weights (what runs 2-3 trained: aim3 and earlier vectors)"""
    ids = groups()[0]
    return np.flatnonzero(np.isin(ids, [i for i, g in enumerate(GROUP_NAMES) if g.startswith("out: ")]))


N_SETUP = len(SETUP_NAMES)
FIXED_V2_SIZE = TRAIN_SIZE - N_SETUP          # runs 4-6 (aim4..aim6): the v2 setup fully on, not learned


def full(x):
    """trained weights -> the full x (SIZE). Accepts: the full x, this layout's vector (TRAIN_SIZE: heads,
    new layer-1 units and the 5 setup dials), a runs 4-6 vector (no dials: read as all dials fully on, the
    way they were trained), or a heads-only vector (aim3 and earlier: dials off)"""
    x = np.asarray(x, float)
    if len(x) == SIZE:
        return x
    out = np.zeros(SIZE)
    tr = trainable()
    if len(x) == TRAIN_SIZE:
        out[tr] = x
    elif len(x) == FIXED_V2_SIZE:
        out[tr[:-N_SETUP]] = x
        out[tr[-N_SETUP:]] = 1.0 / 0.3            # setup scale 0.3: dial value 1
    elif len(x) == len(heads()):
        out[heads()] = x
    else:
        raise ValueError("a network vector of %d weights fits none of the layouts" % len(x))
    return out


def is_v2(x):
    """kit networks from run 4 on (this layout or the fixed-v2 one)"""
    return len(x) in (TRAIN_SIZE, FIXED_V2_SIZE)


# ---- v2 (user 2026-10-25): no acceleration prediction, half the unseen-travel guess, a two-zone deadzone,
# and the network can't see its own past moves (those inputs get fixed typical values) ----
V2 = dict(use_acc=False, extrap_base=0.5, dz_in=0.15, dz_out=0.6, dz_click=0.6, dz_stable=3, dz_speed=0.2,
          logk_min=np.log(0.05))          # scoped-in fields of view down to ~6 degrees
# its own past moves: last move x/y/size, wobble, unshown moves x/y, nudge, leftover x/y,
# and (new inputs) last move in counts, sub-count move
BLIND = [18, 19, 20, 26, 42, 43, 47, 55, 56, OLD_IN + 17, OLD_IN + 18]
# (input 45, "frames since my last click", is NOT blinded from run 5 on: it moves nothing back into the aim,
# and it is what tells the network recoil is coming)
_BLIND_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "blind_values.npy")


def blind_values():
    """the typical value of every input in the current code's own games (fixed, made once, saved)"""
    if os.path.exists(_BLIND_FILE):
        return np.load(_BLIND_FILE)
    raise FileNotFoundError(_BLIND_FILE + " is missing (ChromaScope: it ships with aimkit/)")
    import aimrun                       # here: aimrun imports this module
    _, rec = aimrun.play(64000, [NetRow(np.zeros(len(heads())))], n=64, frames=400, record=(0,))
    v = np.concatenate([f["z"] for f in rec[0]]).mean(0).astype(np.float32)
    np.save(_BLIND_FILE, v)
    return v


def set_setup(brain, sl, dials):
    """the 5 setup dials (0..1; 0 = the current code) -> the brain, per game"""
    b, acc, cut, dz, fov = (float(v) for v in dials)
    brain.acc_scale[sl] = 1.0 - acc
    brain.extrap_base[sl] = 1.0 - (1.0 - V2["extrap_base"]) * cut
    brain.dz_in[sl] = V2["dz_in"] * dz
    brain.dz_out[sl] = V2["dz_out"] * dz
    for k in ("dz_click", "dz_stable", "dz_speed"):
        getattr(brain, k)[sl] = V2[k]
    brain.logk_min[sl] = np.log(0.4) * (1.0 - fov) + V2["logk_min"] * fov


def unpack(x):
    """x (SIZE or TRAIN_SIZE,) -> the network's weights"""
    x = full(x)
    w = base() + _SCALE * x
    i, out = 0, {}
    for nm, shape, _, _ in _LAYOUT:
        k = int(np.prod(shape))
        out[nm] = w[i:i + k].reshape(shape)
        i += k
    W1 = np.concatenate([out["W1_old"], out["W1_new"]], 1)
    W2 = np.concatenate([out["W2_old"], out["W2_new"]], 1)
    W3 = np.stack([out["W3_%d" % o] for o in range(OUT)], 1)
    setup = np.clip(out["setup"], 0.0, 1.0)
    f = lambda a: np.ascontiguousarray(a, np.float32)  # noqa: E731
    # split so the old units are computed with exactly the current model's operations (bit for bit)
    return dict(W1o=f(W1[:OLD_IN, :OLD_H]), W1e=f(W1[OLD_IN:, :OLD_H]), W1n=f(W1[:, OLD_H:]), b1=f(out["b1"]),
                W2oo=f(W2[:OLD_H, :OLD_H]), W2no=f(W2[OLD_H:, :OLD_H]), W2n=f(W2[:, OLD_H:]), b2=f(out["b2"]),
                W3oo=f(W3[:OLD_H, :5]), W3no=f(W3[OLD_H:H, :5]), W3k=f(W3[:H, 5:]), b3=f(W3[H]), setup=setup)


FULL_FORMAT = "kit-aimnet-full-1"
_NET_KEYS = ("W1o", "W1e", "W1n", "b1", "W2oo", "W2no", "W2n", "b2", "W3oo", "W3no", "W3k", "b3")


def export_full(x, path):
    """every weight of a network in one file (both hidden layers in full: the copied model.npz units and the
    new ones, the 17 heads and the 5 setup dials), numbers only: it loads without model.npz"""
    net = unpack(x)
    tmp = path + ".tmp.npz"
    np.savez(tmp, format=np.array(FULL_FORMAT), sizes=np.array([OLD_IN, IN, OLD_H, H, OUT]),
             **{k: net[k] for k in _NET_KEYS}, setup=net["setup"])
    os.replace(tmp, path)


def load_file(path):
    """a network file -> the net dict NetRow uses: a full export (self-contained) or a trained-weights vector
    file ("p", rebuilt on top of model.npz)"""
    with np.load(path, allow_pickle=False) as d:
        if "format" in d.files:
            if str(d["format"]) != FULL_FORMAT or list(d["sizes"]) != [OLD_IN, IN, OLD_H, H, OUT]:
                raise ValueError("%s is a different network layout (%s %s)" % (path, d["format"], list(d["sizes"])))
            net = {k: np.ascontiguousarray(d[k], np.float32) for k in _NET_KEYS}
            net["setup"] = np.clip(np.asarray(d["setup"], float), 0.0, 1.0)
            return net
        return unpack(np.asarray(d["p"], float))


def forward(net, z):
    """z (n, IN) float32 -> (outputs (n, OUT), hidden 1, hidden 2)"""
    z = np.asarray(z, np.float32)
    f, e = z[:, :OLD_IN], z[:, OLD_IN:]
    h1o = np.tanh((f @ net["W1o"] + e @ net["W1e"]) + net["b1"][:OLD_H])
    h1n = np.tanh(z @ net["W1n"] + net["b1"][OLD_H:])
    h1 = np.concatenate([h1o, h1n], 1)
    h2o = np.tanh((h1o @ net["W2oo"] + h1n @ net["W2no"]) + net["b2"][:OLD_H])
    h2n = np.tanh(h1 @ net["W2n"] + net["b2"][OLD_H:])
    h2 = np.concatenate([h2o, h2n], 1)
    raw = (h2o @ net["W3oo"] + h2n @ net["W3no"]) + net["b3"][:5]
    knobs = h2 @ net["W3k"] + net["b3"][5:]
    return np.concatenate([raw, knobs], 1), h1, h2


def _on(t, state):
    """sticky switch: on above +0.5, off below -0.5, else keeps its state"""
    return np.where(t > 0.5, True, np.where(t < -0.5, False, state))


def apply_knobs(brain, sl, o):
    """the 10 new outputs -> the brain's knobs, inside their hard limits"""
    t = np.tanh(o[:, 5:])
    e = brain.est
    e.k_gain[sl] = np.exp(np.log(4.0) * t[:, 0])
    e.k_vel[sl] = np.exp(np.log(3.0) * t[:, 1])
    brain.k_extrap[sl] = np.minimum(1.0, np.exp(3.0 * t[:, 2]))
    # gate and cap fade in smoothly from "off" (a hair over 0 is ~200 sizes: no effect yet)
    brain.k_gate[sl] = np.where(t[:, 3] > 0, 2 + 200 * (1 - t[:, 3]) ** 2, np.inf)
    brain.k_cap[sl] = np.where(t[:, 4] > 0, 1.5 + 200 * (1 - t[:, 4]) ** 2, np.inf)
    e.k_fast[sl] = t[:, 5] > 0.5
    brain.fix[sl] = _on(t[:, 6:9], brain.fix[sl])
    brain.k_y[sl] = np.exp(np.log(2.0) * t[:, 9])
    brain.k_dead[sl] = 0.4 * np.maximum(t[:, 10], 0)
    brain.k_smooth[sl] = 1 - 0.7 * np.maximum(t[:, 11], 0)


def unmap(brain, sl, counts, hand):
    """what to send so the view does `counts`, given the axis fix in use; and what that does to the view"""
    fx, fy, sw = brain.fix[sl, 0], brain.fix[sl, 1], brain.fix[sl, 2]
    ky = brain.k_y[sl]
    s = counts.copy()
    s[:, 1] = s[:, 1] / ky
    s[:, 1] = np.where(fy, -s[:, 1], s[:, 1])
    s[:, 0] = np.where(fx, -s[:, 0], s[:, 0])
    s = np.where(sw[:, None], s[:, ::-1], s)
    h = np.where(sw[:, None], hand[:, ::-1], hand)          # the player's own moves, as the fix believes they act
    h = h.copy()
    h[:, 0] = np.where(fx, -h[:, 0], h[:, 0])
    h[:, 1] = np.where(fy, -h[:, 1], h[:, 1]) * ky
    return s, counts + h


class NetRow:
    """the network driving the aimer (rides on the code: brain2 at neutral knobs + the old five outputs).
    Its 5 setup dials (net["setup"], 0 = the current code) set the v2 setup per game."""

    def __init__(self, x):
        """x: a trained-weights vector, a net dict (unpack / load_file), or a network file path"""
        if isinstance(x, str):
            x = load_file(x)
        self.net = x if isinstance(x, dict) else unpack(np.asarray(x, float))
        self.setup = self.net["setup"]
        self.last = None
        self.prev = None
        self.blind = blind_values() if self.setup[0] > 0 else None

    def before(self, brain, sl):
        set_setup(brain, sl, self.setup)

    def act(self, brain, feat, sl):
        z = np.concatenate([feat, brain.extra[sl]], 1)
        b = self.setup[0]
        if b > 0:                     # blend its own past moves toward fixed typical values (1 = fully blind)
            z[:, BLIND] = (1 - b) * z[:, BLIND] + b * self.blind[BLIND]
        o, h1, h2 = forward(self.net, z)
        apply_knobs(brain, sl, o)
        raw = o[:, :5].copy()
        # smooth snap/click (a new target is never smoothed); a = 1 is the current code exactly
        a = brain.k_smooth[sl][:, None]
        if self.prev is not None and (a < 1).any():
            fresh = (brain.est.since_jump[sl] < 2)[:, None]
            raw[:, 3:5] = np.where(fresh, raw[:, 3:5], a * raw[:, 3:5] + (1 - a) * self.prev)
        self.prev = raw[:, 3:5].copy()
        self.last = (z, h1, h2, o)
        return raw

    def after(self, brain, sl, counts, hand):
        return unmap(brain, sl, counts, hand)


class Knobs:
    """fixed knobs on top of the current code (for the ceiling and the hybrid rules): fn(brain, sl, feat)
    sets them; raw comes from the current model (or classic)"""

    def __init__(self, policy, fn=None, raw_add=None):
        self.policy, self.fn, self.raw_add = policy, fn, raw_add

    def act(self, brain, feat, sl):
        raw = self.policy.mean(feat) if self.policy is not None else np.zeros((len(feat), 5))
        if self.fn:
            self.fn(brain, sl, feat)
        if self.raw_add is not None:
            raw = raw + self.raw_add
        return raw

    def after(self, brain, sl, counts, hand):
        return unmap(brain, sl, counts, hand)
