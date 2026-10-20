"""Makes a plain example NNMM model (a .npz to choose in the NNMM tab): a small network learns to steer a cursor onto moving dots.

Synthetic training: one moving dot, one virtual cursor, and a simple teacher that aims a little ahead of the dot.
The network drives the cursor itself while it learns (so it also learns to recover from its own mistakes), and its
error against the teacher trains it with Adam on mean squared error. Run:  .venv/bin/python train.py OUT.npz [steps]
"""
from __future__ import annotations

import sys
from collections import deque
from pathlib import Path

import numpy as np

import nnio
from nnmm import INPUTS, LEAD, OUTPUTS, VMAX, clip_norm

W, H = 1.6, 1.0                 # world size, in screen heights
DT = 1 / 60
TIMEOUT = 6.0                   # a dot not reached in this long counts as a miss


# ── Network ──────────────────────────────────────────────────────────────────
class MLP:
    """Fully connected net, tanh hidden layers, linear output, trained with Adam on mean squared error."""

    def __init__(self, sizes, seed=None):
        rng = np.random.default_rng(seed)
        self.sizes = sizes
        self.W = [rng.normal(0, np.sqrt(1 / a), (a, b)) for a, b in zip(sizes, sizes[1:])]
        self.b = [np.zeros(b) for b in sizes[1:]]
        self.m = [np.zeros_like(p) for p in self.W + self.b]
        self.v = [np.zeros_like(p) for p in self.W + self.b]
        self.t = 0

    def forward(self, x):
        acts = [x]
        for i, (w, b) in enumerate(zip(self.W, self.b)):
            x = x @ w + b
            if i < len(self.W) - 1: x = np.tanh(x)
            acts.append(x)
        return x, acts

    def train(self, x, y, lr):
        out, acts = self.forward(x)
        g = 2 * (out - y) / len(x)
        gw, gb = [None] * len(self.W), [None] * len(self.W)
        for i in reversed(range(len(self.W))):
            gw[i], gb[i] = acts[i].T @ g, g.sum(0)
            if i: g = (g @ self.W[i].T) * (1 - acts[i] ** 2)
        self.t += 1
        params, grads = self.W + self.b, gw + gb
        for j, (p, d) in enumerate(zip(params, grads)):
            self.m[j] = 0.9 * self.m[j] + 0.1 * d
            self.v[j] = 0.999 * self.v[j] + 0.001 * d * d
            mh, vh = self.m[j] / (1 - 0.9 ** self.t), self.v[j] / (1 - 0.999 ** self.t)
            p -= lr * mh / (np.sqrt(vh) + 1e-8)
        return float(np.mean((out - y) ** 2))


# ── Simulation ───────────────────────────────────────────────────────────────
class Sim:
    """One moving dot, one virtual cursor. The network drives the cursor and learns from a simple teacher."""

    def __init__(self, cfg):
        self.cfg, self.rng = cfg, np.random.default_rng()
        self.rebuild()

    def rebuild(self):
        c = self.cfg
        self.net = MLP([len(INPUTS)] + [int(c["nn_hidden"])] * int(c["nn_layers"]) + [len(OUTPUTS)])
        self.buf_x, self.buf_y, self.n_buf = np.zeros((4096, len(INPUTS))), np.zeros((4096, len(OUTPUTS))), 0
        self.steps, self.loss, self.losses = 0, None, deque(maxlen=240)
        self.hits, self.misses, self.times = 0, 0, deque(maxlen=60)
        self.hit_log = deque(maxlen=200)   # wall-clock-free: sim time of each hit
        self.clock, self.last_x = 0.0, np.zeros(len(INPUTS))
        self.cur, self.cv = np.array([W / 2, H / 2]), np.zeros(2)
        self.trail = deque(maxlen=40)
        self.spawn()

    def spawn(self):
        while True:
            p = self.rng.uniform([0.08, 0.08], [W - 0.08, H - 0.08])
            if np.hypot(*(p - self.cur)) > 0.45: break
        a = self.rng.uniform(0, 2 * np.pi)
        self.dot, self.dv, self.age, self.dwell = p, np.array([np.cos(a), np.sin(a)]), 0.0, 0.0

    def radius(self): return self.cfg["nn_size"] / 100

    def state(self):
        speed = self.cfg["nn_speed"] / 100 * 0.8
        return np.clip(np.concatenate([(self.dot - self.cur) / 0.8, self.dv * speed / 0.8, self.cv / VMAX]), -3, 3)

    def teacher(self):
        speed = self.cfg["nn_speed"] / 100 * 0.8
        aim = (self.dot - self.cur) + self.dv * speed * LEAD
        return clip_norm(aim * 6 / VMAX)

    def step(self):
        c = self.cfg
        x, y = self.state(), self.teacher()
        out = self.net.forward(x[None])[0][0]
        self.last_x = x
        cmd = y if c["nn_driver"] == "teacher" else clip_norm(out)
        if c["nn_train"]:
            i = self.n_buf % len(self.buf_x); self.buf_x[i], self.buf_y[i] = x, y; self.n_buf += 1
            if self.n_buf >= 64:
                idx = self.rng.integers(0, min(self.n_buf, len(self.buf_x)), 32)
                l = self.net.train(self.buf_x[idx], self.buf_y[idx], c["nn_lr"])
                self.loss = l if self.loss is None else 0.97 * self.loss + 0.03 * l
                self.steps += 1
                if self.steps % 10 == 0: self.losses.append(self.loss)
        # move the virtual cursor
        self.cv = cmd * VMAX
        self.cur = np.clip(self.cur + self.cv * DT, 0, [W, H])
        if self.steps % 2 == 0: self.trail.append(self.cur.copy())
        # move the dot: wander and bounce
        if self.rng.random() < c["nn_turns"] / 100 * 0.05:
            a = self.rng.normal(0, 1.2); ca, sa = np.cos(a), np.sin(a)
            self.dv = np.array([ca * self.dv[0] - sa * self.dv[1], sa * self.dv[0] + ca * self.dv[1]])
        self.dot = self.dot + self.dv * (c["nn_speed"] / 100 * 0.8) * DT
        r = self.radius()
        for k, hi in ((0, W), (1, H)):
            if self.dot[k] < r or self.dot[k] > hi - r:
                self.dv[k] = -self.dv[k]; self.dot[k] = np.clip(self.dot[k], r, hi - r)
        # hit = cursor stays on the dot for 0.2 s
        self.clock += DT; self.age += DT
        self.dwell = self.dwell + DT if np.hypot(*(self.dot - self.cur)) < r else 0.0
        if self.dwell >= 0.2:
            self.hits += 1; self.times.append(self.age); self.hit_log.append(self.clock); self.spawn()
        elif self.age > TIMEOUT:
            self.misses += 1; self.times.append(TIMEOUT); self.spawn()

    def hits_per_min(self):
        recent = [t for t in self.hit_log if t > self.clock - 30]
        span = min(30.0, self.clock)
        return len(recent) / span * 60 if span > 1 else 0.0


def main():
    if len(sys.argv) < 2: sys.exit("usage: train.py OUT.npz [steps]")
    out = Path(sys.argv[1])
    steps = int(sys.argv[2]) if len(sys.argv) > 2 else 120_000
    rng = np.random.default_rng(7)
    cfg = {"nn_hidden": 16, "nn_layers": 2, "nn_lr": 0.003, "nn_train": True, "nn_driver": "net",
           "nn_speed": 35, "nn_turns": 30, "nn_size": 3}
    sim = Sim(cfg)
    for i in range(steps):
        if i % 2000 == 0:                                       # vary the dots so it sees slow, fast, straight and twisty ones
            cfg.update(nn_speed=int(rng.integers(0, 90)), nn_turns=int(rng.integers(0, 80)), nn_size=int(rng.integers(2, 6)))
        cfg["nn_lr"] = 0.003 if i < steps * 0.7 else 0.0005
        sim.step()
    cfg["nn_train"] = False
    h0, m0 = sim.hits, sim.misses
    for _ in range(20_000): sim.step()
    print(f"loss {sim.loss:.5f}  evaluation: {sim.hits - h0} hits, {sim.misses - m0} misses")
    nnio.save(out.parent, out.stem, sim.net)
    print(f"wrote {out.parent / (out.stem + '.npz')}")


if __name__ == "__main__":
    main()
