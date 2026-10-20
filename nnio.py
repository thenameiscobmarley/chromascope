"""NNMM model files: .npz weight files, turned into an ONNX network in memory and run by OpenCV DNN.

A model is a small fully connected network: 6 inputs (target offset x/y, target speed x/y, current speed x/y, in screen
heights) → hidden layers → 2 outputs (speed x/y, 1.0 = top speed); see nnmm.INPUTS / OUTPUTS. load_npz() understands
the usual ways such weights get saved (see its docstring). The ONNX is written by hand (a few protobuf fields), so no
onnx/protobuf package is needed. Plain .onnx files with the same 6 → 2 shape work too.
"""
from __future__ import annotations

import re
from pathlib import Path

import cv2
import numpy as np


# ── minimal protobuf writer ─────────────────────────────────────────────────
def _varint(n):
    out = bytearray()
    while True:
        b = n & 0x7F; n >>= 7
        out.append(b | (0x80 if n else 0))
        if not n: return bytes(out)


def _int(field, n): return _varint(field << 3) + _varint(n)
def _bytes(field, b): return _varint(field << 3 | 2) + _varint(len(b)) + b
def _str(field, s): return _bytes(field, s.encode())


def _tensor(name, a):
    a = np.ascontiguousarray(a, np.float32)
    return b"".join(_int(1, d) for d in a.shape) + _int(2, 1) + _str(8, name) + _bytes(9, a.tobytes())


def _value_info(name, dims):
    shape = b"".join(_bytes(1, _int(1, d)) for d in dims)
    return _str(1, name) + _bytes(2, _bytes(1, _int(1, 1) + _bytes(2, shape)))


def _node(op, ins, out):
    return b"".join(_str(1, i) for i in ins) + _str(2, out) + _str(3, out) + _str(4, op)


ACTS = {"tanh": "Tanh", "relu": "Relu", "sigmoid": "Sigmoid"}


def to_onnx(W, b, act="tanh", out_act="linear"):
    """Fully connected net → ONNX bytes. Input "state" [1, n_in] → output "velocity" [1, n_out]."""
    nodes, inits, x = [], [], "state"
    for i, (w, bias) in enumerate(zip(W, b)):
        last = i == len(W) - 1
        f = out_act if last else act
        inits += [_tensor(f"W{i}", w), _tensor(f"b{i}", bias)]
        out = "velocity" if last and f == "linear" else f"z{i}"
        nodes += [_node("MatMul", [x, f"W{i}"], f"mm{i}"), _node("Add", [f"mm{i}", f"b{i}"], out)]
        if f != "linear": x = "velocity" if last else f"h{i}"; nodes.append(_node(ACTS[f], [out], x))
    graph = (b"".join(_bytes(1, n) for n in nodes) + _str(2, "nnmm") + b"".join(_bytes(5, t) for t in inits)
             + _bytes(11, _value_info("state", [1, W[0].shape[0]])) + _bytes(12, _value_info("velocity", [1, W[-1].shape[1]])))
    return _int(1, 8) + _str(2, "ChromaScope NNMM") + _bytes(7, graph) + _bytes(8, _int(2, 13))


# ── files ────────────────────────────────────────────────────────────────────
def save(folder: Path, name, net, act="tanh"):
    folder.mkdir(parents=True, exist_ok=True)
    np.savez(folder / f"{name}.npz", sizes=np.array(net.sizes), activation=np.array(act),
             **{f"W{i}": w for i, w in enumerate(net.W)}, **{f"b{i}": v for i, v in enumerate(net.b)})


def _natural(k): return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", k)]


def load_npz(path, n_in=6):
    """(W, b, hidden activation, output activation) from a .npz of layer weights. Understands:
      * W0, b0, W1, b1 … / W1, b1 … / w_0, b_0 … / weights0, biases0 … (any of these names, any start number)
      * PyTorch-style names: fc1.weight, fc1.bias, net.0.weight … (stored out × in, transposed here)
      * np.savez(w1, b1, w2, b2 …) without names: arr_0, arr_1 … alternating weights and biases
      * object arrays "weights" / "biases" holding one array per layer
    Matrices may be stored in × out or out × in; the layer sizes decide. Optional string keys "activation"
    (tanh / relu / sigmoid, default tanh) and "output_activation" (linear / tanh / sigmoid, default linear)."""
    z = np.load(path, allow_pickle=False)
    try: keys = list(z.keys())
    except ValueError: keys = []
    def text(k, default):
        try: return str(z[k]).strip().lower() if k in keys else default
        except ValueError: return default
    act, out_act = text("activation", "tanh"), text("output_activation", "linear")
    for a in (act, out_act):
        if a not in ACTS and a != "linear": raise ValueError(f"unknown activation '{a}' (use tanh, relu or sigmoid)")
    mats, vecs, torch = [], [], False
    wk = [k for k in keys if re.fullmatch(r"(w|weight|weights|kernel|layer)_?\d+", k, re.I)]
    bk = [k for k in keys if re.fullmatch(r"(b|bias|biases)_?\d+", k, re.I)]
    tk = [k for k in keys if k.endswith(".weight")]
    if wk:
        mats = [z[k] for k in sorted(wk, key=_natural)]
        bias = {int(re.search(r"\d+", k).group()): z[k] for k in bk}
        first = min(int(re.search(r"\d+", k).group()) for k in wk)
        vecs = [bias.get(first + i) for i in range(len(mats))]
    elif tk:
        torch = True
        tk = sorted(tk, key=_natural)
        mats = [z[k] for k in tk]; vecs = [z[k[:-7] + ".bias"] if k[:-7] + ".bias" in keys else None for k in tk]
    elif keys and all(re.fullmatch(r"arr_\d+", k) for k in keys):
        arrs = [z[k] for k in sorted(keys, key=_natural)]
        for a in arrs:
            if a.ndim == 2: mats.append(a); vecs.append(None)
            elif a.ndim == 1 and mats: vecs[-1] = a
    elif "weights" in keys:
        z = np.load(path, allow_pickle=True)                     # per-layer lists are stored as object arrays
        mats = list(z["weights"]); vecs = list(z["biases"]) if "biases" in keys else [None] * len(mats)
    if not mats: raise ValueError(f"no layer weights found (keys: {', '.join(keys[:8]) or 'none'})")
    W, b, n, flip = [], [], n_in, torch                          # flip: stored out × in; square layers follow the others
    for i, (m, v) in enumerate(zip(mats, vecs)):
        m = np.asarray(m, np.float32)
        if m.ndim != 2: raise ValueError(f"layer {i + 1} weights are {m.ndim}-D, expected a matrix")
        if m.shape[0] != m.shape[1]: flip = m.shape[0] != n and m.shape[1] == n
        if flip and m.shape[1] == n: m = m.T
        if m.shape[0] != n: raise ValueError(f"layer {i + 1} is {m.shape[0]}×{m.shape[1]} but gets {n} inputs (the model must take 6 inputs)")
        W.append(m); n = m.shape[1]
        b.append(np.zeros(n, np.float32) if v is None else np.asarray(v, np.float32).reshape(-1))
        if b[-1].size != n: raise ValueError(f"layer {i + 1} bias has {b[-1].size} values, expected {n}")
    if n != 2: raise ValueError(f"the model has {n} outputs, expected 2 (speed x/y)")
    return W, b, act, out_act


def is_aimer(path):
    """True for an aim-trainer model (aimer/ runtime: policy pi0, pi1 … plus obs_dim), not a plain 6 → 2 net."""
    if Path(path).suffix != ".npz": return False
    with np.load(path, allow_pickle=False) as z: return "pi0" in z.files and "obs_dim" in z.files


class CvPolicy:
    """An aim-trainer policy network run by OpenCV DNN (same maths as aimer.policy.Policy.mean)."""

    def __init__(self, path):
        from aimer.policy import Policy                         # widens older 27 / 49-input models
        pol = Policy.load(path)
        P = pol.pi.params
        self.name, self.meta, self.inputs = Path(path).stem, pol.meta, P[0].shape[0]
        self.layers = " → ".join([str(P[0].shape[0])] + [str(w.shape[1]) for w in P[0::2]]) + " (aimer)"
        self.net = cv2.dnn.readNetFromONNX(np.frombuffer(to_onnx(P[0::2], P[1::2]), np.uint8))

    def mean(self, x):
        self.net.setInput(np.asarray(x, np.float32).reshape(1, -1))
        return np.asarray(self.net.forward(), np.float32).reshape(1, -1)


class CvModel:
    """A model run by OpenCV DNN: 6 floats in, 2 floats out. From a .npz (converted to ONNX in memory) or an .onnx file."""

    def __init__(self, path: Path):
        path = Path(path)
        self.name, self.layers = path.stem, ""
        if path.suffix == ".npz":
            W, b, act, out_act = load_npz(path)
            self.layers = " → ".join([str(W[0].shape[0])] + [str(w.shape[1]) for w in W]) + f" ({act})"
            self.net = cv2.dnn.readNetFromONNX(np.frombuffer(to_onnx(W, b, act, out_act), np.uint8))
        else:
            self.net = cv2.dnn.readNetFromONNX(str(path))
        out = self(np.zeros(6, np.float32))
        if out.shape != (2,): raise ValueError(f"{self.name}: expected 2 outputs, got {out.size}")

    def check(self):
        """'' if the model steers towards targets (still ones, from all directions), else what's wrong with it."""
        a = np.linspace(0, 2 * np.pi, 16, endpoint=False)
        good = 0
        for r in (0.05, 0.2, 0.5):
            for e in np.stack([np.cos(a), np.sin(a)], 1) * r:
                out = self(np.concatenate([e / 0.8, [0, 0, 0, 0]]))
                good += float(np.dot(out[:2], e)) > 0
        return "" if good >= 44 else f"it only moves towards the target in {good} of 48 test cases (untrained?)"

    def __call__(self, x):
        self.net.setInput(np.asarray(x, np.float32).reshape(1, -1))
        return np.asarray(self.net.forward(), np.float32).reshape(-1)
