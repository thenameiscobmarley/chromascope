"""Loads a trained model.npz and runs it. Forward pass only: training lives in training/."""
import numpy as np

from .config import OBS_DIM, OBS_DIM_V1


def widen_inputs(W, old_dim):
    """An older model's first layer, extended with zero weights for inputs added since. The
    model then behaves exactly as before and can learn to use the new inputs. New inputs are
    always added at the end, so any older model (27, 49, ... inputs) can be widened."""
    if old_dim == OBS_DIM:
        return W
    if not OBS_DIM_V1 <= old_dim < OBS_DIM:
        raise ValueError(f"model has {old_dim} inputs; this code uses {OBS_DIM}. Retrain it.")
    return np.concatenate([W, np.zeros((OBS_DIM - old_dim, W.shape[1]), W.dtype)], 0)


class MLP:
    """tanh MLP: params = [W0, b0, W1, b1, ...]."""

    def __init__(self, params):
        self.params = params

    @property
    def layers(self):
        return len(self.params) // 2

    def forward(self, x):
        h = x
        for i in range(self.layers):
            h = h @ self.params[2 * i] + self.params[2 * i + 1]
            if i < self.layers - 1:
                h = np.tanh(h)
        return h


class Policy:
    def __init__(self, pi_params, meta=None):
        self.pi = MLP(pi_params)
        self.meta = meta or {}

    def mean(self, x):
        return self.pi.forward(np.asarray(x, np.float32))

    @classmethod
    def load(cls, path):
        d = np.load(path)
        n = sum(1 for k in d.files if k.startswith("pi"))
        params = [d[f"pi{i}"].astype(np.float32) for i in range(n)]
        params[0] = widen_inputs(params[0], int(d["obs_dim"]))
        meta = {k[5:]: d[k] for k in d.files if k.startswith("meta_")}
        return cls(params, meta)
