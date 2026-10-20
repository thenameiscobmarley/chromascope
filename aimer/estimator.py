"""Live estimates of mouse gain, field of view, target velocity and frame delay.

Runs for N aimers at once. Screen positions are first turned into angles
with the estimated field of view (perspective makes a mouse count move
things further near the screen edge than at the center; in angles it is the
same everywhere). Then every frame it compares how the target moved with
the mouse counts that were sent, and fits

    change in target angle = k * v  -  G * (counts sent d frames earlier)

where v is the target's own velocity, G is the mouse gain (the unknown
sensitivity) and d is the delay. The camera part (G * counts) is the "it
moved because I moved" effect; subtracting it is what stops the aimer from
chasing its own movement.

One fit is kept per candidate delay. The one whose predictions have been
most accurate is the one used. Positions called "a" below are in angle
units; "u" means raw screen units.
"""
import numpy as np

from .config import HIST, LOG_G0, N_DELAYS

LOGK_MIN, LOGK_MAX = np.log(0.4), np.log(2.2)    # tan(half FOV): about 44 to 131 degrees


class Estimator:
    def __init__(self, n):
        self.n = n
        D = N_DELAYS
        self.logG = np.full((n, D), LOG_G0)
        self.conf = np.zeros((n, D))
        self.v = np.zeros((n, D, 2))
        self.err = np.full((n, D), 0.05)
        self.pend = np.zeros((n, D))           # a surprising gain seen last move, waiting for confirmation
        self.pend_on = np.zeros((n, D), bool)
        self.switched = np.zeros(n, bool)      # the sensitivity change check fired this frame
        self.expect_new = np.zeros(n, int)     # frames left in which a new target may appear (after a click)
        self.seen = np.zeros(n, int)           # fresh sightings of the current target so far
        self.cur = np.zeros(n, int)            # the delay currently believed (sticky, see _pick)
        self.prev_dobs = np.zeros((n, 2))      # last frame's movement and mouse move (believed delay),
        self.prev_x = np.zeros((n, 2))         # for the speed-free gain check
        self.prev_ok = np.zeros(n, bool)
        self.gain_seen = np.zeros(n, bool)     # a clean speed-free gain measurement happened this frame
        self.acc = np.zeros((n, 2))            # target acceleration (angle units per frame^2), believed delay
        self.vhist = np.zeros((n, 4, 2))       # last 4 measured target speeds (own movement removed)
        self.since_switch = np.full(n, 99)     # frames since the sensitivity-change check fired
        self.since_jump = np.full(n, 99)       # frames since a new target appeared
        self.logk = np.zeros(n)                # log tan(half FOV); 0 means a 90 degree FOV
        self.A = np.zeros((n, HIST, 2))        # A[:, m] = counts applied m+1 frames ago
        self.o_last = np.zeros((n, 2))          # last fresh, visible observation (screen units)
        self.since = np.full(n, HIST, int)      # frames since o_last was observed
        self.have = np.zeros(n, bool)
        self._r = np.arange(n)

    def reset(self, mask):
        self.logG[mask] = LOG_G0
        self.conf[mask] = 0
        self.v[mask] = 0
        self.err[mask] = 0.05
        self.pend[mask] = 0
        self.pend_on[mask] = False
        self.switched[mask] = False
        self.expect_new[mask] = 0
        self.seen[mask] = 0
        self.cur[mask] = 0
        self.prev_ok[mask] = False
        self.gain_seen[mask] = False
        self.acc[mask] = 0
        self.vhist[mask] = 0
        self.since_switch[mask] = 99
        self.since_jump[mask] = 99
        self.logk[mask] = 0
        self.A[mask] = 0
        self.o_last[mask] = 0
        self.since[mask] = HIST
        self.have[mask] = False

    # ---- perspective -------------------------------------------------------
    def to_a(self, u, k=None):
        k = (np.exp(self.logk) if k is None else k)[:, None]
        return np.arctan(k * u) / k

    def to_u(self, a):
        k = np.exp(self.logk)[:, None]
        return np.tan(np.clip(k * a, -1.5, 1.5)) / k

    def a_scale(self, u):
        """How much smaller a size at screen position u is in angle units."""
        k = np.exp(self.logk)
        return 1.0 / (1.0 + (k * np.linalg.norm(u, axis=1)) ** 2)

    def csum(self):
        """C[:, m] = sum of the last m applied mouse moves."""
        C = np.zeros((self.n, HIST + 1, 2))
        np.cumsum(self.A, axis=1, out=C[:, 1:])
        return C

    def update(self, obs, visible, fresh):
        n, D, r = self.n, N_DELAYS, self._r
        self.since += 1
        k = self.since
        valid = fresh & visible & self.have & (k <= HIST - D)

        C = self.csum()
        kc = np.minimum(k, HIST - D)
        d = np.arange(D)
        hi = d[None, :] + kc[:, None]
        lo = np.broadcast_to(d[None, :], (n, D))
        x = C[r[:, None], hi] - C[r[:, None], lo]           # (n, D, 2) counts per candidate delay
        kf = kc[:, None, None].astype(float)
        da = self.to_a(obs) - self.to_a(self.o_last)
        dobs = da[:, None, :]

        G = np.exp(self.logG)
        pred = kf * self.v - G[..., None] * x
        res = np.linalg.norm(dobs - pred, axis=2)            # (n, D)

        # The camera's part of the movement, per candidate delay: should equal G * counts.
        c = kf * self.v - dobs
        cx = (c * x).sum(2)
        xx = (x * x).sum(2)
        cn = np.linalg.norm(c, axis=2)
        aligned = (xx > 0) & (cx > 0.7 * cn * np.sqrt(xx))
        Gobs = cx / np.maximum(xx, 1e-30)

        # A big unexplained jump that is NOT along our own mouse move means a new target.
        # Along our move, it is our own movement with a wrong gain: learn from it instead.
        b = self.cur
        own = G[r, b] * np.linalg.norm(x[r, b], axis=1)
        jump = valid & (res[r, b] > 0.25) & (self.conf[r, b] > 1) & ~aligned[r, b]
        # Right after our own click the target may have been hit and replaced: any clear
        # unexplained jump then is a new target too, even a small one.
        jump |= valid & (self.expect_new > 0) & (res[r, b] > 0.02)
        self.expect_new = np.maximum(self.expect_new - 1, 0)
        upd = valid & ~jump
        U = upd[:, None]

        # Field of view: nudge it toward whichever value explains this move better.
        strong = upd & (own > 0.02) & (self.conf[r, b] > 1)
        if strong.any():
            kk = np.exp(self.logk)
            expect = kc[:, None] * self.v[r, b] - G[r, b][:, None] * x[r, b]

            def miss(scale):
                kq = kk * scale
                return ((self.to_a(obs, kq) - self.to_a(self.o_last, kq) - expect) ** 2).sum(1)

            mp, mm = miss(1.05), miss(1 / 1.05)
            step = (mp - mm) / (mp + mm + 1e-12)
            self.logk = np.clip(np.where(strong, self.logk - 0.03 * step, self.logk), LOGK_MIN, LOGK_MAX)

        conf_before = self.conf.copy()
        self.err = np.where(U, 0.85 * self.err + 0.15 * res, self.err)
        # Predictions keep failing: lower confidence so the gain can relearn quickly.
        self.conf = np.where(U & (res > 0.1), self.conf * 0.8, self.conf)

        # Gain: the part of the movement caused by the camera, projected onto the mouse move.
        # Only learn it when our own camera move clearly dominates the target's own motion;
        # otherwise a wrong gain and a wrong target speed can feed each other and run away.
        cam = G * np.sqrt(xx)
        vmove = kc[:, None] * np.linalg.norm(self.v, axis=2)
        dominant = cam > 3 * vmove + 0.005
        unsure = self.conf < 0.5
        ok = U & aligned & (cn > 0.004) & (dominant | unsure)
        lg = np.log(np.where(ok, Gobs, 1.0))

        # Every move is a move check, done without the target's speed: between two frames
        # the target's speed barely changes, so the change in how the view moved, divided by
        # the change in our mouse move, is the gain alone. If that is clearly different from
        # what we believe (over 1.6x) twice in a row, by about the same amount, the
        # sensitivity has changed: switch to the new gain at once.
        xb, db = x[r, b], dobs[r, 0]
        dx, dd = xb - self.prev_x, db - self.prev_dobs
        dxx = (dx * dx).sum(1)
        gd_raw = -(dd * dx).sum(1) / np.maximum(dxx, 1e-30)
        step_cam = np.exp(self.logG[r, b]) * np.sqrt(dxx)
        dn = np.linalg.norm(dd, axis=1)
        clear = (upd & self.prev_ok & (kc == 1) & (dxx > 0) & (gd_raw > 0) & (step_cam > 0.01)
                 & (-(dd * dx).sum(1) > 0.8 * dn * np.sqrt(dxx)))
        lgd = np.log(np.where(clear, gd_raw, 1.0))
        self.prev_dobs, self.prev_x = db.copy(), xb.copy()
        self.prev_ok = upd & (kc == 1)
        big_b = clear & (conf_before[r, b] >= 2.0) & (np.abs(lgd - self.logG[r, b]) > np.log(1.6))
        big = np.zeros_like(ok)
        big[r, b] = big_b
        lg = lg.copy()
        lg[r, b] = np.where(big_b, lgd, lg[r, b])
        confirm = big & self.pend_on & (np.abs(lg - self.pend) < np.log(1.3))
        self.pend = np.where(big, lg, self.pend)
        seen_b = np.zeros_like(ok)
        seen_b[r, b] = clear
        self.pend_on = np.where(seen_b, big & ~confirm, self.pend_on)
        normal = ok & ~big

        strength = np.clip(cn / 0.03, 0, 1)
        alpha = np.where(unsure, 1.0, 0.25) * strength
        self.logG = np.where(normal, self.logG + alpha * (lg - self.logG), self.logG)
        self.logG = np.where(confirm, lg, self.logG)
        self.logG = np.clip(self.logG, np.log(1e-8), 0.0)
        self.conf = np.where(normal, np.minimum(self.conf + strength, 20), self.conf)
        self.conf = np.where(confirm, np.maximum(self.conf, 2.0), self.conf)
        # Smaller gain errors: the same speed-free measurement corrects them gradually. This is
        # what keeps the gain right while smoothly following a target, where no single move
        # dominates (see the test nudge in brain.py).
        fine = clear & ~big_b & (conf_before[r, b] >= 0.5)
        rate = 0.3 * np.clip(step_cam / 0.03, 0, 1)
        self.logG[r, b] = np.where(fine, self.logG[r, b] + rate * (lgd - self.logG[r, b]), self.logG[r, b])
        self.gain_seen = clear
        self.switched = confirm[r, b]
        self._pick()

        # Target velocity, with our own camera movement removed. The bigger our own move,
        # the less this frame is trusted, so an error in the gain can't leak into the speed.
        G = np.exp(self.logG)
        vobs = (dobs + G[..., None] * x) / kf
        cam = G * np.sqrt(xx)
        trust = 1.0 / (1.0 + (cam / (np.linalg.norm(dobs, axis=2) + 0.005)) ** 2)
        beta = (0.5 * np.maximum(trust, 0.1))[..., None]
        b = self.cur
        v_before = self.v[r, b].copy()
        self.v = np.where(U[..., None], self.v + beta * (vobs - self.v), self.v)
        self.v[jump] = 0

        # Acceleration of the target (for seeing through the delay), and a short speed history.
        dv = (self.v[r, b] - v_before) / kc[:, None]
        self.acc = np.where(upd[:, None], 0.6 * self.acc + 0.4 * dv, self.acc)
        self.acc[jump] = 0
        self.vhist = np.where(upd[:, None, None], np.concatenate([vobs[r, b][:, None], self.vhist[:, :-1]], 1),
                              self.vhist)
        self.vhist[jump] = 0
        self.since_switch = np.where(self.switched, 0, np.minimum(self.since_switch + 1, 99))
        self.since_jump = np.where(jump, 0, np.minimum(self.since_jump + 1, 99))

        new = fresh & visible
        self.seen = np.where(jump, 1, np.where(new, self.seen + 1, self.seen))
        self.o_last = np.where(new[:, None], obs, self.o_last)
        self.since = np.where(new, 0, self.since)
        self.have |= new

    def _pick(self):
        """Switch the believed delay only when another one fits clearly better (no flip-flopping)."""
        r = self._r
        cand = np.argmin(self.err, axis=1)
        better = self.err[r, cand] < 0.7 * self.err[r, self.cur] - 1e-5
        self.cur = np.where(better, cand, self.cur)

    def delay_margin(self):
        """How clearly the believed delay fits better than the runner-up (0 = a tie)."""
        e = np.sort(self.err, axis=1)
        return np.clip(1 - e[:, 0] / np.maximum(e[:, 1], 1e-9), 0, 1)

    def best(self):
        """(delay, gain, confidence, velocity, error) of the believed delay."""
        r, b = self._r, self.cur
        return b, np.exp(self.logG[r, b]), self.conf[r, b], self.v[r, b], self.err[r, b]

    def push(self, counts):
        """Record the mouse counts actually applied this frame."""
        self.A[:, 1:] = self.A[:, :-1]
        self.A[:, 0] = counts
