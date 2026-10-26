"""The aimer with knobs: aimer/brain.py + aimer/estimator.py, copied, with hooks the new network can turn.

With every knob at its neutral value this IS the current code (checked by kit/check_same.py: same counts,
same clicks, frame for frame). Nothing in aimer/ is changed.

Knobs (set per game; the estimator ones are applied at the next frame's update):
  k_gain    x how fast the sensitivity is learned (alpha / rate)            neutral 1, limits 0.25..4
  k_vel     x how fast the target's speed is learned (beta)                  neutral 1, limits 0.2..3
  k_fast    accept a clear sensitivity change at once (no 2nd confirmation)  neutral False
  k_extrap  share of the "speed x frames since seen" guess that is used      neutral 1, limits 0..1
  k_gate    ignore a detection this many sizes away from where it should be  neutral inf (never);
            at most GATE_RUN frames in a row
  k_cap     largest move, in (believed distance + size)                      neutral inf (no cap)
  k_dead    steady: on a calm target, moves shrink to 0 inside k_dead sizes   neutral 0 (off)

The v2 setup (set per game by aimnet.V2, all off by default = the current code):
  use_acc=False      no acceleration prediction
  extrap_base=0.5    only half of the "speed x frames unseen" guess
  dz_in / dz_out     deadzone (in sizes of the target's narrow side): inside dz_in every output is bypassed (only the target's own
                     motion is followed, no probing, no nudge); out to dz_out a careful mode (no snap, no
                     network gain, half the gap per frame, never past the deadzone edge)
  axis fix  flip x / flip y / swap / y ratio: what is sent is mapped back    neutral identity
            (done by the caller: kit/aimnet.py NetRow.after)

New measurements for the network (all code, no network output is fed back):
  monitor M  2x2: how the view really responds to each sent axis (speed-free, from frame-to-frame
             changes, in app counts before any axis fix). Identity = normal, -1 on the diagonal = inverted,
             off-diagonal = swapped
  q_est      the smallest position step the detector shows (pixel grid)
  blind      how often a move that should have shown on screen showed nothing (dead zone, truncation)
  surprise   how far this detection is from where it should be, in target sizes (glitch)
  hold_t     frames the crosshair has been on the target
"""
import numpy as np  # bugcheck: ignore[PY040]

from aimer.brain import ACC_CAP, ANTI_LATENCY, DEFAULT_R, LOST_AFTER, Brain  # bugcheck: ignore[PY040]
from aimer.config import ACT_DIM, ASPECT, HIST, LOG_G0, N_DELAYS  # bugcheck: ignore[PY040]
from aimer.estimator import Estimator  # bugcheck: ignore[PY040]

EXTRA_DIM = 20
GATE_RUN = 3            # the glitch gate ignores at most this many detections in a row


class KitEstimator(Estimator):
    def __init__(self, n):
        super().__init__(n)
        self.k_gain = np.ones(n)
        self.k_vel = np.ones(n)
        self.k_fast = np.zeros(n, bool)
        self.oracle = np.zeros(n, bool)         # ceiling only: the true gain and delay are given
        self.oracle_logG = np.zeros(n)
        self.oracle_d = np.zeros(n, int)
        self.S = np.zeros((n, HIST, 2))         # counts SENT to the app (before any axis fix)
        self.Mxx = np.zeros((n, 2, 2))
        self.Mcx = np.zeros((n, 2, 2))
        self.prev_xs = np.zeros((n, 2))
        self.q_est = np.full(n, 0.01)
        self.blind = np.zeros(n)
        self.surprise = np.zeros(n)
        self.dG10 = np.zeros((n, 10))           # recent log-gain history (for "how fast it changes")

    def reset(self, mask):
        super().reset(mask)
        self.S[mask] = 0
        self.Mxx[mask] = 0
        self.Mcx[mask] = 0
        self.prev_xs[mask] = 0
        self.q_est[mask] = 0.01
        self.blind[mask] = 0
        self.surprise[mask] = 0
        self.dG10[mask] = LOG_G0

    def monitor(self):
        """M: (n, 2, 2) view response per sent axis (identity when normal)"""
        lam = (1e-9 + 0.02 * np.trace(self.Mxx, axis1=1, axis2=2))[:, None, None] * np.eye(2)
        inv = np.linalg.inv(self.Mxx + lam)
        M = np.einsum("nij,njk->nik", self.Mcx, inv)
        known = np.trace(self.Mxx, axis1=1, axis2=2) > 1e-5
        return np.where(known[:, None, None], np.clip(M, -3, 3), np.eye(2)[None])

    def push_sent(self, sent):
        self.S[:, 1:] = self.S[:, :-1]
        self.S[:, 0] = sent

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
        x = C[r[:, None], hi] - C[r[:, None], lo]
        kf = kc[:, None, None].astype(float)
        da = self.to_a(obs) - self.to_a(self.o_last)
        dobs = da[:, None, :]

        G = np.exp(self.logG)
        pred = kf * self.v - G[..., None] * x
        res = np.linalg.norm(dobs - pred, axis=2)

        c = kf * self.v - dobs
        cx = (c * x).sum(2)
        xx = (x * x).sum(2)
        cn = np.linalg.norm(c, axis=2)
        aligned = (xx > 0) & (cx > 0.7 * cn * np.sqrt(xx))
        Gobs = cx / np.maximum(xx, 1e-30)

        b = self.cur
        own = G[r, b] * np.linalg.norm(x[r, b], axis=1)
        # --- new: how surprising this detection is (in target-size units), before anything learns from it
        self.surprise = np.where(valid, res[r, b], 0.0)
        jump = valid & (res[r, b] > 0.25) & (self.conf[r, b] > 1) & ~aligned[r, b]
        jump |= valid & (self.expect_new > 0) & (res[r, b] > 0.02)
        self.expect_new = np.maximum(self.expect_new - 1, 0)
        upd = valid & ~jump
        U = upd[:, None]

        strong = upd & (own > 0.02) & (self.conf[r, b] > 1)
        if strong.any():
            kk = np.exp(self.logk)
            expect = kc[:, None] * self.v[r, b] - G[r, b][:, None] * x[r, b]

            def miss(scale):
                kq = kk * scale
                return ((self.to_a(obs, kq) - self.to_a(self.o_last, kq) - expect) ** 2).sum(1)

            mp, mm = miss(1.05), miss(1 / 1.05)
            step = (mp - mm) / (mp + mm + 1e-12)
            self.logk = np.clip(np.where(strong, self.logk - 0.03 * step, self.logk), self.logk_min, np.log(2.2))

        conf_before = self.conf.copy()
        self.err = np.where(U, 0.85 * self.err + 0.15 * res, self.err)
        self.conf = np.where(U & (res > 0.1), self.conf * 0.8, self.conf)

        cam = G * np.sqrt(xx)
        vmove = kc[:, None] * np.linalg.norm(self.v, axis=2)
        dominant = cam > 3 * vmove + 0.005
        unsure = self.conf < 0.5
        ok = U & aligned & (cn > 0.004) & (dominant | unsure)
        lg = np.log(np.where(ok, Gobs, 1.0))

        xb, db = x[r, b], dobs[r, 0]
        dx, dd = xb - self.prev_x, db - self.prev_dobs
        dxx = (dx * dx).sum(1)
        gd_raw = -(dd * dx).sum(1) / np.maximum(dxx, 1e-30)
        step_cam = np.exp(self.logG[r, b]) * np.sqrt(dxx)
        dn = np.linalg.norm(dd, axis=1)
        clear = (upd & self.prev_ok & (kc == 1) & (dxx > 0) & (gd_raw > 0) & (step_cam > 0.01)
                 & (-(dd * dx).sum(1) > 0.8 * dn * np.sqrt(dxx)))
        lgd = np.log(np.where(clear, gd_raw, 1.0))

        # --- new: the axis monitor, from the same speed-free frame-to-frame change, in SENT counts
        Cs = np.zeros((n, HIST + 1, 2))
        np.cumsum(self.S, axis=1, out=Cs[:, 1:])
        xs = Cs[r, b + kc] - Cs[r, b]
        dxs = xs - self.prev_xs
        gx = np.exp(self.logG[r, b])[:, None] * dxs
        seen_move = upd & self.prev_ok & (kc == 1) & (np.linalg.norm(gx, axis=1) > 2 * self.q_est + 0.004)
        Wm = seen_move[:, None, None] * 1.0
        self.Mxx = 0.97 ** seen_move[:, None, None] * self.Mxx + Wm * np.einsum("ni,nj->nij", gx, gx)
        self.Mcx = 0.97 ** seen_move[:, None, None] * self.Mcx + Wm * np.einsum("ni,nj->nij", -dd, gx)
        # --- new: the pixel grid, and moves that should have shown but didn't
        step_seen = np.abs(db).max(1)
        small = upd & (step_seen > 1e-7)
        self.q_est = np.where(small, np.minimum(self.q_est * 1.02, np.maximum(step_seen, self.q_est * 0.7)), self.q_est)
        expect_vis = upd & self.prev_ok & (kc == 1) & (step_cam > 3 * self.q_est)
        nothing = dn < 0.5 * self.q_est
        self.blind = np.where(expect_vis, 0.9 * self.blind + 0.1 * nothing, self.blind)
        self.prev_xs = xs.copy()

        self.prev_dobs, self.prev_x = db.copy(), xb.copy()
        self.prev_ok = upd & (kc == 1)
        big_b = clear & (conf_before[r, b] >= 2.0) & (np.abs(lgd - self.logG[r, b]) > np.log(1.6))
        big = np.zeros_like(ok)
        big[r, b] = big_b
        lg = lg.copy()
        lg[r, b] = np.where(big_b, lgd, lg[r, b])
        confirm = big & ((self.pend_on & (np.abs(lg - self.pend) < np.log(1.3))) | self.k_fast[:, None])
        self.pend = np.where(big, lg, self.pend)
        seen_b = np.zeros_like(ok)
        seen_b[r, b] = clear
        self.pend_on = np.where(seen_b, big & ~confirm, self.pend_on)
        normal = ok & ~big

        strength = np.clip(cn / 0.03, 0, 1)
        alpha = np.minimum(np.where(unsure, 1.0, 0.25) * strength * self.k_gain[:, None], 1.0)
        self.logG = np.where(normal, self.logG + alpha * (lg - self.logG), self.logG)
        self.logG = np.where(confirm, lg, self.logG)
        self.logG = np.clip(self.logG, np.log(1e-8), 0.0)
        self.conf = np.where(normal, np.minimum(self.conf + strength, 20), self.conf)
        self.conf = np.where(confirm, np.maximum(self.conf, 2.0), self.conf)
        fine = clear & ~big_b & (conf_before[r, b] >= 0.5)
        rate = np.minimum(0.3 * np.clip(step_cam / 0.03, 0, 1) * self.k_gain, 1.0)
        self.logG[r, b] = np.where(fine, self.logG[r, b] + rate * (lgd - self.logG[r, b]), self.logG[r, b])
        self.gain_seen = clear
        self.switched = confirm[r, b]
        self._pick()
        if self.oracle.any():            # ceiling: perfect knowledge of the hidden settings
            o = self.oracle
            self.logG[o] = self.oracle_logG[o, None]
            self.conf[o] = 20.0
            self.cur = np.where(o, self.oracle_d, self.cur)
            self.err[o] = np.where(np.arange(D)[None] == self.oracle_d[o, None], 0.001, 0.05)

        G = np.exp(self.logG)
        vobs = (dobs + G[..., None] * x) / kf
        cam = G * np.sqrt(xx)
        trust = 1.0 / (1.0 + (cam / (np.linalg.norm(dobs, axis=2) + 0.005)) ** 2)
        beta = np.minimum(0.5 * np.maximum(trust, 0.1) * self.k_vel[:, None], 0.95)[..., None]
        b = self.cur
        v_before = self.v[r, b].copy()
        self.v = np.where(U[..., None], self.v + beta * (vobs - self.v), self.v)
        self.v[jump] = 0

        dv = (self.v[r, b] - v_before) / kc[:, None]
        self.acc = np.where(upd[:, None], 0.6 * self.acc + 0.4 * dv, self.acc)
        self.acc[jump] = 0
        self.vhist = np.where(upd[:, None, None], np.concatenate([vobs[r, b][:, None], self.vhist[:, :-1]], 1),
                              self.vhist)
        self.vhist[jump] = 0
        self.since_switch = np.where(self.switched, 0, np.minimum(self.since_switch + 1, 99))
        self.since_jump = np.where(jump, 0, np.minimum(self.since_jump + 1, 99))
        self.dG10 = np.concatenate([self.logG[r, self.cur][:, None], self.dG10[:, :-1]], 1)

        new = fresh & visible
        self.seen = np.where(jump, 1, np.where(new, self.seen + 1, self.seen))
        self.o_last = np.where(new[:, None], obs, self.o_last)
        self.since = np.where(new, 0, self.since)
        self.have |= new


class KitBrain(Brain):
    def __init__(self, n):
        super().__init__(n)
        self.est = KitEstimator(n)
        self.k_extrap = np.ones(n)
        self.k_gate = np.full(n, np.inf)
        self.k_cap = np.full(n, np.inf)
        self.hold_t = np.zeros(n)
        self.gated = np.zeros(n, bool)
        self.gate_run = np.zeros(n, int)
        self.extra = np.zeros((n, EXTRA_DIM), np.float32)
        self.counts0 = np.zeros((n, 2))
        self.fix = np.zeros((n, 3), bool)       # axis fix in use: flip x, flip y, swap (kept by the caller)
        self.force_stale = np.zeros(n, bool)    # ceiling only: the oracle marks glitched / frozen frames
        self.k_y = np.ones(n)                   # y ratio fix (kept by the caller, like fix)
        self.k_dead = np.zeros(n)               # steady: deadband on a calm target, in target sizes (0 = off)
        self.k_smooth = np.ones(n)              # snap/click smoothing (used by the caller; 1 = off)
        # --- the v2 setup (user 2026-10-25). Defaults = the current code; V2 below turns them on per game
        self.use_acc = np.ones(n, bool)         # the acceleration prediction (anti-latency)
        self.acc_scale = np.ones(n)             # how much of it is used (setup dial "prediction off": 1 - dial)
        self.extrap_base = np.ones(n)           # x the "speed x frames unseen" guess, before k_extrap
        self.dz_in = np.zeros(n)                # deadzone, in target sizes: inside, every output is bypassed
        self.dz_out = np.zeros(n)               # careful zone out to here: no snap, no network gain, stop at the edge
        self.zone = np.zeros(n, int)            # this frame: 0 normal, 1 careful, 2 inside the deadzone
        self.dz_click = np.full(n, 0.6)         # zone click: fire when the move lands within this share of the narrow side
        self.dz_stable = np.zeros(n, int)       # zone click: frames in a row the target must have been in the zones
        self.dz_speed = np.full(n, 0.5)         # zones only when the target moves less than this many narrow sides per frame
        self.logk_min = np.full(n, np.log(0.4))  # narrowest field of view the estimator believes (0.4 = ~44 deg)
        self.est.logk_min = self.logk_min        # the same array: setting it per game reaches the estimator
        self.zone_run = np.zeros(n, int)        # frames in a row in the zones so far

    def reset(self, mask):
        mask = np.asarray(mask, bool)
        super().reset(mask)
        self.hold_t[mask] = 0
        self.gated[mask] = False
        self.gate_run[mask] = 0
        self.fix[mask] = False
        if hasattr(self, "zone_run"):
            self.zone_run[mask] = 0

    def observe(self, obs, visible, fresh, radius=None, box=None, dt=None, whole=None, hold=None, zoomed=None):
        e, n = self.est, self.n
        if dt is not None:
            self.dt = np.broadcast_to(np.asarray(dt, float), (n,)).copy()
        if whole is not None:
            self.whole = np.broadcast_to(np.asarray(whole, bool), (n,)).copy()
        if hold is not None:
            self.hold = np.broadcast_to(np.asarray(hold, bool), (n,)).copy()
        if zoomed is not None:
            z = np.broadcast_to(np.asarray(zoomed, bool), (n,))
            self.since_zoom = np.where(z != self.zoomed, 0, np.minimum(self.since_zoom + 1, 99))
            self.zoomed = z.copy()
        visible = np.asarray(visible, bool)
        fresh = np.asarray(fresh, bool) & ~self.force_stale
        obs = np.asarray(obs, float)
        # glitch gate: a detection far from where it must be is not trusted (the network sets how far)
        if np.isfinite(self.k_gate).any() and hasattr(self, "p"):
            pred_u = e.to_u(self.p + self.vnext) if hasattr(self, "vnext") else self.p_u
            dist = np.linalg.norm(obs - pred_u, axis=1) / np.maximum(self.radius, 1e-3)
            # at most GATE_RUN frames in a row: something that stays there is real (a new target)
            self.gated = fresh & visible & e.have & (dist > self.k_gate) & (self.gate_run < GATE_RUN)
            self.gate_run = np.where(self.gated, self.gate_run + 1, 0)
            fresh = fresh & ~self.gated
        else:
            self.gated = np.zeros(n, bool)
        e.update(obs, visible, fresh)
        self.no_check = np.where(e.gain_seen, 0, self.no_check + 1)
        b, G, conf, v, err = e.best()
        r = np.arange(n)

        o_a = e.to_a(e.o_last)
        lag = np.minimum(e.since + b, HIST)
        inflight = e.csum()[r, lag]
        ex = self.k_extrap * self.extrap_base
        p = o_a + (lag * ex)[:, None] * v - G[:, None] * inflight
        a = np.clip(e.acc, -ACC_CAP, ACC_CAP) if ANTI_LATENCY else np.zeros_like(e.acc)
        a = np.where(self.use_acc[:, None], a, 0.0) * self.acc_scale[:, None]
        p = p + a * (lag * (lag + 1) / 2 * ex)[:, None]
        p = np.where(e.have[:, None], np.clip(p, -3, 3), 0.0)
        self.vnext = v + a
        self.inflight = inflight
        self.o_a = o_a
        self.p_u = e.to_u(p)

        self.d, self.G, self.conf, self.v, self.err, self.p = b, G, conf, v, err, p
        self.visible = visible
        scale = e.a_scale(e.o_last)
        radius = np.full(n, np.nan) if radius is None else np.asarray(radius, float)
        self.size_known = np.isfinite(radius)
        radius = np.where(self.size_known, radius, DEFAULT_R)
        box = np.repeat(radius[:, None], 2, 1) if box is None else np.asarray(box, float)
        box = np.where(np.isfinite(box), box, radius[:, None])
        self.radius = np.maximum(radius * scale, 1e-3)
        self.box = box * scale[:, None]
        self.click_r = np.maximum(np.minimum(self.radius, self.box.min(1)), 1e-3)
        self.lost = ~e.have | (e.since > LOST_AFTER)
        on_screen = (np.abs(p[:, 0]) < 1) & (np.abs(p[:, 1]) < 1 / ASPECT)
        moved = np.linalg.norm(G[:, None] * inflight, axis=1) > self.radius
        self.searching = e.have & ~visible & (e.since > b + 2) & ~self.lost & on_screen & moved
        self.probing = (conf < 0.5) & ~self.lost
        self.w_logit = 3 * (np.linalg.norm(p, axis=1) / (3 * self.radius + 0.02) - 1)
        counts0, _, _, _, logit0 = self._plan(np.zeros((n, ACT_DIM)))
        self.counts0 = counts0

        trend = np.log(self.radius / np.maximum(self.rad_hist[:, 2], 1e-3))
        self.rad_hist = np.concatenate([self.radius[:, None], self.rad_hist[:, :-1]], 1)
        onehot = np.zeros((n, N_DELAYS))
        onehot[r, b] = 1
        f = [
            np.clip(p, -2, 2),
            np.clip(o_a, -2, 2),
            np.clip(v * 10, -5, 5),
            ((np.log(G) - LOG_G0) / 4)[:, None],
            np.tanh(conf / 3)[:, None],
            onehot,
            np.clip(self.radius * 10, 0, 5)[:, None],
            np.clip(self.box * 10, 0, 5),
            visible[:, None],
            fresh[:, None],
            np.clip(G[:, None] * self.last, -2, 2),
            (np.log1p(np.linalg.norm(self.last, axis=1)) / 8)[:, None],
            (np.minimum(e.since, 30) / 30)[:, None],
            self.probing[:, None],
            np.clip(err * 10, 0, 5)[:, None],
            np.tanh(self.w_logit / 3)[:, None],
            np.clip(logit0 / 4, -3, 3)[:, None],
            self.osc[:, None],
            np.clip(e.vhist.reshape(n, -1) * 10, -5, 5),
            np.clip(e.acc * 100, -5, 5),
            np.clip(trend, -1, 1)[:, None],
            e.logk[:, None],
            (np.minimum(e.since_switch, 30) / 30)[:, None],
            (np.minimum(e.since_jump, 30) / 30)[:, None],
            e.delay_margin()[:, None],
            np.clip(G[:, None] * inflight, -2, 2),
            (np.minimum(e.seen, 10) / 10)[:, None],
            (np.minimum(self.since_click, 30) / 30)[:, None],
            self.searching[:, None],
            (self.nudge > 0)[:, None],
            self.size_known[:, None],
            np.clip(self.dt - 1, -1, 3)[:, None],
            self.whole[:, None],
            self.hold[:, None],
            self.zoomed[:, None],
            (np.minimum(self.since_zoom, 30) / 30)[:, None],
            np.clip((np.log(G / self.radius) + 4.6) / 3, -3, 3)[:, None],
            np.clip(G[:, None] * self.carry / self.radius[:, None], -2, 2),
        ]
        # --- the new measurements
        dist = np.linalg.norm(p, axis=1)
        on = visible & (dist < self.radius)
        self.hold_t = np.where(on, self.hold_t + 1, 0)
        M = e.monitor()
        move0 = np.linalg.norm(G[:, None] * counts0, axis=1)
        q = np.maximum(e.q_est, 1e-5)
        self.extra = np.concatenate([
            M.reshape(n, 4) - np.eye(2).reshape(1, 4),                                  # 0-3 axis response - identity
            np.tanh(np.log(np.trace(e.Mxx, axis1=1, axis2=2) + 1e-9) / 10 + 1)[:, None],   # 4 how much axis evidence
            np.clip(np.log(q / 0.002) / 3, -2, 2)[:, None],                              # 5 pixel step
            np.clip(np.log1p(move0 / q) / 4, 0, 3)[:, None],                             # 6 planned move in pixels
            e.blind[:, None],                                                            # 7 moves that never showed
            (np.minimum(self.hold_t, 200) / 200)[:, None],                               # 8 time held on target
            np.clip(np.log1p(dist / self.radius) / 3, 0, 3)[:, None],                    # 9 distance in sizes
            np.clip(np.log1p(lag * np.linalg.norm(v, axis=1) / self.radius) / 3, 0, 3)[:, None],   # 10 guessed travel unseen
            np.clip(np.log1p(e.surprise / self.radius) / 3, 0, 3)[:, None],             # 11 glitch surprise
            self.gated[:, None],                                                         # 12 this detection was ignored
            np.clip(np.log1p(move0 / (dist + self.radius)) / 2, 0, 3)[:, None],          # 13 move vs needed (fling risk)
            np.clip((e.dG10[:, 0] - e.dG10[:, -1]) / 2, -2, 2)[:, None],                 # 14 gain change, 10 frames
            np.clip(np.abs(np.diff(e.dG10, axis=1)).sum(1) / 3, 0, 2)[:, None],          # 15 gain jitter
            (np.minimum(lag, 30) / 30)[:, None],                                         # 16 frames of guessing
            np.clip(np.log1p(np.linalg.norm(self.last, axis=1)) / 3 - 0.3, -1, 3)[:, None],   # 17 last move, counts
            (np.linalg.norm(self.last, axis=1) < 1)[:, None],                            # 18 sub-count move
            np.clip(np.linalg.norm(e.acc, axis=1) * 100, 0, 3)[:, None],                 # 19 target acceleration
        ], 1).astype(np.float32)
        return np.concatenate(f, axis=1).astype(np.float32)

    def _plan(self, raw):
        counts, click, w, s_after, logit = super()._plan(raw)
        if np.isfinite(self.k_cap).any() and hasattr(self, "p"):
            limit = self.k_cap * (np.linalg.norm(self.p, axis=1) + self.radius + 0.01)
            move = np.linalg.norm(self.G[:, None] * counts, axis=1)
            f = np.where(move > limit, limit / np.maximum(move, 1e-12), 1.0)
            counts = counts * f[:, None]
        if (self.k_dead > 0).any() and hasattr(self, "p"):
            # steady: on a calm, visible target, don't chase wobbles smaller than k_dead target sizes
            d = self.k_dead
            dist = np.linalg.norm(self.p, axis=1) / self.radius
            calm = (self.visible & ~self.probing & ~self.searching & ~self.lost
                    & (np.linalg.norm(self.v, axis=1) < 0.3 * self.radius) & (d > 0))
            f = np.clip((dist - d) / np.maximum(d, 1e-6), 0, 1)
            counts = np.where(calm[:, None], counts * f[:, None], counts)
        self.zone = np.zeros(self.n, int)
        if (self.dz_in > 0).any() and hasattr(self, "p"):
            # Deadzone: on the target there is nowhere to go. Inside dz_in sizes every output is bypassed:
            # only the target's own motion is followed. Out to dz_out: careful, no snap, no network gain or
            # offset, close half the gap and never aim past the deadzone edge.
            dist = np.linalg.norm(self.p, axis=1) / self.click_r       # in sizes of the narrow side
            calm = np.linalg.norm(self.vnext, axis=1) < self.dz_speed * self.click_r
            ok = self.visible & ~self.lost & ~self.searching & (self.dz_in > 0) & calm
            inner = ok & (dist < self.dz_in)
            care = ok & ~inner & (dist < self.dz_out)
            damp = (1.0 / (1.0 + 2.0 * self.osc))[:, None]
            G = self.G[:, None]
            s_in = self.vnext
            gap = np.clip(1 - self.dz_in / np.maximum(dist, 1e-6), 0, 1)[:, None]
            s_care = self.vnext + 0.5 * self.p * gap
            new = np.where(inner[:, None], s_in / G, np.where(care[:, None], s_care / G * damp, counts))
            # judge the click on the move really sent; on the target the sensitivity needn't be known yet,
            # so not knowing it (probing) doesn't block the click here (everything else still does)
            zoned = inner | care
            miss_new = np.linalg.norm(self.p + self.vnext - G * new, axis=1) / self.click_r
            far = np.linalg.norm(self.p, axis=1) / self.radius
            logit_z = 4 * (self.dz_click - miss_new) - 1.5 * np.maximum(0.0, far - 1.5)   # the network's click: bypassed too
            blocked = self.lost | ~self.visible | (self.cool > 0) | (self.est.seen < 3) | self.searching
            blocked |= self.zone_run < self.dz_stable
            logit = np.where(zoned, np.where(blocked, -10.0, logit_z), logit)
            click = logit > 0
            s_after = np.where(zoned[:, None], self.p + self.vnext - G * new, s_after)
            counts = new
            self.zone = np.where(inner, 2, np.where(care, 1, 0))
        return counts, click, w, s_after, logit

    def act(self, raw):
        if not (self.dz_in > 0).any():
            return super().act(raw)
        self.zone_run = np.where(self.zone > 0, self.zone_run + 1, 0)       # last frame's zone (set in observe's plan)
        probe, wait = self.probe.copy(), self.wait.copy()
        counts, click = super().act(raw)
        # inside the deadzone: no probing growth and no test nudge (nothing to measure, nowhere to go)
        z = self.zone > 0
        self.probe = np.where(z, probe, self.probe)
        self.wait = np.where(z, wait, self.wait)
        self.nudge = np.where(z, 0, self.nudge)
        return counts, click

    def commit(self, counts, sent=None):
        super().commit(counts)
        self.est.push_sent(counts if sent is None else sent)
