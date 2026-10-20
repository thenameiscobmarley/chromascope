"""The aimer: estimator + snap/track controllers + the network's corrections.

Each frame:
  feat = brain.observe(target_pos, visible, fresh, radius)   # screen units; features for the network
  counts, click = brain.act(raw)                             # raw = network output (zeros = classic)
  brain.commit(counts_actually_applied)

Positions inside (p, v, s_after) are in the estimator's angle units, so a
mouse count moves things by the same amount anywhere on screen. p_u is p
back in screen units.

The network does not output mouse counts directly. It nudges the analytic
plan: a gain correction, an aim offset, the snap-vs-track blend and the
click decision. With raw = 0 you get the plain hand-written aimer, which is
also what an untrained model starts from.
"""
import numpy as np

from .config import ACT_DIM, ASPECT, HIST, LOG_G0, N_DELAYS
from .estimator import Estimator

KP = 0.35           # share of the remaining gap track mode closes per frame
PROBE0 = 0.25       # first probe in counts: tiny even at the highest sensitivity
PROBE_GROW = 4.0    # each probe that gets no clear response is this much bigger
PROBE_WAIT = 5      # frames to wait after a probe: longer than the largest delay
CHECK_AFTER = 15    # frames of following without a gain measurement before a test nudge
DEFAULT_R = 0.02    # aim-zone size assumed when the detector gives position only (screen units)
ANTI_LATENCY = True # use the target's acceleration to see through the delay
ACC_CAP = 0.01      # largest acceleration trusted (angle units per frame^2), so noise can't fling the aim
LOST_AFTER = 45     # frames without seeing the target before giving up on it


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


class Brain:
    def __init__(self, n):
        self.n = n
        self.est = Estimator(n)
        self.probe = np.full(n, PROBE0)
        self.wait = np.zeros(n, int)
        self.no_check = np.zeros(n, int)    # frames since the gain was last measured cleanly
        self.nudge = np.zeros(n, int)       # 2: nudge out this frame, 1: nudge back, 0: none
        self.rad_hist = np.full((n, 3), DEFAULT_R)
        self.since_click = np.full(n, 99)
        self.cool = np.zeros(n, int)
        self.last = np.zeros((n, 2))
        self.osc = np.zeros(n)              # wobble meter: recent direction reversals of our own moves
        self.prev_shift = np.zeros((n, 2))
        self.info = {}
        # What the app tells us (see observe): whole counts only, hold-to-fire, frame time, zoom.
        self.whole = np.zeros(n, bool)
        self.hold = np.zeros(n, bool)
        self.dt = np.ones(n)
        self.zoomed = np.zeros(n, bool)
        self.since_zoom = np.full(n, 99)
        self.carry = np.zeros((n, 2))       # whole counts: the part of a move not sent yet

    def reset(self, mask):
        mask = np.asarray(mask, bool)
        if not mask.any():
            return
        self.est.reset(mask)
        self.probe[mask] = PROBE0
        self.wait[mask] = 0
        self.no_check[mask] = 0
        self.nudge[mask] = 0
        self.rad_hist[mask] = DEFAULT_R
        self.since_click[mask] = 99
        self.cool[mask] = 0
        self.last[mask] = 0
        self.osc[mask] = 0
        self.prev_shift[mask] = 0
        self.since_zoom[mask] = 99
        self.carry[mask] = 0

    def observe(self, obs, visible, fresh, radius=None, box=None, dt=None, whole=None, hold=None, zoomed=None):
        """radius: size of the aim zone; box: (n, 2) half width/height of the whole target.
        All in screen units. A missing size (None or NaN) means the detector gives position
        only: a default size is used and the network is told the size is unknown.
        What the app knows (all optional): dt = this frame's time (1 = normal), whole = it only
        takes whole mouse counts, hold = fire is held (tracking), zoomed = the player is scoped in."""
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
        fresh = np.asarray(fresh, bool)
        e.update(np.asarray(obs, float), visible, fresh)
        self.no_check = np.where(e.gain_seen, 0, self.no_check + 1)
        b, G, conf, v, err = e.best()
        r = np.arange(n)

        # Where the target is *now* relative to the crosshair: the last sighting, plus the
        # target's own motion since that (delayed) frame, minus the mouse moves already
        # sent that the screen has not shown yet (so it never chases its own movement).
        o_a = e.to_a(e.o_last)
        lag = np.minimum(e.since + b, HIST)
        inflight = e.csum()[r, lag]
        p = o_a + lag[:, None] * v - G[:, None] * inflight
        # Anti-latency: the target's speed keeps changing while the screen is behind, so add
        # its acceleration over the delay too (capped, so a noisy estimate can't fling the aim).
        a = np.clip(e.acc, -ACC_CAP, ACC_CAP) if ANTI_LATENCY else np.zeros_like(e.acc)
        p = p + a * (lag * (lag + 1) / 2)[:, None]
        p = np.where(e.have[:, None], np.clip(p, -3, 3), 0.0)
        self.vnext = v + a                  # its speed during the next frame
        self.inflight = inflight
        self.o_a = o_a
        self.p_u = e.to_u(p)

        self.d, self.G, self.conf, self.v, self.err, self.p = b, G, conf, v, err, p
        self.visible = visible
        # Sizes come in screen units; near the edge they are smaller in angle units.
        scale = e.a_scale(e.o_last)
        radius = np.full(n, np.nan) if radius is None else np.asarray(radius, float)
        self.size_known = np.isfinite(radius)
        radius = np.where(self.size_known, radius, DEFAULT_R)
        box = np.repeat(radius[:, None], 2, 1) if box is None else np.asarray(box, float)
        box = np.where(np.isfinite(box), box, radius[:, None])
        self.radius = np.maximum(radius * scale, 1e-3)
        self.box = box * scale[:, None]
        # Clicks are judged on the narrow side: a tall, thin zone (a dummy's body, a stretched
        # box) is only as forgiving sideways as its half width, not its average size.
        self.click_r = np.maximum(np.minimum(self.radius, self.box.min(1)), 1e-3)
        self.lost = ~e.have | (e.since > LOST_AFTER)
        # We believe the target is on screen, but it hasn't shown up for a few frames: if we
        # moved a fair way since we last saw it, our gain must be wrong (we overshot), so go back
        # the way we came and re-measure. If we barely moved, it just flickered out or went
        # behind cover: stay put and wait for it.
        on_screen = (np.abs(p[:, 0]) < 1) & (np.abs(p[:, 1]) < 1 / ASPECT)
        moved = np.linalg.norm(G[:, None] * inflight, axis=1) > self.radius
        self.searching = e.have & ~visible & (e.since > b + 2) & ~self.lost & on_screen & moved
        self.probing = (conf < 0.5) & ~self.lost
        self.w_logit = 3 * (np.linalg.norm(p, axis=1) / (3 * self.radius + 0.02) - 1)
        _, _, _, _, logit0 = self._plan(np.zeros((n, ACT_DIM)))

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
            # --- added inputs (28-49); older models start with zero weight on these ---
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
            # --- added inputs (50-57) ---
            np.clip(self.dt - 1, -1, 3)[:, None],
            self.whole[:, None],
            self.hold[:, None],
            self.zoomed[:, None],
            (np.minimum(self.since_zoom, 30) / 30)[:, None],
            # How far one mouse count moves the aim, compared with the target's size: with whole
            # counts and a high sensitivity, small corrections are impossible.
            np.clip((np.log(G / self.radius) + 4.6) / 3, -3, 3)[:, None],
            np.clip(G[:, None] * self.carry / self.radius[:, None], -2, 2),
        ]
        return np.concatenate(f, axis=1).astype(np.float32)

    def _plan(self, raw):
        raw = np.clip(raw, -3, 3)
        p, v, G = self.p, self.vnext, self.G

        # Snap: move onto where the target will be once this move lands. Track: match its
        # speed and close part of the gap. Neither aims further ahead than this one move.
        w = _sigmoid(self.w_logit + 2 * raw[:, 3])[:, None]
        s = w * (p + v) + (1 - w) * (KP * p + v) + 0.03 * raw[:, 1:3]
        sn = np.linalg.norm(s, axis=1, keepdims=True)
        s = s * np.minimum(1.0, 1.5 / np.maximum(sn, 1e-9))
        gain = np.exp(0.5 * raw[:, 0])[:, None]
        self.gain_used = gain[:, 0]
        # Wobbling means our moves are overshooting: damp them until it settles.
        damp = (1.0 / (1.0 + 2.0 * self.osc))[:, None]
        counts = s / G[:, None] * gain * damp

        # Unknown sensitivity: one small test move toward the last sighting, then wait
        # until the screen must have shown it, then a bigger one if it was too small to see.
        on = np.linalg.norm(self.o_a, axis=1, keepdims=True)
        direction = np.where(on > 1e-6, self.o_a / np.maximum(on, 1e-9), np.array([1.0, 0.0]))
        pulse = (self.wait == 0)[:, None]
        probe = np.where(self.whole, np.maximum(self.probe, 1.0), self.probe)   # at least one whole count
        probe_counts = np.where(pulse, direction * probe[:, None] * gain, 0.0)
        counts = np.where(self.probing[:, None], probe_counts, counts)
        # Undo every mouse move since the last sighting, count for count: this returns the
        # camera to where the target was visible without needing to know the gain.
        counts = np.where(self.searching[:, None], -self.inflight, counts)
        counts = np.where(self.lost[:, None], 0.0, counts)

        # Test nudge while following: a little extra one way, then the same back next frame.
        # It adds up to nothing, but the change between the two moves is clearly ours, so the
        # estimator can measure the gain from it even while matching a moving target's speed.
        size = np.maximum(0.4 * self.radius, 0.006)
        out_back = np.where(self.nudge == 2, 1.0, np.where(self.nudge == 1, -1.0, 0.0))
        out_back = out_back * (~self.probing & ~self.lost & ~self.searching)
        counts = counts + (out_back * size / G)[:, None] * np.array([1.0, 0.0])

        s_after = p + v - G[:, None] * counts
        miss = np.linalg.norm(s_after, axis=1) / self.click_r
        # The further the target was before this move, the less a long flick's landing
        # spot is trusted: flick first, see where it landed, then click.
        far = np.linalg.norm(p, axis=1) / self.radius
        logit = 4 * (0.6 - miss) - 1.5 * np.maximum(0.0, far - 1.5) + 3 * raw[:, 4]
        # Don't click a new target before its speed has been seen at least twice.
        settling = self.est.seen < 3
        blocked = self.probing | self.lost | ~self.visible | (self.cool > 0) | settling | self.searching
        logit = np.where(blocked, -10.0, logit)
        return counts, logit > 0, w[:, 0], s_after, logit

    def act(self, raw):
        counts, click, w, s_after, _ = self._plan(np.asarray(raw, float))
        # Whole counts only: send the rounded move and keep the rest for the next frame.
        want = counts + self.carry
        sent = np.where(self.whole[:, None], np.round(want), counts)
        self.carry = np.where(self.whole[:, None], want - sent, 0.0)
        counts = sent

        pulse = self.probing & (self.wait == 0)
        self.probe = np.where(pulse, np.minimum(self.probe * PROBE_GROW, 1e7), self.probe)
        self.wait = np.where(pulse, PROBE_WAIT, np.where(self.probing, np.maximum(self.wait - 1, 0), 0))
        # After a click, wait until the screen could show the result before clicking again.
        # Holding fire is different: keep firing while on target.
        self.cool = np.maximum(self.cool - 1, 0)
        self.cool = np.where(click & ~self.hold, self.d + 1, self.cool)
        self.since_click = np.where(click, 0, np.minimum(self.since_click + 1, 99))
        self.est.expect_new = np.where(click & ~self.hold, self.d + 3, self.est.expect_new)
        # g_used: the sensitivity this move was planned with (estimate and network correction).
        self.info = dict(w=w, s_after=s_after, g_used=self.G / self.gain_used)
        following = ~self.probing & ~self.lost & ~self.searching & self.visible
        start = (self.nudge == 0) & following & (self.no_check > CHECK_AFTER)
        self.nudge = np.where(start, 2, np.maximum(self.nudge - 1, 0))
        self.no_check = np.where(start, 0, self.no_check)
        return counts, click

    def commit(self, counts):
        counts = np.asarray(counts, float)
        self.est.push(counts)
        self.last = counts
        # A reversal is a move that undoes the previous one while both are meaningful in size.
        shift = self.G[:, None] * counts
        big = 0.2 * self.radius
        rev = ((shift * self.prev_shift).sum(1) < 0) & (np.linalg.norm(shift, axis=1) > big) \
            & (np.linalg.norm(self.prev_shift, axis=1) > big)
        self.osc = 0.8 * self.osc + 0.2 * rev
        self.prev_shift = shift
