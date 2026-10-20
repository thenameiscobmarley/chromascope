"""A virtual USB mouse through /dev/uinput (plain ioctl, no extra packages).

Games see it as a real mouse, so relative moves turn the camera even when the pointer is locked (Minecraft).
Needs write access to /dev/uinput; most desktops grant it to the logged-in user, otherwise see README.
"""
from __future__ import annotations

import fcntl
import os
import struct
import time

EV_SYN, EV_KEY, EV_REL = 0, 1, 2
REL_X, REL_Y = 0, 1
BTN_LEFT, BTN_RIGHT = 0x110, 0x111
UI_SET_EVBIT, UI_SET_KEYBIT, UI_SET_RELBIT = 0x40045564, 0x40045565, 0x40045566
UI_DEV_CREATE, UI_DEV_DESTROY = 0x5501, 0x5502
NAME = b"ChromaScope NNMM mouse"


class VirtualMouse:
    def __init__(self):
        self.fd = os.open("/dev/uinput", os.O_WRONLY | os.O_NONBLOCK)
        try:
            fcntl.ioctl(self.fd, UI_SET_EVBIT, EV_KEY); fcntl.ioctl(self.fd, UI_SET_EVBIT, EV_REL)
            for b in (BTN_LEFT, BTN_RIGHT): fcntl.ioctl(self.fd, UI_SET_KEYBIT, b)
            for r in (REL_X, REL_Y): fcntl.ioctl(self.fd, UI_SET_RELBIT, r)
            # legacy uinput_user_dev: name[80], input_id (bus, vendor, product, version), ff_effects_max, 4×abs[64]
            os.write(self.fd, struct.pack("80sHHHHi", NAME, 0x03, 0x1209, 0xC5C5, 1, 0) + bytes(4 * 64 * 4))
            fcntl.ioctl(self.fd, UI_DEV_CREATE)
        except OSError:
            os.close(self.fd); raise
        self.rx = self.ry = 0.0          # sub-count remainders, so slow moves still add up
        self.born = time.monotonic()

    @property
    def ready(self): return time.monotonic() - self.born > 0.4      # the compositor needs a moment to pick it up

    def _ev(self, t, c, v):
        s = time.time(); os.write(self.fd, struct.pack("llHHi", int(s), int(s % 1 * 1e6), t, c, v))

    def move(self, dx, dy):
        """Relative move in mouse counts; fractions carry over to the next call. Returns the whole counts sent."""
        self.rx += dx; self.ry += dy
        ix, iy = int(self.rx), int(self.ry)
        if not (ix or iy) or not self.ready: return 0, 0
        self.rx -= ix; self.ry -= iy
        if ix: self._ev(EV_REL, REL_X, ix)
        if iy: self._ev(EV_REL, REL_Y, iy)
        self._ev(EV_SYN, 0, 0)
        return ix, iy

    def click(self, button=BTN_LEFT):
        if not self.ready: return
        self._ev(EV_KEY, button, 1); self._ev(EV_SYN, 0, 0)
        self._ev(EV_KEY, button, 0); self._ev(EV_SYN, 0, 0)

    def close(self):
        if self.fd is None: return
        try: fcntl.ioctl(self.fd, UI_DEV_DESTROY)
        finally: os.close(self.fd); self.fd = None

    def __del__(self):
        try: self.close()
        except Exception: pass
