"""Hold-to-aim: watches your real mice and touchpads for the right button, in any window.

Modes: "mouse" = a regular mouse's right button, "touchpad" = a touchpad's right click, "both". Mice plugged in later
are picked up within a couple of seconds.

Wayland apps can't see clicks in other windows, so this reads the kernel's input devices directly (read-only, mice and
touchpads only). Needs read access to them: setup.sh adds a udev rule that gives the logged-in user that (uaccess).
Touchpads (clickpads) have one physical button; like KDE, a press in the bottom-right corner or with two fingers
counts as a right click.
"""
from __future__ import annotations

import fcntl
import os
import re
import select
import struct
import threading
import time

from PySide6.QtCore import QObject, Signal

from vmouse import NAME as OUR_MOUSE

EV_SYN, EV_KEY, EV_ABS, BTN_LEFT, BTN_RIGHT, BTN_TOOL_DOUBLETAP = 0, 1, 3, 0x110, 0x111, 0x14D
ABS_X, ABS_Y = 0, 1
EVENT = struct.Struct("llHHi")


def pointer_devices():
    """[(name, /dev/input/eventN)] for every mouse and touchpad except our own virtual mouse."""
    try: text = open("/proc/bus/input/devices").read()
    except OSError: return []
    out = []
    for block in text.split("\n\n"):
        name = re.search(r'^N: Name="(.*)"', block, re.M)
        handlers = re.search(r"^H: Handlers=(.*)$", block, re.M)
        if not (name and handlers) or name.group(1) == OUR_MOUSE.decode(): continue
        hs = handlers.group(1).split()
        ev = next((h for h in hs if h.startswith("event")), None)
        if ev and any(h.startswith("mouse") for h in hs): out.append((name.group(1), f"/dev/input/{ev}"))
    return out


def _abs(fd, code):
    """(min, max, resolution in units/mm) of an absolute axis, or None if the device has no such axis."""
    buf = bytearray(24)
    try: fcntl.ioctl(fd, 0x80184540 + code, buf)               # EVIOCGABS(code)
    except OSError: return None
    _v, lo, hi, _f, _fl, res = struct.unpack("6i", buf)
    return (lo, hi, res) if hi > lo else None


class RightButton(QObject):
    """Emits held(True) when any real mouse's right button goes down and held(False) when all are up."""
    held = Signal(bool)

    def __init__(self, mode="both"):
        super().__init__()
        self.mode = mode                                         # "mouse", "touchpad" or "both"
        self.fds, self.paths, self.denied, self.devices = {}, {}, [], []
        self.down, self.two_fingers, self.left_as_right = set(), set(), set()
        self.area, self.pos = {}, {}                             # touchpads: fd -> right-button corner (x from, y from); finger x, y
        self.pending = set()                                     # clickpad presses to sort out once the finger position is in
        self.thread, self.stop_flag = None, threading.Event()

    @property
    def ok(self): return bool(self.fds)

    def names(self, kind=None):
        """Names of the devices being listened to; kind = "mouse" or "touchpad" to filter."""
        fds = dict(self.fds)
        return sorted({n for fd, n in fds.items() if kind is None or (fd in self.area) == (kind == "touchpad")})

    def _scan(self):
        """Open any mouse / touchpad not open yet (also ones plugged in after start)."""
        self.devices = pointer_devices()
        denied = []
        for name, path in self.devices:
            if path in self.paths.values(): continue
            try: fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK)
            except OSError: denied.append(name); continue
            x, y = _abs(fd, ABS_X), _abs(fd, ABS_Y)
            if x and y:                                          # KDE/libinput's software buttons: bottom ~10 mm, right half
                self.area[fd] = ((x[0] + x[1]) / 2, y[1] - max(10 * (y[2] or 1), 0.12 * (y[1] - y[0])))
                self.pos[fd] = [0, 0]
            self.paths[fd] = path; self.fds[fd] = name
        self.denied = denied

    def _drop(self, fd):
        """A device went away (unplugged)."""
        for d in (self.fds, self.paths, self.area, self.pos): d.pop(fd, None)
        for d in (self.pending, self.two_fingers, self.left_as_right): d.discard(fd)
        was = bool(self.down); self.down.discard(fd)
        try: os.close(fd)
        except OSError: pass
        if was and not self.down: self.held.emit(False)

    def start(self):
        if self.thread: return self.ok
        self._scan()
        self.stop_flag.clear()
        self.thread = threading.Thread(target=self._run, daemon=True); self.thread.start()
        return True

    def stop(self):
        if not self.thread: return
        self.stop_flag.set(); self.thread.join(1); self.thread = None
        for fd in self.fds: os.close(fd)
        self.fds, self.paths, self.denied, self.area, self.pos = {}, {}, [], {}, {}
        if self.down: self.down.clear(); self.held.emit(False)

    def _run(self):
        next_scan = time.monotonic() + 2
        while not self.stop_flag.is_set():
            if time.monotonic() > next_scan: self._scan(); next_scan = time.monotonic() + 2
            r, _, _ = select.select(list(self.fds), [], [], 0.2) if self.fds else (time.sleep(0.2) or [], 0, 0)
            for fd in r:
                try: data = os.read(fd, EVENT.size * 64)
                except BlockingIOError: continue
                except OSError: self._drop(fd); continue
                for i in range(0, len(data) - EVENT.size + 1, EVENT.size):
                    _s, _us, typ, code, val = EVENT.unpack_from(data, i)
                    if typ == EV_KEY: self._key(fd, code, val)
                    elif typ == EV_ABS and code in (ABS_X, ABS_Y) and fd in self.pos: self.pos[fd][code] = val
                    elif typ == EV_SYN and fd in self.pending: self.pending.discard(fd); self._press(fd)

    def _key(self, fd, code, val):
        was = bool(self.down)
        if code == BTN_TOOL_DOUBLETAP: (self.two_fingers.add if val else self.two_fingers.discard)(fd)
        elif code == BTN_RIGHT:                                  # a real right button (mice; some touchpads have one too)
            if self.mode != "touchpad" or fd in self.area: (self.down.add if val else self.down.discard)(fd)
        elif code == BTN_LEFT:                                   # clickpad: decided at the end of this event batch
            if val:
                if fd in self.area and self.mode != "mouse": self.pending.add(fd)
            else:
                self.pending.discard(fd)
                if fd in self.left_as_right: self.left_as_right.discard(fd); self.down.discard(fd)
        if bool(self.down) != was: self.held.emit(bool(self.down))

    def _press(self, fd):
        """Clickpad press: a right click with two fingers down or in the bottom-right corner, else a left click (ignored)."""
        corner = fd in self.area and self.pos[fd][0] >= self.area[fd][0] and self.pos[fd][1] >= self.area[fd][1]
        if fd in self.two_fingers or corner:
            was = bool(self.down); self.left_as_right.add(fd); self.down.add(fd)
            if not was: self.held.emit(True)
