"""Screen capture on Wayland (KDE Plasma) through xdg-desktop-portal ScreenCast + PipeWire + GStreamer.

The portal shows KDE's "pick a screen" dialog once; the restore token it hands back is saved so later runs start
straight away. Frames arrive as BGRx numpy arrays (height, width, 4).
"""
from __future__ import annotations

import os
import secrets
import threading
import time

import gi
gi.require_version("Gst", "1.0")
from gi.repository import Gio, GLib, Gst  # noqa: E402

import numpy as np  # noqa: E402

Gst.init(None)

BUS, PATH = "org.freedesktop.portal.Desktop", "/org/freedesktop/portal/desktop"
IFACE = "org.freedesktop.portal.ScreenCast"
MONITOR, WINDOW = 1, 2


class PortalError(RuntimeError):
    pass


class Portal:
    """Runs the ScreenCast handshake. Blocking; call from a worker thread."""

    def __init__(self):
        self.ctx = GLib.MainContext.new()
        self.ctx.push_thread_default()
        self.bus = Gio.bus_get_sync(Gio.BusType.SESSION, None)
        self.sender = self.bus.get_unique_name()[1:].replace(".", "_")
        self.session = None

    def close(self):
        self.ctx.pop_thread_default()

    def _request(self, method, args_fmt, args, timeout=120):
        token = "cs" + secrets.token_hex(6)
        path = f"{PATH}/request/{self.sender}/{token}"
        box = {}

        def on_response(_c, _s, _p, _i, _sig, params):
            box["r"] = params.unpack()

        sub = self.bus.signal_subscribe(BUS, "org.freedesktop.portal.Request", "Response", path, None,
                                        Gio.DBusSignalFlags.NO_MATCH_RULE, on_response)
        try:
            args = list(args)
            args[-1] = dict(args[-1], handle_token=GLib.Variant("s", token))
            self.bus.call_sync(BUS, PATH, IFACE, method, GLib.Variant(args_fmt, tuple(args)),
                               GLib.VariantType("(o)"), Gio.DBusCallFlags.NONE, -1, None)
            end = time.monotonic() + timeout
            while "r" not in box:
                if time.monotonic() > end: raise PortalError(f"{method}: no answer from the portal")
                self.ctx.iteration(True)
        finally:
            self.bus.signal_unsubscribe(sub)
        code, results = box["r"]
        if code == 1: raise PortalError("Screen sharing was cancelled")
        if code != 0: raise PortalError(f"{method} failed (code {code})")
        return results

    def start(self, source=MONITOR, cursor=False, restore_token=None):
        """Returns (pipewire_fd, node_id, restore_token, (width, height))."""
        r = self._request("CreateSession", "(a{sv})",
                          [{"session_handle_token": GLib.Variant("s", "cs" + secrets.token_hex(6))}])
        self.session = r["session_handle"]
        opts = {"types": GLib.Variant("u", source), "multiple": GLib.Variant("b", False),
                "cursor_mode": GLib.Variant("u", 2 if cursor else 1), "persist_mode": GLib.Variant("u", 2)}
        if restore_token: opts["restore_token"] = GLib.Variant("s", restore_token)
        self._request("SelectSources", "(oa{sv})", [self.session, opts])
        r = self._request("Start", "(osa{sv})", [self.session, "", {}])
        streams = r.get("streams") or []
        if not streams: raise PortalError("The portal returned no stream")
        node, props = streams[0]
        size = tuple(props.get("size", (0, 0)))
        res, fds = self.bus.call_with_unix_fd_list_sync(
            BUS, PATH, IFACE, "OpenPipeWireRemote", GLib.Variant("(oa{sv})", (self.session, {})),
            GLib.VariantType("(h)"), Gio.DBusCallFlags.NONE, -1, None, None)
        fd = fds.get(res.unpack()[0])
        return fd, node, r.get("restore_token"), size

    def stop(self):
        if self.session:
            try:
                self.bus.call_sync(BUS, self.session, "org.freedesktop.portal.Session", "Close", None, None,
                                   Gio.DBusCallFlags.NONE, -1, None)
            except GLib.Error:
                pass
            self.session = None


class Stream:
    """GStreamer pipeline pulling BGRx frames from a portal PipeWire node."""

    def __init__(self, fd, node):
        self.fd = fd
        self.pipe = Gst.parse_launch(
            f"pipewiresrc fd={fd} path={node} do-timestamp=true keepalive-time=250 always-copy=true ! "
            "videoconvert n-threads=2 ! video/x-raw,format=BGRx ! "
            "appsink name=sink drop=true max-buffers=1 sync=false emit-signals=false")
        self.sink = self.pipe.get_by_name("sink")
        self.pipe.set_state(Gst.State.PLAYING)

    def pull(self, timeout_ms=200):
        """Next frame as a (h, w, 4) uint8 array (a copy), or None on timeout."""
        sample = self.sink.emit("try-pull-sample", timeout_ms * Gst.MSECOND)
        if sample is None:
            self._check_bus()
            return None
        s = sample.get_caps().get_structure(0)
        w, h = s.get_value("width"), s.get_value("height")
        buf = sample.get_buffer()
        ok, info = buf.map(Gst.MapFlags.READ)
        if not ok: return None
        try:
            stride = len(info.data) // h
            arr = np.frombuffer(info.data, np.uint8, h * stride).reshape(h, stride)[:, :w * 4].reshape(h, w, 4).copy()
        finally:
            buf.unmap(info)
        return arr

    def _check_bus(self):
        msg = self.pipe.get_bus().pop_filtered(Gst.MessageType.ERROR | Gst.MessageType.EOS)
        if msg is None: return
        if msg.type == Gst.MessageType.EOS: raise PortalError("The screen share ended")
        err, dbg = msg.parse_error()
        raise PortalError(f"PipeWire stream error: {err.message}")

    def close(self):
        self.pipe.set_state(Gst.State.NULL)
        try: os.close(self.fd)
        except OSError: pass


class Capture:
    """Portal + stream in one object, with a background thread keeping only the newest frame."""

    def __init__(self):
        self.frame, self.frame_id, self.error = None, 0, None
        self.size, self.restore_token = (0, 0), None
        self._stop = threading.Event()
        self._thread = None
        self._cv = threading.Condition()
        self.fps = 0.0

    def start(self, source=MONITOR, cursor=False, restore_token=None, on_ready=None):
        self._stop.clear(); self.error = None
        self._thread = threading.Thread(target=self._run, args=(source, cursor, restore_token, on_ready), daemon=True)
        self._thread.start()

    def _run(self, source, cursor, token, on_ready):
        portal = stream = None
        try:
            portal = Portal()
            try:
                fd, node, self.restore_token, self.size = portal.start(source, cursor, token)
            except PortalError:
                if not token: raise
                fd, node, self.restore_token, self.size = portal.start(source, cursor, None)  # stale token
            stream = Stream(fd, node)
            if on_ready: on_ready(self)
            n, t0 = 0, time.monotonic()
            while not self._stop.is_set():
                f = stream.pull()
                if f is None: continue
                with self._cv:
                    self.frame, self.frame_id = f, self.frame_id + 1
                    self._cv.notify_all()
                n += 1
                dt = time.monotonic() - t0
                if dt >= 0.5: self.fps, n, t0 = n / dt, 0, time.monotonic()
        except Exception as e:  # noqa: BLE001 — surfaced in the UI
            self.error = str(e)
            with self._cv: self._cv.notify_all()
        finally:
            if stream: stream.close()
            if portal: portal.stop(); portal.close()
            self.fps = 0.0

    def wait(self, last_id, timeout=0.5):
        """Block until a frame newer than last_id arrives; returns (frame, id) or (None, last_id)."""
        with self._cv:
            if self.frame_id == last_id and not self.error:
                self._cv.wait(timeout)
            if self.frame_id == last_id: return None, last_id
            return self.frame, self.frame_id

    def stop(self):
        self._stop.set()
        if self._thread: self._thread.join(2); self._thread = None
        self.frame = None

    @property
    def running(self):
        return self._thread is not None and self._thread.is_alive()
