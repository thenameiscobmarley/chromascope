# ChromaScope

Live colour detection on your screen for Wayland / KDE Plasma. The UI is built with UiVault3, the UI Vault widgets for PySide6.

**Install:** run `./setup.sh`. It installs the system packages and Python dependencies, then adds ChromaScope and its icon to the app menu. `./setup.sh --uninstall` takes them out again.

After that, open **ChromaScope** from the app menu or run `./run.sh`. Pass `--start` to begin capturing as soon as it opens.

* **Capture:** xdg-desktop-portal ScreenCast → PipeWire → GStreamer. KDE asks which screen to share the first time, and the app remembers your choice after that.
* **Detect:** sets the target colour (presets, a custom colour, or a pixel picked from the preview), hue tolerance, minimum saturation and brightness, minimum size, merge distance, max targets and sort order.
* **FOV:** sets the size, circle or square shape, X/Y offset, ring, crosshair and max FPS.
* **Draw:** picks Box / Glow (detected pixels turn a red shade with a halo) / Both / Corners, plus glow radius, thickness, box colour, labels and HUD.
* **Screen:** shows the overlay on your main screen. It's click-through and stays on top through a KWin rule. You also pick the monitor, monitor or window capture, the cursor setting, and whether the app minimizes while it captures.
* **NNMM:** empty placeholder tab.
* **Look:** defaults to win98 controls on the paper theme. All 18 UI Vault kits and 87 themes are available.
* **No feedback:** KWin rules (`chromascope-overlay`, `chromascope-window`) leave both the overlay and the app window out of the screen capture. They're on screen, but the detector never sees its own boxes, swatches or preview.
* Keys: F8 starts or stops capture, F9 turns the overlay on or off, F10 switches the preview between Screen, FOV zoom and Mask.

Settings are saved in `~/.config/chromascope/settings.json`.

Files: `capture.py` (portal + stream), `detect.py` (HSV mask → boxes, glow tint) and `app.py` (UI, overlay).
