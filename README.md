# ChromaScope

Live colour detection on your screen for Wayland / KDE Plasma. The UI is built with UiVault3, the UI Vault widgets for PySide6.

**Install:** run `./setup.sh`. It installs the system packages and Python dependencies, then adds ChromaScope and its icon to the app menu. `./setup.sh --uninstall` takes them out again.

After that, open **ChromaScope** from the app menu or run `./run.sh`. Pass `--start` to begin capturing as soon as it opens.

* **Capture:** xdg-desktop-portal ScreenCast → PipeWire → GStreamer. KDE asks which screen to share the first time, and the app remembers your choice after that.
* **Detect:** find up to 6 colours at once, each with its own name, box colour and on/off switch. Double-click a colour to rename it. You can set a colour from a preset, a custom colour, or **Sample from preview**: drag a box over the thing you want and its hue, saturation and brightness range is worked out from those pixels. Also: minimum size, merge distance, max boxes and sort order.
* **Profiles:** save and load named setups (colours, FOV, drawing). They're stored in `~/.config/chromascope/profiles/`.
* **FOV:** drag the ring in the preview to move it, or drag its edge to resize it. Shapes are circle, square or full screen, and you can save named regions. Large areas are scanned at half resolution to keep it fast.
* **Draw:** Box / Glow (detected pixels turn a red shade with a halo) / Both / Corners / None, plus glow radius, thickness, labels and HUD.
* **Colour-blind assist:** recolours the detected colours with stripes, a solid fill or an outline in a colour you pick. With the FOV set to full screen and "Show on main screen" on, it works over every app.
* **Screen:** shows the overlay on your main screen. It's click-through and stays on top through a KWin rule. You also pick the monitor, monitor or window capture, the cursor setting, and whether the app minimizes while it captures.
* **NNMM:** empty placeholder tab.
* **Look:** defaults to win98 controls on the paper theme. All 18 UI Vault kits and 87 themes are available.
* **No feedback:** KWin rules (`chromascope-overlay`, `chromascope-window`) leave both the overlay and the app window out of the screen capture. They're on screen, but the detector never sees its own boxes, swatches or preview.
* Keys: F8 starts or stops capture, F9 turns the overlay on or off, F10 switches the preview between Screen, FOV zoom and Mask.

Settings are saved in `~/.config/chromascope/settings.json`.

Files: `capture.py` (portal + stream), `detect.py` (HSV mask → boxes, glow tint) and `app.py` (UI, overlay).
