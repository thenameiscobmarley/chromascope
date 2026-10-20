# ChromaScope

Live colour detection on your screen for Wayland / KDE Plasma. The UI is built with UiVault3, the UI Vault widgets for PySide6.

**Install:** run `./setup.sh`. It installs the system packages and Python dependencies, then adds ChromaScope and its icon to the app menu. `./setup.sh --uninstall` takes them out again.

After that, open **ChromaScope** from the app menu or run `./run.sh`. Pass `--start` to begin capturing as soon as it opens. The first launch shows a short tour; you can replay it from Setup → Help.

## Tabs

* **Detect:** find up to 6 colours at once, each with its own name, box colour and on/off switch. Double-click a colour to rename it. Set a colour from a preset, a custom colour, or **Sample from preview**: drag a box over the thing you want and its hue, saturation and brightness range is worked out from those pixels. Saved profiles live at the top of this tab.
* **FOV:** drag the ring in the preview to move it, or drag its edge to resize it. Shapes are circle, square or full screen, and you can save named regions. Large areas are scanned at half resolution to keep it fast.
* **Draw:** Box / Glow / Both / Corners / None, plus glow radius, thickness, labels and HUD.
  * **Colour-blind assist** recolours the detected colours with stripes, a solid fill or an outline. With the FOV set to full screen and "Show on main screen" on, it works over every app.
* **Tools:**
  * **Eyedropper:** a magnifier over the preview showing HEX / RGB / HSV and a plain colour name. Click a pixel to copy it; picks are kept in a history, and "Add as target" turns one into a colour to find.
  * **Contrast checker:** pick a text colour and a background colour from the screen to get the WCAG ratio with AA / AAA pass or fail, plus a live text sample.
  * **Palette:** the 8 main colours in the FOV, each with a share percentage, a name and "+ Add".
* **Setup:**
  * **Overlay:** draws on your real screen. It's click-through, stays on top and is hidden from the capture.
  * **Capture:** monitor or window, cursor on or off, and minimize while capturing.
  * **Auto-pause:** while the screen is locked, after the screen stops changing for a set time, or while a chosen app isn't focused. A tiny KWin script reports the focused app.
  * **Share profiles:** export or import a file, copy or paste a short `CSP1:` share code, or add the example profiles (traffic lights, status LEDs, red for colour-blind eyes, highlighter yellow).
  * **Look:** 18 UI Vault kits and 87 themes. The default is win98 controls on the paper theme.
* **NNMM (neural-network mouse):** choose a model file, start capture, turn on **Enable** (F7, or **NN mouse** in the bottom bar), then **hold right click in any window**: OpenCV finds the boxes on the captured screen, the model (run by OpenCV DNN) decides how to move, and a virtual mouse moves. Let go to stop; every hold starts fresh. **Aim with**: **Pointer (2D)** moves the desktop pointer onto the box; **Crosshair (3D)** turns a 3D game's camera until the box is at the screen centre (aim-trainer models only, since a game locks the pointer). While it moves, the preview and overlay mark the locked box. **Go for** picks one of the Detect tab's colours (or any). **Hold right click on** picks a regular mouse's right button, a touchpad, or either; mice plugged in later are picked up within 2 s. On a touchpad, press and hold in the bottom-right corner (or press with two fingers).
  * **Aim-trainer models** (the `expert-…` ones, from the aim-trainer project) run with its `aimer/` runtime, bundled here: an estimator that works out the pointer's sensitivity, acceleration and delay by itself (it makes a few small test moves first), snap and track controllers, and the network correcting the plan. In Crosshair (3D) it gets the box as seen, with the screen centre as the crosshair, exactly as it was trained; in Pointer (2D) it gets the box relative to the pointer, so the pointer plays the crosshair. The network runs in OpenCV DNN.
  * Plain models are `.npz` weight files (OpenCV DNN runs them; they're turned into an ONNX network in memory). Nothing is built in: **Choose model file…** picks one, used where it is (not copied). `train.py OUT.npz` makes a plain example. Plain models take 6 inputs (target offset x/y, target speed x/y, current speed x/y, in screen heights) → hidden layers → 2 outputs (speed x/y). Weights can be saved as `W0, b0, W1, b1…` (or `W1…`, `w_0…`, `weights_0…`), PyTorch-style `fc1.weight / fc1.bias`, unnamed `np.savez(w1, b1, w2, b2…)`, or object arrays `weights` / `biases`; in × out or out × in. Optional keys `activation` (tanh, relu, sigmoid; default tanh) and `output_activation` (default linear). Models that don't steer towards targets are refused, with the reason. `.onnx` files with the same shape also work.
  * `./setup.sh` adds a udev rule so the logged-in user can create the virtual mouse (`/dev/uinput`) and read mouse/touchpad buttons (not keyboards). On a touchpad, a press in the bottom-right corner or a two-finger press counts as right click.

**Self-check:** KWin rules normally hide the overlay and the ChromaScope window from the capture. Where they don't work (older KWin, X11, other desktops) the app would detect its own drawings and loop: boxes inside boxes that shrink and grow. So when capture starts (and when the overlay is turned on) it briefly draws a tiny green/magenta mark on the overlay and in its window and looks for it in the captured frames. If the overlay is seen, it draws in white (detection needs colour) without the red glow; if the window is seen, that area is cut out of detection. A badge under the preview says when either is active.

**Undo / redo** (↶ ↷, Ctrl+Z, Ctrl+Shift+Z or Ctrl+Y) covers every detection, FOV and drawing change, and each tab has **Reset this tab**.

Keys: F8 starts or stops capture, F9 turns the overlay on or off, F10 switches the preview between Screen, FOV zoom and Mask, and Esc leaves sampling or the eyedropper.

## How it works

* **Capture:** xdg-desktop-portal ScreenCast → PipeWire → GStreamer. KDE asks which screen to share the first time and the app remembers your choice.
* **No feedback:** KWin rules (`chromascope-overlay`, `chromascope-window`) keep both the overlay and the app window out of the screen capture. They're on screen, but the detector never sees its own boxes, swatches or preview.

Settings are saved in `~/.config/chromascope/` (`settings.json`, `profiles/`).

Files: `capture.py` (portal + stream), `detect.py` (HSV masks → boxes, sampling, glow, assist layer), `tools.py` (colour info and names, contrast, palette, share codes), `extras.py` (undo, auto-pause watchers, tour, example profiles), `nnmm.py` (live mouse controller), `nnio.py` (.npz model loading → ONNX → OpenCV DNN), `vmouse.py` (uinput virtual mouse), `inputwatch.py` (right-click hold), `train.py` (makes a plain example model), `aimer/` (aim-trainer runtime) and `app.py` (UI, overlay).
