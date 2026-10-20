#!/usr/bin/env bash
# ChromaScope setup: installs the system packages + Python dependencies and adds ChromaScope to the app menu.
#   ./setup.sh             install / update
#   ./setup.sh --uninstall remove the menu entry, icon, KWin rules and udev rule (keeps this folder and your settings)
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
APPS="$HOME/.local/share/applications"
ICONS="$HOME/.local/share/icons/hicolor/scalable/apps"
say() { printf '\033[1m==>\033[0m %s\n' "$*"; }

refresh_menu() {
    command -v update-desktop-database >/dev/null && update-desktop-database "$APPS" 2>/dev/null || true
    command -v kbuildsycoca6 >/dev/null && kbuildsycoca6 >/dev/null 2>&1 || true
}

if [[ "${1:-}" == "--uninstall" ]]; then
    rm -f "$APPS/chromascope.desktop" "$ICONS/chromascope.svg"
    [[ -f /etc/udev/rules.d/70-chromascope.rules ]] && sudo rm -f /etc/udev/rules.d/70-chromascope.rules && sudo udevadm control --reload
    if command -v kwriteconfig6 >/dev/null; then
        rules=$(kreadconfig6 --file kwinrulesrc --group General --key rules | tr ',' '\n' | { grep -v '^chromascope-' || true; } | paste -sd, -)
        kwriteconfig6 --file kwinrulesrc --group General --key rules "$rules"
        kwriteconfig6 --file kwinrulesrc --group General --key count "$(awk -F, '{print ($0=="")?0:NF}' <<<"$rules")"
        sed -i '/^\[chromascope-/,/^$/d' "$HOME/.config/kwinrulesrc" 2>/dev/null || true
        dbus-send --session --type=signal /KWin org.kde.KWin.reloadConfig || true
    fi
    refresh_menu; say "ChromaScope removed from the app menu."; exit 0
fi

# 1. System packages: GStreamer PipeWire source, PyGObject, the KDE screen-share portal, kreadconfig6/kwriteconfig6
say "Checking system packages"
if command -v pacman >/dev/null; then
    need=()
    for p in python python-gobject gstreamer gst-plugins-base gst-plugin-pipewire pipewire xdg-desktop-portal-kde kconfig; do
        pacman -Q "$p" >/dev/null 2>&1 || need+=("$p")
    done
    if ((${#need[@]})); then say "Installing: ${need[*]}"; sudo pacman -S --needed --noconfirm "${need[@]}"; fi
elif command -v apt-get >/dev/null; then
    sudo apt-get install -y python3-venv python3-gi gir1.2-gstreamer-1.0 gstreamer1.0-pipewire gstreamer1.0-plugins-base \
        xdg-desktop-portal-kde libkf6config-bin
elif command -v dnf >/dev/null; then
    sudo dnf install -y python3-gobject gstreamer1 gstreamer1-plugins-base pipewire-gstreamer xdg-desktop-portal-kde kf6-kconfig
else
    say "Unknown distro: install PyGObject, GStreamer (+ pipewire plugin) and xdg-desktop-portal-kde yourself."
fi

# 2. Python environment (system site-packages so the distro's PyGObject/GStreamer bindings are visible)
say "Setting up the Python environment"
[[ -x "$DIR/.venv/bin/python" ]] || python3 -m venv --system-site-packages "$DIR/.venv"
"$DIR/.venv/bin/pip" install -q --upgrade pip
"$DIR/.venv/bin/pip" install -q -r "$DIR/requirements.txt"
if [[ -d "$HOME/Projects/ui-vault/uivault3" ]]; then
    "$DIR/.venv/bin/pip" install -q -e "$HOME/Projects/ui-vault/uivault3"
else
    "$DIR/.venv/bin/pip" install -q "git+https://github.com/thenameiscobmarley/ui-vault#subdirectory=uivault3"
fi

# 3. Device access for the NNMM tab: create the virtual mouse (/dev/uinput) and read mouse / touchpad buttons for
#    hold-right-click-to-aim. uaccess = only the user logged in at the screen; keyboards stay off limits.
RULES=/etc/udev/rules.d/70-chromascope.rules
if [[ ! -f "$RULES" ]]; then
    say "Allowing ChromaScope to use a virtual mouse and read mouse buttons (udev rule, asks for your password)"
    sudo tee "$RULES" >/dev/null <<'UDEV'
# ChromaScope NNMM: virtual mouse + read-only mouse/touchpad buttons for the logged-in user
KERNEL=="uinput", SUBSYSTEM=="misc", TAG+="uaccess", OPTIONS+="static_node=uinput"
SUBSYSTEM=="input", KERNEL=="event*", ENV{ID_INPUT_MOUSE}=="1", TAG+="uaccess"
SUBSYSTEM=="input", KERNEL=="event*", ENV{ID_INPUT_TOUCHPAD}=="1", TAG+="uaccess"
UDEV
    sudo udevadm control --reload && sudo udevadm trigger --subsystem-match=input --subsystem-match=misc
fi

# 4. App menu entry + icon
say "Adding ChromaScope to the app menu"
mkdir -p "$APPS" "$ICONS"
cp "$DIR/assets/chromascope.svg" "$ICONS/chromascope.svg"
chmod +x "$DIR/run.sh"
cat > "$APPS/chromascope.desktop" <<DESKTOP
[Desktop Entry]
Type=Application
Name=ChromaScope
GenericName=Screen Colour Detector
Comment=Live colour detection on your screen with FOV, boxes, glow and an on-screen overlay
Exec=$DIR/run.sh
Icon=chromascope
Terminal=false
Categories=Graphics;Utility;
Keywords=colour;color;detect;screen;overlay;
StartupWMClass=chromascope
DESKTOP
refresh_menu

# 5. Self-check
say "Checking"
"$DIR/.venv/bin/python" - <<'PY'
import gi; gi.require_version("Gst", "1.0")
from gi.repository import Gst; Gst.init(None)
import cv2, numpy, PySide6, uivault3
assert Gst.ElementFactory.find("pipewiresrc"), "GStreamer pipewiresrc plugin missing"
print(f"  OK: GStreamer {Gst.version_string()[10:]}, OpenCV {cv2.__version__}, PySide6 {PySide6.__version__}")
PY
say "Done. Open ChromaScope from the app menu, or run $DIR/run.sh"
