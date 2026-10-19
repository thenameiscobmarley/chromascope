#!/bin/sh
# ChromaScope — live colour detection on your screen (Wayland / KDE Plasma)
cd "$(dirname "$0")" && exec .venv/bin/python app.py "$@"
