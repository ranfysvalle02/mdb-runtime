#!/bin/bash
set -e
echo "=== Starting Click Tracker ===" >&2
echo "Working directory: $(pwd)" >&2
echo "Python path: $(which python)" >&2
echo "Python version: $(python --version)" >&2
echo "Files in /app/apps/click_tracker:" >&2
ls -la /app/apps/click_tracker/ >&2 || true
echo "=== Running Click Tracker ===" >&2
exec python -u /app/apps/click_tracker/web.py
