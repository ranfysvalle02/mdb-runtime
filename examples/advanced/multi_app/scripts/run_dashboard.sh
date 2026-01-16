#!/bin/bash
set -e
echo "=== Starting Dashboard ===" >&2
echo "Working directory: $(pwd)" >&2
echo "Python path: $(which python)" >&2
echo "Python version: $(python --version)" >&2
echo "Files in /app/apps/dashboard:" >&2
ls -la /app/apps/dashboard/ >&2 || true
echo "=== Running Dashboard ===" >&2
exec python -u /app/apps/dashboard/web.py
