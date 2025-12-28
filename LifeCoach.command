#!/bin/bash

# ============================================================
# LifeCoach AI - Launcher Script
# Double-click to update and run the app
# ============================================================

# Change to app directory
cd "$(dirname "$0")"

echo "🔄 Checking for updates..."
git pull --quiet origin main

echo "📦 Syncing dependencies..."
uv sync --quiet 2>/dev/null || echo "Dependencies up to date"

echo "🚀 Launching Life Coach AI..."
echo "   (Close this terminal window to stop the app)"
echo ""

uv run streamlit run app.py
