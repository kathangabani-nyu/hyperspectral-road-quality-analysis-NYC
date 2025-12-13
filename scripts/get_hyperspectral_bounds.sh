#!/bin/bash
# Shell wrapper for get_hyperspectral_bounds.py
# 
# Usage:
#   ./scripts/get_hyperspectral_bounds.sh [--config advanced_config] [--window x y width height]
#
# This script requires Python with rasterio installed.
# Install rasterio: pip install rasterio

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Change to project root
cd "$PROJECT_ROOT"

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    if ! command -v python &> /dev/null; then
        echo "ERROR: Python not found. Please install Python 3."
        exit 1
    else
        PYTHON_CMD="python"
    fi
else
    PYTHON_CMD="python3"
fi

# Check if virtual environment exists and activate it
if [ -d "venv" ]; then
    if [ -f "venv/bin/activate" ]; then
        source venv/bin/activate
    elif [ -f "venv/Scripts/activate" ]; then
        source venv/Scripts/activate
    fi
fi

# Run the Python script with all arguments
"$PYTHON_CMD" "$SCRIPT_DIR/get_hyperspectral_bounds.py" "$@"

