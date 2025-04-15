#!/bin/bash
# Setup script for Render deployment

# Install specific NumPy version first to avoid compatibility issues
pip install numpy==1.24.3

# Install other requirements
pip install -r requirements.txt

# Create cache directory for numba (if needed)
mkdir -p /tmp/numba_cache

# Set environment variables
export NUMBA_CACHE_DIR=/tmp/numba_cache

echo "Setup complete. NumPy version:"
pip show numpy | grep Version

echo "PyTorch version:"
pip show torch | grep Version