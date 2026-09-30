#!/usr/bin/env bash
set -euo pipefail

echo "=== Setting up PoolSLAM Development Environment ==="

# 1. Update and install core build toolchain and dependencies
sudo apt-get update
sudo apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    ninja-build \
    git \
    pkg-config \
    libeigen3-dev \
    python3 \
    python3-dev \
    python3-pip \
    python3-venv

# 2. Configure Python Virtual Environment for Calibration & Tooling
VENV_DIR=".venv"
if [ ! -d "${VENV_DIR}" ]; then
    echo "Creating Python virtual environment in ${VENV_DIR}..."
    python3 -m venv "${VENV_DIR}"
fi

# Activate virtualenv and install math dependencies (strictly excluding OpenCV)
# shellcheck source=/dev/null
source "${VENV_DIR}/bin/activate"
pip install --upgrade pip
pip install numpy scipy pyyaml

echo "=== Setup Complete. Activate your environment with: source .venv/bin/activate ==="