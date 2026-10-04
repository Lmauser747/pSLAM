#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${PROJECT_ROOT}/build"
BENCH_DIR="${BUILD_DIR}/benchmarks"

# Ensure the benchmarks directory/target exists; compile if missing
if [ ! -d "${BENCH_DIR}" ]; then
    echo "Benchmarks not configured. Building with benchmarks enabled..."
    "${PROJECT_ROOT}/build.sh" --benchmarks
fi

# Locate all compiled benchmark binaries in the benchmarks folder
BENCH_BINARIES=$(find "${BENCH_DIR}" -maxdepth 1 -type f -executable)

if [ -z "${BENCH_BINARIES}" ]; then
    echo "No benchmark executables found in ${BENCH_DIR}. Rebuilding..."
    "${PROJECT_ROOT}/build.sh" --benchmarks
    BENCH_BINARIES=$(find "${BENCH_DIR}" -maxdepth 1 -type f -executable)
fi

echo "=== Running PoolSLAM Micro-Benchmarks ==="
for bench in ${BENCH_BINARIES}; do
    echo "Executing: $(basename "${bench}")"
    # Forward all CLI arguments passed to this script into the benchmark binary
    "${bench}" "$@"
done