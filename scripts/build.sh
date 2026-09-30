#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${PROJECT_ROOT}/build"

# Default configurations
BUILD_TYPE="Release"
BUILD_TESTS="ON"
BUILD_BENCH="OFF"
BUILD_PYTHON="OFF"
SANITIZE="OFF"
CLEAN="OFF"
JOBS=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)

usage() {
    echo "Usage: $0 [OPTIONS]"
    echo "Options:"
    echo "  -d, --debug        Build with Debug symbols"
    echo "  -r, --release      Build in Release mode (default)"
    echo "  -c, --clean        Clean build directory before compiling"
    echo "  -s, --sanitize     Enable AddressSanitizer and UndefinedBehaviorSanitizer"
    echo "  -m, --benchmarks   Enable micro-benchmark targets"
    echo "  -p, --python       Enable Pybind11 Python bindings"
    echo "  --no-tests         Disable building test suites"
    echo "  -j, --jobs N       Compile using N parallel threads (default: ${JOBS})"
    echo "  -h, --help         Display this help message"
    exit 0
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -d|--debug)      BUILD_TYPE="Debug"; shift ;;
        -r|--release)    BUILD_TYPE="Release"; shift ;;
        -c|--clean)      CLEAN="ON"; shift ;;
        -s|--sanitize)   SANITIZE="ON"; BUILD_TYPE="Debug"; shift ;;
        -m|--benchmarks) BUILD_BENCH="ON"; shift ;;
        -p|--python)     BUILD_PYTHON="ON"; shift ;;
        --no-tests)      BUILD_TESTS="OFF"; shift ;;
        -j|--jobs)       JOBS="$2"; shift 2 ;;
        -h|--help)       usage ;;
        *) echo "Unknown option: $1"; usage ;;
    esac
done

if [ "${CLEAN}" = "ON" ] && [ -d "${BUILD_DIR}" ]; then
    echo "Cleaning build directory: ${BUILD_DIR}"
    rm -rf "${BUILD_DIR}"
fi

CMAKE_FLAGS=(
    -G Ninja
    -DCMAKE_BUILD_TYPE="${BUILD_TYPE}"
    -DBUILD_TESTING="${BUILD_TESTS}"
    -DBUILD_BENCHMARKS="${BUILD_BENCH}"
    -DBUILD_PYTHON_BINDINGS="${BUILD_PYTHON}"
)

if [ "${SANITIZE}" = "ON" ]; then
    echo "Enabling ASan/UBSan memory diagnostics..."
    CMAKE_FLAGS+=(
        -DCMAKE_CXX_FLAGS="-fsanitize=address,undefined -fno-omit-frame-pointer"
    )
fi

echo "Configuring CMake with Ninja (${BUILD_TYPE})..."
cmake -B "${BUILD_DIR}" -S "${PROJECT_ROOT}" "${CMAKE_FLAGS[@]}"

echo "Compiling targets with ${JOBS} parallel jobs..."
cmake --build "${BUILD_DIR}" --parallel "${JOBS}"