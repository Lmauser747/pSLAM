#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${PROJECT_ROOT}/build"

if [ ! -d "${BUILD_DIR}" ]; then
    echo "Error: Build directory not found. Please run ./build.sh first."
    exit 1
fi

FILTER=""
VERBOSE=""

usage() {
    echo "Usage: $0 [OPTIONS]"
    echo "Options:"
    echo "  -u, --unit         Run only component unit tests (HAL, Frontend, Backend)"
    echo "  -s, --system       Run only multi-threaded system integration tests"
    echo "  -f, --filter REGEX Run tests matching custom regular expression"
    echo "  -v, --verbose      Show detailed output on failure"
    echo "  -h, --help         Show this help message"
    exit 0
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        -u|--unit)    FILTER="unit_tests_"; shift ;;
        -s|--system)  FILTER="system_tests"; shift ;;
        -f|--filter)  FILTER="$2"; shift 2 ;;
        -v|--verbose) VERBOSE="--verbose"; shift ;;
        -h|--help)    usage ;;
        *) echo "Unknown option: $1"; usage ;;
    esac
done

cd "${BUILD_DIR}"

CTEST_ARGS=(
    --output-on-failure
    --force-new-ctest-process
)

if [ -n "${VERBOSE}" ]; then
    CTEST_ARGS+=("${VERBOSE}")
fi

if [ -n "${FILTER}" ]; then
    echo "Running tests matching regex pattern: '${FILTER}'..."
    ctest -R "${FILTER}" "${CTEST_ARGS[@]}"
else
    echo "Running all registered test suites..."
    ctest "${CTEST_ARGS[@]}"
fi