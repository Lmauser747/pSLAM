#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${PROJECT_ROOT}/build"
TEST_DIR="${BUILD_DIR}/tests"

if [ ! -d "${BUILD_DIR}" ]; then
    echo "Error: Build directory not found. Please compile the project first via ./build.sh"
    exit 1
fi

FILTER=""
VERBOSE=""
USE_GDB="OFF"
GDB_BATCH="OFF"
TARGET_BIN=""

usage() {
    cat <<EOF
Usage: $0 [OPTIONS]

Testing Modes:
  -u, --unit               Run only unit tests (HAL, Frontend, Backend)
  -s, --system             Run only multi-threaded system integration tests
  -f, --filter PATTERN     Filter tests by regex (CTest) or GTest filter (GDB)
  -v, --verbose            Show verbose test output

GDB Debugging:
  -g, --gdb [BIN]          Run test binary inside an interactive GDB session.
                           If BIN is omitted, an interactive menu will display available targets.
  -b, --batch              Use with --gdb to run non-interactively and print a backtrace on crash.

Examples:
  $0 -u                                          # Run all unit tests via CTest
  $0 -g unit_tests_hal -f "DepthStreamTest.*"    # Debug a specific GTest fixture in GDB
  $0 -g system_tests --batch                     # Run system tests in GDB and dump stack trace on crash
EOF
    exit 0
}

# Parse command line options
while [[ $# -gt 0 ]]; do
    case "$1" in
        -u|--unit)
            FILTER="unit_tests_"
            shift
            ;;
        -s|--system)
            FILTER="system_tests"
            shift
            ;;
        -f|--filter)
            FILTER="$2"
            shift 2
            ;;
        -v|--verbose)
            VERBOSE="--verbose"
            shift
            ;;
        -g|--gdb)
            USE_GDB="ON"
            if [[ $# -gt 1 && ! "$2" =~ ^- ]]; then
                TARGET_BIN="$2"
                shift 2
            else
                shift 1
            fi
            ;;
        -b|--batch)
            GDB_BATCH="ON"
            shift
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo "Unknown option: $1"
            usage
            ;;
    esac
done

# --- GDB Execution Mode ---
if [ "${USE_GDB}" = "ON" ]; then
    if ! command -v gdb &>/dev/null; then
        echo "Error: 'gdb' is not installed. Install it via: sudo apt install gdb"
        exit 1
    fi

    # Discover all compiled test binaries under build/tests
    mapfile -t AVAILABLE_TESTS < <(find "${TEST_DIR}" -maxdepth 2 -type f -executable ! -name "*.sh" ! -name "*.cmake" 2>/dev/null | sort)

    if [ ${#AVAILABLE_TESTS[@]} -eq 0 ]; then
        echo "Error: No test executables found in ${TEST_DIR}. Ensure you built the project with tests enabled."
        exit 1
    fi

    EXEC_PATH=""

    # Resolve executable path if target name was provided
    if [ -n "${TARGET_BIN}" ]; then
        for test_exec in "${AVAILABLE_TESTS[@]}"; do
            if [[ "$(basename "${test_exec}")" == "${TARGET_BIN}" ]]; then
                EXEC_PATH="${test_exec}"
                break
            fi
        done
        if [ -z "${EXEC_PATH}" ]; then
            echo "Error: Target executable '${TARGET_BIN}' not found in ${TEST_DIR}."
            exit 1
        fi
    elif [ ${#AVAILABLE_TESTS[@]} -eq 1 ]; then
        EXEC_PATH="${AVAILABLE_TESTS[0]}"
    else
        # Interactive selection menu if multiple binaries exist and none was specified
        echo "Select test executable to debug under GDB:"
        select selected_test in "${AVAILABLE_TESTS[@]}"; do
            if [ -n "${selected_test}" ]; then
                EXEC_PATH="${selected_test}"
                break
            else
                echo "Invalid selection. Please choose a valid number."
            fi
        done
    fi

    echo "Debugging target: ${EXEC_PATH}"

    # Prepare GoogleTest filter arguments
    GTEST_ARGS=()
    if [ -n "${FILTER}" ]; then
        # Strip CTest prefix wildcard if the user passed standard test names
        GTEST_FILTER_CLEAN="${FILTER#unit_tests_.}"
        GTEST_FILTER_CLEAN="${GTEST_FILTER_CLEAN#system_tests_.}"
        GTEST_ARGS+=("--gtest_filter=${GTEST_FILTER_CLEAN}")
        echo "Applying filter: ${GTEST_ARGS[*]}"
    fi

    cd "$(dirname "${EXEC_PATH}")"

    if [ "${GDB_BATCH}" = "ON" ]; then
        echo "Running non-interactive GDB post-mortem crash trace..."
        # Runs the binary; prints stack backtrace if a signal (SIGSEGV, SIGABRT) occurs
        gdb -quiet -batch \
            -return-child-result \
            -ex "set print thread-events off" \
            -ex "run" \
            -ex "thread apply all bt full" \
            --args "${EXEC_PATH}" "${GTEST_ARGS[@]}"
    else
        echo "Starting interactive GDB session..."
        gdb -quiet \
            -ex "set print thread-events off" \
            -ex "set pagination off" \
            --args "${EXEC_PATH}" "${GTEST_ARGS[@]}"
    fi
    exit 0
fi

# --- Standard CTest Execution Mode ---
cd "${BUILD_DIR}"

CTEST_ARGS=(
    --output-on-failure
    --force-new-ctest-process
)

if [ -n "${VERBOSE}" ]; then
    CTEST_ARGS+=("${VERBOSE}")
fi

if [ -n "${FILTER}" ]; then
    echo "Running tests matching filter pattern: '${FILTER}'..."
    ctest -R "${FILTER}" "${CTEST_ARGS[@]}"
else
    echo "Running all registered test suites..."
    ctest "${CTEST_ARGS[@]}"
fi