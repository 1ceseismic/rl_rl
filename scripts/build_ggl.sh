#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-$ROOT/build-ggl}"
BUILD_TYPE="${BUILD_TYPE:-Release}"
GGL_ROOT="${GGL_ROOT:-$ROOT/../GigaLearnCPP-Leak}"

CMAKE_EXTRA=()
if [[ -n "${TORCH_PATH:-}" ]]; then
    CMAKE_EXTRA+=("-DCMAKE_PREFIX_PATH=$TORCH_PATH")
fi

# PythonLibs ignores Python_EXECUTABLE
PY_EXE="${GGL_PYTHON:-$ROOT/.venv/bin/python}"
if [[ -x "$PY_EXE" ]]; then
    PY_BASE=$("$PY_EXE" -c 'import sys; print(sys.base_prefix)')
    PY_VER=$("$PY_EXE" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
    PY_INCLUDE="$PY_BASE/include/python$PY_VER"
    PY_LIB="$PY_BASE/lib/libpython$PY_VER.so"
    if [[ -f "$PY_LIB" && -d "$PY_INCLUDE" ]]; then
        CMAKE_EXTRA+=(
            "-DPython_EXECUTABLE=$PY_EXE"
            "-DPYTHON_EXECUTABLE=$PY_EXE"
            "-DPYTHON_INCLUDE_DIR=$PY_INCLUDE"
            "-DPYTHON_LIBRARY=$PY_LIB"
        )
        echo "[build_ggl] Python: $PY_EXE (py$PY_VER, libs at $PY_LIB)"
    else
        echo "[build_ggl] WARNING: python libs missing at $PY_BASE"
    fi
else
    echo "[build_ggl] WARNING: venv python not found at $PY_EXE"
fi

echo "[build_ggl] build dir: $BUILD_DIR"
echo "[build_ggl] GGL_ROOT: $GGL_ROOT"
echo "[build_ggl] BUILD_TYPE: $BUILD_TYPE"
echo "[build_ggl] TORCH_PATH: ${TORCH_PATH:-unset}"

if [[ ! -f "$BUILD_DIR/CMakeCache.txt" ]]; then
    cmake -S "$ROOT" -B "$BUILD_DIR" \
        -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
        -DGGL_ROOT="$GGL_ROOT" \
        "${CMAKE_EXTRA[@]}"
fi

cmake --build "$BUILD_DIR" --parallel

if [[ -d "$ROOT/ggl/python_scripts" ]]; then
    mkdir -p "$BUILD_DIR/python_scripts"
    cp "$ROOT/ggl/python_scripts/"*.py "$BUILD_DIR/python_scripts/"
fi

if [[ -x "$ROOT/rlviser" && ! -e "$BUILD_DIR/rlviser" ]]; then
    ln -s "$ROOT/rlviser" "$BUILD_DIR/rlviser"
fi

# mtime for go skip check
[[ -x "$BUILD_DIR/rl_rl_ggl" ]] && touch "$BUILD_DIR/rl_rl_ggl"

echo
echo "[build_ggl] binary: $BUILD_DIR/rl_rl_ggl"
