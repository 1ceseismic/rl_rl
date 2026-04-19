#!/usr/bin/env bash
# Build the GGL port. Requires LibTorch (CPU or CUDA) extracted somewhere.
# Usage:
#   scripts/build_ggl.sh            # release build into build-ggl/
#   TORCH_PATH=/opt/libtorch scripts/build_ggl.sh
#   BUILD_TYPE=Debug scripts/build_ggl.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-$ROOT/build-ggl}"
BUILD_TYPE="${BUILD_TYPE:-Release}"
GGL_ROOT="${GGL_ROOT:-$ROOT/../GigaLearnCPP-Leak}"

CMAKE_EXTRA=()
if [[ -n "${TORCH_PATH:-}" ]]; then
    CMAKE_EXTRA+=("-DCMAKE_PREFIX_PATH=$TORCH_PATH")
fi

# Force GGL to embed the project's venv Python (which has wandb, torch,
# rlviser_py installed) rather than the system interpreter that CMake
# would otherwise pick by version.
#
# GGL's CMakeLists uses legacy `find_package(PythonLibs REQUIRED)` which
# does NOT honor Python_EXECUTABLE, so we have to pass PYTHON_INCLUDE_DIR
# and PYTHON_LIBRARY explicitly. We derive them from the venv Python's
# base_prefix (for uv-managed Pythons, that's where the real .so lives).
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
        echo "[build_ggl] WARNING: venv python found but libs/includes missing at $PY_BASE; CMake will auto-pick."
    fi
else
    echo "[build_ggl] WARNING: venv python not found at $PY_EXE — CMake will auto-pick. Set GGL_PYTHON to override."
fi

echo "[build_ggl] build dir: $BUILD_DIR"
echo "[build_ggl] GGL_ROOT: $GGL_ROOT"
echo "[build_ggl] BUILD_TYPE: $BUILD_TYPE"
echo "[build_ggl] TORCH_PATH: ${TORCH_PATH:-<unset — CMake will try to find Torch on its own>}"

# Skip the CMake configure step on warm rebuilds — it costs ~1s even when
# nothing changed. CMake's generated Makefile will re-run configure itself
# if any CMakeLists.txt has been touched since the cache was written.
if [[ ! -f "$BUILD_DIR/CMakeCache.txt" ]]; then
    cmake -S "$ROOT" -B "$BUILD_DIR" \
        -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
        -DGGL_ROOT="$GGL_ROOT" \
        "${CMAKE_EXTRA[@]}"
fi

cmake --build "$BUILD_DIR" --parallel

# Override GGL's default RocketSimVis render receiver with our rlviser
# forwarder. GGL's own `configure_file(... COPY)` writes its JSON-to-UDP-9273
# receiver into build-ggl/python_scripts/; we overwrite it after build so
# render mode drives rlviser (Bevy) instead of a tool we don't have installed.
if [[ -f "$ROOT/ggl/python_scripts/render_receiver.py" ]]; then
    mkdir -p "$BUILD_DIR/python_scripts"
    cp "$ROOT/ggl/python_scripts/render_receiver.py" "$BUILD_DIR/python_scripts/render_receiver.py"
fi

# Symlink rlviser into build-ggl/ so rlviser_py finds it via ./rlviser (it
# searches CWD first, then PATH). scripts/go also adds the repo root to PATH
# as a backup; either path works.
if [[ -x "$ROOT/rlviser" && ! -e "$BUILD_DIR/rlviser" ]]; then
    ln -s "$ROOT/rlviser" "$BUILD_DIR/rlviser"
fi

# Bump the binary's mtime so no-change invocations can short-circuit via a
# simple `find -newer` check. cmake --build only rewrites the binary when
# something actually linked; without this touch, every `scripts/go` would
# re-enter cmake just to confirm nothing's out of date.
[[ -x "$BUILD_DIR/rl_rl_ggl" ]] && touch "$BUILD_DIR/rl_rl_ggl"

echo
echo "[build_ggl] binary: $BUILD_DIR/rl_rl_ggl"
