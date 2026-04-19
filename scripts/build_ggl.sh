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

# Force GGL to embed the project's venv Python (which has wandb/torch) rather
# than the system interpreter that CMake would otherwise pick by version.
PY_EXE="${GGL_PYTHON:-$ROOT/.venv/bin/python}"
if [[ -x "$PY_EXE" ]]; then
    CMAKE_EXTRA+=("-DPython_EXECUTABLE=$PY_EXE" "-DPYTHON_EXECUTABLE=$PY_EXE")
    echo "[build_ggl] Python_EXECUTABLE: $PY_EXE"
else
    echo "[build_ggl] WARNING: venv python not found at $PY_EXE — CMake will auto-pick. Set GGL_PYTHON to override."
fi

echo "[build_ggl] build dir: $BUILD_DIR"
echo "[build_ggl] GGL_ROOT: $GGL_ROOT"
echo "[build_ggl] BUILD_TYPE: $BUILD_TYPE"
echo "[build_ggl] TORCH_PATH: ${TORCH_PATH:-<unset — CMake will try to find Torch on its own>}"

cmake -S "$ROOT" -B "$BUILD_DIR" \
    -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
    -DGGL_ROOT="$GGL_ROOT" \
    "${CMAKE_EXTRA[@]}"

cmake --build "$BUILD_DIR" --parallel

echo
echo "[build_ggl] binary: $BUILD_DIR/rl_rl_ggl"
