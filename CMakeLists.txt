cmake_minimum_required(VERSION 3.18)
project(rl_rl_ggl CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Output binaries to build/ so scripts/watch_ggl.py can find them predictably.
set(LIBRARY_OUTPUT_PATH "${CMAKE_BINARY_DIR}")
set(EXECUTABLE_OUTPUT_PATH "${CMAKE_BINARY_DIR}")

# Path to the bundled GGL + RLGymCPP + RocketSim tree (leaked source).
set(GGL_ROOT "${CMAKE_SOURCE_DIR}/../GigaLearnCPP-Leak" CACHE PATH "Path to GigaLearnCPP source root")

if(NOT EXISTS "${GGL_ROOT}/GigaLearnCPP/CMakeLists.txt")
    message(FATAL_ERROR "GGL_ROOT=${GGL_ROOT} does not contain GigaLearnCPP/. Set -DGGL_ROOT=/path/to/GigaLearnCPP-Leak.")
endif()

# ── GGL leaked-source include-path shim ────────────────────────────────
# RLGymCPP's CommonValues.h does `#include "../Framework.h"` from its own
# directory (src/RLGymCPP/), which GCC can't resolve (it would look for
# src/Framework.h but Framework.h actually lives at src/RLGymCPP/Framework.h).
# MSVC is more permissive — presumably the leaked source was only ever
# built on Windows. Rather than patching the out-of-repo GGL tree we add
# an include directory one level below where Framework.h lives, so the
# preprocessor's fallback search finds Gamestates/../Framework.h = the
# real Framework.h. `Gamestates/` is a real existing subdir of src/RLGymCPP/.
include_directories(BEFORE SYSTEM "${GGL_ROOT}/GigaLearnCPP/RLGymCPP/src/RLGymCPP/Gamestates")

# Case-mismatch shim for `../OBSBuilders/OBSBuilder.h` in EnvSet.h. See
# ggl/_shim/OBSBuilders/OBSBuilder.h for the forwarder header. Adding the
# _anchor/ subdir as an include path makes the preprocessor's fallback
# resolve `../OBSBuilders/...` relative to this anchor.
include_directories(BEFORE SYSTEM "${CMAKE_SOURCE_DIR}/ggl/_shim/_anchor")

# Build only the GigaLearnCPP library subtree (skips the RLBot example target).
add_subdirectory("${GGL_ROOT}/GigaLearnCPP" "${CMAKE_BINARY_DIR}/GigaLearnCPP")

file(GLOB_RECURSE RL_RL_GGL_SRC
    "${CMAKE_SOURCE_DIR}/ggl/*.cpp"
    "${CMAKE_SOURCE_DIR}/ggl/*.h"
)

add_executable(rl_rl_ggl ${RL_RL_GGL_SRC})
target_link_libraries(rl_rl_ggl PRIVATE GigaLearnCPP)
target_include_directories(rl_rl_ggl PRIVATE
    "${CMAKE_SOURCE_DIR}/ggl"
)
set_target_properties(rl_rl_ggl PROPERTIES LINKER_LANGUAGE CXX)
