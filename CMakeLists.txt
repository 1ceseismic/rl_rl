cmake_minimum_required(VERSION 3.18)
project(rl_rl_ggl CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# RocketSim TLS relocs
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(LIBRARY_OUTPUT_PATH "${CMAKE_BINARY_DIR}")
set(EXECUTABLE_OUTPUT_PATH "${CMAKE_BINARY_DIR}")

set(GGL_ROOT "${CMAKE_SOURCE_DIR}/../GigaLearnCPP-Leak" CACHE PATH "Path to GigaLearnCPP source root")

if(NOT EXISTS "${GGL_ROOT}/GigaLearnCPP/CMakeLists.txt")
    message(FATAL_ERROR "GGL_ROOT=${GGL_ROOT} does not contain GigaLearnCPP/. Set -DGGL_ROOT=/path/to/GigaLearnCPP-Leak.")
endif()

# resolves ../Framework.h include
include_directories(BEFORE SYSTEM "${GGL_ROOT}/GigaLearnCPP/RLGymCPP/src/RLGymCPP/Gamestates")

# OBSBuilder case shim
include_directories(BEFORE SYSTEM "${CMAKE_SOURCE_DIR}/ggl/_shim/_anchor")

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
