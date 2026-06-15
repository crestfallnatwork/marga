cmake_minimum_required(VERSION 3.20)
project(pictor_graphs)

set(CMAKE_CXX_STANDARD 20)

find_package(OpenMP REQUIRED)

add_subdirectory(pictor_core)

include_directories(include)
include_directories(external)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/generated)

# Shader generation
file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/generated)

find_program(GLSLC_EXECUTABLE NAMES glslc)
if(NOT GLSLC_EXECUTABLE)
    message(FATAL_ERROR "glslc not found, please install the Vulkan SDK")
endif()

# Find slangc from FetchContent or system
set(SLANGC_EXECUTABLE ${CMAKE_CURRENT_BINARY_DIR}/_deps/slang-src/bin/slangc)

function(add_slang_shader SHADER_FILE)
    get_filename_component(SHADER_NAME ${SHADER_FILE} NAME_WE)
    set(SPV_FILE ${CMAKE_CURRENT_BINARY_DIR}/generated/${SHADER_NAME}.spv)
    set(JSON_FILE ${CMAKE_CURRENT_BINARY_DIR}/generated/${SHADER_NAME}.json)
    set(LAYOUT_HEADER ${CMAKE_CURRENT_BINARY_DIR}/generated/${SHADER_NAME}.layout.hpp)
    set(HEADER_FILE ${CMAKE_CURRENT_BINARY_DIR}/generated/${SHADER_NAME}.spv.h)
    
    # 1. Compile Slang to SPIR-V and generate reflection JSON
    # We compile all entry points into a single SPIR-V container
    add_custom_command(
        OUTPUT ${SPV_FILE} ${JSON_FILE}
        COMMAND ${SLANGC_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/src/shaders/${SHADER_FILE} 
                -target spirv -profile spirv_1_5 
                -entry task -entry mesh -entry compute_indirect
                -o ${SPV_FILE} -reflection-json ${JSON_FILE}
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/shaders/${SHADER_FILE}
        COMMENT "Compiling ${SHADER_FILE} to SPIR-V and generating reflection"
    )

    # 2. Generate C++ Layout Header from JSON
    add_custom_command(
        OUTPUT ${LAYOUT_HEADER}
        COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/scripts/generate_layout.py ${JSON_FILE} ${LAYOUT_HEADER}
        DEPENDS ${JSON_FILE} ${CMAKE_CURRENT_SOURCE_DIR}/scripts/generate_layout.py
        COMMENT "Generating C++ layout header for ${SHADER_FILE}"
    )

    # 3. Package SPIR-V as C header
    add_custom_command(
        OUTPUT ${HEADER_FILE}
        COMMAND xxd -i ${SHADER_NAME}.spv ${HEADER_FILE}
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/generated
        DEPENDS ${SPV_FILE}
        COMMENT "Packaging ${SHADER_NAME}.spv as header"
    )

    # 4. Package raw Slang source as C header
    set(SOURCE_HEADER ${CMAKE_CURRENT_BINARY_DIR}/generated/${SHADER_NAME}.slang.h)
    add_custom_command(
        OUTPUT ${SOURCE_HEADER}
        COMMAND xxd -i ${SHADER_FILE} ${SOURCE_HEADER}
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/src/shaders
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/shaders/${SHADER_FILE}
        COMMENT "Packaging ${SHADER_FILE} source as header"
    )

    set(GENERATED_SHADERS ${GENERATED_SHADERS} ${LAYOUT_HEADER} ${HEADER_FILE} ${SOURCE_HEADER} PARENT_SCOPE)
endfunction()

add_slang_shader(bfs.slang)

add_library(pictor_graphs src/csr.cpp src/bfs.cpp src/bfs_validation.cpp ${GENERATED_SHADERS})
target_link_libraries(pictor_graphs pictor_core OpenMP::OpenMP_CXX)


add_executable(test test.cpp)
target_link_libraries(test pictor_graphs pictor_core)

add_executable(bfs src/bfs_main.cpp)
target_link_libraries(bfs pictor_graphs pictor_core)
