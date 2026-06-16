function(pictor_add_shader TARGET SLANG_FILE)
    get_filename_component(SLANG_NAME ${SLANG_FILE} NAME_WE)
    get_filename_component(SLANG_ABS ${SLANG_FILE} ABSOLUTE)
    
    set(GEN_SHADER_H "${CMAKE_CURRENT_BINARY_DIR}/${SLANG_NAME}.shader.h")
    set(GEN_SLANG_H "${CMAKE_CURRENT_BINARY_DIR}/${SLANG_NAME}.slang.h")
    
    find_package(Python3 REQUIRED)
    
    # We assume reflect.py is installed in the same location as this cmake file or passed via a variable
    if(NOT PICTOR_REFLECT_PY)
        set(PICTOR_REFLECT_PY "${PICTOR_CORE_SOURCE_DIR}/reflect.py")
    endif()

    add_custom_command(
        OUTPUT ${GEN_SHADER_H} ${GEN_SLANG_H}
        COMMAND ${Python3_EXECUTABLE} ${PICTOR_REFLECT_PY} ${SLANG_ABS}
        DEPENDS ${SLANG_ABS} ${PICTOR_REFLECT_PY}
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMENT "Generating C++ headers for shader ${SLANG_NAME}"
    )
    
    add_library(${TARGET} INTERFACE)
    target_sources(${TARGET} INTERFACE ${GEN_SHADER_H} ${GEN_SLANG_H})
    target_include_directories(${TARGET} INTERFACE ${CMAKE_CURRENT_BINARY_DIR})
    target_link_libraries(${TARGET} INTERFACE pictor_core)
endfunction()
