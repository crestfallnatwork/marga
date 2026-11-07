function(pictor_add_shader TARGET SLANG_FILE)
    get_filename_component(SLANG_NAME ${SLANG_FILE} NAME_WE)
    get_filename_component(SLANG_ABS ${SLANG_FILE} ABSOLUTE)

    set(GEN_SHADER_H "${CMAKE_CURRENT_BINARY_DIR}/${SLANG_NAME}.shader.h")
    set(GEN_SLANG_H "${CMAKE_CURRENT_BINARY_DIR}/${SLANG_NAME}.slang.h")
    set(GEN_DEPFILE "${CMAKE_CURRENT_BINARY_DIR}/${SLANG_NAME}.shader.d")

    set(SHADER_INCLUDE_DIRS "")
    set(EXPECT_INCLUDE_DIRS FALSE)
    foreach(ARG IN LISTS ARGN)
        if(ARG STREQUAL "INCLUDE_DIRS")
            set(EXPECT_INCLUDE_DIRS TRUE)
        elseif(EXPECT_INCLUDE_DIRS)
            if(IS_ABSOLUTE "${ARG}")
                list(APPEND SHADER_INCLUDE_DIRS "${ARG}")
            else()
                list(APPEND SHADER_INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/${ARG}")
            endif()
        else()
            message(FATAL_ERROR
                "pictor_add_shader(${TARGET}): unexpected argument '${ARG}'; expected INCLUDE_DIRS <dir>...")
        endif()
    endforeach()

    set(INCLUDE_FLAGS "")
    foreach(INCLUDE_DIR IN LISTS SHADER_INCLUDE_DIRS)
        list(APPEND INCLUDE_FLAGS -I "${INCLUDE_DIR}")
    endforeach()

    set(DEPFILE_ARGS "")
    set(DEPFILE_OPTION "")
    if(CMAKE_GENERATOR MATCHES "Ninja|Makefiles")
        list(APPEND DEPFILE_ARGS --depfile "${GEN_DEPFILE}")
        set(DEPFILE_OPTION DEPFILE "${GEN_DEPFILE}")
    endif()

    add_custom_command(
        OUTPUT ${GEN_SHADER_H} ${GEN_SLANG_H}
        COMMAND $<TARGET_FILE:pictor_reflect> ${SLANG_ABS} --outdir ${CMAKE_CURRENT_BINARY_DIR}
                ${INCLUDE_FLAGS} ${DEPFILE_ARGS}
        DEPENDS ${SLANG_ABS} pictor_reflect
        ${DEPFILE_OPTION}
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMENT "Generating C++ headers for shader ${SLANG_NAME}"
    )

    add_library(${TARGET} INTERFACE)
    target_sources(${TARGET} INTERFACE ${GEN_SHADER_H} ${GEN_SLANG_H})
    target_include_directories(${TARGET} INTERFACE ${CMAKE_CURRENT_BINARY_DIR})
    target_link_libraries(${TARGET} INTERFACE pictor_core)
endfunction()
