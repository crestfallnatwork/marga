pictor_add_shader(pictor_scan_shader shaders/scan.slang INCLUDE_DIRS shaders)
pictor_add_shader(pictor_sort_shader shaders/sort.slang INCLUDE_DIRS shaders)

add_library(pictor_algorithms
    src/scan.cpp
    src/sort.cpp
)

target_include_directories(pictor_algorithms
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

target_link_libraries(pictor_algorithms
    PUBLIC
        pictor_core
        pictor_scan_shader
        pictor_sort_shader
)

if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
    add_executable(pictor_scan_test test/scan_test.cpp)
    target_link_libraries(pictor_scan_test
        PRIVATE
            pictor_algorithms
    )
    add_test(NAME scan COMMAND pictor_scan_test)

    add_executable(pictor_sort_test test/sort_test.cpp)
    target_link_libraries(pictor_sort_test
        PRIVATE
            pictor_algorithms
    )
    add_test(NAME sort COMMAND pictor_sort_test)
endif()
