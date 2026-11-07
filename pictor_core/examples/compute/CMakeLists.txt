pictor_add_shader(example_compute_shader add.slang)

add_executable(example_compute add.cpp)
target_link_libraries(example_compute
    PRIVATE
        pictor_core
        example_compute_shader
)
add_test(NAME example_compute COMMAND example_compute)
