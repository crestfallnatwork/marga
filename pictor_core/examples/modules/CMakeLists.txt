pictor_add_shader(example_modules_shader mix.slang INCLUDE_DIRS ops)

add_executable(example_modules module.cpp)
target_link_libraries(example_modules
    PRIVATE
        pictor_core
        example_modules_shader
)
add_test(NAME example_modules COMMAND example_modules)
