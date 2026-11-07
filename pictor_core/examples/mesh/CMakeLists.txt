pictor_add_shader(example_mesh_shader tile.slang)

add_executable(example_mesh tile.cpp)
target_link_libraries(example_mesh
    PRIVATE
        pictor_core
        example_mesh_shader
)
add_test(NAME example_mesh COMMAND example_mesh)
