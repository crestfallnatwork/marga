import json
import os
import subprocess
import sys
from collections import OrderedDict

def map_type(t):
    if isinstance(t, str):
        mapping = {
            'uint32': 'uint32_t', 'int32': 'int32_t',
            'uint64': 'uint64_t', 'int64': 'int64_t',
            'float32': 'float', 'float64': 'double',
            'uint': 'uint32_t', 'int': 'int32_t', 'float': 'float',
        }
        return mapping.get(t, t), None

    kind = t.get('kind')
    if kind == 'scalar':
        return map_type(t.get('scalarType'))
    elif kind in ['vector', 'array']:
        elem, _ = map_type(t.get('elementType'))
        return elem, t.get('elementCount')
    elif kind == 'struct':
        return t.get('name'), None
    elif kind == 'pointer':
        elem = t.get('elementType') or t.get('valueType')
        elem_type, _ = map_type(elem)
        return elem_type + "*", None
    return 'void*', None

def generate_struct(struct_name, fields):
    lines = [f"struct {struct_name} {{"]
    current_offset = 0
    for f in fields:
        binding = f.get('binding', {})
        offset = binding.get('offset', 0)
        size = binding.get('size', 0)
        if offset > current_offset:
            lines.append(f"    uint8_t _pad_{current_offset}[{offset - current_offset}];")
        base_type, count = map_type(f['type'])
        lines.append(f"    {base_type} {f['name']}{f'[{count}]' if count else ''};")
        current_offset = offset + size
    lines.append("};")
    return "\n".join(lines)

def extract_struct_deps(t):
    if isinstance(t, str):
        return [t]
    if not isinstance(t, dict):
        return []
    kind = t.get('kind')
    if kind == 'struct':
        n = t.get('name')
        return [n] if n else []
    elif kind in ['constantBuffer', 'parameterBlock', 'array', 'vector', 'pointer']:
        return extract_struct_deps(t.get('elementType') or t.get('valueType'))
    return []

def main():
    if len(sys.argv) < 2:
        print("Usage: python reflect.py <input.slang>")
        sys.exit(1)

    input_file = sys.argv[1]
    base_name = os.path.basename(input_file).split('.')[0]
    output_json = f"{base_name}.reflection.json"
    
    subprocess.run(["slangc", input_file, "-target", "spirv", "-fspv-reflect", "-DWAVE_SIZE=32", "-reflection-json", output_json, "-o", f"{base_name}.spv"], check=True)

    with open(output_json, 'r') as f:
        reflection = json.load(f)
    with open(input_file, 'r') as f:
        slang_source = f.read()

    # Parse explicit @bind annotations from source
    explicit_binds = {}
    lines = slang_source.split('\n')
    import re
    for i, line in enumerate(lines):
        # Strictly match only lines that are exactly an @bind annotation comment
        match = re.match(r'^\s*//\s*@bind\(\s*([a-zA-Z0-9_]+)\s*\)', line)
        if match:
            ep_name = match.group(1)
            # Scan forward up to 10 lines to find the uniform declaration
            for j in range(i+1, min(i+10, len(lines))):
                uniform_match = re.search(r'uniform\s+([a-zA-Z0-9_]+)\s+([a-zA-Z0-9_]+)', lines[j])
                if uniform_match:
                    struct_type = uniform_match.group(1)
                    explicit_binds[ep_name] = struct_type
                    break

    # 1. First Pass: Collect all struct definitions globally
    struct_defs = {}
    def gather_structs(t):
        if isinstance(t, list):
            for item in t: gather_structs(item)
        elif isinstance(t, dict):
            if t.get('kind') == 'struct' and t.get('name'):
                struct_defs[t['name']] = t.get('fields', [])
            for v in t.values():
                gather_structs(v)
                
    gather_structs(reflection)

    # 2. Map global variables to their types
    var_to_type = {}
    for param in reflection.get('parameters', []):
        name = param.get('name')
        t = param.get('type', {})
        if t.get('kind') in ['constantBuffer', 'parameterBlock', 'pointer']:
             t = t.get('elementType') or t.get('valueType')
             if isinstance(t, str):
                 var_to_type[name] = t
                 continue
        if isinstance(t, dict) and t.get('kind') == 'struct':
            var_to_type[name] = t.get('name')

    # 3. Topological sort using the full definitions
    structs_to_gen = OrderedDict()
    visited = set()
    visiting = set()

    def add_struct(name):
        if not name or name in visited or name not in struct_defs:
            return
        if name in visiting:
            return # Cycle detected (e.g. self-referential pointers), ignore
            
        visiting.add(name)
        fields = struct_defs[name]
        
        # Add dependencies first
        for f in fields:
            deps = extract_struct_deps(f.get('type'))
            for dep in deps:
                add_struct(dep)
                
        visiting.remove(name)
        visited.add(name)
        structs_to_gen[name] = fields

    # We only want to generate structs that are actually used in parameters or entry points
    roots = set()
    def find_roots(t):
        if isinstance(t, list):
            for item in t: find_roots(item)
        elif isinstance(t, dict):
            deps = extract_struct_deps(t)
            for d in deps:
                roots.add(d)
                
    for param in reflection.get('parameters', []):
        find_roots(param.get('type'))
    for ep in reflection.get('entryPoints', []):
        for param in ep.get('parameters', []):
            find_roots(param.get('type'))
            
    for root in roots:
        add_struct(root)

    # 4. Generate Headers
    with open(f"{base_name}.slang.h", 'w') as f:
        f.write(f"#pragma once\n#include <string_view>\nnamespace {base_name}_shader_internal {{ struct Source {{ static inline constexpr std::string_view code = R\"(\n{slang_source}\n)\"; }}; }}\n")

    with open(f"{base_name}.shader.h", 'w') as f:
        f.write(f"#pragma once\n#include \"{base_name}.slang.h\"\n#include \"pictor_core/execution/compute_executor.hpp\"\n#include \"pictor_core/execution/mesh_executor.hpp\"\n\nnamespace {base_name}_shader {{\n\n")
        
        for name, fields in structs_to_gen.items():
            f.write(generate_struct(name, fields) + "\n\n")

        # Group entry points by prefix
        groups = {}
        for ep in reflection.get('entryPoints', []):
            prefix = ep['name'].rsplit('_', 1)[0] if '_' in ep['name'] else ep['name']
            suffix = ep['name'].rsplit('_', 1)[1] if '_' in ep['name'] else ""
            if prefix not in groups: groups[prefix] = {}
            groups[prefix][suffix] = ep

        for prefix, eps in groups.items():
            trait_prefix = "".join(x.capitalize() for x in prefix.split('_'))
            
            def resolve_pc_type(ep_dict):
                # 1. Check explicit @bind annotations
                ep_name = ep_dict.get('name')
                if ep_name and ep_name in explicit_binds:
                    return explicit_binds[ep_name]
                
                # 2. Fallback to reflection JSON bindings
                for b in ep_dict.get('bindings', []):
                    if b.get('binding', {}).get('kind') == 'pushConstantBuffer':
                        var_name = b.get('name')
                        if var_name and not var_name.startswith('_'):
                            return var_to_type.get(var_name, "void")
                return "void"

            def parse_thread_dim(dim_str):
                dim_str = dim_str.replace(" ", "")
                if dim_str.startswith("WAVE_SIZE"):
                    if "*" in dim_str:
                        mult = int(dim_str.split("*")[1])
                        return mult, 0
                    else:
                        return 1, 0
                else:
                    return 0, int(dim_str)

            def get_numthreads(ep_dict):
                ep_name = ep_dict.get('name')
                import re
                # Use a regex that doesn't consume other functions' attributes
                pattern = r'\[\s*numthreads\s*\(\s*([^,]+)\s*,\s*([^,]+)\s*,\s*([^)]+)\s*\)\s*\](?:(?!\bvoid\b).)*?\bvoid\s+' + re.escape(ep_name) + r'\b'
                match = re.search(pattern, slang_source, re.DOTALL)
                if match:
                    x_mult, x_const = parse_thread_dim(match.group(1))
                    y_mult, y_const = parse_thread_dim(match.group(2))
                    z_mult, z_const = parse_thread_dim(match.group(3))
                    return (x_mult, x_const), (y_mult, y_const), (z_mult, z_const)
                
                # Fallback to json if possible
                if 'threadGroupSize' in ep_dict:
                    tgs = ep_dict['threadGroupSize']
                    return (0, tgs[0]), (0, tgs[1]), (0, tgs[2])
                
                return (0, 1), (0, 1), (0, 1)

            if 'compute' in eps or (len(eps) == 1 and list(eps.values())[0]['stage'] == 'compute'):
                ep = eps.get('compute') or list(eps.values())[0]
                pc_type = resolve_pc_type(ep)
                (x_m, x_c), (y_m, y_c), (z_m, z_c) = get_numthreads(ep)
                f.write(f"struct {trait_prefix}Compute {{\n")
                f.write(f"    static inline constexpr std::string_view source = {base_name}_shader_internal::Source::code;\n")
                f.write(f'    static inline constexpr std::string_view entry_point = "{ep["name"]}";\n')
                f.write(f'    static inline constexpr uint32_t cSubgroupMultiplierX = {x_m};\n')
                f.write(f'    static inline constexpr uint32_t cGroupSizeX = {x_c};\n')
                f.write(f'    static inline constexpr uint32_t cSubgroupMultiplierY = {y_m};\n')
                f.write(f'    static inline constexpr uint32_t cGroupSizeY = {y_c};\n')
                f.write(f'    static inline constexpr uint32_t cSubgroupMultiplierZ = {z_m};\n')
                f.write(f'    static inline constexpr uint32_t cGroupSizeZ = {z_c};\n')
                f.write("};\n\n")
                f.write(f"using {trait_prefix}ComputeExecutor = pictor::ComputeExecutor<{trait_prefix}Compute, {pc_type}>;\n\n")
            
            if 'task' in eps and 'mesh' in eps:
                task_ep = eps.get('task')
                mesh_ep = eps.get('mesh')
                if task_ep and mesh_ep:
                    pc_type = resolve_pc_type(task_ep)
                    if pc_type == "void": pc_type = resolve_pc_type(mesh_ep)
                        
                    (tx_m, tx_c), (ty_m, ty_c), (tz_m, tz_c) = get_numthreads(task_ep)
                    (mx_m, mx_c), (my_m, my_c), (mz_m, mz_c) = get_numthreads(mesh_ep)

                    f.write(f"struct {trait_prefix}Mesh {{\n")
                    f.write(f"    static inline constexpr std::string_view source = {base_name}_shader_internal::Source::code;\n")
                    f.write(f'    static inline constexpr std::string_view task_entry = "{task_ep["name"]}";\n')
                    f.write(f'    static inline constexpr std::string_view mesh_entry = "{mesh_ep["name"]}";\n')
                    f.write(f'    static inline constexpr uint32_t cTaskSubgroupMultiplierX = {tx_m};\n')
                    f.write(f'    static inline constexpr uint32_t cTaskGroupSizeX = {tx_c};\n')
                    f.write(f'    static inline constexpr uint32_t cTaskSubgroupMultiplierY = {ty_m};\n')
                    f.write(f'    static inline constexpr uint32_t cTaskGroupSizeY = {ty_c};\n')
                    f.write(f'    static inline constexpr uint32_t cTaskSubgroupMultiplierZ = {tz_m};\n')
                    f.write(f'    static inline constexpr uint32_t cTaskGroupSizeZ = {tz_c};\n')
                    f.write(f'    static inline constexpr uint32_t cMeshSubgroupMultiplierX = {mx_m};\n')
                    f.write(f'    static inline constexpr uint32_t cMeshGroupSizeX = {mx_c};\n')
                    f.write(f'    static inline constexpr uint32_t cMeshSubgroupMultiplierY = {my_m};\n')
                    f.write(f'    static inline constexpr uint32_t cMeshGroupSizeY = {my_c};\n')
                    f.write(f'    static inline constexpr uint32_t cMeshSubgroupMultiplierZ = {mz_m};\n')
                    f.write(f'    static inline constexpr uint32_t cMeshGroupSizeZ = {mz_c};\n')
                    f.write("};\n\n")
                    executor_alias = prefix.upper() + "Executor"
                    f.write(f"using {executor_alias} = pictor::MeshExecutor<{trait_prefix}Mesh, {pc_type}>;\n\n")

        f.write(f"}} // namespace {base_name}_shader\n")

if __name__ == "__main__":
    main()
