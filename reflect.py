import json
import os
import subprocess
import sys
from collections import OrderedDict

def map_type(t):
    kind = t.get('kind')
    if kind == 'scalar':
        st = t.get('scalarType')
        mapping = {
            'uint32': 'uint32_t',
            'int32': 'int32_t',
            'uint64': 'uint64_t',
            'int64': 'int64_t',
            'float32': 'float',
            'float64': 'double',
            'uint': 'uint32_t',
            'int': 'int32_t',
        }
        return mapping.get(st, st), None
    elif kind == 'vector':
        count = t.get('elementCount')
        elem, _ = map_type(t.get('elementType'))
        return elem, count
    elif kind == 'array':
        count = t.get('elementCount')
        elem, _ = map_type(t.get('elementType'))
        return elem, count
    elif kind == 'struct':
        return t.get('name'), None
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
        if count is not None:
            lines.append(f"    {base_type} {f['name']}[{count}];")
        else:
            lines.append(f"    {base_type} {f['name']};")
        current_offset = offset + size
    lines.append("};")
    return "\n".join(lines)

def main():
    if len(sys.argv) < 2:
        print("Usage: python reflect.py <input.slang>")
        sys.exit(1)

    input_file = sys.argv[1]
    base_name = os.path.basename(input_file).split('.')[0]
    output_json = f"{base_name}.reflection.json"
    
    subprocess.run([
        "slangc", input_file,
        "-target", "spirv",
        "-fspv-reflect",
        "-reflection-json", output_json,
        "-o", f"{base_name}.spv"
    ], check=True)

    with open(output_json, 'r') as f:
        reflection = json.load(f)

    with open(input_file, 'r') as f:
        slang_source = f.read()

    with open(f"{base_name}.slang.h", 'w') as f:
        f.write("#pragma once\n")
        f.write("#include <string_view>\n\n")
        f.write(f"namespace {base_name}_shader_internal {{\n")
        f.write("struct Source {\n")
        f.write(f'    static inline constexpr std::string_view code = R"(\n{slang_source}\n)";\n')
        f.write("};\n")
        f.write(f"}} // namespace {base_name}_shader_internal\n")

    # Topological sort for structs
    structs_to_gen = OrderedDict()
    
    def collect_structs(t):
        if not isinstance(t, dict): return
        kind = t.get('kind')
        if kind == 'struct':
            name = t.get('name')
            if not name: return
            
            # First collect dependencies
            for f in t.get('fields', []):
                collect_structs(f.get('type'))
            
            # Then add this struct if not already added
            if name not in structs_to_gen:
                structs_to_gen[name] = t.get('fields', [])
        elif kind in ['constantBuffer', 'parameterBlock']:
             collect_structs(t.get('elementType'))
        elif kind in ['array', 'vector']:
             collect_structs(t.get('elementType'))

    for param in reflection.get('parameters', []):
        collect_structs(param.get('type'))

    for ep in reflection.get('entryPoints', []):
        for param in ep.get('parameters', []):
            collect_structs(param.get('type'))

    ep_to_pcs = {}
    for ep in reflection.get('entryPoints', []):
        ep_name = ep['name']
        ep_to_pcs[ep_name] = []
        for binding in ep.get('bindings', []):
            b = binding.get('binding', {})
            if b.get('kind') == 'pushConstantBuffer':
                for param in reflection.get('parameters', []):
                    pb = param.get('binding', {})
                    if pb.get('kind') == 'pushConstantBuffer' and pb.get('index') == b.get('index'):
                        elem_type = param.get('type', {}).get('elementType', {})
                        if elem_type.get('kind') == 'struct':
                            ep_to_pcs[ep_name].append(elem_type['name'])

    def find_best_pc(prefix, ep_name):
        pcs = ep_to_pcs.get(ep_name, [])
        if not pcs: return "void"
        if len(pcs) == 1: return pcs[0]
        for pc in pcs:
            if pc.lower().startswith(prefix.lower()):
                return pc
        return pcs[0]

    with open(f"{base_name}.shader.h", 'w') as f:
        f.write("#pragma once\n")
        f.write(f'#include "{base_name}.slang.h"\n')
        f.write('#include "pictor_core/execution/compute_executor.hpp"\n')
        f.write('#include "pictor_core/execution/mesh_executor.hpp"\n\n')
        f.write(f"namespace {base_name}_shader {{\n\n")

        for name, fields in structs_to_gen.items():
            f.write(generate_struct(name, fields))
            f.write("\n\n")

        entry_points = reflection.get('entryPoints', [])
        groups = {}
        for ep in entry_points:
            name = ep['name']
            if '_' in name:
                prefix = name.rsplit('_', 1)[0]
                suffix = name.rsplit('_', 1)[1]
            else:
                prefix = name
                suffix = ""
            
            if prefix not in groups:
                groups[prefix] = {}
            groups[prefix][suffix] = ep

        for prefix, eps in groups.items():
            trait_prefix = "".join(x.capitalize() for x in prefix.split('_'))
            
            if 'compute' in eps or (len(eps) == 1 and list(eps.values())[0]['stage'] == 'compute'):
                ep = eps.get('compute') or list(eps.values())[0]
                pc_struct = find_best_pc(prefix, ep['name'])
                f.write(f"struct {trait_prefix}Compute {{\n")
                f.write(f"    static inline constexpr std::string_view source = {base_name}_shader_internal::Source::code;\n")
                f.write(f'    static inline constexpr std::string_view entry_point = "{ep["name"]}";\n')
                f.write("};\n\n")
                f.write(f"using {trait_prefix}ComputeExecutor = pictor::ComputeExecutor<{trait_prefix}Compute, {pc_struct}>;\n\n")
            
            if 'task' in eps and 'mesh' in eps:
                task_ep = eps['task']
                mesh_ep = eps['mesh']
                pc_struct = find_best_pc(prefix, task_ep['name'])
                f.write(f"struct {trait_prefix}Mesh {{\n")
                f.write(f"    static inline constexpr std::string_view source = {base_name}_shader_internal::Source::code;\n")
                f.write(f'    static inline constexpr std::string_view task_entry = "{task_ep["name"]}";\n')
                f.write(f'    static inline constexpr std::string_view mesh_entry = "{mesh_ep["name"]}";\n')
                f.write("};\n\n")
                executor_alias = prefix.upper() + "Executor"
                f.write(f"using {executor_alias} = pictor::MeshExecutor<{trait_prefix}Mesh, {pc_struct}>;\n\n")

        f.write(f"}} // namespace {base_name}_shader\n")

if __name__ == "__main__":
    main()
