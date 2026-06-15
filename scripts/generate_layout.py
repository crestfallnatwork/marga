import json
import sys
import os

def slang_resource_to_vulkan_type(resource_shape, access):
    if resource_shape == 'structuredBuffer':
        return 'vk::DescriptorType::eStorageBuffer'
    if resource_shape == 'textureBuffer' or resource_shape == 'uniformTexelBuffer':
        return 'vk::DescriptorType::eUniformTexelBuffer'
    if resource_shape == 'constantBuffer':
        return 'vk::DescriptorType::eConstantBuffer'
    return 'vk::DescriptorType::eStorageBuffer' # Default fallback

def generate_header(json_path, output_path):
    with open(json_path, 'r') as f:
        data = json.load(f)

    header = []
    header.append("#pragma once")
    header.append("#include <cstdint>")
    header.append("#include <vector>")
    header.append("#include <vulkan/vulkan.hpp>")
    header.append("#include \"pictor_core/memory/allocation_type.hpp\"")
    header.append("")
    
    basename = os.path.basename(json_path)
    clean_name = "".join([c if c.isalnum() else "_" for c in basename.split("_reflection.json")[0]])
    header.append(f"namespace {clean_name}_layout {{")
    header.append("")

    # 1. Process Params Struct (Push Constants)
    for param in data.get('parameters', []):
        binding = param.get('binding', {})
        if binding.get('kind') == 'pushConstantBuffer':
            elem_type = param.get('type', {}).get('elementType', {})
            if elem_type.get('kind') == 'struct':
                header.append(f"    struct Params {{")
                current_offset = 0
                for field in elem_type.get('fields', []):
                    f_name = field.get('name')
                    f_binding = field.get('binding', {})
                    off = f_binding.get('offset', 0)
                    size = f_binding.get('size', 4)
                    
                    if off > current_offset:
                        header.append(f"        uint8_t _pad_{current_offset}[{off - current_offset}];")
                    
                    kind = field.get('type', {}).get('kind')
                    cpp_type = "uint32_t"
                    if kind == 'scalar':
                        stype = field.get('type', {}).get('scalarType')
                        if stype == 'uint64': cpp_type = "uint64_t"
                        elif stype == 'int32': cpp_type = "int32_t"
                    
                    header.append(f"        {cpp_type} {f_name};")
                    current_offset = off + size
                header.append(f"    }};")
                header.append("")

    # 2. Process Resources Struct (Bindings)
    resources = []
    for param in data.get('parameters', []):
        binding = param.get('binding', {})
        if binding.get('kind') == 'descriptorTableSlot':
            res_type = param.get('type', {})
            resources.append({
                'name': param.get('name'),
                'index': binding.get('index'),
                'set': binding.get('set', 0),
                'type': slang_resource_to_vulkan_type(res_type.get('resourceShape'), res_type.get('resourceAccess')),
                'raw_type': res_type.get('resourceShape')
            })

    header.append("    struct Resources {")
    for r in resources:
        # Use BufferView for texel buffers, AllocationInfo for everything else
        is_texel = ("Texel" in r['type']) or (r['raw_type'] and "textureBuffer" in r['raw_type'])
        if is_texel:
            header.append(f"        std::pair<pictor::AllocationInfo, vk::BufferView> {r['name']};")
        else:
            header.append(f"        pictor::AllocationInfo {r['name']};")
    header.append("")

    # Static method to get bindings for pipeline creation
    header.append("        static std::vector<vk::DescriptorSetLayoutBinding> get_bindings() {")
    header.append("            return {")
    for r in resources:
        header.append(f"                {{{r['index']}, {r['type']}, 1, vk::ShaderStageFlagBits::eAll, nullptr}},")
    header.append("            };")
    header.append("        }")
    header.append("")

    # Method to fill WriteDescriptorSet for pushing
    header.append("        void fill_writes(std::vector<vk::WriteDescriptorSet>& writes, std::vector<vk::DescriptorBufferInfo>& bufferInfos) const {")
    header.append(f"            bufferInfos.reserve({len(resources)});")
    header.append(f"            writes.reserve({len(resources)});")
    for i, r in enumerate(resources):
        is_texel = ("Texel" in r['type']) or (r['raw_type'] and "textureBuffer" in r['raw_type'])
        if is_texel:
             header.append(f"            writes.push_back({{nullptr, {r['index']}, 0, 1, {r['type']}, nullptr, nullptr, &{r['name']}.second}});")
        else:
             header.append(f"            bufferInfos.push_back({{{r['name']}.buffer, {r['name']}.offset, VK_WHOLE_SIZE}});")
             header.append(f"            writes.push_back({{nullptr, {r['index']}, 0, 1, {r['type']}, nullptr, &bufferInfos.back(), nullptr}});")
    header.append("        }")
    header.append("    };")
    header.append("")

    header.append(f"}} // namespace {clean_name}_layout")
    
    with open(output_path, 'w') as f:
        f.write("\n".join(header))

if __name__ == "__main__":
    generate_header(sys.argv[1], sys.argv[2])
