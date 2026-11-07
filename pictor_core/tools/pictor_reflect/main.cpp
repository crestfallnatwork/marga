/* SPDX-License-Identifier: MPL-2.0 */

#include <slang-com-ptr.h>
#include <slang.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

namespace {

constexpr uint32_t kDefaultWaveSize = 32;

struct Options {
    std::string input;
    std::string outdir{"."};
    std::string depfile;
    std::vector<std::string> includePaths;
    uint32_t waveSize{kDefaultWaveSize};
    bool dump{false};
};

void printUsage(const char* program)
{
    std::cerr
        << "usage: " << program
        << " <shader.slang> --outdir <dir> [--wave-size <n>] [-I <dir>]... [--depfile <file>] "
           "[--dump]\n";
}

bool parseOptions(int argc, char** argv, Options& options)
{
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--outdir" && i + 1 < argc) {
            options.outdir = argv[++i];
        } else if (arg == "--depfile" && i + 1 < argc) {
            options.depfile = argv[++i];
        } else if (arg == "--wave-size" && i + 1 < argc) {
            char* end            = nullptr;
            unsigned long parsed = std::strtoul(argv[++i], &end, 10);
            if (end == argv[i] || *end != '\0' || parsed == 0) {
                std::cerr << "invalid wave size: " << argv[i] << "\n";
                return false;
            }
            options.waveSize = static_cast<uint32_t>(parsed);
        } else if (arg == "-I" && i + 1 < argc) {
            options.includePaths.emplace_back(argv[++i]);
        } else if (arg == "--dump") {
            options.dump = true;
        } else if (!arg.empty() && arg[0] == '-') {
            std::cerr << "unknown option: " << arg << "\n";
            return false;
        } else if (options.input.empty()) {
            options.input = arg;
        } else {
            std::cerr << "unexpected argument: " << arg << "\n";
            return false;
        }
    }
    return !options.input.empty();
}

bool readFile(const std::string& path, std::string& contents)
{
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        return false;
    }
    std::ostringstream buffer;
    buffer << file.rdbuf();
    contents = buffer.str();
    return true;
}

std::filesystem::path normalizePath(const std::filesystem::path& path)
{
    std::error_code error;
    std::filesystem::path canonical = std::filesystem::weakly_canonical(path, error);
    if (error) {
        return std::filesystem::absolute(path).lexically_normal();
    }
    return canonical;
}

std::string stripTrailingSlash(std::string text)
{
    while (text.size() > 1 && text.back() == '/') {
        text.pop_back();
    }
    return text;
}

bool virtualPathFor(const std::filesystem::path& file,
                    const std::vector<std::filesystem::path>& searchPaths, std::string& out)
{
    std::string fileText = file.string();
    size_t bestIndex     = searchPaths.size();
    size_t bestLength    = 0;
    for (size_t i = 0; i < searchPaths.size(); ++i) {
        std::string root = stripTrailingSlash(searchPaths[i].string());
        bool matches     = fileText.size() > root.size() && fileText.starts_with(root) &&
                           (root == "/" || fileText[root.size()] == '/');
        if (matches && root.size() > bestLength) {
            bestIndex  = i;
            bestLength = root.size();
        }
    }
    if (bestIndex == searchPaths.size()) {
        return false;
    }
    size_t offset = bestLength == 1 ? 1 : bestLength + 1;
    out           = "/sp" + std::to_string(bestIndex) + "/" + fileText.substr(offset);
    return true;
}

std::string depfileEscape(const std::string& text)
{
    std::string result;
    for (char c : text) {
        if (c == ' ') {
            result += "\\ ";
        } else if (c == '$') {
            result += "$$";
        } else {
            result += c;
        }
    }
    return result;
}

std::string scalarTypeName(slang::TypeReflection::ScalarType scalar)
{
    switch (scalar) {
        case slang::TypeReflection::ScalarType::Bool:
            return "bool";
        case slang::TypeReflection::ScalarType::Int8:
            return "int8_t";
        case slang::TypeReflection::ScalarType::Int16:
            return "int16_t";
        case slang::TypeReflection::ScalarType::Int32:
            return "int32_t";
        case slang::TypeReflection::ScalarType::Int64:
        case slang::TypeReflection::ScalarType::IntPtr:
            return "int64_t";
        case slang::TypeReflection::ScalarType::UInt8:
            return "uint8_t";
        case slang::TypeReflection::ScalarType::UInt16:
            return "uint16_t";
        case slang::TypeReflection::ScalarType::UInt32:
            return "uint32_t";
        case slang::TypeReflection::ScalarType::UInt64:
        case slang::TypeReflection::ScalarType::UIntPtr:
            return "uint64_t";
        case slang::TypeReflection::ScalarType::Float32:
            return "float";
        case slang::TypeReflection::ScalarType::Float64:
            return "double";
        default:
            return {};
    }
}

std::string stageName(SlangStage stage)
{
    switch (stage) {
        case SLANG_STAGE_VERTEX:
            return "vertex";
        case SLANG_STAGE_FRAGMENT:
            return "fragment";
        case SLANG_STAGE_COMPUTE:
            return "compute";
        case SLANG_STAGE_AMPLIFICATION:
            return "task";
        case SLANG_STAGE_MESH:
            return "mesh";
        case SLANG_STAGE_GEOMETRY:
            return "geometry";
        case SLANG_STAGE_HULL:
            return "hull";
        case SLANG_STAGE_DOMAIN:
            return "domain";
        default:
            return "unknown";
    }
}

struct CppType {
    std::string name;
    std::string elements;
};

bool mapType(slang::TypeLayoutReflection* layout, CppType& type)
{
    using Kind = slang::TypeReflection::Kind;
    switch (layout->getKind()) {
        case Kind::Scalar: {
            type.name = scalarTypeName(layout->getScalarType());
            return !type.name.empty();
        }
        case Kind::Vector: {
            type.name     = scalarTypeName(layout->getElementTypeLayout()->getScalarType());
            type.elements = "[" + std::to_string(layout->getElementCount()) + "]";
            return !type.name.empty();
        }
        case Kind::Array: {
            CppType element;
            if (!mapType(layout->getElementTypeLayout(), element)) {
                return false;
            }
            type.name = element.name;
            type.elements =
                "[" + std::to_string(layout->getElementCount()) + "]" + element.elements;
            return true;
        }
        case Kind::Struct: {
            const auto* name = layout->getType()->getName();
            if (!name) {
                return false;
            }
            type.name = name;
            return true;
        }
        case Kind::Pointer: {
            CppType element;
            if (!mapType(layout->getElementTypeLayout(), element)) {
                return false;
            }
            type.name     = element.name + "*";
            type.elements = element.elements;
            return true;
        }
        default:
            return false;
    }
}

std::string describeType(slang::TypeLayoutReflection* layout)
{
    using Kind = slang::TypeReflection::Kind;
    switch (layout->getKind()) {
        case Kind::Scalar:
            return scalarTypeName(layout->getScalarType());
        case Kind::Vector:
            return scalarTypeName(layout->getElementTypeLayout()->getScalarType()) + "x" +
                   std::to_string(layout->getElementCount());
        case Kind::Array:
            return describeType(layout->getElementTypeLayout()) + "[" +
                   std::to_string(layout->getElementCount()) + "]";
        case Kind::Struct: {
            const auto* name = layout->getType()->getName();
            return name ? std::string(name) : std::string("struct");
        }
        case Kind::Resource:
        case Kind::ShaderStorageBuffer: {
            auto* result      = layout->getResourceResultType();
            std::string inner = result && result->getName() ? result->getName() : "?";
            return "resource(" + inner + ")";
        }
        case Kind::ConstantBuffer:
        case Kind::ParameterBlock:
            return "cbuffer(" + describeType(layout->getElementTypeLayout()) + ")";
        case Kind::Pointer:
            return describeType(layout->getElementTypeLayout()) + "*";
        default:
            return "?";
    }
}

slang::TypeLayoutReflection* structLayoutOf(slang::TypeLayoutReflection* layout)
{
    using Kind = slang::TypeReflection::Kind;
    switch (layout->getKind()) {
        case Kind::Struct:
            return layout;
        case Kind::Array:
        case Kind::Vector:
        case Kind::Pointer:
        case Kind::ConstantBuffer:
        case Kind::ParameterBlock:
            return structLayoutOf(layout->getElementTypeLayout());
        default:
            return nullptr;
    }
}

class StructCollector {
  public:
    void add(slang::TypeLayoutReflection* layout)
    {
        auto* structLayout = structLayoutOf(layout);
        if (!structLayout) {
            return;
        }
        const auto* name = structLayout->getType()->getName();
        if (!name) {
            return;
        }
        std::string structName = name;
        if (std::ranges::find(_visited, structName) != _visited.end()) {
            return;
        }
        _visited.push_back(structName);
        for (unsigned i = 0; i < structLayout->getFieldCount(); ++i) {
            add(structLayout->getFieldByIndex(i)->getTypeLayout());
        }
        _structs.emplace_back(structName, structLayout);
    }

    [[nodiscard]] const std::vector<std::pair<std::string, slang::TypeLayoutReflection*>>&
    structs() const
    {
        return _structs;
    }

  private:
    std::vector<std::pair<std::string, slang::TypeLayoutReflection*>> _structs;
    std::vector<std::string> _visited;
};

std::string renderStruct(const std::string& name, slang::TypeLayoutReflection* layout)
{
    std::ostringstream out;
    out << "struct " << name << " {\n";
    size_t cursor = 0;
    for (unsigned i = 0; i < layout->getFieldCount(); ++i) {
        auto* field       = layout->getFieldByIndex(i);
        auto* fieldLayout = field->getTypeLayout();
        size_t offset     = field->getOffset(slang::ParameterCategory::Uniform);
        size_t size       = fieldLayout->getSize(slang::ParameterCategory::Uniform);
        if (offset > cursor) {
            out << "    uint8_t _pad_" << cursor << "[" << offset - cursor << "];\n";
        }
        CppType type;
        if (!mapType(fieldLayout, type) || size == 0) {
            type = CppType{"uint8_t", "[" + std::to_string(size) + "]"};
        }
        const auto* fieldName = field->getName();
        out << "    " << type.name << " " << (fieldName ? fieldName : "_field") << type.elements
            << ";\n";
        cursor = offset + size;
    }
    out << "};\n";
    return out.str();
}

struct BindingInfo {
    std::string name;
    std::string type;
    uint32_t set{0};
    uint32_t index{0};
    size_t elements{1};
};

struct EntryInfo {
    std::string name;
    SlangStage stage{SLANG_STAGE_NONE};
    std::array<uint32_t, 3> threadGroupSize{1, 1, 1};
    std::string pushConstantType;
    uint32_t colorAttachmentCount{0};
    std::vector<BindingInfo> bindings;
};

std::string entryPrefix(const std::string& name)
{
    auto pos = name.rfind('_');
    return pos == std::string::npos ? name : name.substr(0, pos);
}

std::string traitPrefix(const std::string& prefix)
{
    std::string result;
    size_t start = 0;
    while (start < prefix.size()) {
        size_t end = prefix.find('_', start);
        if (end == std::string::npos) {
            end = prefix.size();
        }
        if (end > start) {
            result += static_cast<char>(std::toupper(prefix[start]));
            for (size_t i = start + 1; i < end; ++i) {
                result += static_cast<char>(std::tolower(prefix[i]));
            }
        }
        start = end + 1;
    }
    return result;
}

std::string upperCase(const std::string& text)
{
    std::string result = text;
    for (char& c : result) {
        c = static_cast<char>(std::toupper(c));
    }
    return result;
}

bool reportDiagnostics(const char* context, slang::IBlob* diagnostics)
{
    std::cerr << "pictor_reflect: " << context << "\n";
    if (diagnostics) {
        std::cerr << static_cast<const char*>(diagnostics->getBufferPointer());
    }
    return false;
}

void dumpReflection(const std::string& input, const std::vector<EntryInfo>& entries,
                    const StructCollector& structs)
{
    std::cout << "module: " << input << "\n";
    for (const auto& entry : entries) {
        std::cout << "entry " << entry.name << " [" << stageName(entry.stage) << "]";
        if (entry.stage == SLANG_STAGE_COMPUTE || entry.stage == SLANG_STAGE_AMPLIFICATION ||
            entry.stage == SLANG_STAGE_MESH) {
            std::cout << " threadgroup " << entry.threadGroupSize[0] << "x"
                      << entry.threadGroupSize[1] << "x" << entry.threadGroupSize[2];
        }
        std::cout << "\n";
        if (!entry.pushConstantType.empty()) {
            std::cout << "  push constant: " << entry.pushConstantType << "\n";
        }
        for (const auto& binding : entry.bindings) {
            std::cout << "  binding set " << binding.set << " index " << binding.index << " "
                      << binding.name << " " << binding.type << " x" << binding.elements << "\n";
        }
        if (entry.stage == SLANG_STAGE_FRAGMENT) {
            std::cout << "  fragment outputs: " << entry.colorAttachmentCount << "\n";
        }
    }
    for (const auto& [name, layout] : structs.structs()) {
        std::cout << "struct " << name << " size "
                  << layout->getSize(slang::ParameterCategory::Uniform) << " align "
                  << layout->getAlignment(slang::ParameterCategory::Uniform) << "\n";
        for (unsigned i = 0; i < layout->getFieldCount(); ++i) {
            auto* field           = layout->getFieldByIndex(i);
            const auto* fieldName = field->getName();
            std::cout << "  " << (fieldName ? fieldName : "?") << " @"
                      << field->getOffset(slang::ParameterCategory::Uniform) << " size "
                      << field->getTypeLayout()->getSize(slang::ParameterCategory::Uniform) << " "
                      << describeType(field->getTypeLayout()) << "\n";
        }
    }
}

bool writeFile(const std::filesystem::path& path, const std::string& contents)
{
    std::ofstream file(path, std::ios::binary);
    if (!file) {
        return false;
    }
    file << contents;
    return static_cast<bool>(file);
}

struct SourceEntry {
    std::filesystem::path realPath;
    std::string virtualPath;
    std::string contents;
};

struct TunableInfo {
    std::string name;
    std::string field;
    std::string type;
    std::string value;
};

std::string_view trimView(std::string_view text)
{
    while (!text.empty() && std::isspace(static_cast<unsigned char>(text.front()))) {
        text.remove_prefix(1);
    }
    while (!text.empty() && std::isspace(static_cast<unsigned char>(text.back()))) {
        text.remove_suffix(1);
    }
    return text;
}

std::string stripComments(std::string_view contents)
{
    std::string out;
    out.reserve(contents.size());
    bool inBlock = false;
    for (size_t i = 0; i < contents.size(); ++i) {
        char c = contents[i];
        if (inBlock) {
            if (c == '*' && i + 1 < contents.size() && contents[i + 1] == '/') {
                inBlock = false;
                ++i;
            } else if (c == '\n') {
                out.push_back('\n');
            }
            continue;
        }
        if (c == '/' && i + 1 < contents.size() && contents[i + 1] == '/') {
            while (i < contents.size() && contents[i] != '\n') {
                ++i;
            }
            if (i < contents.size()) {
                out.push_back('\n');
            }
            continue;
        }
        if (c == '/' && i + 1 < contents.size() && contents[i + 1] == '*') {
            inBlock = true;
            ++i;
            continue;
        }
        out.push_back(c);
    }
    return out;
}

std::vector<std::string> cleanedLines(std::string_view contents)
{
    constexpr std::string_view kBom = "\xEF\xBB\xBF";
    if (contents.starts_with(kBom)) {
        contents.remove_prefix(kBom.size());
    }
    std::string cleaned = stripComments(contents);
    std::vector<std::string> lines;
    std::istringstream stream(cleaned);
    std::string line;
    while (std::getline(stream, line)) {
        lines.emplace_back(trimView(line));
    }
    return lines;
}

bool isIdentifierChar(char c)
{
    return std::isalnum(static_cast<unsigned char>(c)) != 0 || c == '_';
}

bool matchDirective(std::string_view line, std::string_view directive, std::string_view& rest)
{
    if (!line.starts_with(directive)) {
        return false;
    }
    std::string_view remainder = line.substr(directive.size());
    if (!remainder.empty() && !std::isspace(static_cast<unsigned char>(remainder.front()))) {
        return false;
    }
    rest = trimView(remainder);
    return true;
}

std::string firstIdentifier(std::string_view text)
{
    size_t end = 0;
    while (end < text.size() && isIdentifierChar(text[end])) {
        ++end;
    }
    return std::string(text.substr(0, end));
}

enum class TunableKind {
    Bool,
    UInt32
};

bool parseIntegerLiteral(std::string_view value, uint64_t& out)
{
    std::string digits(trimView(value));
    while (!digits.empty() && (digits.back() == 'u' || digits.back() == 'U' ||
                               digits.back() == 'l' || digits.back() == 'L')) {
        digits.pop_back();
    }
    if (digits.empty()) {
        return false;
    }
    errno                     = 0;
    char* end                 = nullptr;
    unsigned long long parsed = std::strtoull(digits.c_str(), &end, 0);
    if (end != digits.c_str() + digits.size() || errno == ERANGE) {
        return false;
    }
    out = parsed;
    return true;
}

bool classifyLiteral(const std::string& value, TunableKind& kind, std::string& normalized)
{
    if (value == "true" || value == "false") {
        kind       = TunableKind::Bool;
        normalized = value;
        return true;
    }
    uint64_t parsed = 0;
    if (!parseIntegerLiteral(value, parsed) || parsed > 0xFFFFFFFFull) {
        return false;
    }
    kind       = TunableKind::UInt32;
    normalized = std::to_string(parsed);
    return true;
}

std::string fieldNameFor(const std::string& macroName)
{
    constexpr std::string_view kPrefix = "PICTOR_";
    std::string_view stripped          = macroName;
    stripped.remove_prefix(kPrefix.size());
    std::string result;
    result.reserve(stripped.size());
    for (char c : stripped) {
        result.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));
    }
    return result;
}

bool isCppKeyword(std::string_view name)
{
    static constexpr auto kKeywords = std::to_array<std::string_view>({
        "alignas",       "alignof",     "and",
        "and_eq",        "asm",         "auto",
        "bitand",        "bitor",       "bool",
        "break",         "case",        "catch",
        "char",          "char8_t",     "char16_t",
        "char32_t",      "class",       "compl",
        "concept",       "const",       "const_cast",
        "consteval",     "constexpr",   "constinit",
        "continue",      "co_await",    "co_return",
        "co_yield",      "decltype",    "default",
        "delete",        "do",          "double",
        "dynamic_cast",  "else",        "enum",
        "explicit",      "export",      "extern",
        "false",         "float",       "for",
        "friend",        "goto",        "if",
        "inline",        "int",         "long",
        "mutable",       "namespace",   "new",
        "noexcept",      "not",         "not_eq",
        "nullptr",       "operator",    "or",
        "or_eq",         "private",     "protected",
        "public",        "register",    "reinterpret_cast",
        "requires",      "return",      "short",
        "signed",        "sizeof",      "static",
        "static_assert", "static_cast", "struct",
        "switch",        "template",    "this",
        "thread_local",  "throw",       "true",
        "try",           "typedef",     "typeid",
        "typename",      "union",       "unsigned",
        "using",         "virtual",     "void",
        "volatile",      "wchar_t",     "while",
        "xor",           "xor_eq",
    });
    return std::ranges::find(kKeywords, name) != std::end(kKeywords);
}

bool isValidFieldName(std::string_view name)
{
    if (name.empty()) {
        return false;
    }
    if (std::isalpha(static_cast<unsigned char>(name.front())) == 0 && name.front() != '_') {
        return false;
    }
    for (char c : name) {
        if (std::isalnum(static_cast<unsigned char>(c)) == 0 && c != '_') {
            return false;
        }
    }
    return !isCppKeyword(name);
}

struct ConditionalFrame {
    bool active{true};
    bool parentActive{true};
    bool taken{false};
};

bool evaluateCondition(std::string_view expr, const std::map<std::string, std::string>& defined)
{
    expr = trimView(expr);
    if (expr.empty()) {
        return true;
    }
    if (expr.front() == '!') {
        return !evaluateCondition(expr.substr(1), defined);
    }
    if (expr.starts_with("defined")) {
        std::string_view rest = trimView(expr.substr(7));
        if (!rest.empty() && rest.front() == '(' && rest.back() == ')') {
            rest = trimView(rest.substr(1, rest.size() - 2));
        }
        std::string name = firstIdentifier(rest);
        if (name.empty()) {
            return true;
        }
        return defined.contains(name);
    }
    if (std::isalpha(static_cast<unsigned char>(expr.front())) != 0 || expr.front() == '_') {
        std::string name = firstIdentifier(expr);
        if (name.size() == expr.size()) {
            auto it = defined.find(name);
            if (it == defined.end()) {
                return false;
            }
            if (it->second.empty() || it->second == "true") {
                return true;
            }
            if (it->second == "false") {
                return false;
            }
            uint64_t parsed = 0;
            if (parseIntegerLiteral(it->second, parsed)) {
                return parsed != 0;
            }
            return true;
        }
        return true;
    }
    uint64_t parsed = 0;
    if (parseIntegerLiteral(expr, parsed)) {
        return parsed != 0;
    }
    return true;
}

bool scanFileTunables(const SourceEntry& file, std::vector<TunableInfo>& out, std::string& error)
{
    std::vector<std::string> lines = cleanedLines(file.contents);

    std::map<std::string, std::string> defined;
    defined.emplace("WAVE_SIZE", "");
    std::vector<ConditionalFrame> conditionals;

    auto currentActive   = [&]() { return conditionals.empty() || conditionals.back().active; };
    auto pushConditional = [&](bool parentActive, bool condition) {
        bool branchActive = parentActive && condition;
        conditionals.push_back(ConditionalFrame{branchActive, parentActive, branchActive});
    };

    for (size_t i = 0; i < lines.size(); ++i) {
        const std::string& line = lines[i];
        if (line.empty() || line.front() != '#') {
            continue;
        }
        const bool active = currentActive();
        std::string_view rest;

        if (matchDirective(line, "#ifndef", rest)) {
            std::string name  = firstIdentifier(rest);
            bool condition    = !defined.contains(name);
            bool branchActive = active && condition;
            if (branchActive && name.starts_with("PICTOR_")) {
                size_t j = i + 1;
                while (j < lines.size() && lines[j].empty()) {
                    ++j;
                }
                if (j < lines.size()) {
                    std::string_view defineRest;
                    if (matchDirective(lines[j], "#define", defineRest) &&
                        firstIdentifier(defineRest) == name) {
                        std::string valueText =
                            std::string(trimView(defineRest.substr(name.size())));
                        if (valueText.empty()) {
                            error = "guarded define '" + name +
                                    "' has no value; a literal default (true/false or an integer) "
                                    "is required";
                            return false;
                        }
                        TunableKind kind;
                        std::string normalized;
                        if (!classifyLiteral(valueText, kind, normalized)) {
                            std::ostringstream message;
                            message << "guarded define '" << name
                                    << "' has the non-literal default '" << valueText
                                    << "'; only true/false or an integer literal is allowed";
                            error = message.str();
                            return false;
                        }
                        std::string field = fieldNameFor(name);
                        if (!isValidFieldName(field)) {
                            std::ostringstream message;
                            message << "guarded define '" << name
                                    << "' maps to the invalid C++ field name '" << field << "'";
                            error = message.str();
                            return false;
                        }
                        TunableInfo info;
                        info.name  = name;
                        info.field = std::move(field);
                        info.type  = (kind == TunableKind::Bool) ? "bool" : "uint32_t";
                        info.value = normalized;
                        out.push_back(std::move(info));
                        defined[name] = normalized;
                        i             = j;
                    }
                }
            }
            pushConditional(active, condition);
            continue;
        }

        if (matchDirective(line, "#ifdef", rest)) {
            std::string name = firstIdentifier(rest);
            pushConditional(active, defined.contains(name));
            continue;
        }

        if (matchDirective(line, "#if", rest)) {
            pushConditional(active, evaluateCondition(rest, defined));
            continue;
        }

        if (matchDirective(line, "#elif", rest)) {
            if (!conditionals.empty()) {
                ConditionalFrame& frame = conditionals.back();
                if (frame.parentActive && !frame.taken) {
                    frame.active = evaluateCondition(rest, defined);
                    frame.taken  = frame.active;
                } else {
                    frame.active = false;
                }
            }
            continue;
        }

        if (matchDirective(line, "#else", rest)) {
            if (!conditionals.empty()) {
                ConditionalFrame& frame = conditionals.back();
                if (frame.parentActive && !frame.taken) {
                    frame.active = true;
                    frame.taken  = true;
                } else {
                    frame.active = false;
                }
            }
            continue;
        }

        if (matchDirective(line, "#endif", rest)) {
            if (!conditionals.empty()) {
                conditionals.pop_back();
            }
            continue;
        }

        if (!active) {
            continue;
        }

        if (matchDirective(line, "#define", rest)) {
            std::string name = firstIdentifier(rest);
            if (!name.empty()) {
                defined[name] = std::string(trimView(rest.substr(name.size())));
            }
            continue;
        }

        if (matchDirective(line, "#undef", rest)) {
            defined.erase(firstIdentifier(rest));
            continue;
        }
    }
    return true;
}

std::string renderTunables(const std::string& structName, const std::vector<TunableInfo>& tunables)
{
    size_t declWidth = 0;
    size_t lineWidth = 0;
    std::vector<std::string> decls;
    decls.reserve(tunables.size());
    for (const auto& tunable : tunables) {
        std::string decl = tunable.type + " " + tunable.field;
        declWidth        = std::max(declWidth, decl.size());
        decls.push_back(std::move(decl));
    }
    for (const auto& tunable : tunables) {
        lineWidth = std::max(lineWidth, declWidth + 3 + tunable.value.size() + 1);
    }

    std::ostringstream out;
    out << "struct " << structName << " {\n";
    for (size_t i = 0; i < tunables.size(); ++i) {
        std::string line = decls[i] + std::string(declWidth - decls[i].size(), ' ') + " = " +
                           tunables[i].value + ";";
        out << "    " << line << std::string(lineWidth - line.size(), ' ') << "   // "
            << tunables[i].name << "\n";
    }
    out << "\n    [[nodiscard]] ShaderDefines defines() const;\n};\n\n";
    out << "inline ShaderDefines " << structName << "::defines() const\n{\n    ShaderDefines d;\n";
    for (const auto& tunable : tunables) {
        out << "    d.add(\"" << tunable.name << "\", std::to_string(" << tunable.field << "));\n";
    }
    out << "    return d;\n}\n";
    return out.str();
}

struct PushConstantVariable {
    std::string name;
    std::string type;
};

struct SpirvVariableUsage {
    std::vector<PushConstantVariable> push_constants;
    bool unnamed_push_constant{false};
};

std::string stripLayoutSuffix(std::string name)
{
    for (std::string_view suffix : {"_std430", "_std140"}) {
        if (name.ends_with(suffix)) {
            name.resize(name.size() - suffix.size());
            break;
        }
    }
    return name;
}

// Slang's linked ProgramLayout lists every module-level push-constant block from the entry file
// (and its #includes), but not from imported modules, and it cannot say which block an entry
// uses. The entry's SPIR-V keeps only the blocks it references; both the uniform name and the
// pointed-to struct name identify the block.
bool scanSpirvVariables(const void* data, std::size_t bytes, SpirvVariableUsage& usage,
                        std::string& error)
{
    constexpr uint32_t cSpirvMagic          = 0x07230203u;
    constexpr uint32_t cOpName              = 5u;
    constexpr uint32_t cOpTypePointer       = 32u;
    constexpr uint32_t cOpVariable          = 59u;
    constexpr uint32_t cStoragePushConstant = 9u;

    if (bytes < 5 * sizeof(uint32_t)) {
        error = "the SPIR-V blob is too small";
        return false;
    }

    const auto* words           = static_cast<const uint32_t*>(data);
    const std::size_t wordCount = bytes / sizeof(uint32_t);
    if (words[0] != cSpirvMagic) {
        error = "the SPIR-V blob has a bad magic";
        return false;
    }

    std::map<uint32_t, std::string> names;
    std::map<uint32_t, uint32_t> pointee;
    struct Variable {
        uint32_t type;
        uint32_t id;
        uint32_t storage;
    };
    std::vector<Variable> variables;

    for (std::size_t i = 5; i < wordCount;) {
        const uint32_t word   = words[i];
        const uint32_t length = word >> 16u;
        const uint32_t opcode = word & 0xffffu;
        if (length == 0 || i + length > wordCount) {
            error = "malformed SPIR-V instruction stream";
            return false;
        }

        if (opcode == cOpName && length >= 3) {
            const uint32_t id         = words[i + 1];
            const char* text          = reinterpret_cast<const char*>(&words[i + 2]);
            const std::size_t maxSize = (length - 2) * sizeof(uint32_t);
            std::string name;
            for (std::size_t b = 0; b < maxSize && text[b] != '\0'; ++b) {
                name.push_back(text[b]);
            }
            names[id] = std::move(name);
        } else if (opcode == cOpTypePointer && length >= 4) {
            pointee[words[i + 1]] = words[i + 3];
        } else if (opcode == cOpVariable && length >= 4) {
            variables.push_back({words[i + 1], words[i + 2], words[i + 3]});
        }
        i += length;
    }

    for (const auto& variable : variables) {
        if (variable.storage != cStoragePushConstant) {
            continue;
        }

        std::string name;
        if (auto it = names.find(variable.id); it != names.end()) {
            name = it->second;
        }

        std::string type;
        if (auto p = pointee.find(variable.type); p != pointee.end()) {
            if (auto t = names.find(p->second); t != names.end()) {
                type = stripLayoutSuffix(t->second);
            }
        }

        if (name.empty() && type.empty()) {
            usage.unnamed_push_constant = true;
        } else {
            usage.push_constants.push_back({std::move(name), std::move(type)});
        }
    }

    return true;
}

} // namespace

int main(int argc, char** argv)
{
    Options options;
    if (!parseOptions(argc, argv, options)) {
        printUsage(argv[0]);
        return 1;
    }

    std::filesystem::path inputPath = normalizePath(options.input);
    std::string base                = inputPath.filename().string();
    base                            = base.substr(0, base.find('.'));

    Slang::ComPtr<slang::IGlobalSession> globalSession;
    if (SLANG_FAILED(slang::createGlobalSession(globalSession.writeRef()))) {
        std::cerr << "pictor_reflect: failed to create the Slang global session\n";
        return 1;
    }

    slang::TargetDesc targetDesc{};
    targetDesc.format  = SLANG_SPIRV;
    targetDesc.profile = globalSession->findProfile("spirv_1_5");

    std::string waveSize = std::to_string(options.waveSize);
    slang::PreprocessorMacroDesc macro{"WAVE_SIZE", waveSize.c_str()};

    std::vector<std::filesystem::path> searchPathDirs;
    auto addSearchPath = [&](const std::filesystem::path& path) {
        auto normalized = normalizePath(path);
        if (std::ranges::find(searchPathDirs, normalized) == searchPathDirs.end()) {
            searchPathDirs.push_back(std::move(normalized));
        }
    };
    addSearchPath(inputPath.parent_path());
    for (const auto& dir : options.includePaths) {
        addSearchPath(dir);
    }

    std::vector<std::string> searchPathStrings;
    searchPathStrings.reserve(searchPathDirs.size());
    for (const auto& dir : searchPathDirs) {
        searchPathStrings.push_back(dir.string());
    }
    std::vector<const char*> searchPaths;
    searchPaths.reserve(searchPathStrings.size());
    for (const auto& path : searchPathStrings) {
        searchPaths.push_back(path.c_str());
    }

    slang::SessionDesc sessionDesc{};
    sessionDesc.targets                = &targetDesc;
    sessionDesc.targetCount            = 1;
    sessionDesc.preprocessorMacros     = &macro;
    sessionDesc.preprocessorMacroCount = 1;
    sessionDesc.searchPaths            = searchPaths.data();
    sessionDesc.searchPathCount        = static_cast<SlangInt>(searchPaths.size());

    Slang::ComPtr<slang::ISession> session;
    if (SLANG_FAILED(globalSession->createSession(sessionDesc, session.writeRef()))) {
        std::cerr << "pictor_reflect: failed to create the Slang session\n";
        return 1;
    }

    Slang::ComPtr<slang::IBlob> diagnostics;
    Slang::ComPtr<slang::IModule> module(
        session->loadModule(inputPath.string().c_str(), diagnostics.writeRef()));
    if (!module) {
        reportDiagnostics("failed to load the shader module", diagnostics);
        return 1;
    }

    std::vector<SourceEntry> closure;
    for (SlangInt32 i = 0; i < module->getDependencyFileCount(); ++i) {
        SourceEntry entry;
        entry.realPath = normalizePath(module->getDependencyFilePath(i));
        if (!virtualPathFor(entry.realPath, searchPathDirs, entry.virtualPath)) {
            std::cerr << "pictor_reflect: dependency outside the search paths: " << entry.realPath
                      << "\n";
            return 1;
        }
        if (!readFile(entry.realPath.string(), entry.contents)) {
            std::cerr << "pictor_reflect: cannot read " << entry.realPath << "\n";
            return 1;
        }
        closure.push_back(std::move(entry));
    }
    std::ranges::sort(closure, [](const SourceEntry& a, const SourceEntry& b) {
        return a.virtualPath < b.virtualPath;
    });

    std::string entryVirtualPath;
    for (const auto& entry : closure) {
        if (entry.realPath == inputPath) {
            entryVirtualPath = entry.virtualPath;
            break;
        }
    }
    if (entryVirtualPath.empty() && !virtualPathFor(inputPath, searchPathDirs, entryVirtualPath)) {
        std::cerr << "pictor_reflect: the shader is outside the search paths\n";
        return 1;
    }

    std::vector<TunableInfo> tunables;
    for (const auto& file : closure) {
        std::vector<TunableInfo> fileTunables;
        std::string error;
        if (!scanFileTunables(file, fileTunables, error)) {
            std::cerr << "pictor_reflect: " << file.realPath << ": " << error << "\n";
            return 1;
        }
        for (auto& tunable : fileTunables) {
            auto it = std::ranges::find_if(tunables, [&](const TunableInfo& existing) {
                return existing.name == tunable.name;
            });
            if (it == tunables.end()) {
                auto collision = std::ranges::find_if(tunables, [&](const TunableInfo& existing) {
                    return existing.field == tunable.field;
                });
                if (collision != tunables.end()) {
                    std::cerr << "pictor_reflect: tunables '" << collision->name << "' and '"
                              << tunable.name << "' both map to the field '" << tunable.field
                              << "'\n";
                    return 1;
                }
                tunables.push_back(std::move(tunable));
            } else if (it->type != tunable.type || it->value != tunable.value) {
                std::cerr << "pictor_reflect: conflicting guarded defaults for " << tunable.name
                          << " (" << it->value << " vs " << tunable.value << ")\n";
                return 1;
            }
        }
    }

    bool importedPushConstants = false;
    for (const auto& file : closure) {
        if (file.realPath != inputPath &&
            file.contents.find("push_constant") != std::string::npos) {
            importedPushConstants = true;
            break;
        }
    }

    StructCollector structs;
    std::vector<EntryInfo> entries;
    std::vector<Slang::ComPtr<slang::IComponentType>> programs;
    SlangInt32 entryCount = module->getDefinedEntryPointCount();
    for (SlangInt32 i = 0; i < entryCount; ++i) {
        Slang::ComPtr<slang::IEntryPoint> entryPoint;
        if (SLANG_FAILED(module->getDefinedEntryPoint(i, entryPoint.writeRef()))) {
            std::cerr << "pictor_reflect: failed to get entry point " << i << "\n";
            return 1;
        }

        std::vector<slang::IComponentType*> components = {module.get(), entryPoint.get()};
        Slang::ComPtr<slang::IComponentType> program;
        diagnostics.setNull();
        if (SLANG_FAILED(session->createCompositeComponentType(
                components.data(), static_cast<SlangInt>(components.size()), program.writeRef(),
                diagnostics.writeRef()))) {
            reportDiagnostics("failed to link an entry point", diagnostics);
            return 1;
        }

        programs.push_back(program);

        diagnostics.setNull();
        slang::ProgramLayout* layout = program->getLayout(0, diagnostics.writeRef());
        if (!layout) {
            reportDiagnostics("failed to get the program layout", diagnostics);
            return 1;
        }

        slang::EntryPointReflection* entryLayout = layout->getEntryPointByIndex(0);
        if (!entryLayout || !entryLayout->getName()) {
            std::cerr << "pictor_reflect: entry point " << i << " has no layout\n";
            return 1;
        }

        EntryInfo entry;
        entry.name  = entryLayout->getName();
        entry.stage = entryLayout->getStage();

        std::array<SlangUInt, 3> threadGroupSize{1, 1, 1};
        entryLayout->getComputeThreadGroupSize(3, threadGroupSize.data());
        entry.threadGroupSize = {static_cast<uint32_t>(threadGroupSize[0]),
                                 static_cast<uint32_t>(threadGroupSize[1]),
                                 static_cast<uint32_t>(threadGroupSize[2])};

        struct PushConstantParameter {
            std::string name;
            std::string type;
        };
        std::vector<PushConstantParameter> pushConstantParameters;

        for (unsigned p = 0; p < layout->getParameterCount(); ++p) {
            auto* parameter = layout->getParameterByIndex(p);
            auto category   = parameter->getCategory();
            if (category == slang::ParameterCategory::PushConstantBuffer) {
                auto* pushConstant = parameter->getTypeLayout();
                while (pushConstant->getKind() == slang::TypeReflection::Kind::ConstantBuffer ||
                       pushConstant->getKind() == slang::TypeReflection::Kind::ParameterBlock) {
                    pushConstant = pushConstant->getElementTypeLayout();
                }
                if (pushConstant->getKind() == slang::TypeReflection::Kind::Struct) {
                    const auto* typeName = pushConstant->getType()->getName();
                    pushConstantParameters.push_back(
                        {parameter->getName() ? parameter->getName() : "",
                         typeName ? typeName : ""});
                    structs.add(pushConstant);
                }
            } else if (category == slang::ParameterCategory::DescriptorTableSlot) {
                BindingInfo binding;
                binding.name  = parameter->getName() ? parameter->getName() : "";
                binding.type  = describeType(parameter->getTypeLayout());
                binding.set   = parameter->getBindingSpace();
                binding.index = parameter->getBindingIndex();
                binding.elements =
                    parameter->getTypeLayout()->getKind() == slang::TypeReflection::Kind::Array
                        ? parameter->getTypeLayout()->getElementCount()
                        : 1;
                entry.bindings.push_back(std::move(binding));
            }
        }

        if (pushConstantParameters.empty() && !importedPushConstants) {
            entry.pushConstantType.clear();
        } else {
            std::set<std::string> distinctTypes;
            for (const auto& pc : pushConstantParameters) {
                distinctTypes.insert(pc.type);
            }

            const bool needsResolution = importedPushConstants || distinctTypes.size() > 1;
            if (!needsResolution) {
                entry.pushConstantType = *distinctTypes.begin();
            } else {
                // The runtime links before compiling. An unlinked composite can succeed while
                // silently dropping push-constant blocks from imported modules, so link first.
                Slang::ComPtr<slang::IComponentType> linked;
                Slang::ComPtr<slang::IBlob> linkDiagnostics;
                if (SLANG_FAILED(program->link(linked.writeRef(), linkDiagnostics.writeRef())) ||
                    !linked) {
                    reportDiagnostics("failed to link an entry for push-constant resolution",
                                      linkDiagnostics);
                    return 1;
                }

                Slang::ComPtr<slang::IBlob> code;
                Slang::ComPtr<slang::IBlob> codeDiagnostics;
                const SlangResult codeResult =
                    linked->getEntryPointCode(0, 0, code.writeRef(), codeDiagnostics.writeRef());
                if (SLANG_FAILED(codeResult) || !code) {
                    reportDiagnostics("failed to compile an entry for push-constant resolution",
                                      codeDiagnostics);
                    return 1;
                }

                SpirvVariableUsage usage;
                std::string scanError;
                if (!scanSpirvVariables(code->getBufferPointer(), code->getBufferSize(), usage,
                                        scanError)) {
                    std::cerr << "pictor_reflect: entry " << entry.name
                              << ": cannot read the compiled SPIR-V: " << scanError << "\n";
                    return 1;
                }
                if (usage.unnamed_push_constant) {
                    std::cerr << "pictor_reflect: entry " << entry.name
                              << " uses an unnamed push-constant block; the block cannot be "
                                 "matched to the reflected type\n";
                    return 1;
                }

                std::set<std::string> usedTypes;
                for (const auto& used : usage.push_constants) {
                    const PushConstantParameter* matched = nullptr;

                    if (!used.type.empty()) {
                        for (const auto& pc : pushConstantParameters) {
                            if (pc.type == used.type) {
                                matched = &pc;
                                break;
                            }
                        }
                        if (!matched) {
                            std::cerr << "pictor_reflect: entry " << entry.name
                                      << " uses a push-constant block of type '" << used.type
                                      << "', which is not declared in the entry file; "
                                         "push constants must be declared in the entry file\n";
                            return 1;
                        }
                        if (!used.name.empty()) {
                            for (const auto& pc : pushConstantParameters) {
                                if (pc.name == used.name && pc.type != used.type) {
                                    std::cerr << "pictor_reflect: entry " << entry.name
                                              << " uses push-constant variable '" << used.name
                                              << "' with type '" << used.type
                                              << "' but the entry file declares it as '" << pc.type
                                              << "'\n";
                                    return 1;
                                }
                            }
                        }
                    } else {
                        for (const auto& pc : pushConstantParameters) {
                            if (pc.name == used.name) {
                                matched = &pc;
                                break;
                            }
                        }
                        if (!matched) {
                            std::cerr << "pictor_reflect: entry " << entry.name
                                      << " uses push-constant variable '" << used.name
                                      << "', which is not in the entry file's program layout; "
                                         "push constants must be declared in the entry file\n";
                            return 1;
                        }
                    }

                    if (matched->type.empty()) {
                        std::cerr << "pictor_reflect: entry " << entry.name
                                  << " uses a push-constant block with no reflected type name\n";
                        return 1;
                    }
                    usedTypes.insert(matched->type);
                }

                if (usedTypes.size() > 1) {
                    std::cerr << "pictor_reflect: entry " << entry.name << " uses "
                              << usedTypes.size()
                              << " distinct push-constant layouts; the executor supports one\n";
                    return 1;
                }
                if (usedTypes.size() == 1) {
                    entry.pushConstantType = *usedTypes.begin();
                } else if (distinctTypes.size() == 1) {
                    // One local layout: keep the fast-path behavior and bind it to every entry,
                    // including one that does not reference it.
                    entry.pushConstantType = *distinctTypes.begin();
                } else {
                    entry.pushConstantType.clear();
                }
            }
        }

        if (entry.stage == SLANG_STAGE_FRAGMENT) {
            if (auto* result = entryLayout->getResultVarLayout()) {
                auto* resultLayout = result->getTypeLayout();
                if (resultLayout->getKind() == slang::TypeReflection::Kind::Struct) {
                    entry.colorAttachmentCount = resultLayout->getFieldCount();
                } else if (resultLayout->getKind() == slang::TypeReflection::Kind::Scalar ||
                           resultLayout->getKind() == slang::TypeReflection::Kind::Vector) {
                    entry.colorAttachmentCount = 1;
                }
            }
        }

        entries.push_back(std::move(entry));
    }

    if (entries.empty()) {
        std::cerr << "pictor_reflect: no entry points found in " << options.input << "\n";
        return 1;
    }

    if (options.dump) {
        dumpReflection(options.input, entries, structs);
        std::cout << "sources: " << closure.size() << "\n";
        for (const auto& entry : closure) {
            std::cout << "  " << entry.virtualPath << " <- " << entry.realPath.string() << "\n";
        }
        if (!tunables.empty()) {
            std::cout << "tunables: " << tunables.size() << "\n";
            for (const auto& tunable : tunables) {
                std::cout << "  " << tunable.name << " = " << tunable.value << " (" << tunable.type
                          << ")\n";
            }
        }
    }

    std::string delimiter = "pictor";
    for (;;) {
        bool clash = false;
        for (const auto& entry : closure) {
            if (entry.contents.find(")" + delimiter + "\"") != std::string::npos) {
                clash = true;
                break;
            }
        }
        if (!clash) {
            break;
        }
        delimiter += '_';
    }

    std::ostringstream slangHeader;
    slangHeader << "#pragma once\n#include \"pictor_core/execution/shader_sources.hpp\"\n\n"
                << "#include <array>\n#include <string_view>\n\n"
                << "namespace " << base << "_shader_sources {\n\n"
                << "inline constexpr std::array<std::string_view, " << searchPathDirs.size()
                << "> cSearchPaths{";
    for (size_t i = 0; i < searchPathDirs.size(); ++i) {
        if (i > 0) {
            slangHeader << ", ";
        }
        slangHeader << "\"/sp" << i << "\"";
    }
    slangHeader << "};\n\n"
                << "inline constexpr std::array<pictor::ShaderSourceFile, " << closure.size()
                << "> cFiles{{\n";
    for (const auto& entry : closure) {
        slangHeader << "    {\"" << entry.virtualPath << "\", R\"" << delimiter << "(\n"
                    << entry.contents << "\n)" << delimiter << "\"},\n";
    }
    slangHeader << "}};\n\n"
                << "inline constexpr pictor::ShaderSources cSet{\"" << entryVirtualPath
                << "\", cSearchPaths, cFiles};\n\n"
                << "} // namespace " << base << "_shader_sources\n";

    std::ostringstream shaderHeader;
    shaderHeader << "#pragma once\n#include \"" << base << ".slang.h\"\n"
                 << "#include \"pictor_core/execution/compute_executor.hpp\"\n"
                 << "#include \"pictor_core/execution/mesh_executor.hpp\"\n";
    if (!tunables.empty()) {
        shaderHeader << "#include \"pictor_core/execution/shader_defines.hpp\"\n"
                     << "\n#include <string>\n";
    }
    shaderHeader << "\nnamespace " << base << "_shader {\n\n";
    if (!tunables.empty()) {
        shaderHeader << "using pictor::ShaderDefines;\n\n";
    }

    for (const auto& [name, layout] : structs.structs()) {
        shaderHeader << renderStruct(name, layout) << "\n";
    }

    // Before the traits so each one can alias the struct with `using Tunables = ...`.
    if (!tunables.empty()) {
        shaderHeader << renderTunables(traitPrefix(base) + "Tunables", tunables) << "\n";
    }

    struct Group {
        std::string prefix;
        std::vector<const EntryInfo*> entries;
    };
    std::vector<Group> groups;
    for (const auto& entry : entries) {
        std::string prefix = entryPrefix(entry.name);
        auto it = std::ranges::find_if(groups,
                                       [&](const Group& group) { return group.prefix == prefix; });
        if (it == groups.end()) {
            groups.push_back(Group{prefix, {&entry}});
        } else {
            it->entries.push_back(&entry);
        }
    }

    for (const auto& group : groups) {
        const EntryInfo* computeEntry = nullptr;
        const EntryInfo* taskEntry    = nullptr;
        const EntryInfo* meshEntry    = nullptr;
        for (const auto* entry : group.entries) {
            if (entry->stage == SLANG_STAGE_COMPUTE && !computeEntry) {
                computeEntry = entry;
            }
            if (entry->stage == SLANG_STAGE_AMPLIFICATION && !taskEntry) {
                taskEntry = entry;
            }
            if (entry->stage == SLANG_STAGE_MESH && !meshEntry) {
                meshEntry = entry;
            }
        }

        std::string trait = traitPrefix(group.prefix);
        if (computeEntry) {
            std::string params =
                computeEntry->pushConstantType.empty() ? "void" : computeEntry->pushConstantType;
            shaderHeader << "struct " << trait << "Compute {\n"
                         << "    static constexpr std::string_view entry_point = \""
                         << computeEntry->name << "\";\n"
                         << "    static constexpr const pictor::ShaderSources& sources = " << base
                         << "_shader_sources::cSet;\n";
            if (!tunables.empty()) {
                shaderHeader << "    using Tunables = " << traitPrefix(base) << "Tunables;\n";
            }
            shaderHeader << "};\n\n"
                         << "using " << trait << "ComputeExecutor = pictor::ComputeExecutor<"
                         << trait << "Compute, " << params << ">;\n\n";
        }
        if (taskEntry && meshEntry) {
            if (!taskEntry->pushConstantType.empty() && !meshEntry->pushConstantType.empty() &&
                taskEntry->pushConstantType != meshEntry->pushConstantType) {
                std::cerr << "pictor_reflect: task '" << taskEntry->name << "' uses "
                          << taskEntry->pushConstantType << " but mesh '" << meshEntry->name
                          << "' uses " << meshEntry->pushConstantType
                          << "; a task/mesh pair must share one push-constant layout\n";
                return 1;
            }
            std::string params = taskEntry->pushConstantType.empty() ? meshEntry->pushConstantType
                                                                     : taskEntry->pushConstantType;
            if (params.empty()) {
                params = "void";
            }
            shaderHeader << "struct " << trait << "Mesh {\n"
                         << "    static constexpr std::string_view task_entry = \""
                         << taskEntry->name << "\";\n"
                         << "    static constexpr std::string_view mesh_entry = \""
                         << meshEntry->name << "\";\n"
                         << "    static constexpr const pictor::ShaderSources& sources = " << base
                         << "_shader_sources::cSet;\n";
            if (!tunables.empty()) {
                shaderHeader << "    using Tunables = " << traitPrefix(base) << "Tunables;\n";
            }
            shaderHeader << "};\n\n"
                         << "using " << upperCase(group.prefix)
                         << "Executor = pictor::MeshExecutor<" << trait << "Mesh, " << params
                         << ">;\n\n";
        }
    }

    shaderHeader << "} // namespace " << base << "_shader\n";

    std::error_code ec;
    std::filesystem::create_directories(options.outdir, ec);
    if (ec) {
        std::cerr << "pictor_reflect: cannot create " << options.outdir << "\n";
        return 1;
    }

    std::filesystem::path outdir(options.outdir);
    if (!writeFile(outdir / (base + ".slang.h"), slangHeader.str()) ||
        !writeFile(outdir / (base + ".shader.h"), shaderHeader.str())) {
        std::cerr << "pictor_reflect: failed to write headers into " << options.outdir << "\n";
        return 1;
    }

    if (!options.depfile.empty()) {
        std::ostringstream depfile;
        depfile << depfileEscape((outdir / (base + ".shader.h")).string()) << ' '
                << depfileEscape((outdir / (base + ".slang.h")).string()) << ':';
        for (const auto& entry : closure) {
            depfile << ' ' << depfileEscape(entry.realPath.string());
        }
        depfile << '\n';
        if (!writeFile(options.depfile, depfile.str())) {
            std::cerr << "pictor_reflect: failed to write the depfile " << options.depfile << "\n";
            return 1;
        }
    }

    return 0;
}
