// Semantic presentation runtime gate.
//
// Stage42 compiles inter-slot drawing order into an effective depth sort key
// while retaining canonical camera depth only inside each semantic slot. The
// public RSS therefore labels those numbers as a semantic sort key, never as
// physical camera depth. This wrapper validates that contract, then delegates
// byte production to the already-proven source-owned raster implementation via
// an internal compatibility RSS whose manifest restores the legacy label only
// for that implementation boundary. No order is inferred at runtime.

#include <cstdio>
#include <string>
#include <vector>
#include <unistd.h>

#define main realsas_depth_base_main
#include "runtime_v2_caa_reference.cpp"
#undef main

namespace {

constexpr const char* kSemanticOrderContract =
    "SEMANTIC_SLOT_ORDER_THEN_CANONICAL_DEPTH_WITHIN_SLOT_V1";
constexpr const char* kSemanticOrderOperator =
    "SEMANTIC_SLOT_ORDER_WITH_INTRA_SLOT_CANONICAL_DEPTH_V1";
constexpr const char* kSemanticOrderEncoding = "EFFECTIVE_DEPTH_SORT_KEY_V1";
constexpr const char* kSemanticDepthContract =
    "COMPILED_SEMANTIC_EFFECTIVE_DEPTH_ASCENDING_V1";
constexpr const char* kBaseDepthContract =
    "CANONICAL_CAMERA_DEPTH_ASCENDING__UNRESOLVED_TIES_FAIL_V1";

template <typename T>
void append_scalar(std::vector<std::uint8_t>& out, const T& value) {
    const auto* begin = reinterpret_cast<const std::uint8_t*>(&value);
    out.insert(out.end(), begin, begin + sizeof(T));
}

void write_compat_rss(const std::string& path, const Entries& entries) {
    std::vector<std::uint8_t> raw;
    static constexpr char magic[] = "RSASV2R1";
    raw.insert(raw.end(), magic, magic + 8);
    const auto count = static_cast<std::uint32_t>(entries.size());
    append_scalar(raw, count);
    for (const auto& item : entries) {
        if (item.first.size() > std::numeric_limits<std::uint16_t>::max())
            throw std::runtime_error("SEMANTIC_COMPAT_ENTRY_NAME_TOO_LONG");
        const auto name_size = static_cast<std::uint16_t>(item.first.size());
        const auto payload_size = static_cast<std::uint64_t>(item.second.size());
        append_scalar(raw, name_size);
        raw.insert(raw.end(), item.first.begin(), item.first.end());
        append_scalar(raw, payload_size);
        raw.insert(raw.end(), item.second.begin(), item.second.end());
    }
    write_file(path, raw.data(), raw.size());
}

std::string compatibility_manifest(const std::vector<std::uint8_t>& payload) {
    std::string text(payload.begin(), payload.end());
    const std::string semantic =
        std::string("depth_ownership_contract=") + kSemanticDepthContract;
    const std::string legacy =
        std::string("depth_ownership_contract=") + kBaseDepthContract;
    const auto pos = text.find(semantic);
    if (pos == std::string::npos)
        throw std::runtime_error("SEMANTIC_DEPTH_SORT_KEY_CONTRACT_MISSING");
    if (text.find(semantic, pos + semantic.size()) != std::string::npos)
        throw std::runtime_error("SEMANTIC_DEPTH_SORT_KEY_CONTRACT_DUPLICATE");
    text.replace(pos, semantic.size(), legacy);
    return text;
}

int delegate_semantic_package(int argc, char** argv, const Entries& entries) {
    auto compat = entries;
    const auto manifest_it = compat.find("manifest.txt");
    if (manifest_it == compat.end())
        throw std::runtime_error("SEMANTIC_MANIFEST_MISSING");
    const auto text = compatibility_manifest(manifest_it->second);
    manifest_it->second.assign(text.begin(), text.end());

    const std::string path = std::string(argv[1]) + ".semantic_base_" +
        std::to_string(static_cast<long long>(::getpid())) + ".rss";
    write_compat_rss(path, compat);

    std::vector<std::string> owned;
    owned.reserve(static_cast<std::size_t>(argc));
    for (int i = 0; i < argc; ++i) owned.emplace_back(argv[i] ? argv[i] : "");
    owned[1] = path;
    std::vector<char*> forwarded;
    forwarded.reserve(owned.size());
    for (auto& value : owned) forwarded.push_back(value.data());

    int rc = 1;
    try {
        rc = realsas_depth_base_main(argc, forwarded.data());
    } catch (...) {
        std::remove(path.c_str());
        throw;
    }
    std::remove(path.c_str());
    return rc;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) return realsas_depth_base_main(argc, argv);

    const auto entries = parse_rss(argv[1]);
    const auto manifest_it = entries.find("manifest.txt");
    if (manifest_it == entries.end())
        return realsas_depth_base_main(argc, argv);
    const auto manifest = parse_manifest(manifest_it->second);
    const auto semantic = manifest.find("semantic_order_contract");
    if (semantic == manifest.end())
        return realsas_depth_base_main(argc, argv);  // matched V6/V5 baseline lane

    if (semantic->second != kSemanticOrderContract)
        throw std::runtime_error("SEMANTIC_ORDER_CONTRACT_INVALID");
    if (manifest.at("semantic_order_operator_id") != kSemanticOrderOperator)
        throw std::runtime_error("SEMANTIC_ORDER_OPERATOR_INVALID");
    if (manifest.at("semantic_order_encoding") != kSemanticOrderEncoding)
        throw std::runtime_error("SEMANTIC_ORDER_ENCODING_INVALID");
    if (manifest.at("depth_ownership_contract") != kSemanticDepthContract)
        throw std::runtime_error("SEMANTIC_DEPTH_SORT_KEY_CONTRACT_INVALID");
    if (manifest.at("semantic_physical_depth_final_authority") != "false" ||
        manifest.at("semantic_sort_key_compiled_upstream") != "true" ||
        manifest.at("runtime_semantic_order_inference") != "false")
        throw std::runtime_error("SEMANTIC_RUNTIME_AUTHORITY_CONTRACT_INVALID");
    if (manifest.at("qualified_contact_contract_hash").size() != 64u ||
        manifest.at("presentation_relations_hash").size() != 64u)
        throw std::runtime_error("PRESENTATION_RELATION_HASH_INVALID");
    const auto entry_name = manifest.at("semantic_order_entry");
    if (entries.find(entry_name) == entries.end())
        throw std::runtime_error("SEMANTIC_ORDER_ENTRY_MISSING");

    const int rc = delegate_semantic_package(argc, argv, entries);
    if (rc == 0) {
        std::cout << "semantic_order_consumer=" << kSemanticOrderContract << "\n"
                  << "semantic_depth_sort_key_contract=" << kSemanticDepthContract << "\n"
                  << "semantic_physical_depth_final_authority=false\n";
    }
    return rc;
}
