// Semantic presentation runtime gate.
//
// The proven source-owned visual renderer remains the byte-producing consumer.
// This thin executable validates that Stage43 carried an explicit semantic
// drawing-order contract, then delegates to the same renderer. Stage42 encodes
// inter-slot order into the depth-sort key while preserving canonical depth
// inside each semantic slot; the base renderer therefore performs composition,
// not physical-Z authority, when this contract is present.

#define main realsas_depth_base_main
#include "runtime_v2_caa_reference.cpp"
#undef main

int main(int argc, char** argv) {
    if (argc >= 2) {
        const auto entries = parse_rss(argv[1]);
        const auto manifest_it = entries.find("manifest.txt");
        if (manifest_it != entries.end()) {
            const auto manifest = parse_manifest(manifest_it->second);
            const auto semantic = manifest.find("semantic_order_contract");
            if (semantic != manifest.end()) {
                const std::string expected =
                    "SEMANTIC_SLOT_ORDER_THEN_CANONICAL_DEPTH_WITHIN_SLOT_V1";
                if (semantic->second != expected)
                    throw std::runtime_error("SEMANTIC_ORDER_CONTRACT_INVALID");
                if (manifest.at("semantic_order_operator_id") !=
                    "SEMANTIC_SLOT_ORDER_WITH_INTRA_SLOT_CANONICAL_DEPTH_V1")
                    throw std::runtime_error("SEMANTIC_ORDER_OPERATOR_INVALID");
                if (manifest.at("semantic_order_encoding") !=
                    "EFFECTIVE_DEPTH_SORT_KEY_V1")
                    throw std::runtime_error("SEMANTIC_ORDER_ENCODING_INVALID");
                if (manifest.at("qualified_contact_contract_hash").size() != 64u ||
                    manifest.at("presentation_relations_hash").size() != 64u)
                    throw std::runtime_error("PRESENTATION_RELATION_HASH_INVALID");
                const auto entry_name = manifest.at("semantic_order_entry");
                if (entries.find(entry_name) == entries.end())
                    throw std::runtime_error("SEMANTIC_ORDER_ENTRY_MISSING");
                std::cout << "semantic_order_consumer=" << expected << "\n";
            }
        }
    }
    return realsas_depth_base_main(argc, argv);
}
