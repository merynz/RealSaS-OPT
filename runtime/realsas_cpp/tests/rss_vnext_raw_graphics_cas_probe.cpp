// Real RSS.vNext two-original-image Graphics binary CAS probe.
//
// A native executable MUST consume the Python-created Graphics candidate,
// not merely revalidate a Python JSON object or a separate handwritten
// C++ fixture. Virtual directions are *not* source visibility observations.
#include "../src/graphics_v1.hpp"
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <vector>

int main(int argc, char** argv) {
    try {
        if (argc != 2) throw std::runtime_error("RAW_GFX_CAS_FILE_REQUIRED");
        std::ifstream input(argv[1], std::ios::binary);
        if (!input) throw std::runtime_error("RAW_GFX_CAS_OPEN_FAILED");
        const std::vector<std::uint8_t> bytes{
            std::istreambuf_iterator<char>(input),
            std::istreambuf_iterator<char>()
        };
        const auto graphics = realsas::graphics_v1::parse(bytes);
        if (graphics.views.size() != 8 || graphics.face_fragment_index.empty()
            || graphics.fragment_ids.empty())
            throw std::runtime_error("RAW_GFX_EIGHT_VIRTUAL_VIEW_DOMAIN_REQUIRED");
        for (std::uint32_t v = 0; v < 8; ++v) {
            if (graphics.views[v].view_index != v ||
                graphics.views[v].topological_order.size() != graphics.fragment_ids.size())
                throw std::runtime_error("RAW_GFX_VIRTUAL_VIEW_INDEX_OR_SORT_DRIFT");
            for (std::uint32_t face = 0; face < graphics.face_fragment_index.size(); ++face) {
                // Unknown source visibility means every fragment is a
                // *candidate* for depth testing, not a source-visible label.
                if (!realsas::graphics_v1::face_visible(graphics, v, face))
                    throw std::runtime_error("RAW_GFX_UNOBSERVED_FACE_WRONGLY_CULLED");
            }
            for (std::uint32_t a = 0; a < graphics.face_fragment_index.size(); ++a)
                for (std::uint32_t b = 0; b < graphics.face_fragment_index.size(); ++b)
                    if (a != b && realsas::graphics_v1::explicit_precedence(graphics, v, a, b))
                        throw std::runtime_error("RAW_GFX_UNOBSERVED_ORDER_FORGED");
        }
        std::cout << "RAW_GFX_BINARY_CAS_NATIVE_CONSUMPTION_PASS\n";
        std::cout << "faces=" << graphics.face_fragment_index.size()
                  << " fragments=" << graphics.fragment_ids.size()
                  << " views=" << graphics.views.size() << "\n";
        std::cout << "mesh=" << graphics.canonical_mesh_binding_hash << "\n";
        std::cout << "graphics=" << graphics.compiled_graphics_binding_hash << "\n";
        std::cout << "runtime=" << graphics.runtime_graphics_hash << "\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << "\n";
        return 2;
    }
}
