#include "realsas/realsas_runtime.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

namespace {

int fail(int code, const std::string& message) {
    std::cerr << "P0_NATIVE_PACKAGE_PROBE_FAIL[" << code << "]:" << message << "\n";
    return code;
}

bool same(const char* value, const char* expected) {
    return value && expected && std::strcmp(value, expected) == 0;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 4) {
        return fail(2, "usage: probe <package.rss> <expected_source_binding_sha256> <expected_proof_bundle_sha256>");
    }
    const char* package_path = argv[1];
    const char* expected_binding = argv[2];
    const char* expected_proof = argv[3];

    char error[2048] = {};
    RsRuntime* runtime = nullptr;
    const RsResult opened = rs_runtime_open(package_path, &runtime, error, sizeof(error));
    if (opened != RS_OK || !runtime) {
        return fail(3, std::string("open:") + error);
    }

    int result = 0;
    do {
        if (rs_runtime_abi_version() != 3u) { result = fail(4, "ABI version"); break; }
        if (!same(rs_runtime_binary_schema(runtime), "realSaS.RuntimeBinary.rsr.v2")) { result = fail(5, "binary schema"); break; }
        if (!same(rs_runtime_coordinate_system(runtime), "screen_2d.top_left.x_right.y_down.uv_top_left")) { result = fail(6, "coordinate system"); break; }
        if (!same(rs_runtime_source_manifest_sha256(runtime), expected_binding)) { result = fail(7, "source binding sha mismatch"); break; }
        if (!same(rs_runtime_source_payload_merkle_sha256(runtime), expected_proof)) { result = fail(8, "proof bundle sha mismatch"); break; }
        if (rs_runtime_view_count(runtime) != 8u) { result = fail(9, "view count"); break; }
        if (rs_runtime_clip_count(runtime) != 1u) { result = fail(10, "clip count"); break; }
        if (!rs_runtime_clip_qualified(runtime, 0u)) { result = fail(11, "clip not runtime-qualified"); break; }

        const char* clip_id = rs_runtime_clip_id(runtime, 0u);
        if (!clip_id || rs_runtime_find_clip(runtime, clip_id) != 0u) { result = fail(12, "clip lookup"); break; }
        const float sample_time = rs_runtime_clip_duration(runtime, 0u) * 0.25f;

        for (uint32_t view = 0; view < 8u; ++view) {
            const char* view_id = rs_runtime_view_id(runtime, view);
            if (!view_id || rs_runtime_find_view(runtime, view_id) != view) { result = fail(13, "view lookup"); break; }
            const uint32_t width = rs_runtime_view_width(runtime, view);
            const uint32_t height = rs_runtime_view_height(runtime, view);
            if (width == 0u || height == 0u || rs_runtime_mesh_count(runtime, view) == 0u) { result = fail(14, "view dimensions/mesh count"); break; }

            for (uint32_t mesh = 0; mesh < rs_runtime_mesh_count(runtime, view); ++mesh) {
                const uint32_t vertices = rs_runtime_mesh_vertex_count(runtime, view, mesh);
                if (vertices == 0u || rs_runtime_mesh_triangle_count(runtime, view, mesh) == 0u) { result = fail(15, "mesh dimensions"); break; }
                std::vector<float> sampled(static_cast<size_t>(vertices) * 2u, 0.0f);
                if (rs_runtime_sample_mesh_vertices(runtime, 0u, view, mesh, sample_time, sampled.data(), sampled.size()) != RS_OK) {
                    result = fail(16, "sample mesh vertices"); break;
                }
            }
            if (result) break;

            std::vector<uint8_t> rgba(static_cast<size_t>(width) * height * 4u, 0u);
            if (rs_runtime_render_rgba(runtime, 0u, view, sample_time, rgba.data(), rgba.size()) != RS_OK) {
                result = fail(17, "render rgba"); break;
            }
            const bool any_alpha = std::any_of(rgba.begin() + 3, rgba.end(), [index = size_t{3}](uint8_t) mutable {
                // This lambda shape is not useful for stride checking; handled below.
                ++index;
                return false;
            });
            (void)any_alpha;
            bool visible = false;
            for (size_t i = 3; i < rgba.size(); i += 4) {
                if (rgba[i] != 0u) { visible = true; break; }
            }
            if (!visible) { result = fail(18, "render produced no visible alpha"); break; }
        }
    } while (false);

    rs_runtime_close(runtime);
    if (result == 0) {
        std::cout << "PASS_CURRENT_V4_NATIVE_PACKAGE_OPEN_RENDER\n";
    }
    return result;
}
