#include "realsas/realsas_runtime.hpp"
#include <png.h>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>
#include <limits>

static void write_png(const std::string& path, uint32_t width, uint32_t height, const std::vector<uint8_t>& rgba) {
    png_image image{}; image.version = PNG_IMAGE_VERSION; image.width = width; image.height = height; image.format = PNG_FORMAT_RGBA;
    if (!png_image_write_to_file(&image, path.c_str(), 0, rgba.data(), 0, nullptr)) {
        throw std::runtime_error(std::string("PNG write failed: ") + image.message);
    }
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: realsas_runtime_demo <package.rss|package.realsas> [--clip id|intent] [--view V0] [--time seconds] [--out frame.png] [--post]\n";
        return 2;
    }
    std::string package = argv[1], clip_name = "idle", view_name = "V0", out = "realsas_frame.png"; float time = 0.0f; bool post = false;
    for (int i = 2; i < argc; ++i) {
        std::string key = argv[i];
        if (key == "--clip" && i + 1 < argc) clip_name = argv[++i];
        else if (key == "--view" && i + 1 < argc) view_name = argv[++i];
        else if (key == "--time" && i + 1 < argc) time = std::stof(argv[++i]);
        else if (key == "--out" && i + 1 < argc) out = argv[++i];
        else if (key == "--post") post = true;
        else { std::cerr << "Unknown argument: " << key << "\n"; return 2; }
    }
    try {
        realsas::Runtime runtime(package); const RsRuntime* r = runtime.get();
        std::cout << "abi_version=" << rs_runtime_abi_version() << "\n";
        std::cout << "schema=" << rs_runtime_binary_schema(r) << "\n";
        std::cout << "coordinate_system=" << rs_runtime_coordinate_system(r)
                  << " units_per_pixel=" << rs_runtime_units_per_pixel(r)
                  << " feature_flags=0x" << std::hex << rs_runtime_feature_flags(r) << std::dec << "\n";
        std::cout << "source_manifest_sha256=" << rs_runtime_source_manifest_sha256(r) << "\n";
        std::cout << "views=" << rs_runtime_view_count(r) << " clips=" << rs_runtime_clip_count(r) << "\n";
        for (uint32_t i = 0; i < rs_runtime_clip_count(r); ++i) {
            std::cout << "clip[" << i << "]=" << rs_runtime_clip_display_name(r, i)
                      << " intent=" << rs_runtime_clip_intent(r, i)
                      << " duration=" << rs_runtime_clip_duration(r, i)
                      << " qualified=" << rs_runtime_clip_qualified(r, i) << "\n";
        }
        const uint32_t ci = rs_runtime_find_clip(r, clip_name.c_str()), vi = rs_runtime_find_view(r, view_name.c_str());
        if (ci == UINT32_MAX) throw std::runtime_error("Clip not found: " + clip_name);
        if (vi == UINT32_MAX) throw std::runtime_error("View not found: " + view_name);
        const uint32_t width = rs_runtime_view_width(r, vi), height = rs_runtime_view_height(r, vi);
        std::vector<uint8_t> rgba(static_cast<size_t>(width) * height * 4);
        RsResult render_result = RS_OK;
        if (post) { const RsPostProcessSettings settings = rs_runtime_default_post_process_settings(); render_result = rs_runtime_render_rgba_ex(r, ci, vi, time, &settings, rgba.data(), rgba.size()); }
        else render_result = rs_runtime_render_rgba(r, ci, vi, time, rgba.data(), rgba.size());
        if (render_result != RS_OK) throw std::runtime_error("Render failed");
        std::filesystem::create_directories(std::filesystem::path(out).parent_path().empty() ? "." : std::filesystem::path(out).parent_path());
        write_png(out, width, height, rgba);
        std::cout << "rendered=" << out << " view=" << view_name << " clip=" << clip_name << " time=" << time << " post=" << post << " size=" << width << "x" << height << "\n";
        return 0;
    } catch (const std::exception& exc) {
        std::cerr << "RealSaS runtime error: " << exc.what() << "\n"; return 1;
    }
}
