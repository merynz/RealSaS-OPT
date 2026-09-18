#include "../src/runtime_v3_reference.h"

#include <png.h>

#include <cstdint>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void write_png(
    const std::string& path,
    uint32_t width,
    uint32_t height,
    const std::vector<uint8_t>& rgba) {
    png_image image{};
    image.version = PNG_IMAGE_VERSION;
    image.width = width;
    image.height = height;
    image.format = PNG_FORMAT_RGBA;
    if (!png_image_write_to_file(&image, path.c_str(), 0, rgba.data(), 0, nullptr)) {
        throw std::runtime_error(std::string("PNG write failed: ") + image.message);
    }
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr
            << "Usage: realsas_runtime_v3_demo <package.rss|package.realsas|dir> "
            << "[--clip id|intent] [--view V0] [--time seconds] [--out frame.png]\n";
        return 2;
    }

    std::string package = argv[1];
    std::string clip_name = "idle";
    std::string view_name = "V0";
    std::string out = "realsas_v3_frame.png";
    float time = 0.0f;
    for (int i = 2; i < argc; ++i) {
        const std::string key = argv[i];
        if (key == "--clip" && i + 1 < argc) clip_name = argv[++i];
        else if (key == "--view" && i + 1 < argc) view_name = argv[++i];
        else if (key == "--time" && i + 1 < argc) time = std::stof(argv[++i]);
        else if (key == "--out" && i + 1 < argc) out = argv[++i];
        else {
            std::cerr << "Unknown argument: " << key << "\n";
            return 2;
        }
    }

    try {
        realsas::runtime_v3::ReferenceRuntime runtime(package);
        std::cout << "schema=" << runtime.binary_schema() << "\n";
        std::cout << "coordinate_system=" << runtime.coordinate_system()
                  << " units_per_pixel=" << runtime.units_per_pixel()
                  << " feature_flags=0x" << std::hex << runtime.feature_flags() << std::dec << "\n";
        std::cout << "source_binding_sha256=" << runtime.source_binding_sha256() << "\n";
        std::cout << "source_proof_bundle_hash=" << runtime.source_proof_bundle_hash() << "\n";
        std::cout << "playback_contract_hash=" << runtime.playback_contract_hash() << "\n";
        std::cout << "reference_raster_contract_hash=" << runtime.reference_raster_contract_hash() << "\n";
        std::cout << "alpha_cutout_threshold=" << runtime.alpha_cutout_threshold() << "\n";
        std::cout << "views=" << runtime.view_count() << " clips=" << runtime.clip_count() << "\n";

        for (uint32_t i = 0; i < runtime.clip_count(); ++i) {
            std::cout << "clip[" << i << "]=" << runtime.clip_display_name(i)
                      << " intent=" << runtime.clip_intent(i)
                      << " duration=" << runtime.clip_duration(i)
                      << " qualified=" << runtime.clip_qualified(i) << "\n";
        }

        const uint32_t ci = runtime.find_clip(clip_name);
        const uint32_t vi = runtime.find_view(view_name);
        if (ci == UINT32_MAX) throw std::runtime_error("Clip not found: " + clip_name);
        if (vi == UINT32_MAX) throw std::runtime_error("View not found: " + view_name);

        const uint32_t width = runtime.view_width(vi);
        const uint32_t height = runtime.view_height(vi);
        const auto rgba = runtime.render_rgba(ci, vi, time);
        if (rgba.size() != static_cast<size_t>(width) * height * 4) {
            throw std::runtime_error("Runtime-v3 renderer returned invalid RGBA buffer size");
        }

        const std::filesystem::path target(out);
        if (!target.parent_path().empty()) std::filesystem::create_directories(target.parent_path());
        write_png(out, width, height, rgba);
        std::cout << "rendered=" << out
                  << " view=" << view_name
                  << " clip=" << clip_name
                  << " time=" << time
                  << " size=" << width << "x" << height
                  << " renderer=RUNTIME_V3_POSED_DEPTH_REFERENCE\n";
        return 0;
    } catch (const std::exception& exc) {
        std::cerr << "RealSaS runtime-v3 error: " << exc.what() << "\n";
        return 1;
    }
}
