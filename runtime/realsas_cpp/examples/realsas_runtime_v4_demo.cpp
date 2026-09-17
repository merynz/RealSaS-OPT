#include "../src/runtime_v4_reference.h"
#include <png.h>

#include <cstdint>
#include <filesystem>
#include <fstream>
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

std::vector<std::string> split_tabs(const std::string& line) {
    std::vector<std::string> out;
    size_t start = 0;
    for (;;) {
        const size_t pos = line.find('\t', start);
        if (pos == std::string::npos) {
            out.push_back(line.substr(start));
            return out;
        }
        out.push_back(line.substr(start, pos - start));
        start = pos + 1;
    }
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr
            << "Usage: realsas_runtime_v4_demo <package.rss|dir> "
            << "[--clip id] [--view V0] [--time seconds] [--out frame.png] "
            << "[--batch-file requests.tsv]\n";
        return 2;
    }

    std::string package = argv[1];
    std::string clip_name = "idle";
    std::string view_name = "V0";
    std::string out = "realsas_v4_frame.png";
    std::string batch_file;
    float time = 0.0f;

    for (int i = 2; i < argc; ++i) {
        const std::string key = argv[i];
        if (key == "--clip" && i + 1 < argc) clip_name = argv[++i];
        else if (key == "--view" && i + 1 < argc) view_name = argv[++i];
        else if (key == "--time" && i + 1 < argc) time = std::stof(argv[++i]);
        else if (key == "--out" && i + 1 < argc) out = argv[++i];
        else if (key == "--batch-file" && i + 1 < argc) batch_file = argv[++i];
        else {
            std::cerr << "Unknown argument: " << key << "\n";
            return 2;
        }
    }

    try {
        realsas::runtime_v4::ReferenceRuntime runtime(package);

        auto render_one = [&](const std::string& clip,
                              const std::string& view,
                              float t,
                              const std::string& target_text) {
            const uint32_t ci = runtime.find_clip(clip);
            const uint32_t vi = runtime.find_view(view);
            if (ci == UINT32_MAX) throw std::runtime_error("Clip not found: " + clip);
            if (vi == UINT32_MAX) throw std::runtime_error("View not found: " + view);
            auto rgba = runtime.render_rgba(ci, vi, t);
            const std::filesystem::path target(target_text);
            if (!target.parent_path().empty()) {
                std::filesystem::create_directories(target.parent_path());
            }
            write_png(target_text, runtime.view_width(vi), runtime.view_height(vi), rgba);
        };

        std::cout << "schema=" << runtime.binary_schema() << "\n"
                  << "source_binding_sha256=" << runtime.source_binding_sha256() << "\n";

        if (!batch_file.empty()) {
            std::ifstream plan(batch_file);
            if (!plan) throw std::runtime_error("Cannot open batch file: " + batch_file);
            size_t request_count = 0;
            std::string line;
            while (std::getline(plan, line)) {
                if (line.empty()) continue;
                const auto fields = split_tabs(line);
                if (fields.size() != 4 || fields[0].empty() || fields[1].empty() || fields[3].empty()) {
                    throw std::runtime_error("Invalid batch row; expected clip<TAB>view<TAB>time<TAB>out");
                }
                const float row_time = std::stof(fields[2]);
                render_one(fields[0], fields[1], row_time, fields[3]);
                ++request_count;
            }
            if (request_count == 0) throw std::runtime_error("Batch file contains no render requests");
            std::cout << "batch_rendered=" << request_count
                      << " renderer=RUNTIME_V4_SHARED_CANONICAL_DEPTH_REFERENCE\n";
            return 0;
        }

        render_one(clip_name, view_name, time, out);
        std::cout << "rendered=" << out
                  << " view=" << view_name
                  << " clip=" << clip_name
                  << " time=" << time
                  << " renderer=RUNTIME_V4_SHARED_CANONICAL_DEPTH_REFERENCE\n";
        return 0;
    } catch (const std::exception& exc) {
        std::cerr << "RealSaS runtime-v4 error: " << exc.what() << "\n";
        return 1;
    }
}
