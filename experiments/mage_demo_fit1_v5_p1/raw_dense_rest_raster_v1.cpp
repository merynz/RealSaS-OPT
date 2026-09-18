#include "../../runtime/realsas_cpp/src/reference_raster_v3.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

using realsas::reference_raster_v3::DepthSample;
using realsas::reference_raster_v3::Vertex;
using realsas::reference_raster_v3::commit_depth;
using realsas::reference_raster_v3::cover_pixel_center;
using realsas::reference_raster_v3::depth_test_passes;
using realsas::reference_raster_v3::interpolate_depth;

namespace {

constexpr uint32_t kInputMagic = 0x31524452u;   // "RDR1"
constexpr uint32_t kOutputMagic = 0x314F4452u;  // "RDO1"
constexpr uint32_t kVersion = 1u;

template <typename T>
void read_exact(std::ifstream& in, T& value) {
    in.read(reinterpret_cast<char*>(&value), sizeof(T));
    if (!in) throw std::runtime_error("RAW_DENSE_RASTER_INPUT_TRUNCATED");
}

template <typename T>
void read_array(std::ifstream& in, std::vector<T>& values) {
    if (values.empty()) return;
    in.read(reinterpret_cast<char*>(values.data()),
            static_cast<std::streamsize>(values.size() * sizeof(T)));
    if (!in) throw std::runtime_error("RAW_DENSE_RASTER_INPUT_ARRAY_TRUNCATED");
}

template <typename T>
void write_exact(std::ofstream& out, const T& value) {
    out.write(reinterpret_cast<const char*>(&value), sizeof(T));
    if (!out) throw std::runtime_error("RAW_DENSE_RASTER_OUTPUT_WRITE_FAILED");
}

template <typename T>
void write_array(std::ofstream& out, const std::vector<T>& values) {
    if (values.empty()) return;
    out.write(reinterpret_cast<const char*>(values.data()),
              static_cast<std::streamsize>(values.size() * sizeof(T)));
    if (!out) throw std::runtime_error("RAW_DENSE_RASTER_OUTPUT_ARRAY_WRITE_FAILED");
}

struct Face {
    uint32_t a = 0;
    uint32_t b = 0;
    uint32_t c = 0;
};

}  // namespace

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "Usage: raw_dense_rest_raster_v1 <input.bin> <output.bin>\n";
        return 2;
    }

    try {
        std::ifstream in(argv[1], std::ios::binary);
        if (!in) throw std::runtime_error("RAW_DENSE_RASTER_INPUT_OPEN_FAILED");

        uint32_t magic = 0, version = 0, width = 0, height = 0;
        uint64_t vertex_count = 0, face_count = 0;
        read_exact(in, magic);
        read_exact(in, version);
        read_exact(in, width);
        read_exact(in, height);
        read_exact(in, vertex_count);
        read_exact(in, face_count);
        if (magic != kInputMagic || version != kVersion || width == 0 || height == 0) {
            throw std::runtime_error("RAW_DENSE_RASTER_INPUT_HEADER_INVALID");
        }
        if (vertex_count < 3 || face_count == 0) {
            throw std::runtime_error("RAW_DENSE_RASTER_INPUT_EMPTY");
        }

        std::vector<float> xyz(static_cast<size_t>(vertex_count) * 3u);
        std::vector<uint32_t> face_raw(static_cast<size_t>(face_count) * 3u);
        read_array(in, xyz);
        read_array(in, face_raw);

        std::vector<Vertex> vertices(static_cast<size_t>(vertex_count));
        for (uint64_t i = 0; i < vertex_count; ++i) {
            const float x = xyz[static_cast<size_t>(i) * 3u + 0u];
            const float y = xyz[static_cast<size_t>(i) * 3u + 1u];
            const float z = xyz[static_cast<size_t>(i) * 3u + 2u];
            if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z)) {
                throw std::runtime_error("RAW_DENSE_RASTER_NONFINITE_VERTEX");
            }
            vertices[static_cast<size_t>(i)] = Vertex{x, y, z, 0.0f, 0.0f};
        }

        std::vector<Face> faces(static_cast<size_t>(face_count));
        for (uint64_t i = 0; i < face_count; ++i) {
            Face f{
                face_raw[static_cast<size_t>(i) * 3u + 0u],
                face_raw[static_cast<size_t>(i) * 3u + 1u],
                face_raw[static_cast<size_t>(i) * 3u + 2u],
            };
            if (f.a >= vertex_count || f.b >= vertex_count || f.c >= vertex_count) {
                throw std::runtime_error("RAW_DENSE_RASTER_FACE_INDEX_OUT_OF_RANGE");
            }
            faces[static_cast<size_t>(i)] = f;
        }

        const size_t pixel_count = static_cast<size_t>(width) * static_cast<size_t>(height);
        std::vector<DepthSample> depth(pixel_count);
        std::vector<int32_t> first_hit_face(pixel_count, -1);

        uint64_t tested_faces = 0;
        uint64_t bbox_pixel_tests = 0;
        for (uint64_t face_id = 0; face_id < face_count; ++face_id) {
            const Face& f = faces[static_cast<size_t>(face_id)];
            const Vertex& a = vertices[f.a];
            const Vertex& b = vertices[f.b];
            const Vertex& c = vertices[f.c];

            if (a.z <= 0.0f && b.z <= 0.0f && c.z <= 0.0f) continue;

            const float min_x = std::min({a.x, b.x, c.x});
            const float max_x = std::max({a.x, b.x, c.x});
            const float min_y = std::min({a.y, b.y, c.y});
            const float max_y = std::max({a.y, b.y, c.y});
            if (max_x < 0.0f || max_y < 0.0f ||
                min_x > static_cast<float>(width) ||
                min_y > static_cast<float>(height)) {
                continue;
            }

            int x0 = std::max(0, static_cast<int>(std::floor(min_x)) - 1);
            int x1 = std::min(static_cast<int>(width) - 1,
                              static_cast<int>(std::ceil(max_x)) + 1);
            int y0 = std::max(0, static_cast<int>(std::floor(min_y)) - 1);
            int y1 = std::min(static_cast<int>(height) - 1,
                              static_cast<int>(std::ceil(max_y)) + 1);
            if (x0 > x1 || y0 > y1) continue;
            ++tested_faces;

            for (int y = y0; y <= y1; ++y) {
                for (int x = x0; x <= x1; ++x) {
                    ++bbox_pixel_tests;
                    const auto bc = cover_pixel_center(a, b, c, x, y);
                    if (!bc.covered) continue;
                    const float z = interpolate_depth(bc, a, b, c);
                    if (!std::isfinite(z) || z <= 0.0f) continue;
                    const size_t p = static_cast<size_t>(y) * width + static_cast<size_t>(x);
                    if (!depth_test_passes(depth[p], z, 0u)) continue;
                    commit_depth(depth[p], z, 0u);
                    first_hit_face[p] = static_cast<int32_t>(face_id);
                }
            }
        }

        uint64_t covered_pixels = 0;
        std::vector<float> depth_out(pixel_count, std::numeric_limits<float>::infinity());
        for (size_t i = 0; i < pixel_count; ++i) {
            if (depth[i].occupied) {
                ++covered_pixels;
                depth_out[i] = depth[i].z;
            }
        }

        std::ofstream out(argv[2], std::ios::binary);
        if (!out) throw std::runtime_error("RAW_DENSE_RASTER_OUTPUT_OPEN_FAILED");
        write_exact(out, kOutputMagic);
        write_exact(out, kVersion);
        write_exact(out, width);
        write_exact(out, height);
        write_array(out, first_hit_face);
        write_array(out, depth_out);

        std::cout << "RAW_DENSE_RASTER_PASS"
                  << " vertices=" << vertex_count
                  << " faces=" << face_count
                  << " tested_faces=" << tested_faces
                  << " covered_pixels=" << covered_pixels
                  << " bbox_pixel_tests=" << bbox_pixel_tests
                  << "\n";
        return 0;
    } catch (const std::exception& exc) {
        std::cerr << "RAW_DENSE_RASTER_FAIL:" << exc.what() << "\n";
        return 1;
    }
}
