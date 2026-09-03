#include "realsas/realsas_runtime.h"

#include <archive.h>
#include <archive_entry.h>
#include <png.h>
#include <zlib.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

struct Vec2 { float x = 0.0f; float y = 0.0f; };
struct Vertex { Vec2 rest; Vec2 uv; };
struct Mesh {
    std::string id;
    std::vector<Vertex> vertices;
    std::vector<std::array<uint32_t, 3>> triangles;
};
struct View {
    std::string id;
    std::string texture_path;
    uint32_t texture_crc32 = 0;
    uint32_t width = 0;
    uint32_t height = 0;
    std::vector<uint8_t> texture_rgba;
    std::vector<Mesh> meshes;
};
struct FrameView {
    std::vector<std::vector<Vec2>> mesh_vertices;
    std::vector<uint32_t> draw_order;
};
struct Frame {
    float time = 0.0f;
    std::vector<FrameView> views;
};
struct Clip {
    std::string id;
    std::string display_name;
    std::string intent;
    float duration = 0.0f;
    float fps = 30.0f;
    bool loop = false;
    bool qualified = false;
    float runtime_envelope_factor = 1.0f;
    std::vector<Frame> frames;
};

struct RuntimeData {
    std::string binary_schema;
    std::string source_manifest_sha256;
    std::string source_payload_merkle_sha256;
    std::string coordinate_system = "screen_2d.top_left.x_right.y_down.uv_top_left";
    float units_per_pixel = 1.0f;
    uint32_t feature_flags = 0;
    std::vector<View> views;
    std::vector<Clip> clips;
};

class Reader {
public:
    explicit Reader(const std::vector<uint8_t>& bytes) : bytes_(bytes) {}

    void require(size_t count) const {
        if (offset_ > bytes_.size() || count > bytes_.size() - offset_) {
            throw std::runtime_error("Truncated RealSaS runtime binary");
        }
    }
    uint8_t u8() { require(1); return bytes_[offset_++]; }
    uint32_t u32() {
        require(4);
        uint32_t value = static_cast<uint32_t>(bytes_[offset_]) |
            (static_cast<uint32_t>(bytes_[offset_ + 1]) << 8) |
            (static_cast<uint32_t>(bytes_[offset_ + 2]) << 16) |
            (static_cast<uint32_t>(bytes_[offset_ + 3]) << 24);
        offset_ += 4;
        return value;
    }
    float f32() {
        const uint32_t bits = u32();
        float value = 0.0f;
        static_assert(sizeof(float) == sizeof(uint32_t));
        std::memcpy(&value, &bits, sizeof(value));
        if (!std::isfinite(value)) throw std::runtime_error("Non-finite value in RealSaS runtime binary");
        return value;
    }
    std::string string() {
        const uint32_t size = u32();
        if (size > 16u * 1024u * 1024u) throw std::runtime_error("Unreasonable string length in RealSaS runtime binary");
        require(size);
        std::string out(reinterpret_cast<const char*>(bytes_.data() + offset_), size);
        offset_ += size;
        return out;
    }
    void magic(const uint8_t* expected, size_t count) {
        require(count);
        if (std::memcmp(bytes_.data() + offset_, expected, count) != 0) {
            throw std::runtime_error("Invalid RealSaS runtime binary magic");
        }
        offset_ += count;
    }
    size_t offset() const noexcept { return offset_; }
private:
    const std::vector<uint8_t>& bytes_;
    size_t offset_ = 0;
};

std::vector<uint8_t> read_file(const std::filesystem::path& path) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) throw std::runtime_error("Cannot open file: " + path.string());
    stream.seekg(0, std::ios::end);
    const auto size = stream.tellg();
    if (size < 0) throw std::runtime_error("Cannot determine file size: " + path.string());
    stream.seekg(0, std::ios::beg);
    std::vector<uint8_t> bytes(static_cast<size_t>(size));
    if (!bytes.empty()) stream.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    if (!stream && !bytes.empty()) throw std::runtime_error("Cannot read file: " + path.string());
    return bytes;
}

std::vector<uint8_t> read_archive_entry(const std::filesystem::path& archive_path, std::string_view wanted) {
    archive* raw = archive_read_new();
    if (!raw) throw std::runtime_error("libarchive allocation failed");
    std::unique_ptr<archive, decltype(&archive_read_free)> holder(raw, &archive_read_free);
    archive_read_support_filter_all(raw);
    archive_read_support_format_all(raw);
    if (archive_read_open_filename(raw, archive_path.string().c_str(), 10240) != ARCHIVE_OK) {
        throw std::runtime_error(std::string("Cannot open runtime archive: ") + archive_error_string(raw));
    }
    archive_entry* entry = nullptr;
    while (archive_read_next_header(raw, &entry) == ARCHIVE_OK) {
        const char* name = archive_entry_pathname(entry);
        if (name && wanted == name) {
            const auto declared = archive_entry_size(entry);
            if (declared < 0 || declared > static_cast<la_int64_t>(2ull * 1024ull * 1024ull * 1024ull)) {
                throw std::runtime_error("Invalid runtime archive entry size");
            }
            std::vector<uint8_t> out(static_cast<size_t>(declared));
            size_t offset = 0;
            while (offset < out.size()) {
                const la_ssize_t got = archive_read_data(raw, out.data() + offset, out.size() - offset);
                if (got < 0) throw std::runtime_error(std::string("Cannot read runtime archive entry: ") + archive_error_string(raw));
                if (got == 0) break;
                offset += static_cast<size_t>(got);
            }
            if (offset != out.size()) {
                throw std::runtime_error("Truncated runtime archive entry: " + std::string(wanted));
            }
            return out;
        }
        archive_read_data_skip(raw);
    }
    throw std::runtime_error("Runtime archive entry not found: " + std::string(wanted));
}

std::vector<uint8_t> read_package_asset(const std::filesystem::path& package, std::string_view relative) {
    if (std::filesystem::is_directory(package)) return read_file(package / std::filesystem::path(relative));
    return read_archive_entry(package, relative);
}

std::vector<uint8_t> decode_png_rgba(const std::vector<uint8_t>& payload, uint32_t expected_w, uint32_t expected_h) {
    png_image image{};
    image.version = PNG_IMAGE_VERSION;
    if (!png_image_begin_read_from_memory(&image, payload.data(), payload.size())) {
        throw std::runtime_error("libpng could not decode RealSaS texture header");
    }
    image.format = PNG_FORMAT_RGBA;
    if ((expected_w && image.width != expected_w) || (expected_h && image.height != expected_h)) {
        png_image_free(&image);
        throw std::runtime_error("RealSaS texture dimensions disagree with runtime binary");
    }
    std::vector<uint8_t> rgba(PNG_IMAGE_SIZE(image));
    if (!png_image_finish_read(&image, nullptr, rgba.data(), 0, nullptr)) {
        const std::string message = image.message;
        png_image_free(&image);
        throw std::runtime_error("libpng could not decode RealSaS texture: " + message);
    }
    png_image_free(&image);
    return rgba;
}

RuntimeData parse_runtime_binary(const std::vector<uint8_t>& bytes) {
    if (bytes.size() < 16) throw std::runtime_error("RealSaS runtime binary is too small");
    const uint32_t stored_crc = static_cast<uint32_t>(bytes[bytes.size() - 4]) |
        (static_cast<uint32_t>(bytes[bytes.size() - 3]) << 8) |
        (static_cast<uint32_t>(bytes[bytes.size() - 2]) << 16) |
        (static_cast<uint32_t>(bytes[bytes.size() - 1]) << 24);
    const uint32_t actual_crc = static_cast<uint32_t>(crc32(0L, bytes.data(), static_cast<uInt>(bytes.size() - 4)));
    if (stored_crc != actual_crc) throw std::runtime_error("RealSaS runtime binary CRC32 mismatch");

    std::vector<uint8_t> body(bytes.begin(), bytes.end() - 4);
    Reader r(body);
    const uint8_t magic_v1[8] = {'R','S','R','T',0,1,0,0};
    const uint8_t magic_v2[8] = {'R','S','R','T',0,2,0,0};
    uint32_t expected_version = 0;
    if (body.size() >= sizeof(magic_v2) && std::memcmp(body.data(), magic_v2, sizeof(magic_v2)) == 0) {
        r.magic(magic_v2, sizeof(magic_v2));
        expected_version = 2;
    } else {
        r.magic(magic_v1, sizeof(magic_v1));
        expected_version = 1;
    }
    const uint32_t version = r.u32();
    if (version != expected_version || (version != 1 && version != 2)) {
        throw std::runtime_error("Unsupported or inconsistent RealSaS runtime binary version");
    }
    RuntimeData data;
    data.binary_schema = r.string();
    data.source_manifest_sha256 = r.string();
    data.source_payload_merkle_sha256 = r.string();
    if (version >= 2) {
        data.coordinate_system = r.string();
        data.units_per_pixel = r.f32();
        data.feature_flags = r.u32();
        if (data.coordinate_system.empty() || data.units_per_pixel <= 0.0f) {
            throw std::runtime_error("Invalid RealSaS runtime coordinate contract");
        }
    }
    const uint32_t view_count = r.u32();
    if (view_count == 0 || view_count > 64) throw std::runtime_error("Invalid RealSaS runtime view count");
    data.views.reserve(view_count);
    for (uint32_t vi = 0; vi < view_count; ++vi) {
        View view;
        view.id = r.string();
        view.texture_path = r.string();
        if (version >= 2) view.texture_crc32 = r.u32();
        view.width = r.u32();
        view.height = r.u32();
        if (view.width == 0 || view.height == 0 || view.width > 32768 || view.height > 32768) {
            throw std::runtime_error("Invalid RealSaS runtime texture dimensions");
        }
        const uint32_t mesh_count = r.u32();
        if (mesh_count == 0 || mesh_count > 4096) throw std::runtime_error("Invalid RealSaS runtime mesh count");
        view.meshes.reserve(mesh_count);
        for (uint32_t mi = 0; mi < mesh_count; ++mi) {
            Mesh mesh;
            mesh.id = r.string();
            const uint32_t vertex_count = r.u32();
            const uint32_t triangle_count = r.u32();
            if (vertex_count == 0 || vertex_count > 10'000'000u || triangle_count > 20'000'000u) {
                throw std::runtime_error("Invalid RealSaS runtime mesh dimensions");
            }
            mesh.vertices.resize(vertex_count);
            for (auto& vertex : mesh.vertices) {
                vertex.rest = {r.f32(), r.f32()};
                vertex.uv = {r.f32(), r.f32()};
            }
            mesh.triangles.resize(triangle_count);
            for (auto& tri : mesh.triangles) {
                tri = {r.u32(), r.u32(), r.u32()};
                if (tri[0] >= vertex_count || tri[1] >= vertex_count || tri[2] >= vertex_count) {
                    throw std::runtime_error("RealSaS runtime triangle index out of range");
                }
            }
            view.meshes.push_back(std::move(mesh));
        }
        data.views.push_back(std::move(view));
    }
    const uint32_t clip_count = r.u32();
    if (clip_count == 0 || clip_count > 4096) throw std::runtime_error("Invalid RealSaS runtime clip count");
    data.clips.reserve(clip_count);
    for (uint32_t ci = 0; ci < clip_count; ++ci) {
        Clip clip;
        clip.id = r.string();
        clip.display_name = r.string();
        clip.intent = r.string();
        clip.duration = r.f32();
        clip.fps = r.f32();
        clip.loop = r.u8() != 0;
        clip.qualified = r.u8() != 0;
        (void)r.u8(); (void)r.u8();
        clip.runtime_envelope_factor = r.f32();
        const uint32_t frame_count = r.u32();
        if (frame_count == 0 || frame_count > 1'000'000u) throw std::runtime_error("Invalid RealSaS runtime frame count");
        clip.frames.reserve(frame_count);
        for (uint32_t fi = 0; fi < frame_count; ++fi) {
            Frame frame;
            frame.time = r.f32();
            frame.views.resize(data.views.size());
            for (size_t vi = 0; vi < data.views.size(); ++vi) {
                const View& view = data.views[vi];
                FrameView& fv = frame.views[vi];
                fv.mesh_vertices.resize(view.meshes.size());
                for (size_t mi = 0; mi < view.meshes.size(); ++mi) {
                    const size_t count = view.meshes[mi].vertices.size();
                    auto& points = fv.mesh_vertices[mi];
                    points.resize(count);
                    for (auto& point : points) point = {r.f32(), r.f32()};
                }
                const uint32_t order_count = r.u32();
                if (order_count > view.meshes.size()) throw std::runtime_error("Invalid RealSaS runtime draw-order count");
                fv.draw_order.resize(order_count);
                for (auto& index : fv.draw_order) {
                    index = r.u32();
                    if (index >= view.meshes.size()) throw std::runtime_error("RealSaS runtime draw-order index out of range");
                }
            }
            clip.frames.push_back(std::move(frame));
        }
        data.clips.push_back(std::move(clip));
    }
    if (r.offset() != body.size()) throw std::runtime_error("Trailing bytes in RealSaS runtime binary");
    return data;
}

struct SampleSpan {
    const Frame* a = nullptr;
    const Frame* b = nullptr;
    float alpha = 0.0f;
};

float normalized_time(const Clip& clip, float time) {
    if (!std::isfinite(time)) time = 0.0f;
    time = std::max(0.0f, time);
    if (clip.duration > 0.0f) {
        if (clip.loop) {
            time = std::fmod(time, clip.duration);
            if (time < 0.0f) time += clip.duration;
        } else {
            time = std::min(time, clip.duration);
        }
    }
    return time;
}

SampleSpan sample_span(const Clip& clip, float time) {
    if (clip.frames.empty()) throw std::runtime_error("RealSaS runtime clip has no frames");
    time = normalized_time(clip, time);
    auto upper = std::lower_bound(clip.frames.begin(), clip.frames.end(), time,
        [](const Frame& frame, float value) { return frame.time < value; });
    if (upper == clip.frames.begin()) return {&clip.frames.front(), &clip.frames.front(), 0.0f};
    if (upper == clip.frames.end()) return {&clip.frames.back(), &clip.frames.back(), 0.0f};
    const Frame* b = &*upper;
    const Frame* a = &*(upper - 1);
    const float denom = b->time - a->time;
    const float alpha = denom > 1e-8f ? std::clamp((time - a->time) / denom, 0.0f, 1.0f) : 0.0f;
    return {a, b, alpha};
}

Vec2 lerp(Vec2 a, Vec2 b, float t) { return {a.x + (b.x - a.x) * t, a.y + (b.y - a.y) * t}; }

void write_error(char* buffer, size_t size, const std::string& message) {
    if (!buffer || size == 0) return;
    const size_t count = std::min(size - 1, message.size());
    std::memcpy(buffer, message.data(), count);
    buffer[count] = '\0';
}

float edge(Vec2 a, Vec2 b, float x, float y) {
    return (x - a.x) * (b.y - a.y) - (y - a.y) * (b.x - a.x);
}

std::array<float, 4> sample_bilinear(const View& view, float u, float v) {
    u = std::clamp(u, 0.0f, 1.0f);
    v = std::clamp(v, 0.0f, 1.0f);
    const float x = u * static_cast<float>(std::max(1u, view.width - 1));
    const float y = v * static_cast<float>(std::max(1u, view.height - 1));
    const uint32_t x0 = static_cast<uint32_t>(std::floor(x));
    const uint32_t y0 = static_cast<uint32_t>(std::floor(y));
    const uint32_t x1 = std::min(view.width - 1, x0 + 1);
    const uint32_t y1 = std::min(view.height - 1, y0 + 1);
    const float tx = x - static_cast<float>(x0);
    const float ty = y - static_cast<float>(y0);
    auto pixel = [&](uint32_t px, uint32_t py) {
        const size_t i = (static_cast<size_t>(py) * view.width + px) * 4;
        return std::array<float, 4>{
            view.texture_rgba[i] / 255.0f, view.texture_rgba[i + 1] / 255.0f,
            view.texture_rgba[i + 2] / 255.0f, view.texture_rgba[i + 3] / 255.0f};
    };
    const auto p00 = pixel(x0, y0), p10 = pixel(x1, y0), p01 = pixel(x0, y1), p11 = pixel(x1, y1);
    std::array<float, 4> out{};
    for (size_t c = 0; c < 4; ++c) {
        const float a = p00[c] + (p10[c] - p00[c]) * tx;
        const float b = p01[c] + (p11[c] - p01[c]) * tx;
        out[c] = a + (b - a) * ty;
    }
    return out;
}

void blend_pixel(uint8_t* dst, const std::array<float, 4>& src) {
    const float sa = std::clamp(src[3], 0.0f, 1.0f);
    const float da = dst[3] / 255.0f;
    const float oa = sa + da * (1.0f - sa);
    if (oa <= 1e-8f) { dst[0] = dst[1] = dst[2] = dst[3] = 0; return; }
    for (int c = 0; c < 3; ++c) {
        const float dc = dst[c] / 255.0f;
        const float oc = (src[c] * sa + dc * da * (1.0f - sa)) / oa;
        dst[c] = static_cast<uint8_t>(std::clamp(std::lround(oc * 255.0f), 0l, 255l));
    }
    dst[3] = static_cast<uint8_t>(std::clamp(std::lround(oa * 255.0f), 0l, 255l));
}

std::vector<float> box_blur_scalar(const std::vector<float>& src, uint32_t width, uint32_t height, int radius) {
    if (radius <= 0 || src.empty()) return src;
    radius = std::min(radius, 48);
    std::vector<float> temp(src.size(), 0.0f), out(src.size(), 0.0f);
    for (uint32_t y = 0; y < height; ++y) {
        float sum = 0.0f; int count = 0;
        for (int x = -radius; x <= radius; ++x) if (x >= 0 && x < static_cast<int>(width)) { sum += src[static_cast<size_t>(y) * width + x]; ++count; }
        for (uint32_t x = 0; x < width; ++x) {
            temp[static_cast<size_t>(y) * width + x] = count ? sum / count : 0.0f;
            const int remove = static_cast<int>(x) - radius, add = static_cast<int>(x) + radius + 1;
            if (remove >= 0) { sum -= src[static_cast<size_t>(y) * width + remove]; --count; }
            if (add < static_cast<int>(width)) { sum += src[static_cast<size_t>(y) * width + add]; ++count; }
        }
    }
    for (uint32_t x = 0; x < width; ++x) {
        float sum = 0.0f; int count = 0;
        for (int y = -radius; y <= radius; ++y) if (y >= 0 && y < static_cast<int>(height)) { sum += temp[static_cast<size_t>(y) * width + x]; ++count; }
        for (uint32_t y = 0; y < height; ++y) {
            out[static_cast<size_t>(y) * width + x] = count ? sum / count : 0.0f;
            const int remove = static_cast<int>(y) - radius, add = static_cast<int>(y) + radius + 1;
            if (remove >= 0) { sum -= temp[static_cast<size_t>(remove) * width + x]; --count; }
            if (add < static_cast<int>(height)) { sum += temp[static_cast<size_t>(add) * width + x]; ++count; }
        }
    }
    return out;
}

void apply_reference_post_process(uint8_t* rgba, uint32_t width, uint32_t height, const RsPostProcessSettings& cfg) {
    const size_t pixels = static_cast<size_t>(width) * height;
    std::vector<uint8_t> base(rgba, rgba + pixels * 4);
    std::vector<float> alpha(pixels), luminance(pixels);
    for (size_t i = 0; i < pixels; ++i) {
        alpha[i] = base[i * 4 + 3] / 255.0f;
        luminance[i] = alpha[i] * (0.2126f * base[i * 4] + 0.7152f * base[i * 4 + 1] + 0.0722f * base[i * 4 + 2]) / 255.0f;
    }
    const auto shadow = box_blur_scalar(alpha, width, height, static_cast<int>(std::lround(std::max(0.0f, cfg.shadow_blur_radius))));
    const auto outer = box_blur_scalar(alpha, width, height, std::max(1, static_cast<int>(std::lround(std::max(1.0f, cfg.bloom_radius * 0.5f)))));
    const auto bloom = box_blur_scalar(luminance, width, height, static_cast<int>(std::lround(std::max(0.0f, cfg.bloom_radius))));
    std::fill(rgba, rgba + pixels * 4, 0);
    const int sx = static_cast<int>(std::lround(cfg.shadow_offset_x)), sy = static_cast<int>(std::lround(cfg.shadow_offset_y));
    for (uint32_t y = 0; y < height; ++y) for (uint32_t x = 0; x < width; ++x) {
        const int source_x = static_cast<int>(x) - sx, source_y = static_cast<int>(y) - sy;
        if (source_x < 0 || source_y < 0 || source_x >= static_cast<int>(width) || source_y >= static_cast<int>(height)) continue;
        const float a = std::clamp(shadow[static_cast<size_t>(source_y) * width + source_x] * cfg.shadow_strength, 0.0f, 0.88f);
        blend_pixel(rgba + (static_cast<size_t>(y) * width + x) * 4, {0.005f, 0.012f, 0.022f, a});
    }
    for (size_t i = 0; i < pixels; ++i) {
        const std::array<float,4> src{base[i*4]/255.0f,base[i*4+1]/255.0f,base[i*4+2]/255.0f,base[i*4+3]/255.0f};
        blend_pixel(rgba+i*4,src);
    }
    const float exposure = std::max(0.0f, cfg.exposure);
    const float saturation = std::max(0.0f, cfg.saturation);
    for (uint32_t y = 0; y < height; ++y) for (uint32_t x = 0; x < width; ++x) {
        const size_t i = static_cast<size_t>(y) * width + x; uint8_t* p = rgba + i * 4;
        float r = p[0] / 255.0f, g = p[1] / 255.0f, b = p[2] / 255.0f;
        const float rim = std::max(0.0f, outer[i] - alpha[i]) * cfg.rim_strength;
        const float glow = bloom[i] * cfg.bloom_strength;
        r += rim * cfg.rim_r + glow; g += rim * cfg.rim_g + glow; b += rim * cfg.rim_b + glow;
        const float l = 0.2126f*r+0.7152f*g+0.0722f*b; r=(l+(r-l)*saturation)*exposure; g=(l+(g-l)*saturation)*exposure; b=(l+(b-l)*saturation)*exposure;
        const float nx=(static_cast<float>(x)+0.5f)/std::max(1u,width)*2.0f-1.0f, ny=(static_cast<float>(y)+0.5f)/std::max(1u,height)*2.0f-1.0f;
        const float vig=1.0f-std::clamp((nx*nx+ny*ny)*cfg.vignette,0.0f,0.82f); r*=vig;g*=vig;b*=vig;
        p[0]=static_cast<uint8_t>(std::clamp(std::lround(r*255.0f),0l,255l)); p[1]=static_cast<uint8_t>(std::clamp(std::lround(g*255.0f),0l,255l)); p[2]=static_cast<uint8_t>(std::clamp(std::lround(b*255.0f),0l,255l));
        const float effect_alpha=std::clamp(std::max(alpha[i],rim*.55f+glow*.22f),0.0f,1.0f); p[3]=static_cast<uint8_t>(std::clamp(std::lround(effect_alpha*255.0f),0l,255l));
    }
}

} // namespace

struct RsRuntime {
    std::filesystem::path package_path;
    RuntimeData data;
};

extern "C" {

uint32_t rs_runtime_abi_version(void) { return 3u; }

RsResult rs_runtime_open(const char* package_path, RsRuntime** out_runtime, char* error_buffer, size_t error_buffer_size) {
    if (!package_path || !out_runtime) return RS_ERROR_INVALID_ARGUMENT;
    *out_runtime = nullptr;
    try {
        auto runtime = std::make_unique<RsRuntime>();
        runtime->package_path = std::filesystem::path(package_path);
        auto binary = read_package_asset(runtime->package_path, "runtime/realsas_runtime.rsr");
        runtime->data = parse_runtime_binary(binary);
        for (auto& view : runtime->data.views) {
            auto texture = read_package_asset(runtime->package_path, view.texture_path);
            if (view.texture_crc32 != 0) {
                const uint32_t actual = static_cast<uint32_t>(
                    crc32(0L, texture.data(), static_cast<uInt>(texture.size())));
                if (actual != view.texture_crc32) {
                    throw std::runtime_error("RealSaS runtime texture CRC32 mismatch: " + view.texture_path);
                }
            }
            view.texture_rgba = decode_png_rgba(texture, view.width, view.height);
        }
        *out_runtime = runtime.release();
        write_error(error_buffer, error_buffer_size, "");
        return RS_OK;
    } catch (const std::exception& exc) {
        write_error(error_buffer, error_buffer_size, exc.what());
        return RS_ERROR_FORMAT;
    }
}

void rs_runtime_close(RsRuntime* runtime) { delete runtime; }
uint32_t rs_runtime_view_count(const RsRuntime* r) { return r ? static_cast<uint32_t>(r->data.views.size()) : 0; }
const char* rs_runtime_view_id(const RsRuntime* r, uint32_t i) { return r && i < r->data.views.size() ? r->data.views[i].id.c_str() : nullptr; }
uint32_t rs_runtime_view_width(const RsRuntime* r, uint32_t i) { return r && i < r->data.views.size() ? r->data.views[i].width : 0; }
uint32_t rs_runtime_view_height(const RsRuntime* r, uint32_t i) { return r && i < r->data.views.size() ? r->data.views[i].height : 0; }
const uint8_t* rs_runtime_view_texture_rgba(const RsRuntime* r, uint32_t i, size_t* count) {
    if (!r || i >= r->data.views.size()) return nullptr;
    if (count) *count = r->data.views[i].texture_rgba.size();
    return r->data.views[i].texture_rgba.data();
}
uint32_t rs_runtime_mesh_count(const RsRuntime* r, uint32_t vi) { return r && vi < r->data.views.size() ? static_cast<uint32_t>(r->data.views[vi].meshes.size()) : 0; }
const char* rs_runtime_mesh_id(const RsRuntime* r, uint32_t vi, uint32_t mi) {
    return r && vi < r->data.views.size() && mi < r->data.views[vi].meshes.size() ? r->data.views[vi].meshes[mi].id.c_str() : nullptr;
}
uint32_t rs_runtime_mesh_vertex_count(const RsRuntime* r, uint32_t vi, uint32_t mi) {
    return r && vi < r->data.views.size() && mi < r->data.views[vi].meshes.size() ? static_cast<uint32_t>(r->data.views[vi].meshes[mi].vertices.size()) : 0;
}
uint32_t rs_runtime_mesh_triangle_count(const RsRuntime* r, uint32_t vi, uint32_t mi) {
    return r && vi < r->data.views.size() && mi < r->data.views[vi].meshes.size() ? static_cast<uint32_t>(r->data.views[vi].meshes[mi].triangles.size()) : 0;
}
RsResult rs_runtime_mesh_rest_vertices(const RsRuntime* r, uint32_t vi, uint32_t mi, float* out, size_t count) {
    if (!r || vi >= r->data.views.size() || mi >= r->data.views[vi].meshes.size() || !out) return RS_ERROR_INVALID_ARGUMENT;
    const auto& vertices = r->data.views[vi].meshes[mi].vertices;
    if (count < vertices.size() * 4) return RS_ERROR_BUFFER_TOO_SMALL;
    size_t j = 0; for (const auto& v : vertices) { out[j++] = v.rest.x; out[j++] = v.rest.y; out[j++] = v.uv.x; out[j++] = v.uv.y; }
    return RS_OK;
}
RsResult rs_runtime_mesh_triangles(const RsRuntime* r, uint32_t vi, uint32_t mi, uint32_t* out, size_t count) {
    if (!r || vi >= r->data.views.size() || mi >= r->data.views[vi].meshes.size() || !out) return RS_ERROR_INVALID_ARGUMENT;
    const auto& triangles = r->data.views[vi].meshes[mi].triangles;
    if (count < triangles.size() * 3) return RS_ERROR_BUFFER_TOO_SMALL;
    size_t j = 0; for (const auto& t : triangles) { out[j++] = t[0]; out[j++] = t[1]; out[j++] = t[2]; }
    return RS_OK;
}
uint32_t rs_runtime_clip_count(const RsRuntime* r) { return r ? static_cast<uint32_t>(r->data.clips.size()) : 0; }
const char* rs_runtime_clip_id(const RsRuntime* r, uint32_t i) { return r && i < r->data.clips.size() ? r->data.clips[i].id.c_str() : nullptr; }
const char* rs_runtime_clip_display_name(const RsRuntime* r, uint32_t i) { return r && i < r->data.clips.size() ? r->data.clips[i].display_name.c_str() : nullptr; }
const char* rs_runtime_clip_intent(const RsRuntime* r, uint32_t i) { return r && i < r->data.clips.size() ? r->data.clips[i].intent.c_str() : nullptr; }
float rs_runtime_clip_duration(const RsRuntime* r, uint32_t i) { return r && i < r->data.clips.size() ? r->data.clips[i].duration : 0.0f; }
float rs_runtime_clip_fps(const RsRuntime* r, uint32_t i) { return r && i < r->data.clips.size() ? r->data.clips[i].fps : 0.0f; }
int rs_runtime_clip_loop(const RsRuntime* r, uint32_t i) { return r && i < r->data.clips.size() && r->data.clips[i].loop; }
int rs_runtime_clip_qualified(const RsRuntime* r, uint32_t i) { return r && i < r->data.clips.size() && r->data.clips[i].qualified; }
uint32_t rs_runtime_find_clip(const RsRuntime* r, const char* name) {
    if (!r || !name) return UINT32_MAX;
    for (uint32_t i = 0; i < r->data.clips.size(); ++i) {
        const auto& c = r->data.clips[i]; if (c.id == name || c.intent == name || c.display_name == name) return i;
    }
    return UINT32_MAX;
}
uint32_t rs_runtime_find_view(const RsRuntime* r, const char* id) {
    if (!r || !id) return UINT32_MAX;
    for (uint32_t i = 0; i < r->data.views.size(); ++i) if (r->data.views[i].id == id) return i;
    return UINT32_MAX;
}
RsResult rs_runtime_sample_mesh_vertices(const RsRuntime* r, uint32_t ci, uint32_t vi, uint32_t mi, float time, float* out, size_t count) {
    if (!r || ci >= r->data.clips.size() || vi >= r->data.views.size() || mi >= r->data.views[vi].meshes.size() || !out) return RS_ERROR_INVALID_ARGUMENT;
    const auto span = sample_span(r->data.clips[ci], time);
    const auto& a = span.a->views[vi].mesh_vertices[mi]; const auto& b = span.b->views[vi].mesh_vertices[mi];
    if (count < a.size() * 2) return RS_ERROR_BUFFER_TOO_SMALL;
    size_t j = 0; for (size_t i = 0; i < a.size(); ++i) { const Vec2 p = lerp(a[i], b[i], span.alpha); out[j++] = p.x; out[j++] = p.y; }
    return RS_OK;
}
RsResult rs_runtime_sample_mesh_vertices_blend(const RsRuntime* r, uint32_t from_ci, float from_time, uint32_t to_ci, float to_time, float mix_alpha, uint32_t vi, uint32_t mi, float* out, size_t count) {
    if (!r || from_ci >= r->data.clips.size() || to_ci >= r->data.clips.size() || vi >= r->data.views.size() || mi >= r->data.views[vi].meshes.size() || !out || !std::isfinite(mix_alpha)) return RS_ERROR_INVALID_ARGUMENT;
    const auto from_span = sample_span(r->data.clips[from_ci], from_time);
    const auto to_span = sample_span(r->data.clips[to_ci], to_time);
    const auto& from_a = from_span.a->views[vi].mesh_vertices[mi]; const auto& from_b = from_span.b->views[vi].mesh_vertices[mi];
    const auto& to_a = to_span.a->views[vi].mesh_vertices[mi]; const auto& to_b = to_span.b->views[vi].mesh_vertices[mi];
    if (from_a.size() != to_a.size() || from_b.size() != to_b.size()) return RS_ERROR_FORMAT;
    if (count < from_a.size() * 2) return RS_ERROR_BUFFER_TOO_SMALL;
    const float alpha = std::clamp(mix_alpha, 0.0f, 1.0f);
    size_t j = 0;
    for (size_t i = 0; i < from_a.size(); ++i) {
        const Vec2 source = lerp(from_a[i], from_b[i], from_span.alpha);
        const Vec2 target = lerp(to_a[i], to_b[i], to_span.alpha);
        const Vec2 p = lerp(source, target, alpha);
        out[j++] = p.x; out[j++] = p.y;
    }
    return RS_OK;
}
uint32_t rs_runtime_sample_draw_order_count(const RsRuntime* r, uint32_t ci, uint32_t vi, float time) {
    if (!r || ci >= r->data.clips.size() || vi >= r->data.views.size()) return 0;
    const auto span = sample_span(r->data.clips[ci], time); return static_cast<uint32_t>(span.alpha < 0.5f ? span.a->views[vi].draw_order.size() : span.b->views[vi].draw_order.size());
}
RsResult rs_runtime_sample_draw_order(const RsRuntime* r, uint32_t ci, uint32_t vi, float time, uint32_t* out, size_t count) {
    if (!r || ci >= r->data.clips.size() || vi >= r->data.views.size() || !out) return RS_ERROR_INVALID_ARGUMENT;
    const auto span = sample_span(r->data.clips[ci], time); const auto& order = span.alpha < 0.5f ? span.a->views[vi].draw_order : span.b->views[vi].draw_order;
    if (count < order.size()) return RS_ERROR_BUFFER_TOO_SMALL; std::copy(order.begin(), order.end(), out); return RS_OK;
}
uint32_t rs_runtime_sample_draw_order_blend_count(const RsRuntime* r, uint32_t from_ci, float from_time, uint32_t to_ci, float to_time, float mix_alpha, uint32_t vi) {
    if (!r || from_ci >= r->data.clips.size() || to_ci >= r->data.clips.size() || vi >= r->data.views.size() || !std::isfinite(mix_alpha)) return 0;
    const uint32_t selected = mix_alpha < 0.5f ? from_ci : to_ci;
    const float selected_time = mix_alpha < 0.5f ? from_time : to_time;
    return rs_runtime_sample_draw_order_count(r, selected, vi, selected_time);
}
RsResult rs_runtime_sample_draw_order_blend(const RsRuntime* r, uint32_t from_ci, float from_time, uint32_t to_ci, float to_time, float mix_alpha, uint32_t vi, uint32_t* out, size_t count) {
    if (!r || from_ci >= r->data.clips.size() || to_ci >= r->data.clips.size() || vi >= r->data.views.size() || !out || !std::isfinite(mix_alpha)) return RS_ERROR_INVALID_ARGUMENT;
    const uint32_t selected = mix_alpha < 0.5f ? from_ci : to_ci;
    const float selected_time = mix_alpha < 0.5f ? from_time : to_time;
    return rs_runtime_sample_draw_order(r, selected, vi, selected_time, out, count);
}
static RsResult render_raw_rgba(const RsRuntime* r, uint32_t ci, uint32_t vi, float time, uint8_t* out, size_t count) {
    if (!r || ci >= r->data.clips.size() || vi >= r->data.views.size() || !out) return RS_ERROR_INVALID_ARGUMENT;
    const View& view = r->data.views[vi]; const size_t needed = static_cast<size_t>(view.width) * view.height * 4;
    if (count < needed) return RS_ERROR_BUFFER_TOO_SMALL; std::fill(out, out + needed, 0);
    const auto span = sample_span(r->data.clips[ci], time); const FrameView& fa = span.a->views[vi]; const FrameView& fb = span.b->views[vi];
    const auto& order = span.alpha < 0.5f ? fa.draw_order : fb.draw_order;
    std::vector<uint32_t> fallback; const std::vector<uint32_t>* draw = &order;
    if (order.empty()) { fallback.resize(view.meshes.size()); for (uint32_t i = 0; i < fallback.size(); ++i) fallback[i] = i; draw = &fallback; }
    for (uint32_t mi : *draw) {
        const Mesh& mesh = view.meshes[mi]; const auto& va = fa.mesh_vertices[mi]; const auto& vb = fb.mesh_vertices[mi];
        for (const auto& tri : mesh.triangles) {
            const Vec2 p0 = lerp(va[tri[0]], vb[tri[0]], span.alpha), p1 = lerp(va[tri[1]], vb[tri[1]], span.alpha), p2 = lerp(va[tri[2]], vb[tri[2]], span.alpha);
            const float area = edge(p0, p1, p2.x, p2.y); if (std::abs(area) < 1e-8f) continue;
            const int min_x = std::max(0, static_cast<int>(std::floor(std::min({p0.x, p1.x, p2.x}))));
            const int max_x = std::min(static_cast<int>(view.width) - 1, static_cast<int>(std::ceil(std::max({p0.x, p1.x, p2.x}))));
            const int min_y = std::max(0, static_cast<int>(std::floor(std::min({p0.y, p1.y, p2.y}))));
            const int max_y = std::min(static_cast<int>(view.height) - 1, static_cast<int>(std::ceil(std::max({p0.y, p1.y, p2.y}))));
            const Vec2 uv0 = mesh.vertices[tri[0]].uv, uv1 = mesh.vertices[tri[1]].uv, uv2 = mesh.vertices[tri[2]].uv;
            for (int y = min_y; y <= max_y; ++y) for (int x = min_x; x <= max_x; ++x) {
                const float px = x + 0.5f, py = y + 0.5f;
                const float w0 = edge(p1, p2, px, py) / area, w1 = edge(p2, p0, px, py) / area, w2 = edge(p0, p1, px, py) / area;
                if (w0 < -1e-5f || w1 < -1e-5f || w2 < -1e-5f) continue;
                const float u = uv0.x * w0 + uv1.x * w1 + uv2.x * w2, v = uv0.y * w0 + uv1.y * w1 + uv2.y * w2;
                blend_pixel(out + (static_cast<size_t>(y) * view.width + x) * 4, sample_bilinear(view, u, v));
            }
        }
    }
    return RS_OK;
}
RsPostProcessSettings rs_runtime_default_post_process_settings(void) {
    RsPostProcessSettings cfg{}; cfg.struct_version=1u; cfg.shadow_strength=.34f; cfg.shadow_offset_x=0.0f; cfg.shadow_offset_y=5.0f; cfg.shadow_blur_radius=9.0f; cfg.rim_strength=.46f; cfg.rim_r=.24f; cfg.rim_g=.72f; cfg.rim_b=1.0f; cfg.bloom_strength=.11f; cfg.bloom_radius=8.0f; cfg.exposure=1.04f; cfg.saturation=1.06f; cfg.vignette=.13f; return cfg;
}
RsResult rs_runtime_render_rgba(const RsRuntime* r, uint32_t ci, uint32_t vi, float time, uint8_t* out, size_t count) { return render_raw_rgba(r,ci,vi,time,out,count); }
RsResult rs_runtime_render_rgba_ex(const RsRuntime* r, uint32_t ci, uint32_t vi, float time, const RsPostProcessSettings* settings, uint8_t* out, size_t count) {
    const RsResult result=render_raw_rgba(r,ci,vi,time,out,count); if(result!=RS_OK)return result; if(!settings)return RS_OK; if(settings->struct_version!=1u)return RS_ERROR_INVALID_ARGUMENT; const View& view=r->data.views[vi]; apply_reference_post_process(out,view.width,view.height,*settings); return RS_OK;
}
const char* rs_runtime_binary_schema(const RsRuntime* r) { return r ? r->data.binary_schema.c_str() : nullptr; }
const char* rs_runtime_source_manifest_sha256(const RsRuntime* r) { return r ? r->data.source_manifest_sha256.c_str() : nullptr; }
const char* rs_runtime_source_payload_merkle_sha256(const RsRuntime* r) { return r ? r->data.source_payload_merkle_sha256.c_str() : nullptr; }
const char* rs_runtime_coordinate_system(const RsRuntime* r) { return r ? r->data.coordinate_system.c_str() : nullptr; }
float rs_runtime_units_per_pixel(const RsRuntime* r) { return r ? r->data.units_per_pixel : 0.0f; }
uint32_t rs_runtime_feature_flags(const RsRuntime* r) { return r ? r->data.feature_flags : 0u; }
uint32_t rs_runtime_view_texture_crc32(const RsRuntime* r, uint32_t i) {
    return r && i < r->data.views.size() ? r->data.views[i].texture_crc32 : 0u;
}

} // extern C
