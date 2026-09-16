#include "runtime_v3_reference.h"
#include "reference_raster_v3.h"

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
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace realsas::runtime_v3 {
namespace {

using reference_raster_v3::Barycentric;
using reference_raster_v3::DepthSample;
using reference_raster_v3::DepthWritePolicy;
using reference_raster_v3::Vertex;

constexpr uint32_t kRuntimeV3Version = 3u;
constexpr uint8_t kMagic[8] = {'R','S','R','T',0,3,0,0};
constexpr uint32_t kFeatureBakedXyzFrames = 1u << 1;
constexpr uint32_t kFeatureReferenceDepthRenderer = 1u << 4;
constexpr uint32_t kFeaturePosedVertexDepth = 1u << 5;
constexpr uint32_t kFeatureFullSurfaceAuthority = 1u << 6;
constexpr uint32_t kFeatureSlotAttachments = 1u << 7;
constexpr uint32_t kFeatureClipIntervals = 1u << 8;
constexpr uint32_t kRequiredFeatureMask =
    kFeatureBakedXyzFrames |
    kFeatureReferenceDepthRenderer |
    kFeaturePosedVertexDepth |
    kFeatureFullSurfaceAuthority |
    kFeatureSlotAttachments;

struct Vec2 {
    float x = 0.0f;
    float y = 0.0f;
};

struct Vec3 {
    float x = 0.0f;
    float y = 0.0f;
    float z = 0.0f;
};

struct RestVertex {
    Vec3 rest;
    Vec2 uv;
};

enum class AttachmentKind : uint8_t {
    DeformableBody = 0,
    RigidComponent = 1,
    Clipping = 2,
};

enum class TopologyClass : uint8_t {
    Static = 0,
    AttachmentDynamic = 1,
    ClipDynamic = 2,
    DrawOrderDynamic = 3,
};

struct Slot {
    std::string id;
    std::string bone_id;
    uint32_t setup_order = 0;
    std::string default_attachment_id;
};

struct Mesh {
    std::string id;
    uint32_t slot_index = 0;
    std::string attachment_id;
    AttachmentKind attachment_kind = AttachmentKind::DeformableBody;
    TopologyClass topology_class = TopologyClass::Static;
    std::vector<RestVertex> vertices;
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

struct ClipInterval {
    std::string clip_attachment_id;
    uint32_t start_slot_index = 0;
    uint32_t end_slot_index = 0;
    bool inverse = false;
};

struct FrameView {
    std::vector<std::vector<Vec3>> mesh_vertices;
    std::vector<uint32_t> draw_order_slots;
    std::vector<std::string> active_attachment_by_slot;
    std::vector<ClipInterval> clip_intervals;
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
    std::vector<Frame> frames;
};

struct RuntimeData {
    std::string binary_schema;
    std::string source_binding_sha256;
    std::string source_proof_bundle_hash;
    std::string coordinate_system;
    float units_per_pixel = 1.0f;
    uint32_t feature_flags = 0;
    std::string playback_contract_hash;
    std::string reference_raster_contract_hash;
    float alpha_cutout_threshold = 0.5f;
    std::vector<Slot> slots;
    std::vector<View> views;
    std::vector<Clip> clips;
};

struct ActiveClipMask {
    uint32_t start_order = 0;
    uint32_t end_order = 0;
    std::vector<uint8_t> mask;
};

class Reader {
public:
    explicit Reader(const std::vector<uint8_t>& bytes) : bytes_(bytes) {}

    void require(size_t count) const {
        if (offset_ > bytes_.size() || count > bytes_.size() - offset_) {
            throw std::runtime_error("Truncated RealSaS runtime-v3 binary");
        }
    }

    uint8_t u8() {
        require(1);
        return bytes_[offset_++];
    }

    uint32_t u32() {
        require(4);
        const uint32_t value = static_cast<uint32_t>(bytes_[offset_]) |
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
        if (!std::isfinite(value)) {
            throw std::runtime_error("Non-finite value in RealSaS runtime-v3 binary");
        }
        return value;
    }

    std::string string() {
        const uint32_t size = u32();
        if (size > 16u * 1024u * 1024u) {
            throw std::runtime_error("Unreasonable string length in RealSaS runtime-v3 binary");
        }
        require(size);
        std::string out(reinterpret_cast<const char*>(bytes_.data() + offset_), size);
        offset_ += size;
        return out;
    }

    void magic(const uint8_t* expected, size_t count) {
        require(count);
        if (std::memcmp(bytes_.data() + offset_, expected, count) != 0) {
            throw std::runtime_error("Invalid RealSaS runtime-v3 binary magic");
        }
        offset_ += count;
    }

    size_t offset() const noexcept { return offset_; }

private:
    const std::vector<uint8_t>& bytes_;
    size_t offset_ = 0;
};

uint32_t crc32_payload(const uint8_t* data, size_t size) {
    uLong crc = crc32(0L, Z_NULL, 0);
    size_t offset = 0;
    while (offset < size) {
        const size_t remaining = size - offset;
        const uInt chunk = static_cast<uInt>(std::min<size_t>(remaining, std::numeric_limits<uInt>::max()));
        crc = crc32(crc, data + offset, chunk);
        offset += chunk;
    }
    return static_cast<uint32_t>(crc);
}

std::vector<uint8_t> read_file(const std::filesystem::path& path) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) throw std::runtime_error("Cannot open file: " + path.string());
    stream.seekg(0, std::ios::end);
    const auto size = stream.tellg();
    if (size < 0) throw std::runtime_error("Cannot determine file size: " + path.string());
    stream.seekg(0, std::ios::beg);
    std::vector<uint8_t> bytes(static_cast<size_t>(size));
    if (!bytes.empty()) {
        stream.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    }
    if (!stream && !bytes.empty()) throw std::runtime_error("Cannot read file: " + path.string());
    return bytes;
}

std::filesystem::path safe_directory_asset(
    const std::filesystem::path& package,
    std::string_view relative) {
    const auto root = std::filesystem::weakly_canonical(package);
    const auto candidate = std::filesystem::weakly_canonical(root / std::filesystem::path(relative));
    if (candidate != root) {
        auto root_it = root.begin();
        auto candidate_it = candidate.begin();
        for (; root_it != root.end() && candidate_it != candidate.end(); ++root_it, ++candidate_it) {
            if (*root_it != *candidate_it) {
                throw std::runtime_error("Runtime-v3 asset path escapes package root");
            }
        }
        if (root_it != root.end()) {
            throw std::runtime_error("Runtime-v3 asset path escapes package root");
        }
    }
    return candidate;
}

std::vector<uint8_t> read_archive_entry(
    const std::filesystem::path& archive_path,
    std::string_view wanted) {
    archive* raw = archive_read_new();
    if (!raw) throw std::runtime_error("libarchive allocation failed");
    std::unique_ptr<archive, decltype(&archive_read_free)> holder(raw, &archive_read_free);
    archive_read_support_filter_all(raw);
    archive_read_support_format_all(raw);
    if (archive_read_open_filename(raw, archive_path.string().c_str(), 10240) != ARCHIVE_OK) {
        throw std::runtime_error(std::string("Cannot open runtime-v3 archive: ") + archive_error_string(raw));
    }

    archive_entry* entry = nullptr;
    while (archive_read_next_header(raw, &entry) == ARCHIVE_OK) {
        const char* name = archive_entry_pathname(entry);
        if (name && wanted == name) {
            const auto declared = archive_entry_size(entry);
            if (declared < 0 || declared > static_cast<la_int64_t>(3ull * 1024ull * 1024ull * 1024ull)) {
                throw std::runtime_error("Invalid runtime-v3 archive entry size");
            }
            std::vector<uint8_t> out(static_cast<size_t>(declared));
            size_t offset = 0;
            while (offset < out.size()) {
                const la_ssize_t got = archive_read_data(raw, out.data() + offset, out.size() - offset);
                if (got < 0) {
                    throw std::runtime_error(std::string("Cannot read runtime-v3 archive entry: ") + archive_error_string(raw));
                }
                if (got == 0) break;
                offset += static_cast<size_t>(got);
            }
            if (offset != out.size()) {
                throw std::runtime_error("Truncated runtime-v3 archive entry: " + std::string(wanted));
            }
            return out;
        }
        archive_read_data_skip(raw);
    }
    throw std::runtime_error("Runtime-v3 archive entry not found: " + std::string(wanted));
}

std::vector<uint8_t> read_package_asset(
    const std::filesystem::path& package,
    std::string_view relative) {
    if (std::filesystem::is_directory(package)) {
        return read_file(safe_directory_asset(package, relative));
    }
    return read_archive_entry(package, relative);
}

std::vector<uint8_t> decode_png_rgba(
    const std::vector<uint8_t>& payload,
    uint32_t expected_w,
    uint32_t expected_h) {
    png_image image{};
    image.version = PNG_IMAGE_VERSION;
    if (!png_image_begin_read_from_memory(&image, payload.data(), payload.size())) {
        throw std::runtime_error("libpng could not decode runtime-v3 texture header");
    }
    image.format = PNG_FORMAT_RGBA;
    if (image.width != expected_w || image.height != expected_h) {
        png_image_free(&image);
        throw std::runtime_error("Runtime-v3 texture dimensions disagree with binary");
    }
    std::vector<uint8_t> rgba(PNG_IMAGE_SIZE(image));
    if (!png_image_finish_read(&image, nullptr, rgba.data(), 0, nullptr)) {
        const std::string message = image.message;
        png_image_free(&image);
        throw std::runtime_error("libpng could not decode runtime-v3 texture: " + message);
    }
    png_image_free(&image);
    return rgba;
}

AttachmentKind parse_attachment_kind(uint8_t value) {
    if (value > static_cast<uint8_t>(AttachmentKind::Clipping)) {
        throw std::runtime_error("Invalid runtime-v3 attachment kind");
    }
    return static_cast<AttachmentKind>(value);
}

TopologyClass parse_topology_class(uint8_t value) {
    if (value > static_cast<uint8_t>(TopologyClass::DrawOrderDynamic)) {
        throw std::runtime_error("Invalid runtime-v3 topology class");
    }
    return static_cast<TopologyClass>(value);
}

void validate_exact_slot_permutation(
    const std::vector<uint32_t>& order,
    size_t slot_count) {
    if (order.size() != slot_count) {
        throw std::runtime_error("Runtime-v3 draw order is not an exact slot permutation");
    }
    std::vector<bool> seen(slot_count, false);
    for (uint32_t index : order) {
        if (index >= slot_count || seen[index]) {
            throw std::runtime_error("Runtime-v3 draw order is not an exact slot permutation");
        }
        seen[index] = true;
    }
}

RuntimeData parse_runtime_v3_binary(const std::vector<uint8_t>& bytes) {
    if (bytes.size() < 16) throw std::runtime_error("RealSaS runtime-v3 binary is too small");
    const size_t body_size = bytes.size() - 4;
    const uint32_t stored_crc = static_cast<uint32_t>(bytes[body_size]) |
        (static_cast<uint32_t>(bytes[body_size + 1]) << 8) |
        (static_cast<uint32_t>(bytes[body_size + 2]) << 16) |
        (static_cast<uint32_t>(bytes[body_size + 3]) << 24);
    const uint32_t actual_crc = crc32_payload(bytes.data(), body_size);
    if (stored_crc != actual_crc) throw std::runtime_error("RealSaS runtime-v3 binary CRC32 mismatch");

    std::vector<uint8_t> body(bytes.begin(), bytes.begin() + static_cast<std::ptrdiff_t>(body_size));
    Reader r(body);
    r.magic(kMagic, sizeof(kMagic));
    if (r.u32() != kRuntimeV3Version) throw std::runtime_error("Unsupported RealSaS runtime-v3 version");

    RuntimeData data;
    data.binary_schema = r.string();
    data.source_binding_sha256 = r.string();
    data.source_proof_bundle_hash = r.string();
    data.coordinate_system = r.string();
    data.units_per_pixel = r.f32();
    data.feature_flags = r.u32();
    data.playback_contract_hash = r.string();
    data.reference_raster_contract_hash = r.string();
    data.alpha_cutout_threshold = r.f32();

    if (data.binary_schema != "RealSaS.RuntimeBinary.rsr.v3") {
        throw std::runtime_error("Unexpected RealSaS runtime-v3 binary schema");
    }
    if (data.source_binding_sha256.size() != 64 || data.source_proof_bundle_hash.size() != 64 ||
        data.playback_contract_hash.size() != 64 || data.reference_raster_contract_hash.size() != 64) {
        throw std::runtime_error("Runtime-v3 source/contract identity is not SHA-256 length");
    }
    if (data.coordinate_system != "screen_2d.top_left.x_right.y_down.depth_camera_forward") {
        throw std::runtime_error("Runtime-v3 coordinate contract mismatch");
    }
    if (!(data.units_per_pixel > 0.0f)) throw std::runtime_error("Invalid runtime-v3 unit scale");
    if ((data.feature_flags & kRequiredFeatureMask) != kRequiredFeatureMask) {
        throw std::runtime_error("Runtime-v3 package lacks required playback/depth capabilities");
    }
    if (!(data.alpha_cutout_threshold >= 0.0f && data.alpha_cutout_threshold <= 1.0f)) {
        throw std::runtime_error("Invalid runtime-v3 alpha cutout threshold");
    }

    const uint32_t slot_count = r.u32();
    if (slot_count == 0 || slot_count > 4096) throw std::runtime_error("Invalid runtime-v3 slot count");
    data.slots.reserve(slot_count);
    std::unordered_set<std::string> slot_ids;
    std::unordered_set<uint32_t> setup_orders;
    for (uint32_t si = 0; si < slot_count; ++si) {
        Slot slot;
        slot.id = r.string();
        slot.bone_id = r.string();
        slot.setup_order = r.u32();
        slot.default_attachment_id = r.string();
        if (slot.id.empty() || !slot_ids.insert(slot.id).second) {
            throw std::runtime_error("Duplicate/empty runtime-v3 slot id");
        }
        if (!setup_orders.insert(slot.setup_order).second) {
            throw std::runtime_error("Runtime-v3 slot setup order must be total");
        }
        data.slots.push_back(std::move(slot));
    }

    const uint32_t view_count = r.u32();
    if (view_count == 0 || view_count > 64) throw std::runtime_error("Invalid runtime-v3 view count");
    data.views.reserve(view_count);
    std::unordered_set<std::string> view_ids;
    std::unordered_set<std::string> mesh_ids;
    std::unordered_map<std::string, std::pair<uint32_t, AttachmentKind>> attachment_semantics;
    for (uint32_t vi = 0; vi < view_count; ++vi) {
        View view;
        view.id = r.string();
        view.texture_path = r.string();
        view.texture_crc32 = r.u32();
        view.width = r.u32();
        view.height = r.u32();
        if (view.id.empty() || !view_ids.insert(view.id).second) {
            throw std::runtime_error("Duplicate/empty runtime-v3 view id");
        }
        if (view.texture_path.empty()) throw std::runtime_error("Empty runtime-v3 texture path");
        if (view.width == 0 || view.height == 0 || view.width > 32768 || view.height > 32768) {
            throw std::runtime_error("Invalid runtime-v3 texture dimensions");
        }
        const uint32_t mesh_count = r.u32();
        if (mesh_count == 0 || mesh_count > 4096) throw std::runtime_error("Invalid runtime-v3 mesh count");
        view.meshes.reserve(mesh_count);
        std::unordered_set<std::string> view_attachments;
        for (uint32_t mi = 0; mi < mesh_count; ++mi) {
            Mesh mesh;
            mesh.id = r.string();
            mesh.slot_index = r.u32();
            mesh.attachment_id = r.string();
            mesh.attachment_kind = parse_attachment_kind(r.u8());
            mesh.topology_class = parse_topology_class(r.u8());
            (void)r.u8();
            (void)r.u8();
            const uint32_t vertex_count = r.u32();
            const uint32_t triangle_count = r.u32();

            if (mesh.id.empty() || !mesh_ids.insert(mesh.id).second) {
                throw std::runtime_error("Duplicate/empty runtime-v3 mesh id");
            }
            if (mesh.slot_index >= data.slots.size()) throw std::runtime_error("Runtime-v3 mesh slot index out of range");
            if (mesh.attachment_id.empty() || !view_attachments.insert(mesh.attachment_id).second) {
                throw std::runtime_error("Duplicate/empty runtime-v3 attachment variant in view");
            }
            const auto semantic = std::make_pair(mesh.slot_index, mesh.attachment_kind);
            const auto found = attachment_semantics.find(mesh.attachment_id);
            if (found == attachment_semantics.end()) {
                attachment_semantics.emplace(mesh.attachment_id, semantic);
            } else if (found->second != semantic) {
                throw std::runtime_error("Runtime-v3 attachment semantic drift across views");
            }
            if (vertex_count == 0 || vertex_count > 10'000'000u || triangle_count == 0 || triangle_count > 20'000'000u) {
                throw std::runtime_error("Invalid runtime-v3 mesh dimensions");
            }

            mesh.vertices.resize(vertex_count);
            for (auto& vertex : mesh.vertices) {
                vertex.rest = {r.f32(), r.f32(), r.f32()};
                vertex.uv = {r.f32(), r.f32()};
                if (vertex.uv.x < 0.0f || vertex.uv.x > 1.0f || vertex.uv.y < 0.0f || vertex.uv.y > 1.0f) {
                    throw std::runtime_error("Runtime-v3 UV outside [0,1]");
                }
            }
            mesh.triangles.resize(triangle_count);
            for (auto& tri : mesh.triangles) {
                tri = {r.u32(), r.u32(), r.u32()};
                if (tri[0] >= vertex_count || tri[1] >= vertex_count || tri[2] >= vertex_count) {
                    throw std::runtime_error("Runtime-v3 triangle index out of range");
                }
            }
            view.meshes.push_back(std::move(mesh));
        }
        data.views.push_back(std::move(view));
    }

    for (const auto& slot : data.slots) {
        if (slot.default_attachment_id.empty()) continue;
        const auto found = attachment_semantics.find(slot.default_attachment_id);
        if (found == attachment_semantics.end()) {
            throw std::runtime_error("Runtime-v3 default attachment not present in mesh set");
        }
    }

    const uint32_t clip_count = r.u32();
    if (clip_count == 0 || clip_count > 4096) throw std::runtime_error("Invalid runtime-v3 clip count");
    data.clips.reserve(clip_count);
    std::unordered_set<std::string> clip_ids;
    for (uint32_t ci = 0; ci < clip_count; ++ci) {
        Clip clip;
        clip.id = r.string();
        clip.display_name = r.string();
        clip.intent = r.string();
        clip.duration = r.f32();
        clip.fps = r.f32();
        clip.loop = r.u8() != 0;
        clip.qualified = r.u8() != 0;
        (void)r.u8();
        (void)r.u8();
        const uint32_t frame_count = r.u32();
        if (clip.id.empty() || !clip_ids.insert(clip.id).second) throw std::runtime_error("Duplicate/empty runtime-v3 clip id");
        if (!(clip.duration > 0.0f) || !(clip.fps > 0.0f) || frame_count == 0 || frame_count > 1'000'000u) {
            throw std::runtime_error("Invalid runtime-v3 clip metadata");
        }
        clip.frames.reserve(frame_count);
        float prior_time = -1.0f;
        for (uint32_t fi = 0; fi < frame_count; ++fi) {
            Frame frame;
            frame.time = r.f32();
            if (frame.time < prior_time) throw std::runtime_error("Runtime-v3 frame times must be ordered");
            prior_time = frame.time;
            frame.views.resize(data.views.size());

            for (size_t vi = 0; vi < data.views.size(); ++vi) {
                const View& view = data.views[vi];
                FrameView& fv = frame.views[vi];
                fv.mesh_vertices.resize(view.meshes.size());
                for (size_t mi = 0; mi < view.meshes.size(); ++mi) {
                    const size_t count = view.meshes[mi].vertices.size();
                    auto& points = fv.mesh_vertices[mi];
                    points.resize(count);
                    for (auto& point : points) point = {r.f32(), r.f32(), r.f32()};
                }

                const uint32_t draw_count = r.u32();
                if (draw_count > data.slots.size()) throw std::runtime_error("Invalid runtime-v3 draw-order count");
                fv.draw_order_slots.resize(draw_count);
                for (auto& index : fv.draw_order_slots) index = r.u32();
                validate_exact_slot_permutation(fv.draw_order_slots, data.slots.size());

                const uint32_t active_count = r.u32();
                if (active_count != data.slots.size()) {
                    throw std::runtime_error("Runtime-v3 active attachment map must cover all slots");
                }
                fv.active_attachment_by_slot.resize(active_count);
                std::unordered_set<std::string> active_clipping;
                for (uint32_t slot_index = 0; slot_index < active_count; ++slot_index) {
                    auto& attachment = fv.active_attachment_by_slot[slot_index];
                    attachment = r.string();
                    if (attachment.empty()) continue;
                    const auto semantic = attachment_semantics.find(attachment);
                    if (semantic == attachment_semantics.end() || semantic->second.first != slot_index) {
                        throw std::runtime_error("Runtime-v3 active attachment slot mismatch");
                    }
                    if (semantic->second.second == AttachmentKind::Clipping) {
                        active_clipping.insert(attachment);
                    }
                }

                const uint32_t interval_count = r.u32();
                if (interval_count > data.slots.size()) throw std::runtime_error("Invalid runtime-v3 clipping interval count");
                if (interval_count > 0 && (data.feature_flags & kFeatureClipIntervals) == 0) {
                    throw std::runtime_error("Runtime-v3 clipping intervals present without capability flag");
                }
                fv.clip_intervals.resize(interval_count);
                std::unordered_set<std::string> interval_ids;
                std::vector<uint32_t> position(data.slots.size());
                for (uint32_t oi = 0; oi < fv.draw_order_slots.size(); ++oi) {
                    position[fv.draw_order_slots[oi]] = oi;
                }
                for (auto& interval : fv.clip_intervals) {
                    interval.clip_attachment_id = r.string();
                    interval.start_slot_index = r.u32();
                    interval.end_slot_index = r.u32();
                    interval.inverse = r.u8() != 0;
                    (void)r.u8();
                    (void)r.u8();
                    (void)r.u8();
                    if (interval.clip_attachment_id.empty() ||
                        interval.start_slot_index >= data.slots.size() ||
                        interval.end_slot_index >= data.slots.size()) {
                        throw std::runtime_error("Invalid runtime-v3 clipping interval");
                    }
                    if (!interval_ids.insert(interval.clip_attachment_id).second) {
                        throw std::runtime_error("Duplicate runtime-v3 clipping interval");
                    }
                    const auto semantic = attachment_semantics.find(interval.clip_attachment_id);
                    if (semantic == attachment_semantics.end() || semantic->second.second != AttachmentKind::Clipping) {
                        throw std::runtime_error("Runtime-v3 clipping interval requires clipping attachment");
                    }
                    if (semantic->second.first != interval.start_slot_index) {
                        throw std::runtime_error("Runtime-v3 clipping interval start must equal clipping slot");
                    }
                    if (fv.active_attachment_by_slot[interval.start_slot_index] != interval.clip_attachment_id) {
                        throw std::runtime_error("Runtime-v3 clipping attachment is not active at interval start");
                    }
                    if (position[interval.start_slot_index] >= position[interval.end_slot_index]) {
                        throw std::runtime_error("Runtime-v3 clipping interval must cover a subsequent slot");
                    }
                }
                if (active_clipping != interval_ids) {
                    throw std::runtime_error("Runtime-v3 active clipping attachment requires exactly one interval");
                }
            }
            clip.frames.push_back(std::move(frame));
        }
        data.clips.push_back(std::move(clip));
    }

    if (r.offset() != body.size()) throw std::runtime_error("Trailing bytes in RealSaS runtime-v3 binary");
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
    if (clip.frames.empty()) throw std::runtime_error("Runtime-v3 clip has no frames");
    time = normalized_time(clip, time);
    auto upper = std::lower_bound(
        clip.frames.begin(), clip.frames.end(), time,
        [](const Frame& frame, float value) { return frame.time < value; });
    if (upper == clip.frames.begin()) return {&clip.frames.front(), &clip.frames.front(), 0.0f};
    if (upper == clip.frames.end()) return {&clip.frames.back(), &clip.frames.back(), 0.0f};
    const Frame* b = &*upper;
    const Frame* a = &*(upper - 1);
    const float denom = b->time - a->time;
    const float alpha = denom > 1e-8f ? std::clamp((time - a->time) / denom, 0.0f, 1.0f) : 0.0f;
    return {a, b, alpha};
}

Vec3 lerp(Vec3 a, Vec3 b, float t) {
    return {
        a.x + (b.x - a.x) * t,
        a.y + (b.y - a.y) * t,
        a.z + (b.z - a.z) * t,
    };
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
            view.texture_rgba[i] / 255.0f,
            view.texture_rgba[i + 1] / 255.0f,
            view.texture_rgba[i + 2] / 255.0f,
            view.texture_rgba[i + 3] / 255.0f,
        };
    };

    const auto p00 = pixel(x0, y0);
    const auto p10 = pixel(x1, y0);
    const auto p01 = pixel(x0, y1);
    const auto p11 = pixel(x1, y1);
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
    if (sa <= 0.0f) return;
    const float da = dst[3] / 255.0f;
    const float oa = sa + da * (1.0f - sa);
    if (oa <= 1e-8f) {
        dst[0] = dst[1] = dst[2] = dst[3] = 0;
        return;
    }
    for (int c = 0; c < 3; ++c) {
        const float dc = dst[c] / 255.0f;
        const float oc = (src[c] * sa + dc * da * (1.0f - sa)) / oa;
        dst[c] = static_cast<uint8_t>(std::clamp(std::lround(oc * 255.0f), 0l, 255l));
    }
    dst[3] = static_cast<uint8_t>(std::clamp(std::lround(oa * 255.0f), 0l, 255l));
}

const Mesh& resolve_active_mesh(
    const View& view,
    uint32_t slot_index,
    const std::string& attachment_id,
    size_t* mesh_index_out) {
    const Mesh* found = nullptr;
    size_t found_index = 0;
    for (size_t mi = 0; mi < view.meshes.size(); ++mi) {
        const Mesh& mesh = view.meshes[mi];
        if (mesh.slot_index == slot_index && mesh.attachment_id == attachment_id) {
            if (found) throw std::runtime_error("Runtime-v3 active attachment resolves to multiple meshes in one view");
            found = &mesh;
            found_index = mi;
        }
    }
    if (!found) {
        throw std::runtime_error(
            "Runtime-v3 active attachment variant missing in view: slot=" +
            std::to_string(slot_index) + " attachment=" + attachment_id);
    }
    if (mesh_index_out) *mesh_index_out = found_index;
    return *found;
}

DepthWritePolicy depth_write_policy(AttachmentKind kind) {
    switch (kind) {
        case AttachmentKind::DeformableBody: return DepthWritePolicy::On;
        case AttachmentKind::RigidComponent: return DepthWritePolicy::CutoutOnly;
        case AttachmentKind::Clipping: break;
    }
    throw std::runtime_error("Runtime-v3 clipping attachment cannot write color/depth");
}

std::vector<uint8_t> rasterize_clip_mask(
    const View& view,
    const Mesh& mesh,
    const std::vector<Vec3>& posed_a,
    const std::vector<Vec3>& posed_b,
    float frame_alpha,
    bool inverse) {
    if (mesh.attachment_kind != AttachmentKind::Clipping) {
        throw std::runtime_error("Runtime-v3 clip-mask raster requires clipping attachment");
    }
    if (posed_a.size() != mesh.vertices.size() || posed_b.size() != mesh.vertices.size()) {
        throw std::runtime_error("Runtime-v3 clipping frame vertex count drift");
    }

    const size_t pixels = static_cast<size_t>(view.width) * view.height;
    std::vector<uint8_t> mask(pixels, 0);
    for (const auto& tri : mesh.triangles) {
        const Vec3 p0 = lerp(posed_a[tri[0]], posed_b[tri[0]], frame_alpha);
        const Vec3 p1 = lerp(posed_a[tri[1]], posed_b[tri[1]], frame_alpha);
        const Vec3 p2 = lerp(posed_a[tri[2]], posed_b[tri[2]], frame_alpha);
        const Vertex v0{p0.x, p0.y, p0.z, 0.0f, 0.0f};
        const Vertex v1{p1.x, p1.y, p1.z, 0.0f, 0.0f};
        const Vertex v2{p2.x, p2.y, p2.z, 0.0f, 0.0f};

        const float min_fx = std::min({v0.x, v1.x, v2.x});
        const float max_fx = std::max({v0.x, v1.x, v2.x});
        const float min_fy = std::min({v0.y, v1.y, v2.y});
        const float max_fy = std::max({v0.y, v1.y, v2.y});
        const int min_x = std::max(0, static_cast<int>(std::floor(min_fx - 0.5f)));
        const int max_x = std::min(static_cast<int>(view.width) - 1, static_cast<int>(std::ceil(max_fx - 0.5f)));
        const int min_y = std::max(0, static_cast<int>(std::floor(min_fy - 0.5f)));
        const int max_y = std::min(static_cast<int>(view.height) - 1, static_cast<int>(std::ceil(max_fy - 0.5f)));
        if (min_x > max_x || min_y > max_y) continue;

        for (int y = min_y; y <= max_y; ++y) {
            for (int x = min_x; x <= max_x; ++x) {
                if (!reference_raster_v3::cover_pixel_center(v0, v1, v2, x, y).covered) continue;
                mask[static_cast<size_t>(y) * view.width + static_cast<size_t>(x)] = 1;
            }
        }
    }
    if (inverse) {
        for (auto& value : mask) value = value ? 0 : 1;
    }
    return mask;
}

std::vector<ActiveClipMask> build_active_clip_masks(
    const RuntimeData& data,
    const View& view,
    const FrameView& a,
    const FrameView& b,
    const FrameView& composition,
    float frame_alpha) {
    std::vector<ActiveClipMask> out;
    if (composition.clip_intervals.empty()) return out;

    std::vector<uint32_t> slot_position(data.slots.size());
    for (uint32_t order_rank = 0; order_rank < composition.draw_order_slots.size(); ++order_rank) {
        slot_position[composition.draw_order_slots[order_rank]] = order_rank;
    }

    out.reserve(composition.clip_intervals.size());
    for (const auto& interval : composition.clip_intervals) {
        if (interval.start_slot_index >= data.slots.size() || interval.end_slot_index >= data.slots.size()) {
            throw std::runtime_error("Runtime-v3 clipping interval slot index out of range at render");
        }
        if (composition.active_attachment_by_slot[interval.start_slot_index] != interval.clip_attachment_id) {
            throw std::runtime_error("Runtime-v3 clipping attachment inactive at sampled interval start");
        }
        const uint32_t start_order = slot_position[interval.start_slot_index];
        const uint32_t end_order = slot_position[interval.end_slot_index];
        if (start_order >= end_order) {
            throw std::runtime_error("Runtime-v3 clipping interval is not forward in sampled draw order");
        }

        size_t mesh_index = 0;
        const Mesh& mesh = resolve_active_mesh(
            view,
            interval.start_slot_index,
            interval.clip_attachment_id,
            &mesh_index);
        if (mesh.attachment_kind != AttachmentKind::Clipping) {
            throw std::runtime_error("Runtime-v3 clipping interval resolved non-clipping mesh");
        }
        if (mesh_index >= a.mesh_vertices.size() || mesh_index >= b.mesh_vertices.size()) {
            throw std::runtime_error("Runtime-v3 clipping frame/mesh topology drift");
        }

        ActiveClipMask row;
        row.start_order = start_order;
        row.end_order = end_order;
        row.mask = rasterize_clip_mask(
            view,
            mesh,
            a.mesh_vertices[mesh_index],
            b.mesh_vertices[mesh_index],
            frame_alpha,
            interval.inverse);
        out.push_back(std::move(row));
    }
    return out;
}

bool pixel_passes_clipping(
    size_t pixel_index,
    uint32_t order_rank,
    const std::vector<ActiveClipMask>& clip_masks) {
    for (const auto& clip : clip_masks) {
        if (order_rank <= clip.start_order || order_rank > clip.end_order) continue;
        if (pixel_index >= clip.mask.size() || clip.mask[pixel_index] == 0) return false;
    }
    return true;
}

void render_triangle(
    const View& view,
    const Mesh& mesh,
    const std::vector<Vec3>& posed_a,
    const std::vector<Vec3>& posed_b,
    float frame_alpha,
    const std::array<uint32_t, 3>& tri,
    uint32_t order_rank,
    uint32_t semantic_order,
    float cutout_threshold,
    const std::vector<ActiveClipMask>& clip_masks,
    std::vector<DepthSample>& depth,
    std::vector<uint8_t>& rgba) {
    const Vec3 p0 = lerp(posed_a[tri[0]], posed_b[tri[0]], frame_alpha);
    const Vec3 p1 = lerp(posed_a[tri[1]], posed_b[tri[1]], frame_alpha);
    const Vec3 p2 = lerp(posed_a[tri[2]], posed_b[tri[2]], frame_alpha);
    const RestVertex& r0 = mesh.vertices[tri[0]];
    const RestVertex& r1 = mesh.vertices[tri[1]];
    const RestVertex& r2 = mesh.vertices[tri[2]];

    const Vertex v0{p0.x, p0.y, p0.z, r0.uv.x, r0.uv.y};
    const Vertex v1{p1.x, p1.y, p1.z, r1.uv.x, r1.uv.y};
    const Vertex v2{p2.x, p2.y, p2.z, r2.uv.x, r2.uv.y};

    const float min_fx = std::min({v0.x, v1.x, v2.x});
    const float max_fx = std::max({v0.x, v1.x, v2.x});
    const float min_fy = std::min({v0.y, v1.y, v2.y});
    const float max_fy = std::max({v0.y, v1.y, v2.y});
    const int min_x = std::max(0, static_cast<int>(std::floor(min_fx - 0.5f)));
    const int max_x = std::min(static_cast<int>(view.width) - 1, static_cast<int>(std::ceil(max_fx - 0.5f)));
    const int min_y = std::max(0, static_cast<int>(std::floor(min_fy - 0.5f)));
    const int max_y = std::min(static_cast<int>(view.height) - 1, static_cast<int>(std::ceil(max_fy - 0.5f)));
    if (min_x > max_x || min_y > max_y) return;

    const DepthWritePolicy policy = depth_write_policy(mesh.attachment_kind);
    for (int y = min_y; y <= max_y; ++y) {
        for (int x = min_x; x <= max_x; ++x) {
            const Barycentric bc = reference_raster_v3::cover_pixel_center(v0, v1, v2, x, y);
            if (!bc.covered) continue;
            const size_t pixel_index = static_cast<size_t>(y) * view.width + static_cast<size_t>(x);
            if (!pixel_passes_clipping(pixel_index, order_rank, clip_masks)) continue;

            const float z = reference_raster_v3::interpolate_depth(bc, v0, v1, v2);
            if (!reference_raster_v3::depth_test_passes(depth[pixel_index], z, semantic_order)) continue;

            const float u = v0.u * bc.w0 + v1.u * bc.w1 + v2.u * bc.w2;
            const float v = v0.v * bc.w0 + v1.v * bc.w1 + v2.v * bc.w2;
            const auto sample = sample_bilinear(view, u, v);
            const float alpha = std::clamp(sample[3], 0.0f, 1.0f);
            if (alpha <= 0.0f) continue;

            blend_pixel(rgba.data() + pixel_index * 4, sample);
            if (reference_raster_v3::should_write_depth(policy, alpha, cutout_threshold)) {
                reference_raster_v3::commit_depth(depth[pixel_index], z, semantic_order);
            }
        }
    }
}

std::vector<uint8_t> render_frame(
    const RuntimeData& data,
    uint32_t clip_index,
    uint32_t view_index,
    float time_seconds) {
    if (clip_index >= data.clips.size() || view_index >= data.views.size()) {
        throw std::out_of_range("Runtime-v3 clip/view index out of range");
    }
    const Clip& clip = data.clips[clip_index];
    const View& view = data.views[view_index];
    const SampleSpan span = sample_span(clip, time_seconds);
    const FrameView& a = span.a->views[view_index];
    const FrameView& b = span.b->views[view_index];
    const FrameView& composition = span.alpha < 0.5f ? a : b;

    const size_t pixels = static_cast<size_t>(view.width) * view.height;
    std::vector<uint8_t> rgba(pixels * 4, 0);
    std::vector<DepthSample> depth(pixels);

    if (composition.active_attachment_by_slot.size() != data.slots.size()) {
        throw std::runtime_error("Runtime-v3 sampled active attachment map is incomplete");
    }
    const auto clip_masks = build_active_clip_masks(data, view, a, b, composition, span.alpha);

    for (uint32_t order_rank = 0; order_rank < composition.draw_order_slots.size(); ++order_rank) {
        const uint32_t slot_index = composition.draw_order_slots[order_rank];
        if (slot_index >= data.slots.size()) throw std::runtime_error("Runtime-v3 sampled slot index out of range");
        const std::string& active_attachment = composition.active_attachment_by_slot[slot_index];
        if (active_attachment.empty()) continue;

        size_t mesh_index = 0;
        const Mesh& mesh = resolve_active_mesh(view, slot_index, active_attachment, &mesh_index);
        // Clipping attachments create a stencil/clip region; they do not emit
        // color or depth themselves. Their interval applies to subsequent slots
        // through end_slot inclusive.
        if (mesh.attachment_kind == AttachmentKind::Clipping) continue;
        if (mesh_index >= a.mesh_vertices.size() || mesh_index >= b.mesh_vertices.size()) {
            throw std::runtime_error("Runtime-v3 frame/mesh topology drift");
        }
        const auto& posed_a = a.mesh_vertices[mesh_index];
        const auto& posed_b = b.mesh_vertices[mesh_index];
        if (posed_a.size() != mesh.vertices.size() || posed_b.size() != mesh.vertices.size()) {
            throw std::runtime_error("Runtime-v3 frame vertex count drift");
        }

        for (size_t ti = 0; ti < mesh.triangles.size(); ++ti) {
            // Slot order is the semantic depth tie-break. Triangle index makes
            // same-slot exact-depth overlaps deterministic without depending on
            // memory address or hash iteration order.
            const uint32_t local = static_cast<uint32_t>(std::min<size_t>(ti, 0xFFFFFu));
            const uint32_t semantic_order =
                (std::min<uint32_t>(order_rank, 0xFFFu) << 20) | local;
            render_triangle(
                view,
                mesh,
                posed_a,
                posed_b,
                span.alpha,
                mesh.triangles[ti],
                order_rank,
                semantic_order,
                data.alpha_cutout_threshold,
                clip_masks,
                depth,
                rgba);
        }
    }
    return rgba;
}

} // namespace

struct ReferenceRuntime::Impl {
    std::filesystem::path package_path;
    RuntimeData data;
};

ReferenceRuntime::ReferenceRuntime(const std::filesystem::path& package_path)
    : impl_(std::make_unique<Impl>()) {
    impl_->package_path = std::filesystem::absolute(package_path);
    if (!std::filesystem::exists(impl_->package_path)) {
        throw std::runtime_error("Runtime-v3 package does not exist: " + impl_->package_path.string());
    }

    const auto binary = read_package_asset(impl_->package_path, "runtime/realsas_runtime.rsr");
    impl_->data = parse_runtime_v3_binary(binary);

    for (auto& view : impl_->data.views) {
        const auto texture_payload = read_package_asset(impl_->package_path, view.texture_path);
        const uint32_t actual_crc = crc32_payload(texture_payload.data(), texture_payload.size());
        if (actual_crc != view.texture_crc32) {
            throw std::runtime_error("Runtime-v3 texture CRC32 mismatch: " + view.id);
        }
        view.texture_rgba = decode_png_rgba(texture_payload, view.width, view.height);
    }
}

ReferenceRuntime::~ReferenceRuntime() = default;
ReferenceRuntime::ReferenceRuntime(ReferenceRuntime&&) noexcept = default;
ReferenceRuntime& ReferenceRuntime::operator=(ReferenceRuntime&&) noexcept = default;

const std::string& ReferenceRuntime::binary_schema() const noexcept { return impl_->data.binary_schema; }
const std::string& ReferenceRuntime::source_binding_sha256() const noexcept { return impl_->data.source_binding_sha256; }
const std::string& ReferenceRuntime::source_proof_bundle_hash() const noexcept { return impl_->data.source_proof_bundle_hash; }
const std::string& ReferenceRuntime::coordinate_system() const noexcept { return impl_->data.coordinate_system; }
const std::string& ReferenceRuntime::playback_contract_hash() const noexcept { return impl_->data.playback_contract_hash; }
const std::string& ReferenceRuntime::reference_raster_contract_hash() const noexcept { return impl_->data.reference_raster_contract_hash; }
float ReferenceRuntime::units_per_pixel() const noexcept { return impl_->data.units_per_pixel; }
uint32_t ReferenceRuntime::feature_flags() const noexcept { return impl_->data.feature_flags; }
float ReferenceRuntime::alpha_cutout_threshold() const noexcept { return impl_->data.alpha_cutout_threshold; }

uint32_t ReferenceRuntime::view_count() const noexcept { return static_cast<uint32_t>(impl_->data.views.size()); }
const std::string& ReferenceRuntime::view_id(uint32_t view_index) const {
    if (view_index >= impl_->data.views.size()) throw std::out_of_range("Runtime-v3 view index out of range");
    return impl_->data.views[view_index].id;
}
uint32_t ReferenceRuntime::view_width(uint32_t view_index) const {
    if (view_index >= impl_->data.views.size()) throw std::out_of_range("Runtime-v3 view index out of range");
    return impl_->data.views[view_index].width;
}
uint32_t ReferenceRuntime::view_height(uint32_t view_index) const {
    if (view_index >= impl_->data.views.size()) throw std::out_of_range("Runtime-v3 view index out of range");
    return impl_->data.views[view_index].height;
}
uint32_t ReferenceRuntime::find_view(const std::string& id) const noexcept {
    for (uint32_t i = 0; i < impl_->data.views.size(); ++i) {
        if (impl_->data.views[i].id == id) return i;
    }
    return UINT32_MAX;
}

uint32_t ReferenceRuntime::clip_count() const noexcept { return static_cast<uint32_t>(impl_->data.clips.size()); }
const std::string& ReferenceRuntime::clip_id(uint32_t clip_index) const {
    if (clip_index >= impl_->data.clips.size()) throw std::out_of_range("Runtime-v3 clip index out of range");
    return impl_->data.clips[clip_index].id;
}
const std::string& ReferenceRuntime::clip_display_name(uint32_t clip_index) const {
    if (clip_index >= impl_->data.clips.size()) throw std::out_of_range("Runtime-v3 clip index out of range");
    return impl_->data.clips[clip_index].display_name;
}
const std::string& ReferenceRuntime::clip_intent(uint32_t clip_index) const {
    if (clip_index >= impl_->data.clips.size()) throw std::out_of_range("Runtime-v3 clip index out of range");
    return impl_->data.clips[clip_index].intent;
}
float ReferenceRuntime::clip_duration(uint32_t clip_index) const {
    if (clip_index >= impl_->data.clips.size()) throw std::out_of_range("Runtime-v3 clip index out of range");
    return impl_->data.clips[clip_index].duration;
}
bool ReferenceRuntime::clip_qualified(uint32_t clip_index) const {
    if (clip_index >= impl_->data.clips.size()) throw std::out_of_range("Runtime-v3 clip index out of range");
    return impl_->data.clips[clip_index].qualified;
}
uint32_t ReferenceRuntime::find_clip(const std::string& name) const noexcept {
    for (uint32_t i = 0; i < impl_->data.clips.size(); ++i) {
        const auto& clip = impl_->data.clips[i];
        if (clip.id == name || clip.intent == name || clip.display_name == name) return i;
    }
    return UINT32_MAX;
}

std::vector<uint8_t> ReferenceRuntime::render_rgba(
    uint32_t clip_index,
    uint32_t view_index,
    float time_seconds) const {
    return render_frame(impl_->data, clip_index, view_index, time_seconds);
}

} // namespace realsas::runtime_v3
