#pragma once

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace realsas::runtime_v4 {

class ReferenceRuntime final {
public:
    explicit ReferenceRuntime(const std::filesystem::path& package_path);
    ~ReferenceRuntime();

    ReferenceRuntime(ReferenceRuntime&&) noexcept;
    ReferenceRuntime& operator=(ReferenceRuntime&&) noexcept;
    ReferenceRuntime(const ReferenceRuntime&) = delete;
    ReferenceRuntime& operator=(const ReferenceRuntime&) = delete;

    const std::string& binary_schema() const noexcept;
    const std::string& source_binding_sha256() const noexcept;
    const std::string& source_proof_bundle_hash() const noexcept;
    const std::string& coordinate_system() const noexcept;
    const std::string& playback_contract_hash() const noexcept;
    const std::string& reference_raster_contract_hash() const noexcept;
    uint32_t feature_flags() const noexcept;
    float alpha_cutout_threshold() const noexcept;

    uint32_t view_count() const noexcept;
    const std::string& view_id(uint32_t view_index) const;
    uint32_t view_width(uint32_t view_index) const;
    uint32_t view_height(uint32_t view_index) const;
    uint32_t find_view(const std::string& view_id) const noexcept;

    uint32_t clip_count() const noexcept;
    const std::string& clip_id(uint32_t clip_index) const;
    const std::string& clip_display_name(uint32_t clip_index) const;
    const std::string& clip_intent(uint32_t clip_index) const;
    float clip_duration(uint32_t clip_index) const;
    bool clip_qualified(uint32_t clip_index) const;
    uint32_t find_clip(const std::string& name) const noexcept;

    std::vector<uint8_t> render_rgba(
        uint32_t clip_index,
        uint32_t view_index,
        float time_seconds) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace realsas::runtime_v4
