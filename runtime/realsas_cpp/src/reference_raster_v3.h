#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>

namespace realsas::reference_raster_v3 {

// Private/native conformance primitive for Runtime v3. This is deliberately
// engine-neutral and has no subject-specific policy.

struct Vec2 {
    float x = 0.0f;
    float y = 0.0f;
};

struct Vertex {
    float x = 0.0f;
    float y = 0.0f;
    float z = 0.0f; // posed camera-forward depth; smaller is nearer.
    float u = 0.0f;
    float v = 0.0f;
};

enum class DepthWritePolicy : uint8_t {
    Off = 0,
    On = 1,
    CutoutOnly = 2,
};

struct DepthSample {
    float z = std::numeric_limits<float>::infinity();
    uint32_t semantic_order = 0;
    bool occupied = false;
};

struct Barycentric {
    float w0 = 0.0f;
    float w1 = 0.0f;
    float w2 = 0.0f;
    bool covered = false;
};

inline double orient2d(Vec2 a, Vec2 b, double px, double py) noexcept {
    return (static_cast<double>(b.x) - a.x) * (py - a.y)
         - (static_cast<double>(b.y) - a.y) * (px - a.x);
}

// Screen coordinates are +X right, +Y down. For positive orient2d winding, a
// directed edge is top/left when it travels upward, or right along a horizontal
// edge. Negative winding uses the reversed directed edge for the same ownership.
inline bool is_top_left(Vec2 a, Vec2 b) noexcept {
    const float dx = b.x - a.x;
    const float dy = b.y - a.y;
    return (dy < 0.0f) || (dy == 0.0f && dx > 0.0f);
}

inline bool edge_accept(double e, bool top_left) noexcept {
    constexpr double eps = 1e-12;
    if (e > eps) return true;
    if (e < -eps) return false;
    return top_left;
}

inline Barycentric cover_pixel_center(const Vertex& a, const Vertex& b, const Vertex& c, int pixel_x, int pixel_y) noexcept {
    const Vec2 p0{a.x, a.y}, p1{b.x, b.y}, p2{c.x, c.y};
    const double signed_area = orient2d(p0, p1, p2.x, p2.y);
    if (std::abs(signed_area) <= 1e-12) return {};

    const double px = static_cast<double>(pixel_x) + 0.5;
    const double py = static_cast<double>(pixel_y) + 0.5;
    const bool positive = signed_area > 0.0;
    const double sign = positive ? 1.0 : -1.0;
    const double area = std::abs(signed_area);

    // We preserve original vertex identity in w0/w1/w2. Only edge sign and
    // directed-edge ownership are normalized for winding.
    const double e0 = sign * orient2d(p1, p2, px, py);
    const double e1 = sign * orient2d(p2, p0, px, py);
    const double e2 = sign * orient2d(p0, p1, px, py);
    const bool tl0 = positive ? is_top_left(p1, p2) : is_top_left(p2, p1);
    const bool tl1 = positive ? is_top_left(p2, p0) : is_top_left(p0, p2);
    const bool tl2 = positive ? is_top_left(p0, p1) : is_top_left(p1, p0);

    if (!edge_accept(e0, tl0) || !edge_accept(e1, tl1) || !edge_accept(e2, tl2)) {
        return {};
    }

    Barycentric out;
    out.w0 = static_cast<float>(e0 / area);
    out.w1 = static_cast<float>(e1 / area);
    out.w2 = static_cast<float>(e2 / area);
    out.covered = true;
    return out;
}

inline float interpolate_depth(const Barycentric& b, const Vertex& a, const Vertex& c1, const Vertex& c2) noexcept {
    return a.z * b.w0 + c1.z * b.w1 + c2.z * b.w2;
}

inline bool depth_test_passes(const DepthSample& current, float incoming_z, uint32_t semantic_order) noexcept {
    if (!std::isfinite(incoming_z)) return false;
    if (!current.occupied) return true;
    constexpr float eps = 1e-6f;
    if (incoming_z < current.z - eps) return true;
    if (incoming_z > current.z + eps) return false;
    // Equal-depth surfaces are resolved by explicit semantic draw order. A
    // later slot/submission wins; address/order of triangles is never a tie-break.
    return semantic_order >= current.semantic_order;
}

inline bool should_write_depth(DepthWritePolicy policy, float alpha, float cutout_threshold) noexcept {
    if (!std::isfinite(alpha) || !std::isfinite(cutout_threshold)) return false;
    alpha = std::clamp(alpha, 0.0f, 1.0f);
    cutout_threshold = std::clamp(cutout_threshold, 0.0f, 1.0f);
    switch (policy) {
        case DepthWritePolicy::Off: return false;
        case DepthWritePolicy::On: return alpha > 0.0f;
        case DepthWritePolicy::CutoutOnly: return alpha >= cutout_threshold;
    }
    return false;
}

inline void commit_depth(DepthSample& dst, float z, uint32_t semantic_order) noexcept {
    dst.z = z;
    dst.semantic_order = semantic_order;
    dst.occupied = true;
}

} // namespace realsas::reference_raster_v3
