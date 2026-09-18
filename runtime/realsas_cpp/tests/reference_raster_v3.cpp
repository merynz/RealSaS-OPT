#include "../src/reference_raster_v3.h"

#include <cassert>
#include <cmath>

using namespace realsas::reference_raster_v3;

int main() {
    // Half-integer pixel center and top-left ownership.
    const Vertex a{0, 0, 0.5f, 0, 0};
    const Vertex b{2, 0, 0.5f, 1, 0};
    const Vertex c{0, 2, 0.5f, 0, 1};
    const auto inside = cover_pixel_center(a, b, c, 0, 0);
    assert(inside.covered);
    assert(std::abs((inside.w0 + inside.w1 + inside.w2) - 1.0f) < 1e-6f);

    // Two triangles sharing a diagonal must never double-own the same pixel
    // center exactly on the shared edge.
    const Vertex q0{0, 0, 0, 0, 0};
    const Vertex q1{2, 0, 0, 1, 0};
    const Vertex q2{2, 2, 0, 1, 1};
    const Vertex q3{0, 2, 0, 0, 1};
    const bool t0 = cover_pixel_center(q0, q1, q2, 0, 0).covered;
    const bool t1 = cover_pixel_center(q0, q2, q3, 0, 0).covered;
    assert(t0 != t1);

    // Reversing winding must preserve coverage when culling is disabled.
    const auto inside_reversed = cover_pixel_center(c, b, a, 0, 0);
    assert(inside_reversed.covered);
    assert(std::abs((inside_reversed.w0 + inside_reversed.w1 + inside_reversed.w2) - 1.0f) < 1e-6f);

    // The shared-edge sample is owned by exactly one triangle: no crack and no double hit.
    assert((t0 ? 1 : 0) + (t1 ? 1 : 0) == 1);

    // Smaller camera-forward depth wins independent of semantic submission.
    DepthSample depth{};
    assert(depth_test_passes(depth, 3.0f, 10));
    commit_depth(depth, 3.0f, 10);
    assert(depth_test_passes(depth, 2.0f, 1));
    assert(!depth_test_passes(depth, 4.0f, 999));

    // Exact/near-exact depth ties use semantic draw order, never triangle index.
    assert(!depth_test_passes(depth, 3.0f, 9));
    assert(depth_test_passes(depth, 3.0f, 10));
    assert(depth_test_passes(depth, 3.0f, 11));

    // Explicit alpha/depth-write semantics for body/rigid composition.
    assert(should_write_depth(DepthWritePolicy::On, 0.01f, 0.5f));
    assert(!should_write_depth(DepthWritePolicy::On, 0.0f, 0.5f));
    assert(!should_write_depth(DepthWritePolicy::Off, 1.0f, 0.5f));
    assert(!should_write_depth(DepthWritePolicy::CutoutOnly, 0.49f, 0.5f));
    assert(should_write_depth(DepthWritePolicy::CutoutOnly, 0.50f, 0.5f));

    return 0;
}
