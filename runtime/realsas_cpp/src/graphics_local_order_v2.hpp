#pragma once

#include "graphics_v1.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <utility>
#include <vector>

namespace realsas::graphics_v2 {

// Pixel/ray-local precedence. The same face pair may legitimately reverse on a
// different ray; callers must never union these edges into a view-global DAG.
struct LocalOrderEdge {
    std::uint32_t front_face{};
    std::uint32_t back_face{};
};

inline std::vector<std::size_t> order_pixel_layers_local(
    const std::vector<std::uint32_t>& faces,
    const std::vector<double>& depths,
    const std::vector<LocalOrderEdge>& local_edges,
    double epsilon
) {
    if(faces.size()!=depths.size() || epsilon<0 || !std::isfinite(epsilon))
        throw std::runtime_error("GRAPHICS_V2_LOCAL_PIXEL_INPUT_INVALID");
    const std::size_t n=faces.size();
    std::vector<std::size_t> order;order.reserve(n);
    if(n==0) return order;

    for(std::size_t i=0;i<n;++i) {
        if(!std::isfinite(depths[i]))
            throw std::runtime_error("GRAPHICS_V2_LOCAL_PIXEL_DEPTH_NONFINITE");
        for(std::size_t j=i+1;j<n;++j)
            if(faces[i]==faces[j])
                throw std::runtime_error("GRAPHICS_V2_LOCAL_PIXEL_FACE_DUPLICATE");
    }

    std::vector<std::size_t> indegree(n,0);
    std::vector<std::vector<std::size_t>> outgoing(n);
    auto local_index=[&](std::uint32_t face)->std::size_t {
        for(std::size_t i=0;i<n;++i) if(faces[i]==face) return i;
        return n;
    };
    std::vector<std::uint8_t> relation(n*n,0);
    for(const auto& edge:local_edges) {
        if(edge.front_face==edge.back_face)
            throw std::runtime_error("GRAPHICS_V2_LOCAL_EDGE_SELF");
        const auto a=local_index(edge.front_face),b=local_index(edge.back_face);
        // A local rule whose endpoint is not present in this pixel has no
        // consequence here. This is not evidence for or against that relation.
        if(a==n || b==n) continue;
        const auto ab=a*n+b,ba=b*n+a;
        if(relation[ba])
            throw std::runtime_error("GRAPHICS_V2_LOCAL_EDGE_CONTRADICTION");
        if(relation[ab]) continue;
        relation[ab]=1;outgoing[a].push_back(b);++indegree[b];
    }

    std::vector<std::uint8_t> used(n,0);
    while(order.size()<n) {
        std::vector<std::size_t> ready;
        for(std::size_t i=0;i<n;++i)
            if(!used[i] && indegree[i]==0) ready.push_back(i);
        if(ready.empty())
            throw std::runtime_error("GRAPHICS_V2_LOCAL_PIXEL_ORDER_CYCLE");

        std::size_t best=ready.front();
        bool ambiguous=false;
        for(std::size_t k=1;k<ready.size();++k) {
            const auto candidate=ready[k];
            if(depths[candidate] < depths[best]-epsilon) {
                best=candidate;ambiguous=false;
            } else if(std::abs(depths[candidate]-depths[best])<=epsilon) {
                // Neither node precedes the other (both are ready), therefore a
                // depth tie has no measured authority. Do not use face id as a
                // hidden presentation label.
                ambiguous=true;
            }
        }
        if(ambiguous)
            throw std::runtime_error("GRAPHICS_V2_LOCAL_PIXEL_AMBIGUOUS_TIE");

        used[best]=1;order.push_back(best);
        for(const auto next:outgoing[best]) {
            if(indegree[next]==0)
                throw std::runtime_error("GRAPHICS_V2_LOCAL_PIXEL_INDEGREE_UNDERFLOW");
            --indegree[next];
        }
    }
    return order;
}

} // namespace realsas::graphics_v2
