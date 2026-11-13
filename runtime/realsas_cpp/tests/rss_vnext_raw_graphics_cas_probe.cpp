// Real RSS.vNext two-original-image Graphics binary CAS probe.
//
// A native executable MUST consume the Python-created Graphics candidate,
// not merely revalidate a Python JSON object or a separate handwritten
// C++ fixture. Virtual directions are *not* source visibility observations.
#include "../src/graphics_v1.hpp"
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <vector>
#include <array>
#include <algorithm>
#include <cmath>
#include <cstring>

namespace {
std::vector<std::uint8_t> read_bytes(const char* path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) throw std::runtime_error("RAW_GFX_CAS_OPEN_FAILED");
    return std::vector<std::uint8_t>{
        std::istreambuf_iterator<char>(file),
        std::istreambuf_iterator<char>()
    };
}
template <typename T>
T take(const std::vector<std::uint8_t>& bytes, std::size_t& offset) {
    if (offset > bytes.size() || sizeof(T) > bytes.size() - offset)
        throw std::runtime_error("RAW_GFX_RASTER_FIXTURE_TRUNCATED");
    T out{};
    std::memcpy(&out, bytes.data() + offset, sizeof(T));
    offset += sizeof(T);
    return out;
}
struct Point { double x{}, y{}; };
double orient(const Point& a, const Point& b, double px, double py) {
    return (b.x-a.x)*(py-a.y) - (b.y-a.y)*(px-a.x);
}
bool top_left(const Point& a, const Point& b) {
    const double dx=b.x-a.x, dy=b.y-a.y;
    return dy < 0.0 || (dy == 0.0 && dx > 0.0);
}
bool accept_edge(double edge, bool inclusive) {
    constexpr double eps=1e-12;
    if (edge>eps) return true;
    if (edge< -eps) return false;
    return inclusive;
}
void assert_static_source_raster_parity(const char* geometry_file,
                                        const char* expected_file) {
    // Static original-alpha qualification path: Python and C++ must cover
    // EXACTLY the same pixel centers for EXACTLY the same fitted hypotheses.
    // This does NOT prove original camera/depth, source RGB or posed rendering.
    const auto bytes=read_bytes(geometry_file);
    std::size_t offset=0;
    const auto width=take<std::uint32_t>(bytes,offset);
    const auto height=take<std::uint32_t>(bytes,offset);
    const auto count=take<std::uint32_t>(bytes,offset);
    if (width<1 || width>512 || height<1 || height>512 || count>32768 ||
        bytes.size()-offset!=static_cast<std::size_t>(count)*6*sizeof(double))
        throw std::runtime_error("RAW_GFX_RASTER_FIXTURE_DOMAIN_INVALID");
    std::vector<std::uint8_t> raster(static_cast<std::size_t>(width)*height,0);
    for(std::uint32_t face=0;face<count;++face) {
        std::array<Point,3> p{};
        for(auto& q:p){q.x=take<double>(bytes,offset);q.y=take<double>(bytes,offset);
            if(!std::isfinite(q.x)||!std::isfinite(q.y)||std::abs(q.x)>1.e6||
               std::abs(q.y)>1.e6)
                throw std::runtime_error("RAW_GFX_RASTER_COORD_INVALID");
        }
        const auto a=p[0],b=p[1],c=p[2];
        const double area=orient(a,b,c.x,c.y);
        if(std::abs(area)<=1e-12) continue;
        const auto min_x=std::max(0,static_cast<int>(
            std::floor(std::min({a.x,b.x,c.x})-.5)));
        const auto max_x=std::min(static_cast<int>(width)-1,static_cast<int>(
            std::ceil(std::max({a.x,b.x,c.x})-.5)));
        const auto min_y=std::max(0,static_cast<int>(
            std::floor(std::min({a.y,b.y,c.y})-.5)));
        const auto max_y=std::min(static_cast<int>(height)-1,static_cast<int>(
            std::ceil(std::max({a.y,b.y,c.y})-.5)));
        const double sign=area>0?1.:-1.;
        const std::array<bool,3> edges=area>0
            ? std::array<bool,3>{top_left(b,c),top_left(c,a),top_left(a,b)}
            : std::array<bool,3>{top_left(c,b),top_left(a,c),top_left(b,a)};
        for(int y=min_y;y<=max_y;++y)for(int x=min_x;x<=max_x;++x){
            const double px=x+.5,py=y+.5;
            if(accept_edge(sign*orient(b,c,px,py),edges[0])&&
               accept_edge(sign*orient(c,a,px,py),edges[1])&&
               accept_edge(sign*orient(a,b,px,py),edges[2]))
                raster[static_cast<std::size_t>(y)*width+x]=1;
        }
    }
    if(raster!=read_bytes(expected_file))
        throw std::runtime_error("RAW_GFX_NATIVE_STATIC_RASTER_PIXEL_MISMATCH");
    std::cout<<"RAW_GFX_STATIC_G5_RASTER_PIXEL_EQUIVALENCE_PASS\n";
}
}

int main(int argc, char** argv) {
    try {
        if (argc != 2 && argc != 4) throw std::runtime_error("RAW_GFX_CAS_FILE_REQUIRED");
        std::ifstream input(argv[1], std::ios::binary);
        if (!input) throw std::runtime_error("RAW_GFX_CAS_OPEN_FAILED");
        const std::vector<std::uint8_t> bytes{
            std::istreambuf_iterator<char>(input),
            std::istreambuf_iterator<char>()
        };
        const auto graphics = realsas::graphics_v1::parse(bytes);
        if (graphics.views.size() != 8 || graphics.face_fragment_index.empty()
            || graphics.fragment_ids.empty())
            throw std::runtime_error("RAW_GFX_EIGHT_VIRTUAL_VIEW_DOMAIN_REQUIRED");
        for (std::uint32_t v = 0; v < 8; ++v) {
            if (graphics.views[v].view_index != v ||
                graphics.views[v].topological_order.size() != graphics.fragment_ids.size())
                throw std::runtime_error("RAW_GFX_VIRTUAL_VIEW_INDEX_OR_SORT_DRIFT");
            for (std::uint32_t face = 0; face < graphics.face_fragment_index.size(); ++face) {
                // Unknown source visibility means every fragment is a
                // *candidate* for depth testing, not a source-visible label.
                if (!realsas::graphics_v1::face_visible(graphics, v, face))
                    throw std::runtime_error("RAW_GFX_UNOBSERVED_FACE_WRONGLY_CULLED");
            }
            for (std::uint32_t a = 0; a < graphics.face_fragment_index.size(); ++a)
                for (std::uint32_t b = 0; b < graphics.face_fragment_index.size(); ++b)
                    if (a != b && realsas::graphics_v1::explicit_precedence(graphics, v, a, b))
                        throw std::runtime_error("RAW_GFX_UNOBSERVED_ORDER_FORGED");
        }
        if (argc == 4) assert_static_source_raster_parity(argv[2], argv[3]);
        std::cout << "RAW_GFX_BINARY_CAS_NATIVE_CONSUMPTION_PASS\n";
        std::cout << "faces=" << graphics.face_fragment_index.size()
                  << " fragments=" << graphics.fragment_ids.size()
                  << " views=" << graphics.views.size() << "\n";
        std::cout << "mesh=" << graphics.canonical_mesh_binding_hash << "\n";
        std::cout << "graphics=" << graphics.compiled_graphics_binding_hash << "\n";
        std::cout << "runtime=" << graphics.runtime_graphics_hash << "\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << "\n";
        return 2;
    }
}
