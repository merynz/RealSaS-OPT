#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <png.h>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>
#include <zlib.h>

namespace {

constexpr const char* kMagic = "RSASV2R1";

template <typename T>
T read_scalar(const std::vector<std::uint8_t>& data, std::size_t& off) {
    if (off + sizeof(T) > data.size()) throw std::runtime_error("TRUNCATED_SCALAR");
    T value{};
    std::memcpy(&value, data.data() + off, sizeof(T));
    off += sizeof(T);
    return value;
}

std::vector<std::uint8_t> read_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("FILE_OPEN_FAIL:" + path);
    f.seekg(0, std::ios::end);
    const auto n = static_cast<std::size_t>(f.tellg());
    f.seekg(0, std::ios::beg);
    std::vector<std::uint8_t> out(n);
    if (n && !f.read(reinterpret_cast<char*>(out.data()), static_cast<std::streamsize>(n)))
        throw std::runtime_error("FILE_READ_FAIL:" + path);
    return out;
}

void write_file(const std::string& path, const void* data, std::size_t size) {
    std::ofstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("FILE_WRITE_OPEN_FAIL:" + path);
    if (size && !f.write(reinterpret_cast<const char*>(data), static_cast<std::streamsize>(size)))
        throw std::runtime_error("FILE_WRITE_FAIL:" + path);
}

using Entries = std::map<std::string, std::vector<std::uint8_t>>;

Entries parse_rss(const std::string& path) {
    const auto raw = read_file(path);
    std::size_t off = 0;
    if (raw.size() < 12 || std::memcmp(raw.data(), kMagic, 8) != 0)
        throw std::runtime_error("RSS_MAGIC_INVALID");
    off += 8;
    const auto count = read_scalar<std::uint32_t>(raw, off);
    Entries entries;
    for (std::uint32_t i = 0; i < count; ++i) {
        const auto name_len = read_scalar<std::uint16_t>(raw, off);
        if (off + name_len > raw.size()) throw std::runtime_error("RSS_NAME_TRUNCATED");
        std::string name(reinterpret_cast<const char*>(raw.data() + off), name_len);
        off += name_len;
        const auto size = read_scalar<std::uint64_t>(raw, off);
        if (size > raw.size() - off) throw std::runtime_error("RSS_ENTRY_TRUNCATED");
        if (entries.count(name)) throw std::runtime_error("RSS_ENTRY_DUPLICATE");
        entries[name] = std::vector<std::uint8_t>(raw.begin() + static_cast<std::ptrdiff_t>(off),
                                                  raw.begin() + static_cast<std::ptrdiff_t>(off + size));
        off += static_cast<std::size_t>(size);
    }
    if (off != raw.size()) throw std::runtime_error("RSS_TRAILING_BYTES");
    return entries;
}

std::unordered_map<std::string, std::string> parse_manifest(const std::vector<std::uint8_t>& bytes) {
    std::string text(bytes.begin(), bytes.end());
    std::unordered_map<std::string, std::string> out;
    std::size_t start = 0;
    while (start < text.size()) {
        auto end = text.find('\n', start);
        if (end == std::string::npos) end = text.size();
        auto line = text.substr(start, end - start);
        if (!line.empty()) {
            auto eq = line.find('=');
            if (eq == std::string::npos) throw std::runtime_error("MANIFEST_LINE_INVALID");
            out[line.substr(0, eq)] = line.substr(eq + 1);
        }
        start = end + 1;
    }
    return out;
}

struct Vec2 { double x{}, y{}; };
struct Vec3 { double x{}, y{}, z{}; };
struct Proj { double x{}, y{}, z{}; };
struct Camera {
    Vec3 origin, right, up, forward;
    double half{};
    std::uint32_t resolution{};
};

Vec3 read_vec3d(const std::vector<std::uint8_t>& data, std::size_t& off) {
    return {read_scalar<double>(data, off), read_scalar<double>(data, off), read_scalar<double>(data, off)};
}

std::vector<Camera> parse_cameras(const std::vector<std::uint8_t>& data) {
    std::size_t off = 0;
    const auto n = read_scalar<std::uint32_t>(data, off);
    std::vector<Camera> out;
    out.reserve(n);
    for (std::uint32_t i = 0; i < n; ++i) {
        Camera c;
        c.origin = read_vec3d(data, off);
        c.right = read_vec3d(data, off);
        c.up = read_vec3d(data, off);
        c.forward = read_vec3d(data, off);
        c.half = read_scalar<double>(data, off);
        c.resolution = read_scalar<std::uint32_t>(data, off);
        out.push_back(c);
    }
    if (off != data.size()) throw std::runtime_error("CAMERA_BYTES_TRAILING");
    return out;
}

struct Mesh {
    std::uint32_t vertex_count{};
    std::uint32_t face_count{};
    std::vector<std::array<std::uint32_t,3>> faces;
    std::vector<std::array<Vec2,3>> uv;
};

Mesh parse_mesh(const std::vector<std::uint8_t>& data) {
    std::size_t off = 0;
    Mesh m;
    m.vertex_count = read_scalar<std::uint32_t>(data, off);
    m.face_count = read_scalar<std::uint32_t>(data, off);
    const auto vertex_bytes = static_cast<std::size_t>(m.vertex_count) * 3u * sizeof(double);
    if (off + vertex_bytes > data.size()) throw std::runtime_error("MESH_VERTEX_TRUNCATED");
    off += vertex_bytes; // runtime frames carry exact posed XYZ.
    m.faces.resize(m.face_count);
    for (std::uint32_t f = 0; f < m.face_count; ++f) {
        for (int k = 0; k < 3; ++k) m.faces[f][k] = read_scalar<std::uint32_t>(data, off);
    }
    m.uv.resize(m.face_count);
    for (std::uint32_t f = 0; f < m.face_count; ++f) {
        for (int k = 0; k < 3; ++k) {
            m.uv[f][k].x = read_scalar<double>(data, off);
            m.uv[f][k].y = read_scalar<double>(data, off);
        }
    }
    if (off != data.size()) throw std::runtime_error("MESH_BYTES_TRAILING");
    return m;
}

struct TextureSet {
    std::uint32_t views{}, pages{1}, height{}, width{};
    std::vector<std::uint8_t> rgba;
};

std::vector<std::uint8_t> decode_png_rgba(
    const std::vector<std::uint8_t>& payload,
    std::uint32_t expected_w,
    std::uint32_t expected_h
) {
    png_image image{};
    image.version = PNG_IMAGE_VERSION;
    if (!png_image_begin_read_from_memory(&image, payload.data(), payload.size()))
        throw std::runtime_error("PNG_HEADER_DECODE_FAIL");
    image.format = PNG_FORMAT_RGBA;
    if (image.width != expected_w || image.height != expected_h) {
        png_image_free(&image);
        throw std::runtime_error("PNG_DIMENSION_DRIFT");
    }
    std::vector<std::uint8_t> out(PNG_IMAGE_SIZE(image));
    if (!png_image_finish_read(&image, nullptr, out.data(), 0, nullptr)) {
        const std::string message = image.message;
        png_image_free(&image);
        throw std::runtime_error("PNG_DECODE_FAIL:" + message);
    }
    png_image_free(&image);
    return out;
}

TextureSet parse_textures(const std::vector<std::uint8_t>& data) {
    std::size_t off = 0;
    TextureSet t;
    t.views = read_scalar<std::uint32_t>(data, off);
    t.pages = 1;
    t.height = read_scalar<std::uint32_t>(data, off);
    t.width = read_scalar<std::uint32_t>(data, off);
    const auto need = static_cast<std::size_t>(t.views) * t.height * t.width * 4u;
    if (data.size() - off != need) throw std::runtime_error("TEXTURE_BYTES_INVALID");
    t.rgba.assign(data.begin() + static_cast<std::ptrdiff_t>(off), data.end());
    return t;
}

TextureSet parse_paged_textures(
    const Entries& entries,
    const std::unordered_map<std::string, std::string>& manifest
) {
    TextureSet t;
    t.views = static_cast<std::uint32_t>(std::stoul(manifest.at("view_count")));
    t.pages = static_cast<std::uint32_t>(std::stoul(manifest.at("atlas_page_count")));
    t.height = static_cast<std::uint32_t>(std::stoul(manifest.at("atlas_page_height")));
    t.width = static_cast<std::uint32_t>(std::stoul(manifest.at("atlas_page_width")));
    if (t.views != 8 || t.pages == 0 || t.height == 0 || t.width == 0)
        throw std::runtime_error("PAGED_TEXTURE_DIMENSION_INVALID");
    const auto page_bytes = static_cast<std::size_t>(t.height) * t.width * 4u;
    t.rgba.resize(static_cast<std::size_t>(t.views) * t.pages * page_bytes);
    for (std::uint32_t view = 0; view < t.views; ++view) {
        for (std::uint32_t page = 0; page < t.pages; ++page) {
            const auto key =
                "texture." + std::to_string(view) + "." + std::to_string(page) + ".entry";
            const auto entry_name = manifest.at(key);
            const auto it = entries.find(entry_name);
            if (it == entries.end()) throw std::runtime_error("PAGED_TEXTURE_ENTRY_MISSING");
            const auto decoded = decode_png_rgba(it->second, t.width, t.height);
            if (decoded.size() != page_bytes) throw std::runtime_error("PAGED_TEXTURE_DECODE_SIZE_DRIFT");
            const auto offset =
                (static_cast<std::size_t>(view) * t.pages + page) * page_bytes;
            std::copy(decoded.begin(), decoded.end(), t.rgba.begin() + static_cast<std::ptrdiff_t>(offset));
        }
    }
    return t;
}

struct ProvenanceSet {
    std::uint32_t views{}, pages{1}, height{}, width{};
    std::vector<std::uint8_t> value;
    std::vector<std::int16_t> source_view;
};

void validate_provenance(const ProvenanceSet& p) {
    const auto count =
        static_cast<std::size_t>(p.views) * p.pages * p.height * p.width;
    if (p.value.size() != count || p.source_view.size() != count)
        throw std::runtime_error("PROVENANCE_CARDINALITY_DRIFT");
    constexpr auto padding = std::numeric_limits<std::int16_t>::min();
    for (std::size_t i = 0; i < count; ++i) {
        const auto code = p.value[i];
        const auto donor = p.source_view[i];
        if (code == 0 || code == 1) {
            if (!(donor >= 0 && donor < 8))
                throw std::runtime_error("SOURCE_VIEW_IDENTITY_INVALID");
        } else if (code == 2) {
            if (donor != -2)
                throw std::runtime_error("SOURCE_VIEW_HARMONIC_DRIFT");
        } else if (code == 3) {
            if (donor != -4)
                throw std::runtime_error("SOURCE_VIEW_UNSUPPORTED_ABSTAIN_DRIFT");
        } else if (code == 4) {
            if (donor != -3)
                throw std::runtime_error("SOURCE_VIEW_CANONICAL_GLOBAL_DRIFT");
        } else if (code == 255) {
            if (donor != padding)
                throw std::runtime_error("SOURCE_VIEW_PADDING_DRIFT");
        } else {
            throw std::runtime_error("PROVENANCE_CODE_INVALID");
        }
    }
}

ProvenanceSet parse_provenance(const std::vector<std::uint8_t>& data) {
    std::size_t off = 0;
    ProvenanceSet p;
    p.views = read_scalar<std::uint32_t>(data, off);
    p.pages = 1;
    p.height = read_scalar<std::uint32_t>(data, off);
    p.width = read_scalar<std::uint32_t>(data, off);
    const auto count = static_cast<std::size_t>(p.views) * p.height * p.width;
    const auto need = count + count * sizeof(std::int16_t);
    if (data.size() - off != need) throw std::runtime_error("PROVENANCE_BYTES_INVALID");
    p.value.assign(
        data.begin() + static_cast<std::ptrdiff_t>(off),
        data.begin() + static_cast<std::ptrdiff_t>(off + count)
    );
    off += count;
    p.source_view.resize(count);
    for (auto& value : p.source_view) value = read_scalar<std::int16_t>(data, off);
    if (off != data.size()) throw std::runtime_error("PROVENANCE_BYTES_TRAILING");
    validate_provenance(p);
    return p;
}

ProvenanceSet parse_paged_provenance(const std::vector<std::uint8_t>& data) {
    std::size_t off = 0;
    ProvenanceSet p;
    p.views = read_scalar<std::uint32_t>(data, off);
    p.pages = read_scalar<std::uint32_t>(data, off);
    p.height = read_scalar<std::uint32_t>(data, off);
    p.width = read_scalar<std::uint32_t>(data, off);
    const auto uncompressed_bytes = read_scalar<std::uint64_t>(data, off);
    const auto count =
        static_cast<std::size_t>(p.views) * p.pages * p.height * p.width;
    const auto expected = count + count * sizeof(std::int16_t);
    if (uncompressed_bytes != expected) throw std::runtime_error("PAGED_PROVENANCE_SIZE_HEADER_DRIFT");
    std::vector<std::uint8_t> raw(expected);
    uLongf dest_len = static_cast<uLongf>(raw.size());
    const auto* source = reinterpret_cast<const Bytef*>(data.data() + off);
    const auto source_len = static_cast<uLong>(data.size() - off);
    const int rc = uncompress(
        reinterpret_cast<Bytef*>(raw.data()),
        &dest_len,
        source,
        source_len
    );
    if (rc != Z_OK || dest_len != raw.size())
        throw std::runtime_error("PAGED_PROVENANCE_ZLIB_DECODE_FAIL");
    p.value.assign(raw.begin(), raw.begin() + static_cast<std::ptrdiff_t>(count));
    p.source_view.resize(count);
    std::memcpy(
        p.source_view.data(),
        raw.data() + count,
        count * sizeof(std::int16_t)
    );
    validate_provenance(p);
    return p;
}

std::vector<std::uint32_t> parse_face_pages(
    const std::vector<std::uint8_t>& data,
    std::uint32_t expected_faces,
    std::uint32_t page_count
) {
    std::size_t off = 0;
    const auto count = read_scalar<std::uint32_t>(data, off);
    if (count != expected_faces) throw std::runtime_error("FACE_PAGE_COUNT_DRIFT");
    std::vector<std::uint32_t> pages(count);
    for (auto& page : pages) {
        page = read_scalar<std::uint32_t>(data, off);
        if (page >= page_count) throw std::runtime_error("FACE_PAGE_OUT_OF_RANGE");
    }
    if (off != data.size()) throw std::runtime_error("FACE_PAGE_BYTES_TRAILING");
    return pages;
}

struct Clip {
    std::uint32_t frame_count{}, vertex_count{};
    std::vector<double> times;
    std::vector<double> positions;
};

Clip parse_clip(const std::vector<std::uint8_t>& data) {
    std::size_t off = 0;
    Clip c;
    c.frame_count = read_scalar<std::uint32_t>(data, off);
    c.vertex_count = read_scalar<std::uint32_t>(data, off);
    c.times.resize(c.frame_count);
    for (auto& t : c.times) t = read_scalar<double>(data, off);
    const auto n = static_cast<std::size_t>(c.frame_count) * c.vertex_count * 3u;
    c.positions.resize(n);
    for (auto& v : c.positions) v = read_scalar<double>(data, off);
    if (off != data.size()) throw std::runtime_error("CLIP_BYTES_TRAILING");
    return c;
}

Proj project(const Vec3& p, const Camera& c) {
    const Vec3 d{p.x-c.origin.x, p.y-c.origin.y, p.z-c.origin.z};
    const auto dot = [](const Vec3& a, const Vec3& b){ return a.x*b.x+a.y*b.y+a.z*b.z; };
    const double gx = dot(d, c.right) / c.half;
    const double gy = -dot(d, c.up) / c.half;
    return {
        (gx + 1.0) * 0.5 * static_cast<double>(c.resolution),
        (gy + 1.0) * 0.5 * static_cast<double>(c.resolution),
        dot(d, c.forward)
    };
}

double orient(const Proj& a, const Proj& b, double px, double py) {
    return (b.x-a.x)*(py-a.y) - (b.y-a.y)*(px-a.x);
}
bool top_left(const Proj& a, const Proj& b) {
    const double dx=b.x-a.x, dy=b.y-a.y;
    return dy < 0.0 || (dy == 0.0 && dx > 0.0);
}
bool edge_accept(double e, bool tl) {
    constexpr double eps=1e-12;
    if (e > eps) return true;
    if (e < -eps) return false;
    return tl;
}
bool covers(const Proj& a, const Proj& b, const Proj& c, int x, int y) {
    const double area=orient(a,b,c.x,c.y);
    if (std::abs(area)<=1e-12) return false;
    const double px=static_cast<double>(x)+0.5, py=static_cast<double>(y)+0.5;
    const bool positive=area>0.0;
    const double sign=positive?1.0:-1.0;
    const double e0=sign*orient(b,c,px,py);
    const double e1=sign*orient(c,a,px,py);
    const double e2=sign*orient(a,b,px,py);
    const bool tl0=positive?top_left(b,c):top_left(c,b);
    const bool tl1=positive?top_left(c,a):top_left(a,c);
    const bool tl2=positive?top_left(a,b):top_left(b,a);
    return edge_accept(e0,tl0)&&edge_accept(e1,tl1)&&edge_accept(e2,tl2);
}

struct PM { double r{},g{},b{},a{}; };

double srgb_to_linear(double x) {
    x=std::max(0.0,std::min(1.0,x));
    if(x<=0.04045) return x/12.92;
    return std::pow((x+0.055)/1.055,2.4);
}
double linear_to_srgb(double x) {
    x=std::max(0.0,std::min(1.0,x));
    if(x<=0.0031308) return 12.92*x;
    return 1.055*std::pow(x,1.0/2.4)-0.055;
}

PM texel_pm(const TextureSet& t, std::uint32_t view, std::uint32_t page, int x, int y) {
    x=std::max(0,std::min(x,static_cast<int>(t.width)-1));
    y=std::max(0,std::min(y,static_cast<int>(t.height)-1));
    const auto idx=(((((static_cast<std::size_t>(view)*t.pages)+page)*t.height+static_cast<std::size_t>(y))*t.width)+static_cast<std::size_t>(x))*4u;
    const double a=static_cast<double>(t.rgba[idx+3])/255.0;
    return {srgb_to_linear(static_cast<double>(t.rgba[idx])/255.0)*a,
            srgb_to_linear(static_cast<double>(t.rgba[idx+1])/255.0)*a,
            srgb_to_linear(static_cast<double>(t.rgba[idx+2])/255.0)*a,a};
}
PM mix(const PM& a,const PM& b,double t) {
    return {a.r+(b.r-a.r)*t,a.g+(b.g-a.g)*t,a.b+(b.b-a.b)*t,a.a+(b.a-a.a)*t};
}
PM sample_pm(const TextureSet& t,std::uint32_t view,std::uint32_t page,double u,double v) {
    u=std::max(0.0,std::min(1.0,u)); v=std::max(0.0,std::min(1.0,v));
    const double x=u*static_cast<double>(t.width-1), y=v*static_cast<double>(t.height-1);
    const int x0=static_cast<int>(std::floor(x)), y0=static_cast<int>(std::floor(y));
    const int x1=std::min(x0+1,static_cast<int>(t.width)-1), y1=std::min(y0+1,static_cast<int>(t.height)-1);
    const double tx=x-x0, ty=y-y0;
    return mix(mix(texel_pm(t,view,page,x0,y0),texel_pm(t,view,page,x1,y0),tx),
               mix(texel_pm(t,view,page,x0,y1),texel_pm(t,view,page,x1,y1),tx),ty);
}
std::uint8_t provenance_texel(const ProvenanceSet& p,std::uint32_t view,std::uint32_t page,int x,int y) {
    x=std::max(0,std::min(x,static_cast<int>(p.width)-1));
    y=std::max(0,std::min(y,static_cast<int>(p.height)-1));
    const auto idx=(((static_cast<std::size_t>(view)*p.pages)+page)*p.height+static_cast<std::size_t>(y))*p.width+static_cast<std::size_t>(x);
    return p.value[idx];
}
std::uint8_t sample_provenance(const ProvenanceSet& p,std::uint32_t view,std::uint32_t page,double u,double v) {
    u=std::max(0.0,std::min(1.0,u)); v=std::max(0.0,std::min(1.0,v));
    const double x=u*static_cast<double>(p.width-1), y=v*static_cast<double>(p.height-1);
    const int x0=static_cast<int>(std::floor(x)), y0=static_cast<int>(std::floor(y));
    const int x1=std::min(x0+1,static_cast<int>(p.width)-1), y1=std::min(y0+1,static_cast<int>(p.height)-1);
    const double tx=x-x0, ty=y-y0;
    const std::array<double,4> w{
        (1.0-tx)*(1.0-ty), tx*(1.0-ty), (1.0-tx)*ty, tx*ty
    };
    const std::array<std::uint8_t,4> v4{
        provenance_texel(p,view,page,x0,y0),
        provenance_texel(p,view,page,x1,y0),
        provenance_texel(p,view,page,x0,y1),
        provenance_texel(p,view,page,x1,y1)
    };
    int risk=-1;
    for(std::size_t i=0;i<4;++i) if(w[i]>1e-12) risk=std::max(risk,static_cast<int>(v4[i]));
    if(risk<0) throw std::runtime_error("PROVENANCE_BILINEAR_FOOTPRINT_EMPTY");
    return static_cast<std::uint8_t>(risk);
}
std::int16_t source_view_texel(const ProvenanceSet& p,std::uint32_t view,std::uint32_t page,int x,int y) {
    x=std::max(0,std::min(x,static_cast<int>(p.width)-1));
    y=std::max(0,std::min(y,static_cast<int>(p.height)-1));
    const auto idx=(((static_cast<std::size_t>(view)*p.pages)+page)*p.height+static_cast<std::size_t>(y))*p.width+static_cast<std::size_t>(x);
    return p.source_view[idx];
}
std::int16_t sample_source_view(const ProvenanceSet& p,std::uint32_t view,std::uint32_t page,double u,double v) {
    u=std::max(0.0,std::min(1.0,u)); v=std::max(0.0,std::min(1.0,v));
    const double x=u*static_cast<double>(p.width-1), y=v*static_cast<double>(p.height-1);
    const int x0=static_cast<int>(std::floor(x)), y0=static_cast<int>(std::floor(y));
    const int x1=std::min(x0+1,static_cast<int>(p.width)-1), y1=std::min(y0+1,static_cast<int>(p.height)-1);
    const double tx=x-x0, ty=y-y0;
    const std::array<double,4> w{
        (1.0-tx)*(1.0-ty), tx*(1.0-ty), (1.0-tx)*ty, tx*ty
    };
    const std::array<std::int16_t,4> v4{
        source_view_texel(p,view,page,x0,y0),
        source_view_texel(p,view,page,x1,y0),
        source_view_texel(p,view,page,x0,y1),
        source_view_texel(p,view,page,x1,y1)
    };
    constexpr std::int16_t kUnset=std::numeric_limits<std::int16_t>::min();
    constexpr std::int16_t kMixed=-3;
    std::int16_t result=kUnset;
    for(std::size_t i=0;i<4;++i) {
        if(w[i]<=1e-12) continue;
        if(v4[i]==kUnset) throw std::runtime_error("SOURCE_VIEW_PADDING_SAMPLED");
        if(result==kUnset) result=v4[i];
        else if(result!=v4[i]) result=kMixed;
    }
    if(result==kUnset) throw std::runtime_error("SOURCE_VIEW_BILINEAR_FOOTPRINT_EMPTY");
    return result;
}
std::int16_t merge_source_view_diagnostic(std::int16_t current,std::int16_t next) {
    constexpr std::int16_t kUnset=std::numeric_limits<std::int16_t>::min();
    constexpr std::int16_t kMixed=-3;
    if(next==kUnset) return current;
    if(current==kUnset) return next;
    if(current==next) return current;
    return kMixed;
}

std::uint8_t q8(double x) {
    x=std::max(0.0,std::min(1.0,x));
    return static_cast<std::uint8_t>(std::max(0.0,std::min(255.0,std::floor(x*255.0+0.5))));
}

std::string arg_value(int argc,char** argv,const std::string& key,bool required=true) {
    for(int i=2;i+1<argc;++i) if(argv[i]==key) return argv[i+1];
    if(required) throw std::runtime_error("ARG_MISSING:"+key);
    return {};
}

bool has_flag(int argc,char** argv,const std::string& key) {
    for(int i=2;i<argc;++i) if(argv[i]==key) return true;
    return false;
}

int parse_int_exact(const std::string& raw,const std::string& label) {
    if(raw.empty()) throw std::runtime_error(label+"_INTEGER_EMPTY");
    std::size_t consumed=0;
    int value=0;
    try {
        value=std::stoi(raw,&consumed,10);
    } catch(const std::exception&) {
        throw std::runtime_error(label+"_INTEGER_INVALID");
    }
    if(consumed!=raw.size()) throw std::runtime_error(label+"_INTEGER_INVALID");
    return value;
}


struct VisualMesh2D {
    std::uint32_t source_width{}, source_height{}, vertex_count{}, face_count{};
    std::vector<Vec2> uv;
    std::vector<std::array<std::uint32_t,3>> faces;
};

VisualMesh2D parse_visual_mesh_2d(const std::vector<std::uint8_t>& data) {
    if (data.size() < 8 || std::memcmp(data.data(), "RSVM1\0\0\0", 8) != 0)
        throw std::runtime_error("VISUAL_MESH_MAGIC_INVALID");
    std::size_t off=8;
    VisualMesh2D mesh;
    mesh.source_width=read_scalar<std::uint32_t>(data,off);
    mesh.source_height=read_scalar<std::uint32_t>(data,off);
    mesh.vertex_count=read_scalar<std::uint32_t>(data,off);
    mesh.face_count=read_scalar<std::uint32_t>(data,off);
    if(mesh.source_width<1||mesh.source_height<1||mesh.vertex_count<3||mesh.face_count<1)
        throw std::runtime_error("VISUAL_MESH_HEADER_INVALID");
    mesh.uv.resize(mesh.vertex_count);
    for(auto& p:mesh.uv) {
        p.x=read_scalar<double>(data,off);
        p.y=read_scalar<double>(data,off);
        if(!std::isfinite(p.x)||!std::isfinite(p.y))
            throw std::runtime_error("VISUAL_MESH_UV_NONFINITE");
    }
    mesh.faces.resize(mesh.face_count);
    for(auto& face:mesh.faces) {
        for(auto& index:face) {
            index=read_scalar<std::uint32_t>(data,off);
            if(index>=mesh.vertex_count)
                throw std::runtime_error("VISUAL_MESH_FACE_INDEX_INVALID");
        }
    }
    if(off!=data.size()) throw std::runtime_error("VISUAL_MESH_BYTES_TRAILING");
    return mesh;
}

std::vector<Vec2> parse_visual_positions_2d(
    const std::vector<std::uint8_t>& data,
    std::uint32_t expected_vertices,
    int expected_frames, int frame_index
) {
    if(data.size()<8||std::memcmp(data.data(),"RSVP1\0\0\0",8)!=0)
        throw std::runtime_error("VISUAL_POSITIONS_MAGIC_INVALID");
    std::size_t off=8;
    const auto frame_count=read_scalar<std::uint32_t>(data,off);
    const auto vertex_count=read_scalar<std::uint32_t>(data,off);
    if(vertex_count!=expected_vertices||frame_count!=static_cast<std::uint32_t>(expected_frames)||frame_index<0||
       static_cast<std::uint32_t>(frame_index)>=frame_count)
        throw std::runtime_error("VISUAL_POSITIONS_HEADER_INVALID");
    const auto frame_stride=
        static_cast<std::size_t>(vertex_count)*2u*sizeof(double);
    if(data.size()!=16+static_cast<std::size_t>(frame_count)*frame_stride)
        throw std::runtime_error("VISUAL_POSITIONS_BYTES_INVALID");
    const auto target=off+static_cast<std::size_t>(frame_index)*frame_stride;
    if(target+frame_stride>data.size())
        throw std::runtime_error("VISUAL_POSITIONS_TRUNCATED");
    off=target;
    std::vector<Vec2> out(vertex_count);
    for(auto& p:out) {
        p.x=read_scalar<double>(data,off);
        p.y=read_scalar<double>(data,off);
        if(!std::isfinite(p.x)||!std::isfinite(p.y))
            throw std::runtime_error("VISUAL_POSITION_NONFINITE");
    }
    return out;
}

std::vector<double> parse_visual_depth(
    const std::vector<std::uint8_t>& data, std::uint32_t expected_vertices,
    int expected_frames, int frame_index
) {
    if(data.size()<16||std::memcmp(data.data(),"RSVD1\0\0\0",8)!=0)
        throw std::runtime_error("VISUAL_CANONICAL_DEPTH_MAGIC_INVALID");
    std::size_t off=8;
    const auto frames=read_scalar<std::uint32_t>(data,off);
    const auto vertices=read_scalar<std::uint32_t>(data,off);
    const auto stride=static_cast<std::size_t>(vertices)*sizeof(double);
    if(vertices!=expected_vertices||frames!=static_cast<std::uint32_t>(expected_frames)||
       data.size()!=16+static_cast<std::size_t>(frames)*stride||frame_index<0||
       static_cast<std::uint32_t>(frame_index)>=frames)
        throw std::runtime_error("VISUAL_CANONICAL_DEPTH_HEADER_OR_BYTES_INVALID");
    off+=static_cast<std::size_t>(frame_index)*stride;
    std::vector<double> out(vertices);
    for(auto& z:out) {
        z=read_scalar<double>(data,off);
        if(!std::isfinite(z)||z<=0) throw std::runtime_error("VISUAL_CANONICAL_DEPTH_NONFINITE_OR_BEHIND_CAMERA");
    }
    return out;
}

struct StraightRGBA { double r{},g{},b{},a{}; };

StraightRGBA visual_texel(
    const std::vector<std::uint8_t>& rgba,
    std::uint32_t width,
    std::uint32_t height,
    int x,
    int y
) {
    x=std::max(0,std::min(x,static_cast<int>(width)-1));
    y=std::max(0,std::min(y,static_cast<int>(height)-1));
    const auto idx=(static_cast<std::size_t>(y)*width+static_cast<std::size_t>(x))*4u;
    return {
        static_cast<double>(rgba[idx])/255.0,
        static_cast<double>(rgba[idx+1])/255.0,
        static_cast<double>(rgba[idx+2])/255.0,
        static_cast<double>(rgba[idx+3])/255.0
    };
}

StraightRGBA visual_sample_bilinear(
    const std::vector<std::uint8_t>& rgba,
    std::uint32_t width,
    std::uint32_t height,
    double u,
    double v
) {
    u=std::clamp(u,0.0,1.0);
    v=std::clamp(v,0.0,1.0);
    const double x=u*static_cast<double>(width-1);
    const double y=v*static_cast<double>(height-1);
    const int x0=static_cast<int>(std::floor(x));
    const int y0=static_cast<int>(std::floor(y));
    const int x1=std::min(x0+1,static_cast<int>(width)-1);
    const int y1=std::min(y0+1,static_cast<int>(height)-1);
    const double tx=x-x0,ty=y-y0;
    const auto p00=visual_texel(rgba,width,height,x0,y0);
    const auto p10=visual_texel(rgba,width,height,x1,y0);
    const auto p01=visual_texel(rgba,width,height,x0,y1);
    const auto p11=visual_texel(rgba,width,height,x1,y1);
    auto mix1=[](double a,double b,double t){return a*(1.0-t)+b*t;};
    StraightRGBA out;
    out.r=mix1(mix1(p00.r,p10.r,tx),mix1(p01.r,p11.r,tx),ty);
    out.g=mix1(mix1(p00.g,p10.g,tx),mix1(p01.g,p11.g,tx),ty);
    out.b=mix1(mix1(p00.b,p10.b,tx),mix1(p01.b,p11.b,tx),ty);
    out.a=mix1(mix1(p00.a,p10.a,tx),mix1(p01.a,p11.a,tx),ty);
    return out;
}

double visual_orient(const Vec2& a,const Vec2& b,double x,double y) {
    return (b.x-a.x)*(y-a.y)-(b.y-a.y)*(x-a.x);
}

bool visual_top_left(const Vec2& a,const Vec2& b) {
    const double dy=b.y-a.y,dx=b.x-a.x;
    return dy<0.0||(std::abs(dy)<=1e-12&&dx>0.0);
}

int render_source_owned_visual(
    const Entries& entries,
    const std::unordered_map<std::string,std::string>& manifest,
    const std::string& clip_id,
    const std::string& view_id,
    int frame_index,
    const std::string& rgba_path,
    const std::string& prov_path,
    const std::string& source_view_path,
    const std::string& owner_path
) {
    if(manifest.at("presentation_geometry_mode")!="SOURCE_OWNED_VISUAL_PRESENTATION_V1")
        throw std::runtime_error("VISUAL_PRESENTATION_MODE_INVALID");
    if(manifest.at("mechanical_mesh_render_authority")!="0")
        throw std::runtime_error("VISUAL_MECHANICAL_RENDER_AUTHORITY_FORBIDDEN");
    if(manifest.at("runtime_visual_mesh_rebuild")!="0"||
       manifest.at("runtime_binding_solve")!="0"||
       manifest.at("runtime_generation")!="0"||
       manifest.at("donor_search_at_runtime")!="0")
        throw std::runtime_error("VISUAL_RUNTIME_REBUILD_OR_GENERATION_FORBIDDEN");
    const bool material=manifest.find("visual_material_contract")!=manifest.end();
    if(manifest.at("texture_sampling_contract")!=(material?
       "CAA_RGBA8_LINEAR_PM_BILINEAR_VISUAL_V1":"SOURCE_RGBA8_BILINEAR_STRAIGHT_TO_PM_V1"))
        throw std::runtime_error("VISUAL_TEXTURE_SAMPLING_CONTRACT_INVALID");
    if(material && (manifest.at("visual_material_contract")!="CAA_VISUAL_TEXEL_PROVENANCE_AND_SOURCE_VIEW_V1" ||
       manifest.at("depth_ownership_contract")!="CANONICAL_CAMERA_DEPTH_ASCENDING__UNRESOLVED_TIES_FAIL_V1" ||
       std::stod(manifest.at("depth_tie_epsilon"))!=1e-12 ||
       manifest.at("maximum_fragment_layers")!="4"))
        throw std::runtime_error("VISUAL_MATERIAL_OR_DEPTH_CONTRACT_INVALID");
    if(manifest.at("mip_generation_authorized")!="0")
        throw std::runtime_error("MIP_GENERATION_MUST_BE_FORBIDDEN");
    if(view_id.size()!=2||view_id[0]!='V')
        throw std::runtime_error("VIEW_ID_INVALID");
    const int view=parse_int_exact(view_id.substr(1),"VIEW");
    if(view<0||view>=8) throw std::runtime_error("VIEW_INDEX_INVALID");
    if(manifest.at("view."+std::to_string(view)+".id")!=view_id)
        throw std::runtime_error("VISUAL_VIEW_ID_DRIFT");

    const int clip_count=std::stoi(manifest.at("clip_count"));
    int clip_index=-1;
    for(int i=0;i<clip_count;++i) {
        if(manifest.at("clip."+std::to_string(i)+".id")==clip_id) {
            clip_index=i; break;
        }
    }
    if(clip_index<0) throw std::runtime_error("CLIP_NOT_FOUND");
    const auto frame_count=std::stoi(
        manifest.at("clip."+std::to_string(clip_index)+".frame_count")
    );
    if(frame_index<0||frame_index>=frame_count)
        throw std::runtime_error("FRAME_INDEX_INVALID");

    const auto mesh_entry=manifest.at(
        "view."+std::to_string(view)+".mesh_entry"
    );
    const auto texture_entry=manifest.at(
        "view."+std::to_string(view)+".texture_entry"
    );
    const auto positions_entry=manifest.at(
        "clip."+std::to_string(clip_index)+".view."+
        std::to_string(view)+".positions_entry"
    );
    const auto mesh=parse_visual_mesh_2d(entries.at(mesh_entry));
    auto positions=parse_visual_positions_2d(
        entries.at(positions_entry),mesh.vertex_count,frame_count,frame_index
    );
    const auto texture=decode_png_rgba(
        entries.at(texture_entry),mesh.source_width,mesh.source_height
    );
    std::vector<double> depths;
    std::vector<std::uint8_t> material_codes;
    std::vector<std::int16_t> material_donors;
    TextureSet linear_texture;
    if(material) {
        const auto prefix="view."+std::to_string(view)+".";
        const auto& bytes=entries.at(manifest.at(prefix+"material_entry"));
        if(bytes.size()<16 || std::memcmp(bytes.data(),"RSVA1\0\0\0",8)!=0)
            throw std::runtime_error("VISUAL_MATERIAL_MAGIC_INVALID");
        std::size_t offset=8;
        const auto width=read_scalar<std::uint32_t>(bytes,offset);
        const auto height=read_scalar<std::uint32_t>(bytes,offset);
        const auto count=static_cast<std::size_t>(width)*height;
        if(width!=mesh.source_width || height!=mesh.source_height || bytes.size()!=16+count*3)
            throw std::runtime_error("VISUAL_MATERIAL_SIZE_INVALID");
        material_codes.assign(bytes.begin()+16,bytes.begin()+16+count);
        offset=16+count;
        for(std::size_t i=0;i<count;++i) {
            const auto donor=read_scalar<std::int16_t>(bytes,offset);
            const auto code=material_codes[i];
            const bool valid=(code==0&&donor==view)||(code==1&&donor>=0&&donor<8&&donor!=view)||
                (code==2&&donor==-2)||(code==3&&donor==-4)||(code==4&&donor==-3)||
                (code==255&&donor==std::numeric_limits<std::int16_t>::min());
            if(!valid) throw std::runtime_error("VISUAL_MATERIAL_PROVENANCE_SOURCE_VIEW_DRIFT");
            if((code==3||code==255) && (texture[4*i]||texture[4*i+1]||texture[4*i+2]||texture[4*i+3]))
                throw std::runtime_error("VISUAL_MATERIAL_UNSUPPORTED_OR_PADDING_RGBA");
            material_donors.push_back(donor);
        }
        depths=parse_visual_depth(entries.at(manifest.at("clip."+std::to_string(clip_index)+
            ".view."+std::to_string(view)+".depths_entry")),mesh.vertex_count,frame_count,frame_index);
        linear_texture.views=1; linear_texture.pages=1;
        linear_texture.width=width; linear_texture.height=height; linear_texture.rgba=texture;
    }
    const int resolution=std::stoi(
        manifest.at("view."+std::to_string(view)+".resolution")
    );
    if(resolution<=0) throw std::runtime_error("VISUAL_OUTPUT_RESOLUTION_INVALID");

    const double sx=static_cast<double>(resolution)/
        static_cast<double>(mesh.source_width);
    const double sy=static_cast<double>(resolution)/
        static_cast<double>(mesh.source_height);
    for(auto& p:positions) {
        p.x=(p.x+0.5)*sx;
        p.y=(p.y+0.5)*sy;
    }

    struct Accum { double r{},g{},b{},a{}; };
    const auto pixels=static_cast<std::size_t>(resolution)*resolution;
    std::vector<Accum> accum(pixels);
    std::vector<std::int32_t> owner(pixels,-1);
    struct Fragment { double depth; PM color; std::int32_t owner; std::uint8_t code; std::int16_t donor; };
    std::vector<std::vector<Fragment>> fragments(material?pixels:0);
    std::vector<std::uint8_t> compiled_provenance(pixels,255);
    std::vector<std::int16_t> compiled_donor(pixels,std::numeric_limits<std::int16_t>::min());
    std::uint64_t covered_samples=0;

    for(std::size_t fi=0;fi<mesh.faces.size();++fi) {
        const auto face=mesh.faces[fi];
        const auto& a=positions[face[0]];
        const auto& b=positions[face[1]];
        const auto& c=positions[face[2]];
        const double area=visual_orient(a,b,c.x,c.y);
        if(std::abs(area)<=1e-12) continue;
        const double sign=area>0.0?1.0:-1.0;
        const bool positive=area>0.0;
        const int minx=std::max(
            0,static_cast<int>(std::floor(std::min({a.x,b.x,c.x})-0.5))
        );
        const int maxx=std::min(
            resolution-1,
            static_cast<int>(std::ceil(std::max({a.x,b.x,c.x})-0.5))
        );
        const int miny=std::max(
            0,static_cast<int>(std::floor(std::min({a.y,b.y,c.y})-0.5))
        );
        const int maxy=std::min(
            resolution-1,
            static_cast<int>(std::ceil(std::max({a.y,b.y,c.y})-0.5))
        );
        if(minx>maxx||miny>maxy) continue;
        const bool tl0=positive?visual_top_left(b,c):visual_top_left(c,b);
        const bool tl1=positive?visual_top_left(c,a):visual_top_left(a,c);
        const bool tl2=positive?visual_top_left(a,b):visual_top_left(b,a);
        for(int y=miny;y<=maxy;++y) for(int x=minx;x<=maxx;++x) {
            const double px=x+0.5,py=y+0.5;
            const double q0=sign*visual_orient(b,c,px,py);
            const double q1=sign*visual_orient(c,a,px,py);
            const double q2=sign*visual_orient(a,b,px,py);
            auto accept=[](double q,bool tl){
                return q>1e-12||(std::abs(q)<=1e-12&&tl);
            };
            if(!accept(q0,tl0)||!accept(q1,tl1)||!accept(q2,tl2)) continue;
            const double w0=visual_orient(b,c,px,py)/area;
            const double w1=visual_orient(c,a,px,py)/area;
            const double w2=1.0-w0-w1;
            const auto& ua=mesh.uv[face[0]];
            const auto& ub=mesh.uv[face[1]];
            const auto& uc=mesh.uv[face[2]];
            const double u=w0*ua.x+w1*ub.x+w2*uc.x;
            const double v=w0*ua.y+w1*ub.y+w2*uc.y;
            if(material) {
                const double tx=std::clamp(u,0.0,1.0)*(mesh.source_width-1);
                const double ty=std::clamp(v,0.0,1.0)*(mesh.source_height-1);
                const int x0=static_cast<int>(std::floor(tx)),y0=static_cast<int>(std::floor(ty));
                const int x1=std::min(x0+1,static_cast<int>(mesh.source_width)-1);
                const int y1=std::min(y0+1,static_cast<int>(mesh.source_height)-1);
                const double fx=tx-x0,fy=ty-y0;
                const std::array<int,4> xs{x0,x1,x0,x1},ys{y0,y0,y1,y1};
                const std::array<double,4> weights{(1-fx)*(1-fy),fx*(1-fy),(1-fx)*fy,fx*fy};
                std::uint8_t code=255;
                std::int16_t donor=std::numeric_limits<std::int16_t>::min();
                bool unsupported=false;
                for(std::size_t k=0;k<4;++k) {
                    const auto index=static_cast<std::size_t>(ys[k])*mesh.source_width+xs[k];
                    if(weights[k]<=1e-12) continue;
                    const auto candidate=material_codes[index];
                    unsupported=unsupported||candidate==3;
                    if(candidate!=255 && (code==255||candidate>code)) {
                        code=candidate; donor=material_donors[index];
                    }
                }
                if(unsupported) { code=3; donor=-4; }
                const auto sample=sample_pm(linear_texture,0,0,u,v);
                if(sample.a>1e-12||code==3) {
                    auto& rows=fragments[static_cast<std::size_t>(y)*resolution+x];
                    if(rows.size()>=4) throw std::runtime_error("VISUAL_DEPTH_FRAGMENT_OVERFLOW");
                    rows.push_back({w0*depths[face[0]]+w1*depths[face[1]]+w2*depths[face[2]],
                        sample,static_cast<std::int32_t>(fi),code,donor});
                }
                ++covered_samples;
                continue;
            }
            const auto sample=visual_sample_bilinear(
                texture,mesh.source_width,mesh.source_height,u,v
            );
            const double alpha=std::clamp(sample.a,0.0,1.0);
            auto& dst=accum[
                static_cast<std::size_t>(y)*resolution+
                static_cast<std::size_t>(x)
            ];
            const double transmission=1.0-alpha;
            dst.r=sample.r*alpha+dst.r*transmission;
            dst.g=sample.g*alpha+dst.g*transmission;
            dst.b=sample.b*alpha+dst.b*transmission;
            dst.a=alpha+dst.a*transmission;
            if(alpha>1e-12) owner[
                static_cast<std::size_t>(y)*resolution+
                static_cast<std::size_t>(x)
            ]=static_cast<std::int32_t>(fi);
            ++covered_samples;
        }
    }

    if(material) for(std::size_t i=0;i<pixels;++i) {
        auto& rows=fragments[i];
        std::sort(rows.begin(),rows.end(),[](const Fragment& a,const Fragment& b){ return a.depth<b.depth; });
        for(std::size_t k=1;k<rows.size();++k)
            if(std::abs(rows[k].depth-rows[k-1].depth)<=1e-12)
                throw std::runtime_error("VISUAL_DEPTH_UNRESOLVED_TIE");
        auto& dst=accum[i];
        if(!rows.empty()) owner[i]=rows.front().owner;
        for(const auto& row:rows) {
            const double transmission=1.0-dst.a;
            if(row.code==3 && transmission>1e-12)
                throw std::runtime_error("VISUAL_MATERIAL_UNSUPPORTED_FOOTPRINT");
            if(row.color.a*transmission>1e-12 &&
               (compiled_provenance[i]==255||row.code>compiled_provenance[i])) {
                compiled_provenance[i]=row.code; compiled_donor[i]=row.donor;
            }
            dst.r+=row.color.r*transmission; dst.g+=row.color.g*transmission;
            dst.b+=row.color.b*transmission; dst.a+=row.color.a*transmission;
        }
    }
    std::vector<std::uint8_t> rgba(pixels*4u,0);
    std::vector<std::uint8_t> provenance(pixels,255);
    std::vector<std::int16_t> source_view(
        pixels,std::numeric_limits<std::int16_t>::min()
    );
    for(std::size_t i=0;i<pixels;++i) {
        const auto& p=accum[i];
        const double alpha=std::clamp(p.a,0.0,1.0);
        const double inv=alpha>1e-12?1.0/alpha:0.0;
        rgba[i*4u]=q8(material?linear_to_srgb(p.r*inv):p.r*inv);
        rgba[i*4u+1]=q8(material?linear_to_srgb(p.g*inv):p.g*inv);
        rgba[i*4u+2]=q8(material?linear_to_srgb(p.b*inv):p.b*inv);
        rgba[i*4u+3]=q8(alpha);
        if(material ? rgba[i*4u+3]>0 : alpha>1e-12) {
            provenance[i]=material?compiled_provenance[i]:0;
            source_view[i]=material?compiled_donor[i]:static_cast<std::int16_t>(view);
        } else if(material) owner[i]=-1;
    }
    write_file(rgba_path,rgba.data(),rgba.size());
    if(!prov_path.empty())
        write_file(prov_path,provenance.data(),provenance.size());
    if(!source_view_path.empty())
        write_file(
            source_view_path,source_view.data(),
            source_view.size()*sizeof(std::int16_t)
        );
    if(!owner_path.empty())
        write_file(owner_path,owner.data(),owner.size()*sizeof(std::int32_t));

    std::cout<<"renderer=REALSAS_V2_SOURCE_OWNED_VISUAL_2D"
             <<" clip="<<clip_id<<" view="<<view_id
             <<" frame="<<frame_index<<" resolution="<<resolution
             <<" faces="<<mesh.face_count<<" vertices="<<mesh.vertex_count
             <<" covered_samples="<<covered_samples
             <<" sealed_visual_material="<<(material?1:0)
             <<" appearance=SOURCE_RGBA8_FIXED_UV\\n";
    return 0;
}

} // namespace

int main(int argc,char** argv) {
    try {
        if(argc<2) throw std::runtime_error("USAGE: player package --clip ID --view V0 --frame N --out-rgba PATH [--out-provenance PATH] [--out-source-view PATH] [--out-owner PATH]");
        const std::string package_path=argv[1];
        const std::string clip_id=arg_value(argc,argv,"--clip");
        const std::string view_id=arg_value(argc,argv,"--view");
        const int frame_index=parse_int_exact(arg_value(argc,argv,"--frame"),"FRAME");
        const std::string rgba_path=arg_value(argc,argv,"--out-rgba");
        const std::string prov_path=arg_value(argc,argv,"--out-provenance",false);
        const std::string source_view_path=arg_value(argc,argv,"--out-source-view",false);
        const std::string owner_path=arg_value(argc,argv,"--out-owner",false);
        const bool allow_layer_overflow_diagnostic=
            has_flag(argc,argv,"--allow-layer-overflow-diagnostic");

        const auto entries=parse_rss(package_path);
        const auto manifest=parse_manifest(entries.at("manifest.txt"));
        if(manifest.at("playback_sampling_contract")!="SEALED_FRAME_INDEX_ONLY")
            throw std::runtime_error("PLAYBACK_SAMPLING_CONTRACT_INVALID");
        if(manifest.at("view_selection_contract")!="SEALED_DISCRETE_DIRECTION_INDEX_ONLY")
            throw std::runtime_error("VIEW_SELECTION_CONTRACT_INVALID");
        if(manifest.at("cross_direction_blending_authorized")!="0")
            throw std::runtime_error("CROSS_DIRECTION_BLENDING_MUST_BE_FORBIDDEN");
        if(manifest.at("host_interpolation_authorized")!="0")
            throw std::runtime_error("HOST_INTERPOLATION_MUST_BE_FORBIDDEN");
        if(manifest.at("presentation_state_execution_authorized")!="0")
            throw std::runtime_error("PRESENTATION_STATE_EXECUTION_MUST_BE_FORBIDDEN");
        if(manifest.at("clipping_authorized")!="0")
            throw std::runtime_error("RUNTIME_CLIPPING_MUST_BE_FORBIDDEN");
        if(manifest.at("tint_order_visibility_authorized")!="0")
            throw std::runtime_error("RUNTIME_TINT_ORDER_VISIBILITY_MUST_BE_FORBIDDEN");
        const auto presentation_mode_it=manifest.find("presentation_geometry_mode");
        const auto presentation_mode=(
            presentation_mode_it==manifest.end()
            ? std::string("MECHANICAL_CANONICAL_DEPTH_V2")
            : presentation_mode_it->second
        );
        if(presentation_mode=="SOURCE_OWNED_VISUAL_PRESENTATION_V1") {
            return render_source_owned_visual(
                entries,
                manifest,
                clip_id,
                view_id,
                frame_index,
                rgba_path,
                prov_path,
                source_view_path,
                owner_path
            );
        }
        if(presentation_mode!="MECHANICAL_CANONICAL_DEPTH_V2")
            throw std::runtime_error("PRESENTATION_GEOMETRY_MODE_INVALID");
        if(manifest.at("texture_sampling_contract")!="BASE_LEVEL_BILINEAR_LINEAR_PM_ONLY")
            throw std::runtime_error("TEXTURE_SAMPLING_CONTRACT_INVALID");
        if(manifest.at("mip_generation_authorized")!="0")
            throw std::runtime_error("MIP_GENERATION_MUST_BE_FORBIDDEN");
        if(manifest.at("pixel_coverage_contract")!="FIXED_2X2_QUARTER_SUBSAMPLES__LINEAR_PM_AVERAGE")
            throw std::runtime_error("PIXEL_COVERAGE_CONTRACT_INVALID");
        if(manifest.at("pixel_coverage_sample_count")!="4")
            throw std::runtime_error("PIXEL_COVERAGE_SAMPLE_COUNT_INVALID");
        if(manifest.at("depth_buffer_contract")!="IEEE754_FLOAT64_SOFTWARE_SORT")
            throw std::runtime_error("DEPTH_BUFFER_CONTRACT_INVALID");
        if(manifest.at("depth_equivalence_epsilon_camera_z")!="1e-12")
            throw std::runtime_error("DEPTH_EQUIVALENCE_EPSILON_INVALID");
        if(manifest.at("source_view_identity_contract")!="PER_TEXEL_INT16_PRESERVED__DIAGNOSTIC_ONLY")
            throw std::runtime_error("SOURCE_VIEW_IDENTITY_CONTRACT_INVALID");
        if(manifest.at("source_view_identity_render_authority")!="0")
            throw std::runtime_error("SOURCE_VIEW_IDENTITY_RENDER_AUTHORITY_FORBIDDEN");
        if(manifest.at("source_view_identity_compiled_harmonic_code")!="-2")
            throw std::runtime_error("SOURCE_VIEW_HARMONIC_CODE_INVALID");
        if(manifest.at("source_view_identity_unsupported_abstain_code")!="-4")
            throw std::runtime_error("SOURCE_VIEW_UNSUPPORTED_ABSTAIN_CODE_INVALID");
        if(manifest.at("source_view_identity_physical_padding_code")!="INT16_MIN")
            throw std::runtime_error("SOURCE_VIEW_PADDING_CODE_INVALID");
        if(manifest.at("source_view_identity_mixed_sample_code")!="-3")
            throw std::runtime_error("SOURCE_VIEW_MIXED_CODE_INVALID");
        const auto mesh=parse_mesh(entries.at("mesh.bin"));
        const auto cameras=parse_cameras(entries.at("cameras.bin"));
        const auto paging_contract=manifest.at("atlas_paging_contract");
        TextureSet textures;
        ProvenanceSet provenance;
        std::vector<std::uint32_t> face_pages(mesh.face_count,0);
        if(paging_contract=="LEGACY_SINGLE_PAGE_V1") {
            textures=parse_textures(entries.at("textures.bin"));
            provenance=parse_provenance(entries.at("provenance.bin"));
        } else if(paging_contract=="FACE_INDEX_TO_FIXED_PHYSICAL_PAGE_V1") {
            textures=parse_paged_textures(entries,manifest);
            provenance=parse_paged_provenance(entries.at(manifest.at("provenance_entry")));
            face_pages=parse_face_pages(
                entries.at(manifest.at("face_page_entry")),
                mesh.face_count,
                textures.pages
            );
        } else {
            throw std::runtime_error("ATLAS_PAGING_CONTRACT_INVALID");
        }
        if(
            cameras.size()!=8||textures.views!=8||provenance.views!=8||
            textures.pages!=provenance.pages||
            textures.height!=provenance.height||
            textures.width!=provenance.width
        ) throw std::runtime_error("VIEW_OR_ATLAS_DIMENSION_INVALID");
        if(view_id.size()!=2||view_id[0]!='V') throw std::runtime_error("VIEW_ID_INVALID");
        const int view=parse_int_exact(view_id.substr(1),"VIEW");
        if(view<0||view>=8) throw std::runtime_error("VIEW_INDEX_INVALID");

        const int clip_count=std::stoi(manifest.at("clip_count"));
        std::string clip_entry;
        for(int i=0;i<clip_count;++i) {
            if(manifest.at("clip."+std::to_string(i)+".id")==clip_id) {
                clip_entry=manifest.at("clip."+std::to_string(i)+".entry");
                break;
            }
        }
        if(clip_entry.empty()) throw std::runtime_error("CLIP_NOT_FOUND");
        const auto clip=parse_clip(entries.at(clip_entry));
        if(clip.vertex_count!=mesh.vertex_count) throw std::runtime_error("CLIP_MESH_VERTEX_COUNT_DRIFT");
        if(frame_index<0||frame_index>=static_cast<int>(clip.frame_count)) throw std::runtime_error("FRAME_INDEX_INVALID");

        std::vector<Vec3> xyz(mesh.vertex_count);
        const auto base=(static_cast<std::size_t>(frame_index)*mesh.vertex_count)*3u;
        for(std::uint32_t i=0;i<mesh.vertex_count;++i) {
            xyz[i]={clip.positions[base+i*3u],clip.positions[base+i*3u+1u],clip.positions[base+i*3u+2u]};
        }
        constexpr int kCoverageScale=2;
        constexpr int kCoverageSampleCount=kCoverageScale*kCoverageScale;
        constexpr double kDepthEquivalenceEpsilon=1e-12;
        std::vector<Proj> projected(mesh.vertex_count);
        for(std::uint32_t i=0;i<mesh.vertex_count;++i) {
            auto p=project(xyz[i],cameras[view]);
            p.x*=static_cast<double>(kCoverageScale);
            p.y*=static_cast<double>(kCoverageScale);
            projected[i]=p;
        }

        const int resolution=static_cast<int>(cameras[view].resolution);
        const int coverage_resolution=resolution*kCoverageScale;
        const auto pixels=static_cast<std::size_t>(resolution)*resolution;
        const auto coverage_pixels=static_cast<std::size_t>(coverage_resolution)*coverage_resolution;
        constexpr int kMaxLayers=4;
        const std::array<double,kMaxLayers> depth_init{
            std::numeric_limits<double>::infinity(),
            std::numeric_limits<double>::infinity(),
            std::numeric_limits<double>::infinity(),
            std::numeric_limits<double>::infinity()
        };
        const std::array<std::int32_t,kMaxLayers> owner_init{-1,-1,-1,-1};
        const std::array<float,3> bary_nan{NAN,NAN,NAN};
        const std::array<std::array<float,3>,kMaxLayers> bary_init{
            bary_nan,bary_nan,bary_nan,bary_nan
        };
        std::vector<std::array<double,kMaxLayers>> layer_depth(coverage_pixels,depth_init);
        std::vector<std::array<std::int32_t,kMaxLayers>> layer_owner(coverage_pixels,owner_init);
        std::vector<std::array<std::array<float,3>,kMaxLayers>> layer_bary(coverage_pixels,bary_init);
        std::vector<std::uint8_t> layer_overflow(coverage_pixels,0);

        for(std::uint32_t fi=0;fi<mesh.face_count;++fi) {
            const auto face=mesh.faces[fi];
            const auto a=projected[face[0]], b=projected[face[1]], c=projected[face[2]];
            const double area=orient(a,b,c.x,c.y);
            if(std::abs(area)<=1e-12) continue;
            const double minxv=std::min({a.x,b.x,c.x}), maxxv=std::max({a.x,b.x,c.x});
            const double minyv=std::min({a.y,b.y,c.y}), maxyv=std::max({a.y,b.y,c.y});
            const int minx=std::max(0,static_cast<int>(std::floor(minxv-0.5)));
            const int maxx=std::min(coverage_resolution-1,static_cast<int>(std::ceil(maxxv-0.5)));
            const int miny=std::max(0,static_cast<int>(std::floor(minyv-0.5)));
            const int maxy=std::min(coverage_resolution-1,static_cast<int>(std::ceil(maxyv-0.5)));
            for(int y=miny;y<=maxy;++y) for(int x=minx;x<=maxx;++x) {
                if(!covers(a,b,c,x,y)) continue;
                const double px=x+0.5, py=y+0.5;
                const double w0=orient(b,c,px,py)/area;
                const double w1=orient(c,a,px,py)/area;
                const double w2=orient(a,b,px,py)/area;
                const double z=w0*a.z+w1*b.z+w2*c.z;
                if(!std::isfinite(z)) throw std::runtime_error("VISIBILITY_DEPTH_NONFINITE");
                if(z<=kDepthEquivalenceEpsilon) continue;
                const auto idx=static_cast<std::size_t>(y)*coverage_resolution+x;
                int insert_at=-1;
                for(int layer=0;layer<kMaxLayers;++layer) {
                    const auto current_owner=layer_owner[idx][layer];
                    const auto current_depth=layer_depth[idx][layer];
                    if(current_owner<0 || z<current_depth-kDepthEquivalenceEpsilon ||
                       (std::abs(z-current_depth)<=kDepthEquivalenceEpsilon && static_cast<std::int32_t>(fi)<current_owner)) {
                        insert_at=layer;
                        break;
                    }
                }
                if(insert_at<0) {
                    layer_overflow[idx]=1;
                    continue;
                }
                if(layer_owner[idx][kMaxLayers-1]>=0) layer_overflow[idx]=1;
                for(int layer=kMaxLayers-1;layer>insert_at;--layer) {
                    layer_depth[idx][layer]=layer_depth[idx][layer-1];
                    layer_owner[idx][layer]=layer_owner[idx][layer-1];
                    layer_bary[idx][layer]=layer_bary[idx][layer-1];
                }
                layer_depth[idx][insert_at]=z;
                layer_owner[idx][insert_at]=static_cast<std::int32_t>(fi);
                layer_bary[idx][insert_at]={
                    static_cast<float>(w0),
                    static_cast<float>(w1),
                    static_cast<float>(w2)
                };
            }
        }

        std::vector<std::uint8_t> rgba(pixels*4u,0);
        std::vector<std::uint8_t> prov(pixels,255);
        std::vector<std::int16_t> source_view_diag(
            pixels,
            std::numeric_limits<std::int16_t>::min()
        );
        std::vector<std::int32_t> owner(pixels,-1);
        for(int y=0;y<resolution;++y) {
            for(int x=0;x<resolution;++x) {
                const auto idx=static_cast<std::size_t>(y)*resolution+x;
                PM pixel_accum{};
                int pixel_risk=-1;
                std::int16_t pixel_source_view=std::numeric_limits<std::int16_t>::min();
                double representative_depth=std::numeric_limits<double>::infinity();
                std::int32_t representative_owner=-1;

                for(int sy=0;sy<kCoverageScale;++sy) {
                    for(int sx=0;sx<kCoverageScale;++sx) {
                        const int cy=y*kCoverageScale+sy;
                        const int cx=x*kCoverageScale+sx;
                        const auto cidx=static_cast<std::size_t>(cy)*coverage_resolution+cx;
                        PM sample_accum{};
                        int sample_risk=-1;
                        std::int16_t sample_source_view_diag=std::numeric_limits<std::int16_t>::min();
                        for(int layer=0;layer<kMaxLayers;++layer) {
                            const auto fi=layer_owner[cidx][layer];
                            if(fi<0) continue;
                            const auto& uv=mesh.uv[static_cast<std::size_t>(fi)];
                            const auto& w=layer_bary[cidx][layer];
                            const double u=uv[0].x*w[0]+uv[1].x*w[1]+uv[2].x*w[2];
                            const double v=uv[0].y*w[0]+uv[1].y*w[1]+uv[2].y*w[2];
                            const auto page=face_pages[static_cast<std::size_t>(fi)];
                            const auto sampled_provenance=sample_provenance(
                                provenance,
                                static_cast<std::uint32_t>(view),
                                page,
                                u,
                                v
                            );
                            if(sampled_provenance==3)
                                throw std::runtime_error("VISIBLE_UNSUPPORTED_APPEARANCE");
                            if(sampled_provenance==255)
                                throw std::runtime_error("VISIBLE_ATLAS_PADDING");
                            const auto sample=sample_pm(textures,static_cast<std::uint32_t>(view),page,u,v);
                            const double transmission=1.0-std::max(0.0,std::min(1.0,sample_accum.a));
                            if(sample.a*transmission>1e-12) {
                                sample_risk=std::max(
                                    sample_risk,
                                    static_cast<int>(sampled_provenance)
                                );
                                sample_source_view_diag=merge_source_view_diagnostic(
                                    sample_source_view_diag,
                                    sample_source_view(
                                        provenance,
                                        static_cast<std::uint32_t>(view),
                                        page,
                                        u,
                                        v
                                    )
                                );
                            }
                            sample_accum.r+=transmission*sample.r;
                            sample_accum.g+=transmission*sample.g;
                            sample_accum.b+=transmission*sample.b;
                            sample_accum.a+=transmission*sample.a;
                        }
                        const double inv_samples=1.0/static_cast<double>(kCoverageSampleCount);
                        pixel_accum.r+=sample_accum.r*inv_samples;
                        pixel_accum.g+=sample_accum.g*inv_samples;
                        pixel_accum.b+=sample_accum.b*inv_samples;
                        pixel_accum.a+=sample_accum.a*inv_samples;
                        if(sample_risk>=0) pixel_risk=std::max(pixel_risk,sample_risk);
                        if(sample_source_view_diag!=std::numeric_limits<std::int16_t>::min()) {
                            pixel_source_view=merge_source_view_diagnostic(
                                pixel_source_view,
                                sample_source_view_diag
                            );
                        }

                        const double front_depth=layer_depth[cidx][0];
                        if(front_depth<representative_depth) {
                            representative_depth=front_depth;
                            representative_owner=layer_owner[cidx][0];
                        }
                    }
                }

                owner[idx]=representative_owner;
                const auto out=idx*4u;
                const double alpha=std::max(0.0,std::min(1.0,pixel_accum.a));
                if(alpha>1e-12) {
                    rgba[out]=q8(linear_to_srgb(pixel_accum.r/alpha));
                    rgba[out+1]=q8(linear_to_srgb(pixel_accum.g/alpha));
                    rgba[out+2]=q8(linear_to_srgb(pixel_accum.b/alpha));
                }
                rgba[out+3]=q8(alpha);
                if(pixel_risk>=0) prov[idx]=static_cast<std::uint8_t>(pixel_risk);
                source_view_diag[idx]=pixel_source_view;
            }
        }

        const auto overflow_count=static_cast<std::size_t>(
            std::count(layer_overflow.begin(),layer_overflow.end(),static_cast<std::uint8_t>(1))
        );
        if(overflow_count>0 && !allow_layer_overflow_diagnostic)
            throw std::runtime_error("VISIBILITY_LAYER_OVERFLOW");

        write_file(rgba_path,rgba.data(),rgba.size());
        if(!prov_path.empty()) write_file(prov_path,prov.data(),prov.size());
        if(!source_view_path.empty()) write_file(
            source_view_path,
            source_view_diag.data(),
            source_view_diag.size()*sizeof(std::int16_t)
        );
        if(!owner_path.empty()) write_file(owner_path,owner.data(),owner.size()*sizeof(std::int32_t));

        std::cout<<"atlas_pages="<<textures.pages<<"\n";
        std::cout<<"renderer=REALSAS_V2_CAA_CANONICAL_DEPTH"
                 <<" clip="<<clip_id<<" view="<<view_id<<" frame="<<frame_index
                 <<" resolution="<<resolution<<" visibility=SEALED_K4_DEPTH_LAYERS"
                 <<" coverage=FIXED_2X2_QUARTER_SUBSAMPLES"
                 <<" appearance=CAA_LINEAR_PREMULTIPLIED_LAYER_COMPOSITE\\n";
        return 0;
    } catch(const std::exception& e) {
        std::cerr<<"ERROR "<<e.what()<<"\\n";
        return 2;
    }
}
