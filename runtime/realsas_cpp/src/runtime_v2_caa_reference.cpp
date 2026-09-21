#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

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
    std::uint32_t views{}, height{}, width{};
    std::vector<std::uint8_t> rgba;
};

TextureSet parse_textures(const std::vector<std::uint8_t>& data) {
    std::size_t off = 0;
    TextureSet t;
    t.views = read_scalar<std::uint32_t>(data, off);
    t.height = read_scalar<std::uint32_t>(data, off);
    t.width = read_scalar<std::uint32_t>(data, off);
    const auto need = static_cast<std::size_t>(t.views) * t.height * t.width * 4u;
    if (data.size() - off != need) throw std::runtime_error("TEXTURE_BYTES_INVALID");
    t.rgba.assign(data.begin() + static_cast<std::ptrdiff_t>(off), data.end());
    return t;
}

struct ProvenanceSet {
    std::uint32_t views{}, height{}, width{};
    std::vector<std::uint8_t> value;
};

ProvenanceSet parse_provenance(const std::vector<std::uint8_t>& data) {
    std::size_t off = 0;
    ProvenanceSet p;
    p.views = read_scalar<std::uint32_t>(data, off);
    p.height = read_scalar<std::uint32_t>(data, off);
    p.width = read_scalar<std::uint32_t>(data, off);
    const auto need = static_cast<std::size_t>(p.views) * p.height * p.width;
    if (data.size() - off != need) throw std::runtime_error("PROVENANCE_BYTES_INVALID");
    p.value.assign(data.begin() + static_cast<std::ptrdiff_t>(off), data.end());
    return p;
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

PM texel_pm(const TextureSet& t, std::uint32_t view, int x, int y) {
    x=std::max(0,std::min(x,static_cast<int>(t.width)-1));
    y=std::max(0,std::min(y,static_cast<int>(t.height)-1));
    const auto idx=((((static_cast<std::size_t>(view)*t.height)+static_cast<std::size_t>(y))*t.width)+static_cast<std::size_t>(x))*4u;
    const double a=static_cast<double>(t.rgba[idx+3])/255.0;
    return {srgb_to_linear(static_cast<double>(t.rgba[idx])/255.0)*a,
            srgb_to_linear(static_cast<double>(t.rgba[idx+1])/255.0)*a,
            srgb_to_linear(static_cast<double>(t.rgba[idx+2])/255.0)*a,a};
}
PM mix(const PM& a,const PM& b,double t) {
    return {a.r+(b.r-a.r)*t,a.g+(b.g-a.g)*t,a.b+(b.b-a.b)*t,a.a+(b.a-a.a)*t};
}
PM sample_pm(const TextureSet& t,std::uint32_t view,double u,double v) {
    u=std::max(0.0,std::min(1.0,u)); v=std::max(0.0,std::min(1.0,v));
    const double x=u*static_cast<double>(t.width-1), y=v*static_cast<double>(t.height-1);
    const int x0=static_cast<int>(std::floor(x)), y0=static_cast<int>(std::floor(y));
    const int x1=std::min(x0+1,static_cast<int>(t.width)-1), y1=std::min(y0+1,static_cast<int>(t.height)-1);
    const double tx=x-x0, ty=y-y0;
    return mix(mix(texel_pm(t,view,x0,y0),texel_pm(t,view,x1,y0),tx),
               mix(texel_pm(t,view,x0,y1),texel_pm(t,view,x1,y1),tx),ty);
}
std::uint8_t provenance_texel(const ProvenanceSet& p,std::uint32_t view,int x,int y) {
    x=std::max(0,std::min(x,static_cast<int>(p.width)-1));
    y=std::max(0,std::min(y,static_cast<int>(p.height)-1));
    const auto idx=((static_cast<std::size_t>(view)*p.height)+static_cast<std::size_t>(y))*p.width+static_cast<std::size_t>(x);
    return p.value[idx];
}
std::uint8_t sample_provenance(const ProvenanceSet& p,std::uint32_t view,double u,double v) {
    u=std::max(0.0,std::min(1.0,u)); v=std::max(0.0,std::min(1.0,v));
    const double x=u*static_cast<double>(p.width-1), y=v*static_cast<double>(p.height-1);
    const int x0=static_cast<int>(std::floor(x)), y0=static_cast<int>(std::floor(y));
    const int x1=std::min(x0+1,static_cast<int>(p.width)-1), y1=std::min(y0+1,static_cast<int>(p.height)-1);
    const double tx=x-x0, ty=y-y0;
    const std::array<double,4> w{
        (1.0-tx)*(1.0-ty), tx*(1.0-ty), (1.0-tx)*ty, tx*ty
    };
    const std::array<std::uint8_t,4> v4{
        provenance_texel(p,view,x0,y0),
        provenance_texel(p,view,x1,y0),
        provenance_texel(p,view,x0,y1),
        provenance_texel(p,view,x1,y1)
    };
    int risk=-1;
    for(std::size_t i=0;i<4;++i) if(w[i]>1e-12) risk=std::max(risk,static_cast<int>(v4[i]));
    if(risk<0) throw std::runtime_error("PROVENANCE_BILINEAR_FOOTPRINT_EMPTY");
    return static_cast<std::uint8_t>(risk);
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

} // namespace

int main(int argc,char** argv) {
    try {
        if(argc<2) throw std::runtime_error("USAGE: player package --clip ID --view V0 --frame N --out-rgba PATH [--out-provenance PATH] [--out-owner PATH]");
        const std::string package_path=argv[1];
        const std::string clip_id=arg_value(argc,argv,"--clip");
        const std::string view_id=arg_value(argc,argv,"--view");
        const int frame_index=parse_int_exact(arg_value(argc,argv,"--frame"),"FRAME");
        const std::string rgba_path=arg_value(argc,argv,"--out-rgba");
        const std::string prov_path=arg_value(argc,argv,"--out-provenance",false);
        const std::string owner_path=arg_value(argc,argv,"--out-owner",false);

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
        const auto mesh=parse_mesh(entries.at("mesh.bin"));
        const auto cameras=parse_cameras(entries.at("cameras.bin"));
        const auto textures=parse_textures(entries.at("textures.bin"));
        const auto provenance=parse_provenance(entries.at("provenance.bin"));
        if(cameras.size()!=8||textures.views!=8||provenance.views!=8) throw std::runtime_error("VIEW_COUNT_INVALID");
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
        std::vector<std::int32_t> owner(pixels,-1);
        for(int y=0;y<resolution;++y) {
            for(int x=0;x<resolution;++x) {
                const auto idx=static_cast<std::size_t>(y)*resolution+x;
                PM pixel_accum{};
                int pixel_risk=-1;
                double representative_depth=std::numeric_limits<double>::infinity();
                std::int32_t representative_owner=-1;

                for(int sy=0;sy<kCoverageScale;++sy) {
                    for(int sx=0;sx<kCoverageScale;++sx) {
                        const int cy=y*kCoverageScale+sy;
                        const int cx=x*kCoverageScale+sx;
                        const auto cidx=static_cast<std::size_t>(cy)*coverage_resolution+cx;
                        PM sample_accum{};
                        int sample_risk=-1;
                        for(int layer=0;layer<kMaxLayers;++layer) {
                            const auto fi=layer_owner[cidx][layer];
                            if(fi<0) continue;
                            const auto& uv=mesh.uv[static_cast<std::size_t>(fi)];
                            const auto& w=layer_bary[cidx][layer];
                            const double u=uv[0].x*w[0]+uv[1].x*w[1]+uv[2].x*w[2];
                            const double v=uv[0].y*w[0]+uv[1].y*w[1]+uv[2].y*w[2];
                            const auto sample=sample_pm(textures,static_cast<std::uint32_t>(view),u,v);
                            const double transmission=1.0-std::max(0.0,std::min(1.0,sample_accum.a));
                            if(sample.a*transmission>1e-12) {
                                sample_risk=std::max(
                                    sample_risk,
                                    static_cast<int>(sample_provenance(
                                        provenance,
                                        static_cast<std::uint32_t>(view),
                                        u,
                                        v
                                    ))
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
            }
        }

        const auto overflow_count=static_cast<std::size_t>(
            std::count(layer_overflow.begin(),layer_overflow.end(),static_cast<std::uint8_t>(1))
        );
        if(overflow_count>0) throw std::runtime_error("VISIBILITY_LAYER_OVERFLOW");

        write_file(rgba_path,rgba.data(),rgba.size());
        if(!prov_path.empty()) write_file(prov_path,prov.data(),prov.size());
        if(!owner_path.empty()) write_file(owner_path,owner.data(),owner.size()*sizeof(std::int32_t));

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
