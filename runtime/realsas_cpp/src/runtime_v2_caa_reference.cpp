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
    const auto vertex_bytes = static_cast<std::size_t>(m.vertex_count) * 3u * sizeof(float);
    if (off + vertex_bytes > data.size()) throw std::runtime_error("MESH_VERTEX_TRUNCATED");
    off += vertex_bytes; // runtime frames carry exact posed XYZ.
    m.faces.resize(m.face_count);
    for (std::uint32_t f = 0; f < m.face_count; ++f) {
        for (int k = 0; k < 3; ++k) m.faces[f][k] = read_scalar<std::uint32_t>(data, off);
    }
    m.uv.resize(m.face_count);
    for (std::uint32_t f = 0; f < m.face_count; ++f) {
        for (int k = 0; k < 3; ++k) {
            m.uv[f][k].x = static_cast<double>(read_scalar<float>(data, off));
            m.uv[f][k].y = static_cast<double>(read_scalar<float>(data, off));
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
    std::vector<float> positions;
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
    for (auto& v : c.positions) v = read_scalar<float>(data, off);
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

} // namespace

int main(int argc,char** argv) {
    try {
        if(argc<2) throw std::runtime_error("USAGE: player package --clip ID --view V0 --frame N --out-rgba PATH [--out-provenance PATH] [--out-owner PATH]");
        const std::string package_path=argv[1];
        const std::string clip_id=arg_value(argc,argv,"--clip");
        const std::string view_id=arg_value(argc,argv,"--view");
        const int frame_index=std::stoi(arg_value(argc,argv,"--frame"));
        const std::string rgba_path=arg_value(argc,argv,"--out-rgba");
        const std::string prov_path=arg_value(argc,argv,"--out-provenance",false);
        const std::string owner_path=arg_value(argc,argv,"--out-owner",false);

        const auto entries=parse_rss(package_path);
        const auto manifest=parse_manifest(entries.at("manifest.txt"));
        const auto mesh=parse_mesh(entries.at("mesh.bin"));
        const auto cameras=parse_cameras(entries.at("cameras.bin"));
        const auto textures=parse_textures(entries.at("textures.bin"));
        const auto provenance=parse_provenance(entries.at("provenance.bin"));
        if(cameras.size()!=8||textures.views!=8||provenance.views!=8) throw std::runtime_error("VIEW_COUNT_INVALID");
        if(view_id.size()!=2||view_id[0]!='V') throw std::runtime_error("VIEW_ID_INVALID");
        const int view=std::stoi(view_id.substr(1));
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
        std::vector<Proj> projected(mesh.vertex_count);
        for(std::uint32_t i=0;i<mesh.vertex_count;++i) projected[i]=project(xyz[i],cameras[view]);

        const int resolution=static_cast<int>(cameras[view].resolution);
        const auto pixels=static_cast<std::size_t>(resolution)*resolution;
        std::vector<double> depth(pixels,std::numeric_limits<double>::infinity());
        std::vector<std::int32_t> owner(pixels,-1);
        std::vector<std::array<float,3>> bary(pixels,{NAN,NAN,NAN});

        for(std::uint32_t fi=0;fi<mesh.face_count;++fi) {
            const auto face=mesh.faces[fi];
            const auto a=projected[face[0]], b=projected[face[1]], c=projected[face[2]];
            const double area=orient(a,b,c.x,c.y);
            if(std::abs(area)<=1e-12) continue;
            const double minxv=std::min({a.x,b.x,c.x}), maxxv=std::max({a.x,b.x,c.x});
            const double minyv=std::min({a.y,b.y,c.y}), maxyv=std::max({a.y,b.y,c.y});
            const int minx=std::max(0,static_cast<int>(std::floor(minxv-0.5)));
            const int maxx=std::min(resolution-1,static_cast<int>(std::ceil(maxxv-0.5)));
            const int miny=std::max(0,static_cast<int>(std::floor(minyv-0.5)));
            const int maxy=std::min(resolution-1,static_cast<int>(std::ceil(maxyv-0.5)));
            for(int y=miny;y<=maxy;++y) for(int x=minx;x<=maxx;++x) {
                if(!covers(a,b,c,x,y)) continue;
                const double px=x+0.5, py=y+0.5;
                const double w0=orient(b,c,px,py)/area;
                const double w1=orient(c,a,px,py)/area;
                const double w2=orient(a,b,px,py)/area;
                const double z=w0*a.z+w1*b.z+w2*c.z;
                const auto idx=static_cast<std::size_t>(y)*resolution+x;
                if(z<depth[idx]-1e-12 || (std::abs(z-depth[idx])<=1e-12 && static_cast<std::int32_t>(fi)<owner[idx])) {
                    depth[idx]=z; owner[idx]=static_cast<std::int32_t>(fi);
                    bary[idx]={static_cast<float>(w0),static_cast<float>(w1),static_cast<float>(w2)};
                }
            }
        }

        std::vector<std::uint8_t> rgba(pixels*4u,0);
        std::vector<std::uint8_t> prov(pixels,255);
        for(std::size_t idx=0;idx<pixels;++idx) {
            const auto fi=owner[idx];
            if(fi<0) continue;
            const auto& uv=mesh.uv[static_cast<std::size_t>(fi)];
            const auto& w=bary[idx];
            const double u=uv[0].x*w[0]+uv[1].x*w[1]+uv[2].x*w[2];
            const double v=uv[0].y*w[0]+uv[1].y*w[1]+uv[2].y*w[2];
            const auto pm=sample_pm(textures,static_cast<std::uint32_t>(view),u,v);
            const auto out=idx*4u;
            const double alpha=pm.a;
            if(alpha>1e-12) {
                rgba[out]=q8(linear_to_srgb(pm.r/alpha));
                rgba[out+1]=q8(linear_to_srgb(pm.g/alpha));
                rgba[out+2]=q8(linear_to_srgb(pm.b/alpha));
            }
            rgba[out+3]=q8(alpha);
            prov[idx]=sample_provenance(provenance,static_cast<std::uint32_t>(view),u,v);
        }

        write_file(rgba_path,rgba.data(),rgba.size());
        if(!prov_path.empty()) write_file(prov_path,prov.data(),prov.size());
        if(!owner_path.empty()) write_file(owner_path,owner.data(),owner.size()*sizeof(std::int32_t));

        std::cout<<"renderer=REALSAS_V2_CAA_CANONICAL_DEPTH"
                 <<" clip="<<clip_id<<" view="<<view_id<<" frame="<<frame_index
                 <<" resolution="<<resolution<<" visibility=SEALED_FACE_INDEX_ZBUFFER"
                 <<" appearance=CAA_LINEAR_PREMULTIPLIED_BILINEAR\\n";
        return 0;
    } catch(const std::exception& e) {
        std::cerr<<"ERROR "<<e.what()<<"\\n";
        return 2;
    }
}
