#include "runtime_v4_reference.h"
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

namespace realsas::runtime_v4 {
namespace {

using reference_raster_v3::Barycentric;
using reference_raster_v3::DepthSample;
using reference_raster_v3::DepthWritePolicy;
using reference_raster_v3::Vertex;

constexpr uint32_t kVersion = 4u;
constexpr uint8_t kMagic[8] = {'R','S','R','T',0,4,0,0};
constexpr uint32_t kFeatureCanonicalXyz = 1u << 1;
constexpr uint32_t kFeatureDepthRenderer = 1u << 4;
constexpr uint32_t kFeatureRuntimeProjection = 1u << 5;
constexpr uint32_t kFeatureFullSurface = 1u << 6;
constexpr uint32_t kFeatureSlots = 1u << 7;
constexpr uint32_t kFeatureSharedGeometry = 1u << 10;
constexpr uint32_t kRequiredFeatures = kFeatureCanonicalXyz | kFeatureDepthRenderer |
    kFeatureRuntimeProjection | kFeatureFullSurface | kFeatureSlots | kFeatureSharedGeometry;

struct Vec2 { float x=0, y=0; };
struct Vec3 { float x=0, y=0, z=0; };
enum class AttachmentKind:uint8_t { DeformableBody=0, RigidComponent=1, Clipping=2 };
enum class TopologyClass:uint8_t { Static=0, AttachmentDynamic=1, ClipDynamic=2, DrawOrderDynamic=3 };
enum class AppearanceProvenance:uint8_t { DirectSource=0, OtherViewSource=1, UnderRigidSource=2, Unseen=3, Completion=4 };
struct Slot { std::string id,bone_id; uint32_t setup_order=0; std::string default_attachment_id; };
struct Asset { std::string id; uint32_t slot_index=0; std::string attachment_id; AttachmentKind kind=AttachmentKind::DeformableBody; TopologyClass topo=TopologyClass::Static; std::string sealed_source_hash; std::vector<Vec3> rest; std::vector<std::array<uint32_t,3>> faces; };
struct Camera { Vec3 origin,right,up,forward; float half_extent=1; uint32_t resolution=0; };
struct Overlay { uint32_t asset_index=0; std::vector<Vec2> uv; std::vector<AppearanceProvenance> provenance; std::vector<int16_t> donor; };
struct View { std::string id,texture_path; uint32_t texture_crc32=0,width=0,height=0; std::vector<uint8_t> texture_rgba; Camera camera; std::vector<Overlay> overlays; };
struct ClipInterval { std::string clip_attachment_id; uint32_t start_slot_index=0,end_slot_index=0; bool inverse=false; };
struct FrameView { std::vector<uint32_t> draw_order_slots; std::vector<std::string> active_attachment_by_slot; std::vector<ClipInterval> clip_intervals; };
struct Frame { float time=0; std::vector<std::vector<Vec3>> asset_vertices; std::vector<FrameView> views; };
struct Clip { std::string id,display_name,intent; float duration=0,fps=30; bool loop=false,qualified=false; std::vector<Frame> frames; };
struct RuntimeData { std::string binary_schema,source_binding_sha256,source_proof_bundle_hash,coordinate_system,playback_contract_hash,reference_raster_contract_hash; float units_per_pixel=1,alpha_cutout_threshold=.5f; uint32_t feature_flags=0; std::vector<Slot> slots; std::vector<Asset> assets; std::vector<View> views; std::vector<Clip> clips; };

class Reader {
public:
    explicit Reader(const std::vector<uint8_t>& b):b_(b){}
    void require(size_t n) const { if(off_>b_.size()||n>b_.size()-off_) throw std::runtime_error("Truncated RealSaS runtime-v4 binary"); }
    uint8_t u8(){require(1);return b_[off_++];}
    uint32_t u32(){require(4);uint32_t v=uint32_t(b_[off_])|(uint32_t(b_[off_+1])<<8)|(uint32_t(b_[off_+2])<<16)|(uint32_t(b_[off_+3])<<24);off_+=4;return v;}
    int16_t i16(){require(2);uint16_t u=uint16_t(b_[off_])|(uint16_t(b_[off_+1])<<8);off_+=2;return static_cast<int16_t>(u);}
    float f32(){uint32_t bits=u32();float v;std::memcpy(&v,&bits,4);if(!std::isfinite(v))throw std::runtime_error("Non-finite runtime-v4 float");return v;}
    std::string str(){uint32_t n=u32();if(n>16u*1024u*1024u)throw std::runtime_error("Unreasonable runtime-v4 string");require(n);std::string s(reinterpret_cast<const char*>(b_.data()+off_),n);off_+=n;return s;}
    void magic(){require(8);if(std::memcmp(b_.data()+off_,kMagic,8)!=0)throw std::runtime_error("Invalid runtime-v4 magic");off_+=8;}
    size_t offset()const{return off_;}
private:
    const std::vector<uint8_t>& b_;
    size_t off_=0;
};

uint32_t crc_bytes(const uint8_t* data,size_t size){uLong c=crc32(0L,Z_NULL,0);size_t o=0;while(o<size){uInt n=static_cast<uInt>(std::min<size_t>(size-o,std::numeric_limits<uInt>::max()));c=crc32(c,data+o,n);o+=n;}return uint32_t(c);}

std::vector<uint8_t> read_file(const std::filesystem::path&p){std::ifstream f(p,std::ios::binary);if(!f)throw std::runtime_error("Cannot open file: "+p.string());f.seekg(0,std::ios::end);auto n=f.tellg();if(n<0)throw std::runtime_error("Cannot size file");f.seekg(0);std::vector<uint8_t>b(static_cast<size_t>(n));if(!b.empty())f.read(reinterpret_cast<char*>(b.data()),std::streamsize(b.size()));if(!f&&!b.empty())throw std::runtime_error("Cannot read file");return b;}

std::vector<uint8_t> archive_entry_bytes(const std::filesystem::path&p,std::string_view wanted){archive*a=archive_read_new();if(!a)throw std::runtime_error("libarchive allocation failed");std::unique_ptr<archive,decltype(&archive_read_free)>hold(a,&archive_read_free);archive_read_support_filter_all(a);archive_read_support_format_all(a);if(archive_read_open_filename(a,p.string().c_str(),10240)!=ARCHIVE_OK)throw std::runtime_error(std::string("Cannot open archive: ")+archive_error_string(a));archive_entry*e=nullptr;while(archive_read_next_header(a,&e)==ARCHIVE_OK){const char*name=archive_entry_pathname(e);if(name&&wanted==name){la_int64_t dn=archive_entry_size(e);if(dn<0||dn>la_int64_t(3ull<<30))throw std::runtime_error("Invalid archive entry size");std::vector<uint8_t>out(static_cast<size_t>(dn));size_t o=0;while(o<out.size()){la_ssize_t g=archive_read_data(a,out.data()+o,out.size()-o);if(g<0)throw std::runtime_error(std::string("Archive read failed: ")+archive_error_string(a));if(g==0)break;o+=size_t(g);}if(o!=out.size())throw std::runtime_error("Truncated archive entry");return out;}archive_read_data_skip(a);}throw std::runtime_error("Archive entry not found: "+std::string(wanted));}

std::vector<uint8_t> package_asset(const std::filesystem::path&p,std::string_view rel){if(std::filesystem::is_directory(p)){auto root=std::filesystem::weakly_canonical(p);auto c=std::filesystem::weakly_canonical(root/std::filesystem::path(rel));auto ri=root.begin(),ci=c.begin();for(;ri!=root.end()&&ci!=c.end();++ri,++ci)if(*ri!=*ci)throw std::runtime_error("Runtime-v4 path escape");if(ri!=root.end())throw std::runtime_error("Runtime-v4 path escape");return read_file(c);}return archive_entry_bytes(p,rel);}

std::vector<uint8_t> decode_png(const std::vector<uint8_t>&b,uint32_t w,uint32_t h){png_image im{};im.version=PNG_IMAGE_VERSION;if(!png_image_begin_read_from_memory(&im,b.data(),b.size()))throw std::runtime_error("PNG header decode failed");im.format=PNG_FORMAT_RGBA;if(im.width!=w||im.height!=h){png_image_free(&im);throw std::runtime_error("Texture dimensions disagree with binary");}std::vector<uint8_t>out(PNG_IMAGE_SIZE(im));if(!png_image_finish_read(&im,nullptr,out.data(),0,nullptr)){std::string m=im.message;png_image_free(&im);throw std::runtime_error("PNG decode failed: "+m);}png_image_free(&im);return out;}

AttachmentKind parse_kind(uint8_t v){if(v>2)throw std::runtime_error("Invalid attachment kind");return AttachmentKind(v);}
TopologyClass parse_topo(uint8_t v){if(v>3)throw std::runtime_error("Invalid topology class");return TopologyClass(v);}
AppearanceProvenance parse_prov(uint8_t v){if(v>4)throw std::runtime_error("Invalid appearance provenance");return AppearanceProvenance(v);}

void exact_permutation(const std::vector<uint32_t>&o,size_t n){if(o.size()!=n)throw std::runtime_error("Draw order is not exact slot permutation");std::vector<bool>s(n,false);for(uint32_t x:o){if(x>=n||s[x])throw std::runtime_error("Draw order is not exact slot permutation");s[x]=true;}}

RuntimeData parse_binary(const std::vector<uint8_t>&bytes){
    if(bytes.size()<16)throw std::runtime_error("Runtime-v4 binary too small");
    const size_t body_n=bytes.size()-4;
    const uint32_t stored=uint32_t(bytes[body_n])|(uint32_t(bytes[body_n+1])<<8)|(uint32_t(bytes[body_n+2])<<16)|(uint32_t(bytes[body_n+3])<<24);
    if(stored!=crc_bytes(bytes.data(),body_n))throw std::runtime_error("Runtime-v4 CRC32 mismatch");
    std::vector<uint8_t>body(bytes.begin(),bytes.begin()+std::ptrdiff_t(body_n));
    Reader r(body);
    r.magic();
    if(r.u32()!=kVersion)throw std::runtime_error("Unsupported runtime-v4 version");

    RuntimeData d;
    d.binary_schema=r.str();
    d.source_binding_sha256=r.str();
    d.source_proof_bundle_hash=r.str();
    d.coordinate_system=r.str();
    d.units_per_pixel=r.f32();
    d.feature_flags=r.u32();
    d.playback_contract_hash=r.str();
    d.reference_raster_contract_hash=r.str();
    d.alpha_cutout_threshold=r.f32();
    if(d.binary_schema!="RealSaS.RuntimeBinary.rsr.v4")throw std::runtime_error("Unexpected runtime-v4 schema");
    if(d.source_binding_sha256.size()!=64||d.source_proof_bundle_hash.size()!=64||d.playback_contract_hash.size()!=64||d.reference_raster_contract_hash.size()!=64)throw std::runtime_error("Runtime-v4 identity is not SHA-256 length");
    if(d.coordinate_system!="canonical_3d.shared_pose__view_projection_runtime")throw std::runtime_error("Runtime-v4 coordinate mismatch");
    if((d.feature_flags&kRequiredFeatures)!=kRequiredFeatures)throw std::runtime_error("Runtime-v4 lacks required shared-geometry capabilities");
    if(d.alpha_cutout_threshold<0||d.alpha_cutout_threshold>1)throw std::runtime_error("Invalid cutout threshold");

    uint32_t ns=r.u32();
    if(ns==0||ns>4096)throw std::runtime_error("Invalid slot count");
    d.slots.reserve(ns);
    std::unordered_set<std::string>slot_ids;
    std::unordered_set<uint32_t>orders;
    for(uint32_t i=0;i<ns;++i){
        Slot s;
        s.id=r.str();s.bone_id=r.str();s.setup_order=r.u32();s.default_attachment_id=r.str();
        if(s.id.empty()||!slot_ids.insert(s.id).second||!orders.insert(s.setup_order).second)throw std::runtime_error("Invalid/duplicate slot");
        d.slots.push_back(std::move(s));
    }

    uint32_t na=r.u32();
    if(na==0||na>4096)throw std::runtime_error("Invalid asset count");
    d.assets.reserve(na);
    std::unordered_set<std::string>asset_ids,attachment_ids;
    for(uint32_t ai=0;ai<na;++ai){
        Asset a;
        a.id=r.str();a.slot_index=r.u32();a.attachment_id=r.str();a.kind=parse_kind(r.u8());a.topo=parse_topo(r.u8());
        (void)r.u8();(void)r.u8();a.sealed_source_hash=r.str();
        uint32_t nv=r.u32(),nf=r.u32();
        if(a.id.empty()||!asset_ids.insert(a.id).second||a.slot_index>=d.slots.size()||a.attachment_id.empty()||!attachment_ids.insert(a.attachment_id).second||a.sealed_source_hash.size()!=64)throw std::runtime_error("Invalid asset identity");
        if(nv==0||nv>10000000u||nf==0||nf>20000000u)throw std::runtime_error("Invalid asset dimensions");
        a.rest.resize(nv);for(auto&v:a.rest)v={r.f32(),r.f32(),r.f32()};
        a.faces.resize(nf);for(auto&f:a.faces){f={r.u32(),r.u32(),r.u32()};if(f[0]>=nv||f[1]>=nv||f[2]>=nv)throw std::runtime_error("Triangle index out of range");}
        d.assets.push_back(std::move(a));
    }

    uint32_t nview=r.u32();
    if(nview==0||nview>64)throw std::runtime_error("Invalid view count");
    d.views.reserve(nview);
    std::unordered_set<std::string>view_ids;
    for(uint32_t vi=0;vi<nview;++vi){
        View v;
        v.id=r.str();v.texture_path=r.str();v.texture_crc32=r.u32();v.width=r.u32();v.height=r.u32();
        if(v.id.empty()||!view_ids.insert(v.id).second||v.texture_path.empty()||v.width==0||v.height==0)throw std::runtime_error("Invalid view metadata");
        v.camera.origin={r.f32(),r.f32(),r.f32()};v.camera.right={r.f32(),r.f32(),r.f32()};v.camera.up={r.f32(),r.f32(),r.f32()};v.camera.forward={r.f32(),r.f32(),r.f32()};v.camera.half_extent=r.f32();v.camera.resolution=r.u32();
        if(!(v.camera.half_extent>0)||v.camera.resolution==0)throw std::runtime_error("Invalid camera");
        uint32_t no=r.u32();
        if(no!=d.assets.size())throw std::runtime_error("View overlay asset cardinality mismatch");
        v.overlays.reserve(no);
        std::unordered_set<uint32_t>seen;
        for(uint32_t oi=0;oi<no;++oi){
            Overlay o;
            o.asset_index=r.u32();
            if(o.asset_index>=d.assets.size()||!seen.insert(o.asset_index).second)throw std::runtime_error("Invalid/duplicate overlay asset");
            const auto&asset=d.assets[o.asset_index];
            o.uv.resize(asset.rest.size());
            for(auto&uv:o.uv){uv={r.f32(),r.f32()};if(uv.x<0||uv.x>1||uv.y<0||uv.y>1)throw std::runtime_error("UV outside [0,1]");}
            o.provenance.resize(asset.faces.size());for(auto&prov:o.provenance)prov=parse_prov(r.u8());
            o.donor.resize(asset.faces.size());for(auto&donor:o.donor)donor=r.i16();
            v.overlays.push_back(std::move(o));
        }
        d.views.push_back(std::move(v));
    }

    uint32_t nc=r.u32();
    if(nc==0||nc>4096)throw std::runtime_error("Invalid clip count");
    d.clips.reserve(nc);
    std::unordered_set<std::string>clip_ids;
    for(uint32_t ci=0;ci<nc;++ci){
        Clip c;
        c.id=r.str();c.display_name=r.str();c.intent=r.str();c.duration=r.f32();c.fps=r.f32();c.loop=r.u8()!=0;c.qualified=r.u8()!=0;
        (void)r.u8();(void)r.u8();
        uint32_t nf=r.u32();
        if(c.id.empty()||!clip_ids.insert(c.id).second||!(c.duration>0)||!(c.fps>0)||nf==0||nf>1000000u)throw std::runtime_error("Invalid clip");
        c.frames.resize(nf);
        for(auto&frame:c.frames){
            frame.time=r.f32();
            frame.asset_vertices.resize(d.assets.size());
            for(size_t ai=0;ai<d.assets.size();++ai){frame.asset_vertices[ai].resize(d.assets[ai].rest.size());for(auto&v:frame.asset_vertices[ai])v={r.f32(),r.f32(),r.f32()};}
            frame.views.resize(d.views.size());
            for(auto&fv:frame.views){
                uint32_t nord=r.u32();fv.draw_order_slots.resize(nord);for(auto&x:fv.draw_order_slots)x=r.u32();exact_permutation(fv.draw_order_slots,d.slots.size());
                uint32_t nactive=r.u32();if(nactive!=d.slots.size())throw std::runtime_error("Active attachment slot cardinality mismatch");fv.active_attachment_by_slot.resize(nactive);for(auto&s:fv.active_attachment_by_slot)s=r.str();
                uint32_t nclip=r.u32();fv.clip_intervals.resize(nclip);for(auto&x:fv.clip_intervals){x.clip_attachment_id=r.str();x.start_slot_index=r.u32();x.end_slot_index=r.u32();x.inverse=r.u8()!=0;(void)r.u8();(void)r.u8();(void)r.u8();if(x.start_slot_index>=d.slots.size()||x.end_slot_index>=d.slots.size())throw std::runtime_error("Clip interval slot out of range");}
            }
        }
        d.clips.push_back(std::move(c));
    }
    if(r.offset()!=body.size())throw std::runtime_error("Runtime-v4 binary has trailing data before CRC");
    return d;
}

Vec3 project(const Vec3&p,const Camera&c){const float dx=p.x-c.origin.x,dy=p.y-c.origin.y,dz=p.z-c.origin.z;const float gx=(dx*c.right.x+dy*c.right.y+dz*c.right.z)/c.half_extent;const float gy=-(dx*c.up.x+dy*c.up.y+dz*c.up.z)/c.half_extent;const float z=dx*c.forward.x+dy*c.forward.y+dz*c.forward.z;const float res=static_cast<float>(c.resolution);return {(gx+1.0f)*0.5f*res,(gy+1.0f)*0.5f*res,z};}

std::array<float,4> sample_bilinear(const View&v,float u,float vv){u=std::clamp(u,0.0f,1.0f);vv=std::clamp(vv,0.0f,1.0f);const float fx=u*float(v.width-1),fy=vv*float(v.height-1);const uint32_t x0=uint32_t(std::floor(fx)),y0=uint32_t(std::floor(fy)),x1=std::min(x0+1,v.width-1),y1=std::min(y0+1,v.height-1);const float tx=fx-float(x0),ty=fy-float(y0);auto px=[&](uint32_t x,uint32_t y,int c){return float(v.texture_rgba[(size_t(y)*v.width+x)*4+size_t(c)])/255.0f;};std::array<float,4>out{};for(int c=0;c<4;++c){float a=px(x0,y0,c)*(1-tx)+px(x1,y0,c)*tx;float b=px(x0,y1,c)*(1-tx)+px(x1,y1,c)*tx;out[c]=a*(1-ty)+b*ty;}return out;}

void blend_source_over(std::array<float,4>&dst,const std::array<float,4>&src){const float sa=std::clamp(src[3],0.0f,1.0f),da=std::clamp(dst[3],0.0f,1.0f);const float oa=sa+da*(1-sa);if(oa<=1e-8f){dst={0,0,0,0};return;}for(int c=0;c<3;++c)dst[c]=(src[c]*sa+dst[c]*da*(1-sa))/oa;dst[3]=oa;}

size_t choose_frame_pair(const Clip&c,float t,size_t&hi,float&alpha){if(c.frames.size()==1){hi=0;alpha=0;return 0;}float local=t;if(c.loop&&c.duration>0){local=std::fmod(local,c.duration);if(local<0)local+=c.duration;}else local=std::clamp(local,0.0f,c.duration);auto it=std::upper_bound(c.frames.begin(),c.frames.end(),local,[](float x,const Frame&f){return x<f.time;});if(it==c.frames.begin()){hi=0;alpha=0;return 0;}if(it==c.frames.end()){hi=c.frames.size()-1;alpha=0;return hi;}hi=size_t(it-c.frames.begin());size_t lo=hi-1;float dt=c.frames[hi].time-c.frames[lo].time;alpha=dt>0?(local-c.frames[lo].time)/dt:0;return lo;}

const Overlay& overlay_for_asset(const View& view, uint32_t asset_index){
    for(const auto& overlay : view.overlays){
        if(overlay.asset_index==asset_index)return overlay;
    }
    throw std::runtime_error("Missing Runtime-v4 donor view overlay");
}

struct SourceBinding {
    const View* view=nullptr;
    const Overlay* overlay=nullptr;
};

SourceBinding source_binding_for_face(
    const RuntimeData& data,
    uint32_t target_view_index,
    uint32_t asset_index,
    size_t face_index,
    const Overlay& target_overlay){
    const auto provenance=target_overlay.provenance.at(face_index);
    if(provenance==AppearanceProvenance::Unseen)return {};
    if(provenance==AppearanceProvenance::Completion)throw std::runtime_error("Runtime-v4 native completion atlas sampling is not qualified");

    const int16_t donor=target_overlay.donor.at(face_index);
    if(donor<0||static_cast<size_t>(donor)>=data.views.size())throw std::runtime_error("Runtime-v4 source provenance donor out of range");
    if(provenance==AppearanceProvenance::DirectSource&&static_cast<uint32_t>(donor)!=target_view_index)throw std::runtime_error("Runtime-v4 DIRECT_SOURCE donor must match target view");
    if(provenance==AppearanceProvenance::OtherViewSource&&static_cast<uint32_t>(donor)==target_view_index)throw std::runtime_error("Runtime-v4 OTHER_VIEW_SOURCE donor must differ from target view");

    const View& source_view=data.views[static_cast<size_t>(donor)];
    const Overlay& source_overlay=overlay_for_asset(source_view,asset_index);
    return {&source_view,&source_overlay};
}

} // namespace

struct ReferenceRuntime::Impl {
    explicit Impl(const std::filesystem::path& package_path):package(package_path){
        auto binary=package_asset(package,"runtime/realsas_runtime.rsr");
        data=parse_binary(binary);
        for(auto&view:data.views){
            auto raw=package_asset(package,view.texture_path);
            if(crc_bytes(raw.data(),raw.size())!=view.texture_crc32)throw std::runtime_error("Runtime-v4 texture CRC mismatch");
            view.texture_rgba=decode_png(raw,view.width,view.height);
        }
    }
    std::filesystem::path package;
    RuntimeData data;
};

ReferenceRuntime::ReferenceRuntime(const std::filesystem::path&p):impl_(std::make_unique<Impl>(p)){}
ReferenceRuntime::~ReferenceRuntime()=default;
ReferenceRuntime::ReferenceRuntime(ReferenceRuntime&&) noexcept=default;
ReferenceRuntime& ReferenceRuntime::operator=(ReferenceRuntime&&) noexcept=default;
const std::string& ReferenceRuntime::binary_schema()const noexcept{return impl_->data.binary_schema;}
const std::string& ReferenceRuntime::source_binding_sha256()const noexcept{return impl_->data.source_binding_sha256;}
const std::string& ReferenceRuntime::source_proof_bundle_hash()const noexcept{return impl_->data.source_proof_bundle_hash;}
const std::string& ReferenceRuntime::coordinate_system()const noexcept{return impl_->data.coordinate_system;}
const std::string& ReferenceRuntime::playback_contract_hash()const noexcept{return impl_->data.playback_contract_hash;}
const std::string& ReferenceRuntime::reference_raster_contract_hash()const noexcept{return impl_->data.reference_raster_contract_hash;}
uint32_t ReferenceRuntime::feature_flags()const noexcept{return impl_->data.feature_flags;}
float ReferenceRuntime::alpha_cutout_threshold()const noexcept{return impl_->data.alpha_cutout_threshold;}
uint32_t ReferenceRuntime::view_count()const noexcept{return uint32_t(impl_->data.views.size());}
const std::string& ReferenceRuntime::view_id(uint32_t i)const{if(i>=impl_->data.views.size())throw std::out_of_range("view index");return impl_->data.views[i].id;}
uint32_t ReferenceRuntime::view_width(uint32_t i)const{if(i>=impl_->data.views.size())throw std::out_of_range("view index");return impl_->data.views[i].camera.resolution;}
uint32_t ReferenceRuntime::view_height(uint32_t i)const{if(i>=impl_->data.views.size())throw std::out_of_range("view index");return impl_->data.views[i].camera.resolution;}
uint32_t ReferenceRuntime::find_view(const std::string&id)const noexcept{for(uint32_t i=0;i<impl_->data.views.size();++i)if(impl_->data.views[i].id==id)return i;return UINT32_MAX;}
uint32_t ReferenceRuntime::clip_count()const noexcept{return uint32_t(impl_->data.clips.size());}
const std::string& ReferenceRuntime::clip_id(uint32_t i)const{if(i>=impl_->data.clips.size())throw std::out_of_range("clip index");return impl_->data.clips[i].id;}
const std::string& ReferenceRuntime::clip_display_name(uint32_t i)const{if(i>=impl_->data.clips.size())throw std::out_of_range("clip index");return impl_->data.clips[i].display_name;}
const std::string& ReferenceRuntime::clip_intent(uint32_t i)const{if(i>=impl_->data.clips.size())throw std::out_of_range("clip index");return impl_->data.clips[i].intent;}
float ReferenceRuntime::clip_duration(uint32_t i)const{if(i>=impl_->data.clips.size())throw std::out_of_range("clip index");return impl_->data.clips[i].duration;}
bool ReferenceRuntime::clip_qualified(uint32_t i)const{if(i>=impl_->data.clips.size())throw std::out_of_range("clip index");return impl_->data.clips[i].qualified;}
uint32_t ReferenceRuntime::find_clip(const std::string&name)const noexcept{for(uint32_t i=0;i<impl_->data.clips.size();++i)if(impl_->data.clips[i].id==name||impl_->data.clips[i].intent==name)return i;return UINT32_MAX;}

std::vector<uint8_t> ReferenceRuntime::render_rgba(uint32_t ci,uint32_t vi,float time_seconds)const{
    const auto&d=impl_->data;
    if(ci>=d.clips.size()||vi>=d.views.size())throw std::out_of_range("render index");
    const auto&clip=d.clips[ci];
    const auto&view=d.views[vi];
    size_t hi=0;float alpha=0;
    const size_t lo=choose_frame_pair(clip,time_seconds,hi,alpha);
    const Frame&f0=clip.frames[lo];
    const Frame&f1=clip.frames[hi];
    const FrameView&fv=(alpha<0.5f?f0.views[vi]:f1.views[vi]);
    if(!fv.clip_intervals.empty())throw std::runtime_error("Runtime-v4 P0/P1 native clipping is not yet qualified");

    const uint32_t output_width=view.camera.resolution;
    const uint32_t output_height=view.camera.resolution;
    if(output_width==0||output_height==0)throw std::runtime_error("Runtime-v4 output resolution is zero");
    const size_t pixels=size_t(output_width)*output_height;
    std::vector<std::array<float,4>>color(pixels,{0,0,0,0});
    std::vector<DepthSample>depth(pixels);
    std::unordered_map<uint32_t,const Overlay*>overlay_by_asset;
    for(const auto&o:view.overlays)overlay_by_asset.emplace(o.asset_index,&o);

    for(uint32_t semantic_order=0;semantic_order<fv.draw_order_slots.size();++semantic_order){
        uint32_t slot_index=fv.draw_order_slots[semantic_order];
        const std::string&active=fv.active_attachment_by_slot[slot_index];
        if(active.empty())continue;
        uint32_t ai=UINT32_MAX;
        for(uint32_t j=0;j<d.assets.size();++j)if(d.assets[j].attachment_id==active){ai=j;break;}
        if(ai==UINT32_MAX)throw std::runtime_error("Active attachment has no Runtime-v4 asset");
        const auto&asset=d.assets[ai];
        if(asset.slot_index!=slot_index)throw std::runtime_error("Active attachment slot mismatch");
        if(asset.kind==AttachmentKind::Clipping)throw std::runtime_error("Runtime-v4 P0/P1 clipping attachment cannot render before clipping qualification");
        auto oit=overlay_by_asset.find(ai);
        if(oit==overlay_by_asset.end())throw std::runtime_error("Missing Runtime-v4 view overlay");
        const Overlay&ov=*oit->second;

        std::vector<Vertex>verts(asset.rest.size());
        for(size_t i=0;i<verts.size();++i){
            Vec3 p{
                f0.asset_vertices[ai][i].x+(f1.asset_vertices[ai][i].x-f0.asset_vertices[ai][i].x)*alpha,
                f0.asset_vertices[ai][i].y+(f1.asset_vertices[ai][i].y-f0.asset_vertices[ai][i].y)*alpha,
                f0.asset_vertices[ai][i].z+(f1.asset_vertices[ai][i].z-f0.asset_vertices[ai][i].z)*alpha
            };
            Vec3 q=project(p,view.camera);
            verts[i]={q.x,q.y,q.z,0.0f,0.0f};
        }

        for(size_t fi=0;fi<asset.faces.size();++fi){
            SourceBinding binding=source_binding_for_face(d,vi,ai,fi,ov);
            if(binding.view==nullptr)continue;
            const auto&t=asset.faces[fi];
            Vertex a=verts[t[0]],b=verts[t[1]],c=verts[t[2]];
            a.u=binding.overlay->uv[t[0]].x;a.v=binding.overlay->uv[t[0]].y;
            b.u=binding.overlay->uv[t[1]].x;b.v=binding.overlay->uv[t[1]].y;
            c.u=binding.overlay->uv[t[2]].x;c.v=binding.overlay->uv[t[2]].y;
            int minx=std::max(0,int(std::floor(std::min({a.x,b.x,c.x})-0.5f)));
            int maxx=std::min(int(output_width)-1,int(std::ceil(std::max({a.x,b.x,c.x})-0.5f)));
            int miny=std::max(0,int(std::floor(std::min({a.y,b.y,c.y})-0.5f)));
            int maxy=std::min(int(output_height)-1,int(std::ceil(std::max({a.y,b.y,c.y})-0.5f)));
            if(minx>maxx||miny>maxy)continue;
            for(int y=miny;y<=maxy;++y)for(int x=minx;x<=maxx;++x){
                Barycentric bc=reference_raster_v3::cover_pixel_center(a,b,c,x,y);
                if(!bc.covered)continue;
                float z=reference_raster_v3::interpolate_depth(bc,a,b,c);
                size_t pi=size_t(y)*output_width+size_t(x);
                const uint64_t provenance_rank =
                    target_overlay.provenance.at(fi)==AppearanceProvenance::DirectSource ? 2ull : 1ull;
                const uint64_t authority_order =
                    (provenance_rank<<62) | (uint64_t(ai)<<32) | uint64_t(fi);
                if(!reference_raster_v3::depth_test_passes(depth[pi],z,authority_order))continue;
                float u=a.u*bc.w0+b.u*bc.w1+c.u*bc.w2;
                float v=a.v*bc.w0+b.v*bc.w1+c.v*bc.w2;
                auto src=sample_bilinear(*binding.view,u,v);
                if(src[3]<=0)continue;
                // Flattened 8-view art is treated as nearest-surface opaque/cutout
                // appearance authority. Physical pixel ownership is independent of
                // editable slot draw order; stacked translucency is not inferred.
                color[pi]=src;
                const DepthWritePolicy policy=DepthWritePolicy::On;
                if(reference_raster_v3::should_write_depth(policy,src[3],d.alpha_cutout_threshold))
                    reference_raster_v3::commit_depth(depth[pi],z,authority_order);
            }
        }
    }

    std::vector<uint8_t>out(pixels*4);
    for(size_t i=0;i<pixels;++i)for(int c=0;c<4;++c)out[i*4+size_t(c)]=uint8_t(std::lround(std::clamp(color[i][c],0.0f,1.0f)*255.0f));
    return out;
}

} // namespace realsas::runtime_v4
