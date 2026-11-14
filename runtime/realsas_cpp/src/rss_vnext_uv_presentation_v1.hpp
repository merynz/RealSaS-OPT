#pragma once
// RSS.vNext: native, pose-conditioned 2D presentation over canonical 3D mesh.
//
// This reuses GraphicsV1's exact per-pixel partial-order/depth solver.
// Source FRONT/BACK pixels, inferred RGB and UNKNOWN are *separate* layers.
// Selecting FRONT/BACK is a caller hypothesis, not proof of a source camera.
// The texture is an existing typed CAS artifact re-encoded for native memory.
// No reconstruction, resynthesis, hardcoded humanoid or second mesh.
#include "graphics_v1.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace realsas::rss_uv_v1 {
struct Reader {
    const std::vector<std::uint8_t>& input;
    std::size_t pos{0};
    void require(std::size_t n) const {
        if(pos > input.size() || n > input.size()-pos)
            throw std::runtime_error("RSS_UV_NATIVE_TRUNCATED");
    }
    std::uint32_t u32() {
        require(4);
        const auto* a=input.data()+pos;
        std::uint32_t v=static_cast<std::uint32_t>(a[0])
            |(static_cast<std::uint32_t>(a[1])<<8)
            |(static_cast<std::uint32_t>(a[2])<<16)
            |(static_cast<std::uint32_t>(a[3])<<24);
        pos+=4;
        return v;
    }
    std::uint16_t u16() {
        require(2);
        const auto value=static_cast<std::uint16_t>(input[pos])
            |static_cast<std::uint16_t>(input[pos+1]<<8);
        pos+=2;
        return value;
    }
    float f32() {
        const auto bits=u32();
        float v{};
        static_assert(sizeof(float)==4, "IEEE float32 required");
        std::memcpy(&v,&bits,4);
        if(!std::isfinite(v))throw std::runtime_error("RSS_UV_NATIVE_NONFINITE_UV");
        return v;
    }
    std::string digest() {
        if(u32()!=64)throw std::runtime_error("RSS_UV_NATIVE_DIGEST_LENGTH");
        require(64);
        const auto* data=reinterpret_cast<const char*>(input.data()+pos);
        const std::string value(data,64);
        pos+=64;
        for(char c:value)if(!((c>='0'&&c<='9')||(c>='a'&&c<='f')))
            throw std::runtime_error("RSS_UV_NATIVE_DIGEST_HEX");
        return value;
    }
    std::vector<std::uint8_t> bytes(std::size_t n) {
        require(n);
        std::vector<std::uint8_t> value(input.begin()+static_cast<std::ptrdiff_t>(pos),
                                         input.begin()+static_cast<std::ptrdiff_t>(pos+n));
        pos+=n;
        return value;
    }
};
struct Color {
    std::array<double,4> rgba{};
    std::uint8_t authority{0}; // 0=unfilled, 1=original image, 2=same H predicted
};
struct Texture {
    std::uint32_t width{}, height{}, faces{};
    std::string mesh_hash,atlas_cas_sha256,original_source_binding_sha256;
    std::vector<std::array<std::array<float,2>,3>> face_uv;
    std::vector<std::uint8_t> source_rgba;  // [2,H,W,4]
    std::vector<std::uint8_t> source_mask;  // [2,H,W]
    std::vector<std::uint8_t> inferred_rgba; // [H,W,4]
    std::vector<std::uint8_t> inferred_mask; // [H,W]
    std::vector<std::uint8_t> conflict; // [H,W] (never merged silently)
    std::vector<std::uint16_t> inferred_log_uncertainty_half;
};

inline Texture parse_texture(const std::vector<std::uint8_t>& blob) {
    static const char magic[8]={'R','S','U','V','T','0','0','1'};
    if(blob.size()<28||blob.size()>128u*1024u*1024u||
       std::memcmp(blob.data(),magic,8)!=0)
        throw std::runtime_error("RSS_UV_NATIVE_MAGIC_OR_SIZE");
    Reader r{blob,8};
    if(r.u32()!=1)throw std::runtime_error("RSS_UV_NATIVE_VERSION");
    Texture t;
    t.width=r.u32();t.height=r.u32();t.faces=r.u32();
    if(r.u32()!=3 || !t.width || !t.height ||
       t.width>2048 || t.height>2048 || !t.faces || t.faces>4096)
        throw std::runtime_error("RSS_UV_NATIVE_DOMAIN");
    t.mesh_hash=r.digest();
    t.atlas_cas_sha256=r.digest();
    t.original_source_binding_sha256=r.digest();
    const std::size_t pixels=static_cast<std::size_t>(t.width)*t.height;
    t.face_uv.resize(t.faces);
    for(auto& face:t.face_uv)for(auto& uv:face)for(auto& c:uv){
        c=r.f32();
        if(c<0.0f||c>1.0f)throw std::runtime_error("RSS_UV_NATIVE_UV_RANGE");
    }
    t.source_rgba=r.bytes(2*pixels*4);
    t.source_mask=r.bytes(2*pixels);
    t.inferred_rgba=r.bytes(pixels*4);
    t.inferred_mask=r.bytes(pixels);
    t.conflict=r.bytes(pixels);
    t.inferred_log_uncertainty_half.reserve(pixels);
    for(std::size_t i=0;i<pixels;++i) {
        const std::uint16_t v=r.u16();
        if((v&0x7c00u)==0x7c00u)
            throw std::runtime_error("RSS_UV_NATIVE_UNCERTAINTY_NONFINITE");
        t.inferred_log_uncertainty_half.push_back(v);
    }
    for(std::size_t i=0;i<2*pixels;++i) {
        const auto mask=t.source_mask[i];
        if(mask>1)throw std::runtime_error("RSS_UV_NATIVE_SOURCE_MASK_DOMAIN");
        if(mask==0)for(int channel=0;channel<4;++channel)
            if(t.source_rgba[i*4+channel]!=0)
                throw std::runtime_error("RSS_UV_NATIVE_UNOBSERVED_SOURCE_COLOR_FORGED");
    }
    for(std::size_t i=0;i<pixels;++i) {
        if(t.inferred_mask[i]>1||t.conflict[i]>1)
            throw std::runtime_error("RSS_UV_NATIVE_INFERRED_MASK_DOMAIN");
        if(t.inferred_mask[i]==0)for(int channel=0;channel<4;++channel)
            if(t.inferred_rgba[4*i+channel]!=0)
                throw std::runtime_error("RSS_UV_NATIVE_UNKNOWN_INFERRED_COLOR_FORGED");
    }
    if(r.pos!=blob.size())
        throw std::runtime_error("RSS_UV_NATIVE_TRAILING");
    return t;
}

struct Projected {
    double x{},y{},z{};
};
struct PixelImage {
    std::uint32_t width{},height{};
    std::vector<std::uint8_t> rgba;
    std::vector<std::int32_t> front_face;
    std::vector<std::uint8_t> color_authority;
    std::uint64_t overlapping_pixels{},explicit_order_overrides{},uncolored_pixels{};
};

inline double orient(const Projected& a,const Projected& b,double x,double y){
    return (b.x-a.x)*(y-a.y)-(b.y-a.y)*(x-a.x);
}
inline bool top_left(const Projected& a,const Projected& b){
    return b.y-a.y<0.0||(b.y==a.y&&b.x-a.x>0.0);
}
inline bool accept_edge(double v,bool inclusive){
    return v>1e-12||(std::abs(v)<=1e-12&&inclusive);
}

// Sample a given source view ONLY when caller explicitly selects its binding.
// Missing real texels may fall back to the same-checkpoint inferred layer.
// Bilinear taps stay inside this face's UV triangle, not a neighbor's chart.
inline Color sample_texture(const Texture& tex,std::uint32_t face,
                            const std::array<double,3>& bary,
                            int original_view_hint){
    if(face>=tex.faces || original_view_hint < -1 || original_view_hint>1)
        throw std::runtime_error("RSS_UV_NATIVE_SAMPLING_DOMAIN");
    const auto& uv=tex.face_uv[face];
    const double u=bary[0]*uv[0][0]+bary[1]*uv[1][0]+bary[2]*uv[2][0];
    const double v=bary[0]*uv[0][1]+bary[1]*uv[1][1]+bary[2]*uv[2][1];
    const double gx=u*tex.width-.5,gy=v*tex.height-.5;
    const auto x0=static_cast<int>(std::floor(gx)),y0=static_cast<int>(std::floor(gy));
    const double fx=gx-x0,fy=gy-y0;
    const double umin=std::min({static_cast<double>(uv[0][0]),double(uv[1][0]),double(uv[2][0])});
    const double umax=std::max({static_cast<double>(uv[0][0]),double(uv[1][0]),double(uv[2][0])});
    const double vmin=std::min({static_cast<double>(uv[0][1]),double(uv[1][1]),double(uv[2][1])});
    const double vmax=std::max({static_cast<double>(uv[0][1]),double(uv[1][1]),double(uv[2][1])});
    const std::array<std::array<int,2>,4> taps{{
        {x0,y0},{x0+1,y0},{x0,y0+1},{x0+1,y0+1}
    }};
    const std::array<double,4> weights={
        (1-fx)*(1-fy),fx*(1-fy),(1-fx)*fy,fx*fy
    };
    const auto try_layer=[&](int view)->Color{
        Color out;
        double sum=0;
        for(std::size_t i=0;i<4;++i) {
            const int x=taps[i][0],y=taps[i][1];
            if(x<0||y<0||x>=static_cast<int>(tex.width)||y>=static_cast<int>(tex.height))
                continue;
            const double tu=(x+.5)/tex.width,tv=(y+.5)/tex.height;
            if(tu<umin-1e-7||tu>umax+1e-7||tv<vmin-1e-7||tv>vmax+1e-7)
                continue;
            const auto p=static_cast<std::size_t>(y)*tex.width+x;
            const auto idx=(view<0?p:static_cast<std::size_t>(view)*tex.width*tex.height+p);
            const auto& mask=(view<0?tex.inferred_mask:tex.source_mask);
            if(!mask[idx]||weights[i]<=0)continue;
            const auto& colors=(view<0?tex.inferred_rgba:tex.source_rgba);
            for(int ch=0;ch<4;++ch)
                out.rgba[ch]+=weights[i]*colors[4*idx+ch]/255.;
            sum+=weights[i];
        }
        if(sum<=1e-12)return Color{};
        for(auto& c:out.rgba)c/=sum;
        out.authority=static_cast<std::uint8_t>(view<0?2:1);
        return out;
    };
    if(original_view_hint>=0) {
        const auto original=try_layer(original_view_hint);
        if(original.authority)return original;
    }
    return try_layer(-1);
}

// Presentation owns draw order, not UV texture; "z" comes from the projected,
// deformed *canonical* mesh at this frame. Different frames can change overlap.
inline PixelImage render(const graphics_v1::State& graphics,const Texture& texture,
                         const std::vector<std::array<std::uint32_t,3>>& faces,
                         const std::vector<Projected>& posed_vertices,
                         std::uint32_t width,std::uint32_t height,
                         std::uint32_t view,int original_view_hint){
    if(
        texture.mesh_hash!=graphics.canonical_mesh_binding_hash
        ||texture.faces!=faces.size()
        ||graphics.face_fragment_index.size()!=faces.size()
        ||view>=graphics.views.size()
        ||width<1||height<1||width>2048||height>2048
        ||posed_vertices.empty()||posed_vertices.size()>65536
    )throw std::runtime_error("RSS_UV_NATIVE_GRAPHICS_MESH_POSE_DOMAIN");
    struct Layer {std::uint32_t face{};double z{};Color color;};
    const std::size_t pixels=static_cast<std::size_t>(width)*height;
    std::vector<std::vector<Layer>> layers(pixels);
    for(std::uint32_t f=0;f<faces.size();++f){
        if(!graphics_v1::face_visible(graphics,view,f))continue;
        const auto ids=faces[f];
        if(ids[0]>=posed_vertices.size()||ids[1]>=posed_vertices.size()||
           ids[2]>=posed_vertices.size())
            throw std::runtime_error("RSS_UV_NATIVE_FACE_INDEX_INVALID");
        const auto a=posed_vertices[ids[0]],b=posed_vertices[ids[1]],c=posed_vertices[ids[2]];
        for(const auto& point:{a,b,c}){
            if(!std::isfinite(point.x)||!std::isfinite(point.y)||!std::isfinite(point.z)||
               !(point.z>0))
                throw std::runtime_error("RSS_UV_NATIVE_POSED_VERTEX_INVALID");
        }
        const double area=orient(a,b,c.x,c.y);
        if(std::abs(area)<=1e-12)continue;
        const int x0=std::max(0,static_cast<int>(std::floor(std::min({a.x,b.x,c.x})-.5)));
        const int x1=std::min(static_cast<int>(width)-1,static_cast<int>(std::ceil(std::max({a.x,b.x,c.x})-.5)));
        const int y0=std::max(0,static_cast<int>(std::floor(std::min({a.y,b.y,c.y})-.5)));
        const int y1=std::min(static_cast<int>(height)-1,static_cast<int>(std::ceil(std::max({a.y,b.y,c.y})-.5)));
        if(x1<x0||y1<y0)continue;
        const double sign=area>0?1.:-1.;
        const std::array<bool,3> inclusive=area>0
            ?std::array<bool,3>{top_left(b,c),top_left(c,a),top_left(a,b)}
            :std::array<bool,3>{top_left(c,b),top_left(a,c),top_left(b,a)};
        for(int y=y0;y<=y1;++y)for(int x=x0;x<=x1;++x){
            const double e0=orient(b,c,x+.5,y+.5);
            const double e1=orient(c,a,x+.5,y+.5);
            const double e2=orient(a,b,x+.5,y+.5);
            if(!accept_edge(sign*e0,inclusive[0])||
               !accept_edge(sign*e1,inclusive[1])||
               !accept_edge(sign*e2,inclusive[2]))continue;
            // Perspective-correct interpolation in the camera z domain.
            const double w0=e0/area,w1=e1/area,w2=e2/area;
            const double denom=w0/a.z+w1/b.z+w2/c.z;
            if(denom<=0||!std::isfinite(denom))
                throw std::runtime_error("RSS_UV_NATIVE_PERSPECTIVE_DOMAIN");
            const std::array<double,3> bary={
                (w0/a.z)/denom,(w1/b.z)/denom,(w2/c.z)/denom
            };
            const auto color=sample_texture(texture,f,bary,original_view_hint);
            if(color.authority==0||color.rgba[3]<=1e-12)continue;
            const auto p=static_cast<std::size_t>(y)*width+x;
            if(layers[p].size()>=32)
                throw std::runtime_error("RSS_UV_NATIVE_LAYER_OVERFLOW");
            layers[p].push_back({f,1./denom,color});
        }
    }
    PixelImage output;
    output.width=width;output.height=height;
    output.rgba.resize(pixels*4,0);
    output.front_face.resize(pixels,-1);
    output.color_authority.resize(pixels,0);
    const auto q8=[](double x)->std::uint8_t{
        return static_cast<std::uint8_t>(std::floor(std::max(0.,std::min(1.,x))*255.+.5));
    };
    for(std::size_t pixel=0;pixel<pixels;++pixel){
        const auto& hits=layers[pixel];
        if(hits.empty()){++output.uncolored_pixels;continue;}
        if(hits.size()>1)++output.overlapping_pixels;
        std::vector<std::uint32_t> face_ids;
        std::vector<double> depths;
        for(const auto& layer:hits){face_ids.push_back(layer.face);depths.push_back(layer.z);}
        const auto order=graphics_v1::order_pixel_layers(graphics,view,face_ids,depths,1e-9);
        auto physical=order;
        std::stable_sort(physical.begin(),physical.end(),[&](auto i,auto j){
            if(depths[i]<depths[j]-1e-9)return true;
            if(depths[j]<depths[i]-1e-9)return false;
            return face_ids[i]<face_ids[j];
        });
        if(order!=physical)++output.explicit_order_overrides;
        std::array<double,4> accum{};
        for(auto idx:order){
            const auto& rgba=hits[idx].color.rgba;
            const double transmit=1.-accum[3];
            for(int ch=0;ch<3;++ch)accum[ch]+=transmit*rgba[ch]*rgba[3];
            accum[3]+=transmit*rgba[3];
        }
        if(accum[3]>1e-12)
            for(int ch=0;ch<3;++ch)
                output.rgba[4*pixel+ch]=q8(accum[ch]/accum[3]);
        output.rgba[4*pixel+3]=q8(accum[3]);
        output.front_face[pixel]=static_cast<std::int32_t>(hits[order.front()].face);
        output.color_authority[pixel]=hits[order.front()].color.authority;
    }
    return output;
}
} // namespace realsas::rss_uv_v1
