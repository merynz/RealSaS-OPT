#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace realsas::graphics_v1 {

struct Reader {
    const std::vector<std::uint8_t>& data;
    std::size_t off{0};
    void require(std::size_t n) const { if (off > data.size() || n > data.size()-off) throw std::runtime_error("GRAPHICS_V1_TRUNCATED"); }
    std::uint32_t u32(){require(4);std::uint32_t v{};std::memcpy(&v,data.data()+off,4);off+=4;return v;}
    double f64(){require(8);double v{};std::memcpy(&v,data.data()+off,8);off+=8;if(!std::isfinite(v))throw std::runtime_error("GRAPHICS_V1_NONFINITE");return v;}
    std::string text(){auto n=u32();if(n>16u*1024u*1024u)throw std::runtime_error("GRAPHICS_V1_STRING_TOO_LARGE");require(n);std::string s(reinterpret_cast<const char*>(data.data()+off),n);off+=n;return s;}
};

struct View {
    std::uint32_t view_index{};
    std::vector<std::uint8_t> visible_fragment;
    std::vector<std::uint32_t> topological_order;
    std::vector<std::uint8_t> precedes;
};

struct Attachment {
    std::uint32_t fragment_index{};
    std::string parent_mechanical_domain_id;
    std::uint32_t anchor_face_index{};
    double barycentric[3]{};
    double confidence{};
};

struct State {
    std::string canonical_mesh_binding_hash;
    std::string compiled_graphics_binding_hash;
    std::string runtime_graphics_hash;
    std::vector<std::uint32_t> face_fragment_index;
    std::vector<std::string> fragment_ids;
    std::vector<std::uint32_t> support_band_pixels;
    std::vector<View> views;
    std::vector<Attachment> attachments;
};

inline State parse(const std::vector<std::uint8_t>& bytes) {
    static constexpr char kMagic[8]={'R','S','G','F','X','0','0','1'};
    if(bytes.size()<8 || std::memcmp(bytes.data(),kMagic,8)!=0) throw std::runtime_error("GRAPHICS_V1_MAGIC_INVALID");
    Reader r{bytes,8};
    if(r.u32()!=1)throw std::runtime_error("GRAPHICS_V1_VERSION_INVALID");
    const auto face_count=r.u32(),fragment_count=r.u32(),view_count=r.u32(),attachment_count=r.u32();
    if(face_count==0 || fragment_count==0 || view_count==0 || fragment_count>face_count)throw std::runtime_error("GRAPHICS_V1_CARDINALITY_INVALID");
    State out;out.canonical_mesh_binding_hash=r.text();out.compiled_graphics_binding_hash=r.text();out.runtime_graphics_hash=r.text();
    if(out.canonical_mesh_binding_hash.empty()||out.compiled_graphics_binding_hash.empty()||out.runtime_graphics_hash.empty())throw std::runtime_error("GRAPHICS_V1_BINDING_EMPTY");
    out.face_fragment_index.resize(face_count);
    for(auto& x:out.face_fragment_index){x=r.u32();if(x>=fragment_count)throw std::runtime_error("GRAPHICS_V1_FACE_OWNER_INVALID");}
    out.fragment_ids.resize(fragment_count);out.support_band_pixels.resize(fragment_count);
    for(std::uint32_t i=0;i<fragment_count;++i){out.fragment_ids[i]=r.text();if(out.fragment_ids[i].empty())throw std::runtime_error("GRAPHICS_V1_FRAGMENT_ID_EMPTY");out.support_band_pixels[i]=r.u32();if(out.support_band_pixels[i]>4)throw std::runtime_error("GRAPHICS_V1_SUPPORT_BAND_TOO_LARGE");}
    if(std::vector<std::string> ids=out.fragment_ids;[&](){std::sort(ids.begin(),ids.end());return std::adjacent_find(ids.begin(),ids.end())!=ids.end();}())throw std::runtime_error("GRAPHICS_V1_FRAGMENT_ID_DUPLICATE");
    out.views.reserve(view_count);
    for(std::uint32_t vi=0;vi<view_count;++vi){
        View v;v.view_index=r.u32();if(v.view_index!=vi)throw std::runtime_error("GRAPHICS_V1_VIEW_INDEX_DRIFT");v.visible_fragment.assign(fragment_count,0);auto visible_count=r.u32();
        for(std::uint32_t i=0;i<visible_count;++i){auto f=r.u32();if(f>=fragment_count||v.visible_fragment[f])throw std::runtime_error("GRAPHICS_V1_VISIBLE_FRAGMENT_INVALID");v.visible_fragment[f]=1;}
        auto order_count=r.u32();if(order_count!=fragment_count)throw std::runtime_error("GRAPHICS_V1_TOPOLOGICAL_ORDER_INCOMPLETE");v.topological_order.resize(order_count);std::vector<std::uint8_t> seen(fragment_count,0);
        for(auto& f:v.topological_order){f=r.u32();if(f>=fragment_count||seen[f])throw std::runtime_error("GRAPHICS_V1_TOPOLOGICAL_ORDER_INVALID");seen[f]=1;}
        v.precedes.assign(static_cast<std::size_t>(fragment_count)*fragment_count,0);auto edge_count=r.u32();
        for(std::uint32_t i=0;i<edge_count;++i){auto a=r.u32(),b=r.u32();if(a>=fragment_count||b>=fragment_count||a==b)throw std::runtime_error("GRAPHICS_V1_ORDER_EDGE_INVALID");v.precedes[static_cast<std::size_t>(a)*fragment_count+b]=1;}
        for(std::uint32_t k=0;k<fragment_count;++k)for(std::uint32_t a=0;a<fragment_count;++a)if(v.precedes[static_cast<std::size_t>(a)*fragment_count+k])for(std::uint32_t b=0;b<fragment_count;++b)if(v.precedes[static_cast<std::size_t>(k)*fragment_count+b])v.precedes[static_cast<std::size_t>(a)*fragment_count+b]=1;
        for(std::uint32_t i=0;i<fragment_count;++i)if(v.precedes[static_cast<std::size_t>(i)*fragment_count+i])throw std::runtime_error("GRAPHICS_V1_ORDER_CYCLE");
        std::vector<std::uint32_t> rank(fragment_count);for(std::uint32_t i=0;i<fragment_count;++i)rank[v.topological_order[i]]=i;
        for(std::uint32_t a=0;a<fragment_count;++a)for(std::uint32_t b=0;b<fragment_count;++b)if(v.precedes[static_cast<std::size_t>(a)*fragment_count+b]&&rank[a]>=rank[b])throw std::runtime_error("GRAPHICS_V1_TOPOLOGICAL_ORDER_EDGE_DRIFT");
        out.views.push_back(std::move(v));
    }
    out.attachments.reserve(attachment_count);
    for(std::uint32_t i=0;i<attachment_count;++i){Attachment a;a.fragment_index=r.u32();if(a.fragment_index>=fragment_count)throw std::runtime_error("GRAPHICS_V1_ATTACHMENT_FRAGMENT_INVALID");a.parent_mechanical_domain_id=r.text();a.anchor_face_index=r.u32();if(a.anchor_face_index>=face_count||a.parent_mechanical_domain_id.empty())throw std::runtime_error("GRAPHICS_V1_ATTACHMENT_BINDING_INVALID");double sum=0;for(double& x:a.barycentric){x=r.f64();sum+=x;if(x<-1e-9)throw std::runtime_error("GRAPHICS_V1_ATTACHMENT_BARY_INVALID");}if(std::abs(sum-1.0)>1e-6)throw std::runtime_error("GRAPHICS_V1_ATTACHMENT_BARY_SUM_INVALID");a.confidence=r.f64();if(a.confidence<0||a.confidence>1)throw std::runtime_error("GRAPHICS_V1_ATTACHMENT_CONFIDENCE_INVALID");out.attachments.push_back(std::move(a));}
    if(r.off!=bytes.size())throw std::runtime_error("GRAPHICS_V1_TRAILING_BYTES");return out;
}

inline bool face_visible(const State& s,std::uint32_t view,std::uint32_t face){
    if(view>=s.views.size()||face>=s.face_fragment_index.size())throw std::runtime_error("GRAPHICS_V1_VISIBILITY_INDEX_INVALID");return s.views[view].visible_fragment[s.face_fragment_index[face]]!=0;
}
inline bool explicit_precedence(const State& s,std::uint32_t view,std::uint32_t front_face,std::uint32_t back_face){
    if(view>=s.views.size()||front_face>=s.face_fragment_index.size()||back_face>=s.face_fragment_index.size())throw std::runtime_error("GRAPHICS_V1_PRECEDENCE_INDEX_INVALID");
    const std::size_t n=s.fragment_ids.size();
    const std::uint32_t a=s.face_fragment_index[front_face];
    const std::uint32_t b=s.face_fragment_index[back_face];
    return a!=b&&s.views[view].precedes[static_cast<std::size_t>(a)*n+b]!=0;
}
inline int compare_layers(const State& s,std::uint32_t view,std::uint32_t face_a,double depth_a,std::uint32_t face_b,double depth_b,double epsilon){
    if(!std::isfinite(depth_a)||!std::isfinite(depth_b)||epsilon<0)throw std::runtime_error("GRAPHICS_V1_COMPARE_INPUT_INVALID");if(explicit_precedence(s,view,face_a,face_b))return -1;if(explicit_precedence(s,view,face_b,face_a))return 1;if(depth_a<depth_b-epsilon)return -1;if(depth_b<depth_a-epsilon)return 1;return face_a<face_b?-1:(face_b<face_a?1:0);
}

} // namespace realsas::graphics_v1
