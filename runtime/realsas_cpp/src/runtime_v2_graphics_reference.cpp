// vNext Graphics-native reference player.
//
// This is an append-only consumer for RSS packages that explicitly authorize
// ROLE_FREE_CANONICAL_M_GRAPHICS_V1.  The historical V2/CAA player remains an
// immutable regression consumer and rejects these packages.  Both consumers
// therefore cannot claim authority over the same package contract.
#define main realsas_legacy_v2_reference_main
#include "runtime_v2_caa_reference.cpp"
#undef main

#include "graphics_v1.hpp"

namespace {

int render_single_m_graphics(
    const Entries& entries,
    const std::unordered_map<std::string,std::string>& manifest,
    int argc,
    char** argv
) {
    if(manifest.at("graphics_state_execution_authorized")!="1" ||
       manifest.at("graphics_state_contract")!="ROLE_FREE_CANONICAL_M_GRAPHICS_V1" ||
       manifest.at("graphics_face_ownership_required")!="1" ||
       manifest.at("graphics_partial_order_required")!="1" ||
       manifest.at("triangle_array_order_as_presentation_authority")!="0" ||
       manifest.at("independent_visual_mesh_authorized")!="0")
        throw std::runtime_error("GRAPHICS_RUNTIME_CONTRACT_INVALID");
    if(manifest.at("presentation_state_execution_authorized")!="1" ||
       manifest.at("tint_order_visibility_authorized")!="1")
        throw std::runtime_error("GRAPHICS_RUNTIME_EXECUTION_NOT_AUTHORIZED");
    if(manifest.at("texture_sampling_contract")!="SOURCE_WINS_BILINEAR_SRGB__CAA_FALLBACK_LINEAR_PM" ||
       manifest.at("pixel_coverage_contract")!="SINGLE_HALF_INTEGER_TOP_LEFT_APPROVED_BASELINE" ||
       manifest.at("pixel_coverage_sample_count")!="1" ||
       manifest.at("composition_contract")!="STRAIGHT_SRGB_FRONT_TO_BACK" ||
       manifest.at("depth_equivalence_epsilon_camera_z")!="1e-9" ||
       manifest.at("mip_generation_authorized")!="0")
        throw std::runtime_error("GRAPHICS_SOURCE_WINS_MATERIAL_CONTRACT_INVALID");

    const auto graphics_entry=manifest.at("graphics_entry");
    const auto graphics_it=entries.find(graphics_entry);
    if(graphics_it==entries.end()) throw std::runtime_error("GRAPHICS_ENTRY_MISSING");
    const auto graphics=realsas::graphics_v1::parse(graphics_it->second);
    if(graphics.runtime_graphics_hash!=manifest.at("graphics_binding_hash"))
        throw std::runtime_error("GRAPHICS_RUNTIME_BINDING_HASH_DRIFT");
    if(graphics.canonical_mesh_binding_hash!=manifest.at("mesh_hash"))
        throw std::runtime_error("GRAPHICS_RUNTIME_MESH_BINDING_DRIFT");

    const auto material=parse_m_caa(entries.at(manifest.at("single_m_caa_material_entry")));
    const auto mesh=parse_mesh(entries.at("mesh.bin"));
    if(material.vertices!=mesh.vertex_count || material.topology!=mesh.faces)
        throw std::runtime_error("GRAPHICS_MATERIAL_RUNTIME_DOMAIN_DRIFT");
    if(graphics.face_fragment_index.size()!=mesh.face_count || graphics.views.size()!=8)
        throw std::runtime_error("GRAPHICS_RUNTIME_FACE_OR_VIEW_COUNT_DRIFT");

    const auto cameras=parse_cameras(entries.at("cameras.bin"));
    const auto view_id=arg_value(argc,argv,"--view");
    if(view_id.size()!=2 || view_id[0]!='V') throw std::runtime_error("VIEW_ID_INVALID");
    const int view=parse_int_exact(view_id.substr(1),"VIEW");
    if(view<0 || view>=8 || cameras.size()!=8) throw std::runtime_error("VIEW_INDEX_INVALID");

    std::string clip_entry;
    const auto clip_id=arg_value(argc,argv,"--clip");
    for(int i=0;i<std::stoi(manifest.at("clip_count"));++i)
        if(manifest.at("clip."+std::to_string(i)+".id")==clip_id)
            clip_entry=manifest.at("clip."+std::to_string(i)+".entry");
    if(clip_entry.empty()) throw std::runtime_error("CLIP_NOT_FOUND");
    const auto clip=parse_clip(entries.at(clip_entry));
    const int frame=parse_int_exact(arg_value(argc,argv,"--frame"),"FRAME");
    if(frame<0 || frame>=static_cast<int>(clip.frame_count) || clip.vertex_count!=mesh.vertex_count)
        throw std::runtime_error("FRAME_OR_VERTEX_COUNT_INVALID");

    const int res=static_cast<int>(cameras[view].resolution);
    if(res<=0 || res>4096) throw std::runtime_error("GRAPHICS_OUTPUT_RESOLUTION_INVALID");
    std::vector<Proj> points(mesh.vertex_count);
    for(std::size_t i=0;i<points.size();++i) {
        const auto offset=(static_cast<std::size_t>(frame)*points.size()+i)*3;
        points[i]=project(
            {clip.positions[offset],clip.positions[offset+1],clip.positions[offset+2]},
            cameras[view]
        );
        if(!std::isfinite(points[i].x)||!std::isfinite(points[i].y)||!(points[i].z>0))
            throw std::runtime_error("GRAPHICS_POSED_DOMAIN_INVALID");
    }

    struct Fragment {double depth{};std::uint32_t face{};MCAAColor color;};
    std::vector<std::vector<Fragment>> layers(static_cast<std::size_t>(res)*res);
    const auto tl=[](const Proj& a,const Proj& b){
        return b.y-a.y<0 || (std::abs(b.y-a.y)<=1e-12 && b.x-a.x>0);
    };

    std::size_t invisible_faces=0;
    for(std::size_t f=0;f<mesh.faces.size();++f) {
        if(!realsas::graphics_v1::face_visible(
            graphics,static_cast<std::uint32_t>(view),static_cast<std::uint32_t>(f))) {
            ++invisible_faces;
            continue;
        }
        const auto ids=mesh.faces[f];
        const auto a=points[ids[0]],b=points[ids[1]],c=points[ids[2]];
        const double area=orient(a,b,c.x,c.y);
        if(std::abs(area)<=1e-12) continue;
        const int x0=std::max(0,static_cast<int>(std::floor(std::min({a.x,b.x,c.x})-.5)));
        const int x1=std::min(res-1,static_cast<int>(std::ceil(std::max({a.x,b.x,c.x})-.5)));
        const int y0=std::max(0,static_cast<int>(std::floor(std::min({a.y,b.y,c.y})-.5)));
        const int y1=std::min(res-1,static_cast<int>(std::ceil(std::max({a.y,b.y,c.y})-.5)));
        const double sign=area>0?1:-1;
        const std::array<bool,3> edges=area>0
            ? std::array<bool,3>{tl(b,c),tl(c,a),tl(a,b)}
            : std::array<bool,3>{tl(c,b),tl(a,c),tl(b,a)};
        for(int y=y0;y<=y1;++y) for(int x=x0;x<=x1;++x) {
            const double e0=orient(b,c,x+.5,y+.5);
            const double e1=orient(c,a,x+.5,y+.5);
            const double e2=orient(a,b,x+.5,y+.5);
            if(!edge_accept(sign*e0,edges[0]) ||
               !edge_accept(sign*e1,edges[1]) ||
               !edge_accept(sign*e2,edges[2])) continue;
            const double w0=e0/area,w1=e1/area,w2=1-w0-w1;
            auto color=sample_m_caa(material,f,{w0,w1,w2},view);
            if(color.value[3]<=1e-12) continue;
            const auto pixel=static_cast<std::size_t>(y)*res+x;
            const double depth=w0*a.z+w1*b.z+w2*c.z;
            layers[pixel].push_back({depth,static_cast<std::uint32_t>(f),color});
        }
    }

    std::vector<std::uint8_t> rgba(layers.size()*4,0),provenance(layers.size(),255);
    std::vector<std::int32_t> owners(layers.size(),-1);
    std::vector<std::int16_t> donors(layers.size(),std::numeric_limits<std::int16_t>::min());
    std::size_t overlap=0,overflow=0,explicit_override_pixels=0;

    for(std::size_t p=0;p<layers.size();++p) {
        auto& fragments=layers[p];
        if(fragments.empty()) continue;
        if(fragments.size()>1) ++overlap;
        if(fragments.size()>32) ++overflow;

        std::vector<std::uint32_t> faces;
        std::vector<double> depths;
        faces.reserve(fragments.size());depths.reserve(fragments.size());
        for(const auto& fragment:fragments){faces.push_back(fragment.face);depths.push_back(fragment.depth);}
        const auto order=realsas::graphics_v1::order_pixel_layers(
            graphics,static_cast<std::uint32_t>(view),faces,depths,1e-9);

        auto physical=order;
        std::stable_sort(physical.begin(),physical.end(),[&](std::size_t lhs,std::size_t rhs){
            if(depths[lhs]<depths[rhs]-1e-9)return true;
            if(depths[rhs]<depths[lhs]-1e-9)return false;
            return faces[lhs]<faces[rhs];
        });
        if(order!=physical) ++explicit_override_pixels;

        std::array<double,4> accum{};
        for(const auto index:order) {
            const auto& src=fragments[index].color.value;
            const double transmission=1-accum[3];
            for(int channel=0;channel<3;++channel)
                accum[channel]+=src[channel]*src[3]*transmission;
            accum[3]+=src[3]*transmission;
        }
        if(accum[3]>1e-12)
            for(int channel=0;channel<3;++channel) rgba[p*4+channel]=q8(accum[channel]/accum[3]);
        rgba[p*4+3]=q8(accum[3]);
        const auto front=order.front();
        owners[p]=static_cast<std::int32_t>(fragments[front].face);
        provenance[p]=fragments[front].color.code;
        donors[p]=fragments[front].color.donor;
    }

    if(overflow && !has_flag(argc,argv,"--allow-layer-overflow-diagnostic"))
        throw std::runtime_error("GRAPHICS_FRAGMENT_LAYER_OVERFLOW");

    write_file(arg_value(argc,argv,"--out-rgba"),rgba.data(),rgba.size());
    const auto prov=arg_value(argc,argv,"--out-provenance",false);
    const auto owner=arg_value(argc,argv,"--out-owner",false);
    const auto donor=arg_value(argc,argv,"--out-source-view",false);
    if(!prov.empty()) write_file(prov,provenance.data(),provenance.size());
    if(!owner.empty()) write_file(owner,owners.data(),owners.size()*sizeof(std::int32_t));
    if(!donor.empty()) write_file(donor,donors.data(),donors.size()*sizeof(std::int16_t));

    std::cout<<"renderer=REALSAS_VNEXT_SINGLE_M_GRAPHICS"
             <<" overlap_pixels="<<overlap
             <<" explicit_order_override_pixels="<<explicit_override_pixels
             <<" invisible_faces="<<invisible_faces
             <<" fragment_overflow="<<overflow
             <<" graphics_binding="<<graphics.runtime_graphics_hash
             <<"\n";
    return 0;
}

} // namespace

int main(int argc,char** argv) {
    try {
        if(argc<2) throw std::runtime_error(
            "USAGE: graphics-player package --clip ID --view V0 --frame N --out-rgba PATH [--out-provenance PATH] [--out-source-view PATH] [--out-owner PATH]");
        const auto entries=parse_rss(argv[1]);
        const auto manifest=parse_manifest(entries.at("manifest.txt"));
        if(manifest.at("schema")!="RealSaS.RuntimePackage.v2")
            throw std::runtime_error("GRAPHICS_PACKAGE_SCHEMA_INVALID");
        if(manifest.find("graphics_entry")==manifest.end())
            throw std::runtime_error("GRAPHICS_PACKAGE_CONTENT_REQUIRED");
        return render_single_m_graphics(entries,manifest,argc,argv);
    } catch(const std::exception& e) {
        std::cerr<<"ERROR "<<e.what()<<"\n";
        return 2;
    }
}
