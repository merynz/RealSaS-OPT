// Sealed source-wins + CAA transport on the existing M topology and poses.
// Included by the reference player after its common binary/camera helpers.
template <typename T>
std::vector<T> m_caa_array(const std::vector<std::uint8_t>& bytes, std::size_t& off, std::size_t n) {
    if(n>(bytes.size()-off)/sizeof(T)) throw std::runtime_error("M_CAA_ARRAY_TRUNCATED");
    std::vector<T> result(n);
    if(n) std::memcpy(result.data(),bytes.data()+off,n*sizeof(T));
    off+=n*sizeof(T);
    return result;
}

struct MCAA {
    std::uint32_t vertices{},faces{},samples{},width{},height{};
    std::vector<Vec3> rest;
    std::vector<std::array<std::uint32_t,3>> topology;
    std::array<Camera,8> source_cameras;
    std::vector<std::uint8_t> source,rgba,provenance;
    std::vector<std::int32_t> owner;
    std::vector<double> depth;
    std::vector<std::uint64_t> offsets;
    std::vector<std::uint32_t> resolutions;
    std::vector<std::int16_t> donors;
    std::map<std::uint32_t,std::vector<std::uint32_t>> luts;
};

MCAA parse_m_caa(const std::vector<std::uint8_t>& compressed) {
    std::size_t off=0;
    const auto size=read_scalar<std::uint64_t>(compressed,off);
    if(size>1024ull*1024ull*1024ull) throw std::runtime_error("M_CAA_MATERIAL_BYTE_BUDGET");
    std::vector<std::uint8_t> bytes(size);
    uLongf actual=static_cast<uLongf>(size);
    if(uncompress(bytes.data(),&actual,compressed.data()+off,compressed.size()-off)!=Z_OK||actual!=size)
        throw std::runtime_error("M_CAA_MATERIAL_DECOMPRESS_FAIL");
    off=0;MCAA m;
    m.vertices=read_scalar<std::uint32_t>(bytes,off);m.faces=read_scalar<std::uint32_t>(bytes,off);
    m.samples=read_scalar<std::uint32_t>(bytes,off);m.width=read_scalar<std::uint32_t>(bytes,off);
    m.height=read_scalar<std::uint32_t>(bytes,off);
    if(m.width==0||m.width!=m.height||m.vertices==0||m.faces==0||m.samples==0)
        throw std::runtime_error("M_CAA_MATERIAL_DIMENSIONS_INVALID");
    m.rest=m_caa_array<Vec3>(bytes,off,m.vertices);
    m.topology=m_caa_array<std::array<std::uint32_t,3>>(bytes,off,m.faces);
    for(auto& c:m.source_cameras) {
        c.origin=read_vec3d(bytes,off);c.right=read_vec3d(bytes,off);
        c.up=read_vec3d(bytes,off);c.forward=read_vec3d(bytes,off);
        c.half=read_scalar<double>(bytes,off);c.resolution=read_scalar<std::uint32_t>(bytes,off);
        if(c.resolution!=m.width||!(c.half>0)) throw std::runtime_error("M_CAA_SOURCE_CAMERA_DRIFT");
    }
    const std::size_t pixels=8ull*m.width*m.height;
    m.source=m_caa_array<std::uint8_t>(bytes,off,pixels*4);
    m.owner=m_caa_array<std::int32_t>(bytes,off,pixels);
    m.depth=m_caa_array<double>(bytes,off,pixels);
    m.offsets=m_caa_array<std::uint64_t>(bytes,off,m.faces+1);
    m.resolutions=m_caa_array<std::uint32_t>(bytes,off,m.faces);
    m.rgba=m_caa_array<std::uint8_t>(bytes,off,8ull*m.samples*4);
    m.provenance=m_caa_array<std::uint8_t>(bytes,off,8ull*m.samples);
    m.donors=m_caa_array<std::int16_t>(bytes,off,8ull*m.samples);
    const auto count=read_scalar<std::uint32_t>(bytes,off);
    for(std::uint32_t i=0;i<count;++i) {
        const auto n=read_scalar<std::uint32_t>(bytes,off);
        if(n<4||n>4096) throw std::runtime_error("M_CAA_LATTICE_RESOLUTION_INVALID");
        m.luts[n]=m_caa_array<std::uint32_t>(bytes,off,static_cast<std::size_t>(n)*n);
    }
    if(off!=bytes.size()||m.offsets.front()!=0||m.offsets.back()!=m.samples)
        throw std::runtime_error("M_CAA_MATERIAL_ADDRESS_DRIFT");
    for(std::size_t f=0;f<m.faces;++f) {
        const auto n=m.resolutions[f];
        if(n<4||m.offsets[f+1]-m.offsets[f]!=static_cast<std::uint64_t>(n)*(n+1)/2||!m.luts.count(n))
            throw std::runtime_error("M_CAA_FACE_LATTICE_DRIFT");
        for(auto v:m.topology[f]) if(v>=m.vertices) throw std::runtime_error("M_CAA_TOPOLOGY_INVALID");
        for(auto s:m.luts.at(n)) if(s>=m.offsets[f+1]-m.offsets[f]) throw std::runtime_error("M_CAA_SAMPLE_LUT_INVALID");
    }
    return m;
}

struct MCAAColor {std::array<double,4> value{};std::uint8_t code{255};std::int16_t donor{-32768};};

MCAAColor sample_m_caa(const MCAA& m,std::size_t face,std::array<double,3> bary,int view) {
    double sum=0;for(auto& b:bary) {b=std::clamp(b,0.0,1.0);sum+=b;}
    for(auto& b:bary) b/=sum;
    Vec3 rest{};
    for(int k=0;k<3;++k) {const auto& p=m.rest[m.topology[face][k]];rest.x+=bary[k]*p.x;rest.y+=bary[k]*p.y;rest.z+=bary[k]*p.z;}
    const auto& camera=m.source_cameras[view];const auto p=project(rest,camera);
    const int px=static_cast<int>(std::floor(p.x)),py=static_cast<int>(std::floor(p.y));
    const bool inside=px>=0&&py>=0&&px<static_cast<int>(m.width)&&py<static_cast<int>(m.height);
    const auto pixel=(static_cast<std::size_t>(view)*m.height+std::clamp(py,0,static_cast<int>(m.height)-1))*m.width+std::clamp(px,0,static_cast<int>(m.width)-1);
    const bool visible=inside&&m.owner[pixel]>=0&&std::abs(p.z-m.depth[pixel])<=2*camera.half/camera.resolution;
    const double sx=std::clamp(p.x-.5,0.0,static_cast<double>(m.width-1));
    const double sy=std::clamp(p.y-.5,0.0,static_cast<double>(m.height-1));
    const int x0=static_cast<int>(std::floor(sx)),y0=static_cast<int>(std::floor(sy));
    const int x1=std::min(x0+1,static_cast<int>(m.width)-1),y1=std::min(y0+1,static_cast<int>(m.height)-1);
    const double fx=sx-x0,fy=sy-y0;
    MCAAColor out;
    for(int channel=0;channel<4;++channel) {
        const auto texel=[&](int x,int y){return static_cast<double>(m.source[(((static_cast<std::size_t>(view)*m.height+y)*m.width+x)*4)+channel]);};
        out.value[channel]=((texel(x0,y0)*(1-fx)+texel(x1,y0)*fx)*(1-fy)+(texel(x0,y1)*(1-fx)+texel(x1,y1)*fx)*fy)/255;
    }
    if(visible&&out.value[3]>1e-12) {out.code=0;out.donor=static_cast<std::int16_t>(view);return out;}
    out.value={};
    const auto n=m.resolutions[face];const auto& lut=m.luts.at(n);
    const double x=std::clamp(bary[1]*(n-1),0.0,static_cast<double>(n-1));
    const double y=std::clamp(bary[2]*(n-1),0.0,static_cast<double>(n-1));
    const auto ix=static_cast<std::uint32_t>(std::floor(x)),iy=static_cast<std::uint32_t>(std::floor(y));
    const auto jx=std::min(ix+1,n-1),jy=std::min(iy+1,n-1);
    const double tx=x-ix,ty=y-iy;
    const std::array<double,4> weights{(1-tx)*(1-ty),tx*(1-ty),(1-tx)*ty,tx*ty};
    const std::array<std::uint32_t,4> ids{lut[iy*n+ix],lut[iy*n+jx],lut[jy*n+ix],lut[jy*n+jx]};
    constexpr std::array<int,5> ranks{0,1,2,4,3};constexpr std::array<std::uint8_t,5> codes{0,1,2,4,3};
    std::array<double,4> pm{};int worst=0;
    for(int i=0;i<4;++i) {
        const auto id=static_cast<std::size_t>(view)*m.samples+m.offsets[face]+ids[i];
        const auto code=m.provenance[id];if(code>4) throw std::runtime_error("M_CAA_PROVENANCE_INVALID");
        const double alpha=m.rgba[id*4+3]/255.0;
        for(int c=0;c<3;++c) pm[c]+=srgb_to_linear(m.rgba[id*4+c]/255.0)*alpha*weights[i];
        pm[3]+=alpha*weights[i];
        if(weights[i]>1e-12) {worst=std::max(worst,ranks[code]);out.donor=merge_source_view_diagnostic(out.donor,m.donors[id]);}
    }
    out.code=codes[worst];
    if(out.code==3) {out.donor=-4;return out;}
    for(int c=0;c<3;++c) out.value[c]=q8(pm[3]>1e-12?linear_to_srgb(pm[c]/pm[3]):0)/255.0;
    out.value[3]=q8(pm[3])/255.0;
    return out;
}

int render_single_m_native(const Entries& entries,const std::unordered_map<std::string,std::string>& manifest,int argc,char** argv) {
    if(manifest.at("texture_sampling_contract")!="SOURCE_WINS_BILINEAR_SRGB__CAA_FALLBACK_LINEAR_PM"||
       manifest.at("pixel_coverage_contract")!="SINGLE_HALF_INTEGER_TOP_LEFT_APPROVED_BASELINE"||
       manifest.at("pixel_coverage_sample_count")!="1"||manifest.at("composition_contract")!="STRAIGHT_SRGB_FRONT_TO_BACK"||
       manifest.at("depth_equivalence_epsilon_camera_z")!="1e-9"||manifest.at("mip_generation_authorized")!="0")
        throw std::runtime_error("M_CAA_RENDER_CONTRACT_INVALID");
    const auto material=parse_m_caa(entries.at(manifest.at("single_m_caa_material_entry")));
    const auto mesh=parse_mesh(entries.at("mesh.bin"));
    if(material.vertices!=mesh.vertex_count||material.topology!=mesh.faces) throw std::runtime_error("M_CAA_RUNTIME_DOMAIN_DRIFT");
    const auto cameras=parse_cameras(entries.at("cameras.bin"));
    const auto view_id=arg_value(argc,argv,"--view");
    if(view_id.size()!=2||view_id[0]!='V') throw std::runtime_error("VIEW_ID_INVALID");
    const int view=parse_int_exact(view_id.substr(1),"VIEW");
    if(view<0||view>=8||cameras.size()!=8) throw std::runtime_error("VIEW_INDEX_INVALID");
    std::string clip_entry;const auto clip_id=arg_value(argc,argv,"--clip");
    for(int i=0;i<std::stoi(manifest.at("clip_count"));++i) if(manifest.at("clip."+std::to_string(i)+".id")==clip_id) clip_entry=manifest.at("clip."+std::to_string(i)+".entry");
    if(clip_entry.empty()) throw std::runtime_error("CLIP_NOT_FOUND");
    const auto clip=parse_clip(entries.at(clip_entry));
    const int frame=parse_int_exact(arg_value(argc,argv,"--frame"),"FRAME");
    if(frame<0||frame>=static_cast<int>(clip.frame_count)||clip.vertex_count!=mesh.vertex_count) throw std::runtime_error("FRAME_OR_VERTEX_COUNT_INVALID");
    const int res=static_cast<int>(cameras[view].resolution);
    if(res<=0||res>4096) throw std::runtime_error("M_CAA_OUTPUT_RESOLUTION_INVALID");
    std::vector<Proj> points(mesh.vertex_count);
    for(std::size_t i=0;i<points.size();++i) {
        const auto offset=(static_cast<std::size_t>(frame)*points.size()+i)*3;
        points[i]=project({clip.positions[offset],clip.positions[offset+1],clip.positions[offset+2]},cameras[view]);
        if(!std::isfinite(points[i].x)||!std::isfinite(points[i].y)||!(points[i].z>0)) throw std::runtime_error("M_CAA_POSED_DOMAIN_INVALID");
    }
    struct Fragment {double depth{};std::uint32_t face{};MCAAColor color;};
    std::vector<std::vector<Fragment>> layers(static_cast<std::size_t>(res)*res);
    std::vector<double> geometry_depth(layers.size(),std::numeric_limits<double>::infinity());
    std::vector<std::uint8_t> geometry_provenance(layers.size(),255);
    const auto tl=[](const Proj& a,const Proj& b){return b.y-a.y<0||(std::abs(b.y-a.y)<=1e-12&&b.x-a.x>0);};
    for(std::size_t f=0;f<mesh.faces.size();++f) {
        const auto ids=mesh.faces[f];const auto a=points[ids[0]],b=points[ids[1]],c=points[ids[2]];
        const double area=orient(a,b,c.x,c.y);if(std::abs(area)<=1e-12) continue;
        const int x0=std::max(0,static_cast<int>(std::floor(std::min({a.x,b.x,c.x})-.5)));
        const int x1=std::min(res-1,static_cast<int>(std::ceil(std::max({a.x,b.x,c.x})-.5)));
        const int y0=std::max(0,static_cast<int>(std::floor(std::min({a.y,b.y,c.y})-.5)));
        const int y1=std::min(res-1,static_cast<int>(std::ceil(std::max({a.y,b.y,c.y})-.5)));
        const double sign=area>0?1:-1;
        const std::array<bool,3> edges=area>0?std::array<bool,3>{tl(b,c),tl(c,a),tl(a,b)}:std::array<bool,3>{tl(c,b),tl(a,c),tl(b,a)};
        for(int y=y0;y<=y1;++y) for(int x=x0;x<=x1;++x) {
            const double e0=orient(b,c,x+.5,y+.5),e1=orient(c,a,x+.5,y+.5),e2=orient(a,b,x+.5,y+.5);
            if(!edge_accept(sign*e0,edges[0])||!edge_accept(sign*e1,edges[1])||!edge_accept(sign*e2,edges[2])) continue;
            const double w0=e0/area,w1=e1/area,w2=1-w0-w1;
            auto color=sample_m_caa(material,f,{w0,w1,w2},view);
            const auto pixel=static_cast<std::size_t>(y)*res+x;
            const double depth=w0*a.z+w1*b.z+w2*c.z;
            if(depth<geometry_depth[pixel]) {geometry_depth[pixel]=depth;geometry_provenance[pixel]=color.code;}
            if(color.value[3]>1e-12) layers[pixel].push_back({depth,static_cast<std::uint32_t>(f),color});
        }
    }
    std::vector<std::uint8_t> rgba(layers.size()*4),provenance=geometry_provenance;
    std::vector<std::int32_t> owners(layers.size(),-1);std::vector<std::int16_t> donors(layers.size(),-32768);
    std::size_t ties=0,overflow=0,overlap=0;
    for(std::size_t p=0;p<layers.size();++p) {
        auto& fragments=layers[p];if(fragments.empty()) continue;
        std::stable_sort(fragments.begin(),fragments.end(),[](const Fragment& a,const Fragment& b){return a.depth<b.depth;});
        std::array<double,4> accum{};
        for(std::size_t i=0;i<fragments.size();++i) {
            if(i>0&&std::abs(fragments[i].depth-fragments[i-1].depth)<=1e-9) ++ties;
            if(i>=32) ++overflow;
            const auto& src=fragments[i].color.value;const double transmission=1-accum[3];
            for(int c=0;c<3;++c) accum[c]+=src[c]*src[3]*transmission;
            accum[3]+=src[3]*transmission;
        }
        if(fragments.size()>1) ++overlap;
        for(int c=0;c<3;++c) rgba[p*4+c]=q8(accum[c]/accum[3]);
        rgba[p*4+3]=q8(accum[3]);
        owners[p]=static_cast<std::int32_t>(fragments.front().face);
        provenance[p]=fragments.front().color.code;donors[p]=fragments.front().color.donor;
    }
    if(overflow&&!has_flag(argc,argv,"--allow-layer-overflow-diagnostic")) throw std::runtime_error("M_CAA_FRAGMENT_LAYER_OVERFLOW");
    write_file(arg_value(argc,argv,"--out-rgba"),rgba.data(),rgba.size());
    const auto prov=arg_value(argc,argv,"--out-provenance",false),owner=arg_value(argc,argv,"--out-owner",false),donor=arg_value(argc,argv,"--out-source-view",false);
    if(!prov.empty()) write_file(prov,provenance.data(),provenance.size());
    if(!owner.empty()) write_file(owner,owners.data(),owners.size()*sizeof(std::int32_t));
    if(!donor.empty()) write_file(donor,donors.data(),donors.size()*sizeof(std::int16_t));
    std::cout<<"renderer=REALSAS_V2_SINGLE_M_SOURCE_WINS_CAA depth_ties="<<ties<<" fragment_overflow="<<overflow<<" overlap_pixels="<<overlap<<"\n";
    return 0;
}
