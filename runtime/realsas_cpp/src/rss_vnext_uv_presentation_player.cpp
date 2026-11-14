// RSS.vNext native preview consumer: immutable CAS Graphics + canonical UV
// candidate transport + projected deformed canonical mesh -> 2D RGBA.
//
// The .rss packaging bridge will supply these three already-typed assets.
// This executable never claims observed camera rays, source-occlusion truth,
// animation quality or ProductRevision PASS.
#include "rss_vnext_uv_presentation_v1.hpp"
#include <array>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <iterator>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

using realsas::rss_uv_v1::Projected;
using realsas::rss_uv_v1::Texture;
using realsas::graphics_v1::State;

namespace {
std::vector<std::uint8_t> read_file(const char* filename,std::size_t max_bytes){
    std::ifstream file(filename,std::ios::binary);
    if(!file)throw std::runtime_error("RSS_UV_NATIVE_FILE_NOT_FOUND");
    file.seekg(0,std::ios::end);
    const auto size=file.tellg();
    if(size<0||static_cast<std::uint64_t>(size)>max_bytes)
        throw std::runtime_error("RSS_UV_NATIVE_INPUT_RESOURCE_LIMIT");
    file.seekg(0,std::ios::beg);
    return std::vector<std::uint8_t>(
        std::istreambuf_iterator<char>(file),std::istreambuf_iterator<char>()
    );
}
template <typename T>
void write_file(const char* filename,const std::vector<T>& values){
    std::ofstream file(filename,std::ios::binary);
    if(!file)throw std::runtime_error("RSS_UV_NATIVE_OUTPUT_OPEN_FAILED");
    file.write(reinterpret_cast<const char*>(values.data()),
               static_cast<std::streamsize>(values.size()*sizeof(T)));
    if(!file)throw std::runtime_error("RSS_UV_NATIVE_OUTPUT_WRITE_FAILED");
}
struct PoseReader {
    const std::vector<std::uint8_t>& input;
    std::size_t cursor{0};
    void need(std::size_t count)const{
        if(cursor>input.size()||count>input.size()-cursor)
            throw std::runtime_error("RSS_UV_NATIVE_POSE_TRUNCATED");
    }
    std::uint32_t u32(){
        need(4);
        const auto* ptr=input.data()+cursor;
        cursor+=4;
        return static_cast<std::uint32_t>(ptr[0]) |
            (static_cast<std::uint32_t>(ptr[1])<<8) |
            (static_cast<std::uint32_t>(ptr[2])<<16) |
            (static_cast<std::uint32_t>(ptr[3])<<24);
    }
    double f64(){
        need(8);
        std::uint64_t v=0;
        for(int i=0;i<8;++i)v|=static_cast<std::uint64_t>(input[cursor+i])<<(8*i);
        cursor+=8;
        double out{};
        static_assert(sizeof(out)==8,"IEEE float64 required");
        std::memcpy(&out,&v,8);
        if(!std::isfinite(out))throw std::runtime_error("RSS_UV_NATIVE_POSE_NONFINITE");
        return out;
    }
};
struct Poses {
    std::uint32_t width{},height{};
    std::vector<std::array<std::uint32_t,3>> faces;
    std::vector<std::vector<Projected>> frames;
};
Poses parse_poses(const std::vector<std::uint8_t>& input){
    static const char magic[8]={'R','S','P','O','S','E','0','1'};
    if(input.size()<32||std::memcmp(input.data(),magic,8)!=0)
        throw std::runtime_error("RSS_UV_NATIVE_POSE_MAGIC");
    PoseReader r{input,8};
    if(r.u32()!=1)throw std::runtime_error("RSS_UV_NATIVE_POSE_VERSION");
    const auto vertices=r.u32(),faces=r.u32(),frames=r.u32();
    Poses result;
    result.width=r.u32();result.height=r.u32();
    if(
        vertices<3||vertices>65536||faces<1||faces>4096||
        frames<1||frames>512||result.width<1||result.height<1||
        result.width>2048||result.height>2048
    )throw std::runtime_error("RSS_UV_NATIVE_POSE_DOMAIN");
    const std::size_t expected=32u+static_cast<std::size_t>(faces)*12u+
        static_cast<std::size_t>(frames)*vertices*24u;
    if(expected!=input.size())
        throw std::runtime_error("RSS_UV_NATIVE_POSE_BYTE_COUNT");
    result.faces.resize(faces);
    for(auto& face:result.faces)for(auto& vertex:face){
        vertex=r.u32();
        if(vertex>=vertices)throw std::runtime_error("RSS_UV_NATIVE_POSE_FACE_INDEX");
    }
    result.frames.resize(frames,std::vector<Projected>(vertices));
    for(auto& frame:result.frames)for(auto& vertex:frame){
        vertex={r.f64(),r.f64(),r.f64()};
        if(!(vertex.z>0.))throw std::runtime_error("RSS_UV_NATIVE_POSE_Z");
    }
    return result;
}
int parse_index(const char* text){
    const std::string value(text);
    if(value.empty()||value.size()>5)
        throw std::runtime_error("RSS_UV_NATIVE_ARG_DOMAIN");
    const auto v=std::stoi(value);
    if(std::to_string(v)!=value)
        throw std::runtime_error("RSS_UV_NATIVE_ARG_NONCANONICAL");
    return v;
}

// Native archive consumers must verify the same SHA-256 entry receipts as
// Python reopen. This implementation is freestanding (the preview player is
// compiled with g++ alone) and hashes the input without a padded byte copy.
std::string rss_preview_sha256_hex(const std::vector<std::uint8_t>& bytes){
    static constexpr std::uint32_t k[64]={
        0x428a2f98u,0x71374491u,0xb5c0fbcfu,0xe9b5dba5u,
        0x3956c25bu,0x59f111f1u,0x923f82a4u,0xab1c5ed5u,
        0xd807aa98u,0x12835b01u,0x243185beu,0x550c7dc3u,
        0x72be5d74u,0x80deb1feu,0x9bdc06a7u,0xc19bf174u,
        0xe49b69c1u,0xefbe4786u,0x0fc19dc6u,0x240ca1ccu,
        0x2de92c6fu,0x4a7484aau,0x5cb0a9dcu,0x76f988dau,
        0x983e5152u,0xa831c66du,0xb00327c8u,0xbf597fc7u,
        0xc6e00bf3u,0xd5a79147u,0x06ca6351u,0x14292967u,
        0x27b70a85u,0x2e1b2138u,0x4d2c6dfcu,0x53380d13u,
        0x650a7354u,0x766a0abbu,0x81c2c92eu,0x92722c85u,
        0xa2bfe8a1u,0xa81a664bu,0xc24b8b70u,0xc76c51a3u,
        0xd192e819u,0xd6990624u,0xf40e3585u,0x106aa070u,
        0x19a4c116u,0x1e376c08u,0x2748774cu,0x34b0bcb5u,
        0x391c0cb3u,0x4ed8aa4au,0x5b9cca4fu,0x682e6ff3u,
        0x748f82eeu,0x78a5636fu,0x84c87814u,0x8cc70208u,
        0x90befffau,0xa4506cebu,0xbef9a3f7u,0xc67178f2u
    };
    std::array<std::uint32_t,8> h={{
        0x6a09e667u,0xbb67ae85u,0x3c6ef372u,0xa54ff53au,
        0x510e527fu,0x9b05688cu,0x1f83d9abu,0x5be0cd19u
    }};
    const auto rotate=[](std::uint32_t x,unsigned s){
        return (x>>s)|(x<<(32u-s));
    };
    if(bytes.size()>160u*1024u*1024u)
        throw std::runtime_error("RSS_UV_PREVIEW_DIGEST_RESOURCE_BUDGET");
    const std::uint64_t bit_length=static_cast<std::uint64_t>(bytes.size())*8u;
    const std::size_t padded=((bytes.size()+9u+63u)/64u)*64u;
    for(std::size_t base=0;base<padded;base+=64u){
        std::uint32_t words[64]={};
        for(std::size_t i=0;i<64u;++i){
            const std::size_t index=base+i;
            std::uint8_t byte=0;
            if(index<bytes.size())byte=bytes[index];
            else if(index==bytes.size())byte=0x80u;
            else if(index>=padded-8u){
                const auto shift=static_cast<unsigned>(
                    8u*(padded-1u-index)
                );
                byte=static_cast<std::uint8_t>((bit_length>>shift)&0xffu);
            }
            words[i/4u]|=static_cast<std::uint32_t>(byte)
                <<static_cast<unsigned>(24u-8u*(i%4u));
        }
        for(std::size_t i=16;i<64u;++i){
            const auto a=words[i-15u],b=words[i-2u];
            const auto s0=rotate(a,7u)^rotate(a,18u)^(a>>3u);
            const auto s1=rotate(b,17u)^rotate(b,19u)^(b>>10u);
            words[i]=words[i-16u]+s0+words[i-7u]+s1;
        }
        auto a=h[0],b=h[1],c=h[2],d=h[3],
             e=h[4],f=h[5],g=h[6],z=h[7];
        for(std::size_t i=0;i<64u;++i){
            const auto s1=rotate(e,6u)^rotate(e,11u)^rotate(e,25u);
            const auto choice=(e&f)^(~e&g);
            const auto t1=z+s1+choice+k[i]+words[i];
            const auto s0=rotate(a,2u)^rotate(a,13u)^rotate(a,22u);
            const auto majority=(a&b)^(a&c)^(b&c);
            const auto t2=s0+majority;
            z=g;g=f;f=e;e=d+t1;d=c;c=b;b=a;a=t1+t2;
        }
        h[0]+=a;h[1]+=b;h[2]+=c;h[3]+=d;
        h[4]+=e;h[5]+=f;h[6]+=g;h[7]+=z;
    }
    static constexpr char digits[]="0123456789abcdef";
    std::string result;
    result.reserve(64);
    for(auto word:h)for(int shift=28;shift>=0;shift-=4)
        result.push_back(digits[(word>>shift)&0x0fu]);
    return result;
}

using RSSPreviewEntries=std::map<std::string,std::vector<std::uint8_t>>;
RSSPreviewEntries parse_existing_rss_archive_v1(
    const std::vector<std::uint8_t>& archive
){
    static constexpr char magic[8]={'R','S','A','S','V','2','R','1'};
    if(archive.size()<12||std::memcmp(archive.data(),magic,8)!=0)
        throw std::runtime_error("RSS_UV_PREVIEW_ARCHIVE_MAGIC");
    std::size_t pos=8;
    auto need=[&](std::size_t size){
        if(pos>archive.size()||size>archive.size()-pos)
            throw std::runtime_error("RSS_UV_PREVIEW_ARCHIVE_TRUNCATED");
    };
    auto u16=[&](){
        need(2);
        const auto value=static_cast<std::uint16_t>(archive[pos])|
            static_cast<std::uint16_t>(archive[pos+1]<<8);
        pos+=2;
        return value;
    };
    auto u32=[&](){
        need(4);
        std::uint32_t result=0;
        for(unsigned i=0;i<4;++i)
            result|=static_cast<std::uint32_t>(archive[pos+i])<<(i*8);
        pos+=4;
        return result;
    };
    auto u64=[&](){
        need(8);
        std::uint64_t result=0;
        for(unsigned i=0;i<8;++i)
            result|=static_cast<std::uint64_t>(archive[pos+i])<<(i*8);
        pos+=8;
        return result;
    };
    if(u32()!=8)throw std::runtime_error("RSS_UV_PREVIEW_ARCHIVE_ENTRY_COUNT");
    const std::set<std::string> required={
        "manifest.txt","mesh.json","rig.json","skin.json",
        "graphics.bin","texture.bin","pose.bin","controls.json",
    };
    RSSPreviewEntries entries;
    for(int i=0;i<8;++i){
        const auto name_length=u16();
        if(!name_length||name_length>128)
            throw std::runtime_error("RSS_UV_PREVIEW_ENTRY_NAME_LENGTH");
        need(name_length);
        const std::string name(
            reinterpret_cast<const char*>(archive.data()+pos),name_length
        );
        pos+=name_length;
        const auto length=u64();
        if(!required.count(name)||entries.count(name)||
           length>128u*1024u*1024u)
            throw std::runtime_error("RSS_UV_PREVIEW_UNKNOWN_OR_DUPLICATE_ENTRY");
        need(static_cast<std::size_t>(length));
        entries[name]=std::vector<std::uint8_t>(
            archive.begin()+static_cast<std::ptrdiff_t>(pos),
            archive.begin()+static_cast<std::ptrdiff_t>(pos+length)
        );
        pos+=static_cast<std::size_t>(length);
    }
    if(entries.size()!=required.size()||pos!=archive.size())
        throw std::runtime_error("RSS_UV_PREVIEW_ARCHIVE_TRAILING_OR_MISSING");
    return entries;
}
std::map<std::string,std::string> preview_manifest(
    const std::vector<std::uint8_t>& bytes
){
    const std::string source(bytes.begin(),bytes.end());
    if(source.size()>8192)
        throw std::runtime_error("RSS_UV_PREVIEW_MANIFEST_SIZE");
    std::map<std::string,std::string> result;
    std::size_t start=0;
    while(start<source.size()){
        auto end=source.find('\n',start);
        if(end==std::string::npos)end=source.size();
        const auto line=source.substr(start,end-start);
        if(!line.empty()){
            const auto eq=line.find('=');
            if(eq==std::string::npos||eq==0||
               !result.emplace(line.substr(0,eq),line.substr(eq+1)).second)
                throw std::runtime_error("RSS_UV_PREVIEW_MANIFEST_DUPLICATE");
        }
        start=end+1;
    }
    if(
        result.at("schema")!="RealSaS.RSSvNext.NativeInteractivePuppetPreview.v1"
        ||result.at("admission")!="DEVELOPER_PREVIEW_NOT_PRODUCT_REVISION"
        ||result.at("product_revision_minted")!="0"
        ||result.at("original_source_camera_observed")!="0"
        ||result.at("source_rgb_surface_ownership_qualified")!="0"
        ||result.at("native_texture_preview_rendering_enabled")!="1"
        ||result.at("view_index")!="0"
    )throw std::runtime_error("RSS_UV_PREVIEW_FALSE_SOURCE_OR_PRODUCT_AUTHORITY");
    return result;
}

int self_test(){
    if(rss_preview_sha256_hex({})!=
           "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
       ||rss_preview_sha256_hex({'a','b','c'})!=
           "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")
        throw std::runtime_error("RSS_UV_PREVIEW_SHA256_SELFTEST");
    Texture tex;
    tex.width=8;tex.height=8;tex.faces=2;
    tex.mesh_hash=std::string(64,'a');
    tex.face_uv={
        {{{0.1875f,0.1875f},{0.4375f,0.1875f},{0.1875f,0.4375f}}},
        {{{0.5625f,0.1875f},{0.8125f,0.1875f},{0.5625f,0.4375f}}},
    };
    tex.source_rgba.resize(2*8*8*4,0);
    tex.source_mask.resize(2*8*8,0);
    tex.inferred_rgba.resize(8*8*4,0);
    tex.inferred_mask.resize(8*8,0);
    tex.conflict.resize(8*8,0);
    for(int y=0;y<8;++y)for(int x=0;x<8;++x){
        const auto p=y*8+x;
        if((x>=1&&x<=3)||(x>=4&&x<=6)){
            tex.inferred_mask[p]=1;
            tex.inferred_rgba[4*p]=(x<4?255:0);
            tex.inferred_rgba[4*p+2]=(x>=4?255:0);
            tex.inferred_rgba[4*p+3]=255;
        }
        if(x>=1&&x<=3){
            tex.source_mask[p]=1;
            tex.source_rgba[4*p+1]=255;
            tex.source_rgba[4*p+3]=255;
        }
    }
    State graphics;
    graphics.canonical_mesh_binding_hash=std::string(64,'a');
    graphics.face_fragment_index={0,1};
    graphics.fragment_ids={"red","blue"};
    graphics.views.resize(8);
    for(unsigned i=0;i<8;++i){
        graphics.views[i].view_index=i;
        graphics.views[i].visible_fragment={1,1};
        graphics.views[i].topological_order={0,1};
        graphics.views[i].precedes={0,0,0,0};
    }
    const std::vector<std::array<std::uint32_t,3>> faces={
        {{0,1,2}},{{3,4,5}},
    };
    const auto posed=[](double z0,double z1,double shift){
        return std::vector<Projected>{
            {10+shift,10,z0},{54+shift,10,z0},{10+shift,54,z0},
            {10+shift,10,z1},{54+shift,10,z1},{10+shift,54,z1},
        };
    };
    const auto at=[](const auto& img,int x,int y){
        const std::size_t idx=4*(static_cast<std::size_t>(y)*img.width+x);
        return std::array<int,4>{
            img.rgba[idx],img.rgba[idx+1],img.rgba[idx+2],img.rgba[idx+3]
        };
    };
    auto original=realsas::rss_uv_v1::render(graphics,tex,faces,
        posed(1.,2.,0),64,64,0,-1);
    auto back=realsas::rss_uv_v1::render(graphics,tex,faces,
        posed(2.,1.,0),64,64,0,-1);
    if(at(original,20,20)!=std::array<int,4>{255,0,0,255}||
       at(back,20,20)!=std::array<int,4>{0,0,255,255}||
       original.overlapping_pixels==0)
        throw std::runtime_error("RSS_UV_NATIVE_POSE_DEPTH_ORDER_TEST_FAIL");
    graphics.views[0].precedes[2]=1; // fragment #1 in front of fragment #0
    auto override=realsas::rss_uv_v1::render(graphics,tex,faces,
        posed(1.,2.,0),64,64,0,-1);
    if(at(override,20,20)!=std::array<int,4>{0,0,255,255}||
       override.explicit_order_overrides==0)
        throw std::runtime_error("RSS_UV_NATIVE_GRAPHICS_PRECEDENCE_TEST_FAIL");
    graphics.views[0].precedes[2]=0;
    auto source=realsas::rss_uv_v1::render(graphics,tex,faces,
        posed(1.,2.,0),64,64,0,0);
    if(at(source,20,20)!=std::array<int,4>{0,255,0,255}||
       source.color_authority[20*64+20]!=1)
        throw std::runtime_error("RSS_UV_NATIVE_SOURCE_LAYER_POLICY_TEST_FAIL");
    auto unknown_source=realsas::rss_uv_v1::render(graphics,tex,faces,
        posed(1.,2.,0),64,64,0,1);
    if(at(unknown_source,20,20)!=std::array<int,4>{255,0,0,255}||
       unknown_source.color_authority[20*64+20]!=2)
        throw std::runtime_error("RSS_UV_NATIVE_UNKNOWN_SOURCE_FALLBACK_TEST_FAIL");
    auto moved=realsas::rss_uv_v1::render(graphics,tex,faces,
        posed(1.,2.,12.),64,64,0,-1);
    if(at(moved,20,20)==at(original,20,20)||at(moved,42,20)!=at(original,20,20))
        throw std::runtime_error("RSS_UV_NATIVE_POSED_TRANSLATION_TEST_FAIL");
    std::cout<<"RSS_UV_NATIVE_POSED_DEPTH_AND_PARTIAL_ORDER_PASS\n";
    return 0;
}
}
int main(int argc,char** argv){
    try {
        if(argc==2&&std::string(argv[1])=="--self-test")return self_test();
        const bool archive_mode=argc>1&&std::string(argv[1])=="--rss";
        if(archive_mode ? (argc!=7&&argc!=9) : (argc!=8&&argc!=10))
            throw std::runtime_error(
                "USAGE: rss_uv_player graphics.bin uv.bin pose.bin frame view"
                " source_view output.rgba [owner.i32 authority.u8]"
                " OR --rss preview.rss frame view source_view output.rgba"
                " [owner.i32 authority.u8]"
            );
        std::vector<std::uint8_t> graphics_blob,texture_blob,pose_blob;
        int frame=0,view=0,source_view=-1;
        const char* output_path=nullptr;
        const char* owner_path=nullptr;
        const char* authority_path=nullptr;
        std::map<std::string,std::string> manifest;
        if(archive_mode){
            const auto entries=parse_existing_rss_archive_v1(
                read_file(argv[2],160u*1024u*1024u)
            );
            manifest=preview_manifest(entries.at("manifest.txt"));
            for(const auto& row:entries){
                if(row.first=="manifest.txt")continue;
                const auto key=std::string("sha256:")+row.first;
                const auto expected=manifest.find(key);
                if(expected==manifest.end()||
                   expected->second!=rss_preview_sha256_hex(row.second))
                    throw std::runtime_error(
                        "RSS_UV_PREVIEW_ENTRY_SHA256_MISMATCH:"+row.first
                    );
            }
            graphics_blob=entries.at("graphics.bin");
            texture_blob=entries.at("texture.bin");
            pose_blob=entries.at("pose.bin");
            frame=parse_index(argv[3]);
            view=parse_index(argv[4]);
            source_view=parse_index(argv[5]);
            output_path=argv[6];
            if(argc==9){owner_path=argv[7];authority_path=argv[8];}
            if(view!=0)
                throw std::runtime_error("RSS_UV_PREVIEW_SINGLE_PROJECTED_VIEW_ONLY");
        }else{
            graphics_blob=read_file(argv[1],16u*1024u*1024u);
            texture_blob=read_file(argv[2],128u*1024u*1024u);
            pose_blob=read_file(argv[3],64u*1024u*1024u);
            frame=parse_index(argv[4]);
            view=parse_index(argv[5]);
            source_view=parse_index(argv[6]);
            output_path=argv[7];
            if(argc==10){owner_path=argv[8];authority_path=argv[9];}
        }
        const auto graphics=realsas::graphics_v1::parse(graphics_blob);
        const auto texture=realsas::rss_uv_v1::parse_texture(texture_blob);
        const auto poses=parse_poses(pose_blob);
        if(frame<0||frame>=static_cast<int>(poses.frames.size())||
           view<0||view>=8||source_view < -1||source_view>1)
            throw std::runtime_error("RSS_UV_NATIVE_ARGUMENT_RANGE");
        if(archive_mode){
            if(
                manifest.at("graphics_runtime_hash")!=graphics.runtime_graphics_hash
                ||manifest.at("canonical_mesh_lineage_hash")!=texture.mesh_hash
                ||parse_index(manifest.at("frame_count").c_str())
                    !=static_cast<int>(poses.frames.size())
            )throw std::runtime_error("RSS_UV_PREVIEW_MANIFEST_RUNTIME_BINDING_DRIFT");
        }
        const auto output=realsas::rss_uv_v1::render(
            graphics,texture,poses.faces,poses.frames[frame],
            poses.width,poses.height,static_cast<std::uint32_t>(view),source_view
        );
        write_file(output_path,output.rgba);
        if(owner_path){
            write_file(owner_path,output.front_face);
            write_file(authority_path,output.color_authority);
        }
        std::uint64_t source_pixels=0,predicted_pixels=0;
        for(const auto authority:output.color_authority){
            if(authority==1)++source_pixels;
            if(authority==2)++predicted_pixels;
        }
        std::cout<<"RSS_UV_NATIVE_POSED_2D_RENDER_PASS"
                 <<" frame="<<frame<<" view="<<view
                 <<" source_layer="<<source_view
                 <<" rss_archive="<<(archive_mode?1:0)
                 <<" overlap_pixels="<<output.overlapping_pixels
                 <<" graphics_order_overrides="<<output.explicit_order_overrides
                 <<" source_candidate_pixels="<<source_pixels
                 <<" inferred_pixels="<<predicted_pixels
                 <<" unknown_pixels="<<output.uncolored_pixels
                 <<"\n";
        return 0;
    }catch(const std::exception& exc){
        std::cerr<<"ERROR "<<exc.what()<<"\n";
        return 2;
    }
}
