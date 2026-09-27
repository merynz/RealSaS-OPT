#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include <png.h>

namespace {

struct Image {
    std::uint32_t width=0,height=0;
    std::vector<std::uint8_t> rgba;
};

struct Mesh {
    std::uint32_t source_width=0,source_height=0;
    std::vector<std::array<double,2>> uv;
    std::vector<std::array<std::uint32_t,3>> faces;
};

template <class T>
T read_scalar(std::ifstream& in) {
    T value{};
    in.read(reinterpret_cast<char*>(&value), sizeof(T));
    if(!in) throw std::runtime_error("BINARY_READ_FAILED");
    return value;
}

Mesh read_mesh(const std::string& path) {
    std::ifstream in(path,std::ios::binary);
    if(!in) throw std::runtime_error("MESH_OPEN_FAILED");
    char magic[8]{};
    in.read(magic,8);
    if(!in || std::string(magic,magic+5)!="RSVM1")
        throw std::runtime_error("MESH_MAGIC_INVALID");
    Mesh mesh;
    mesh.source_width=read_scalar<std::uint32_t>(in);
    mesh.source_height=read_scalar<std::uint32_t>(in);
    const auto vertex_count=read_scalar<std::uint32_t>(in);
    const auto face_count=read_scalar<std::uint32_t>(in);
    if(vertex_count<3||face_count<1||mesh.source_width<1||mesh.source_height<1)
        throw std::runtime_error("MESH_HEADER_INVALID");
    mesh.uv.resize(vertex_count);
    for(auto& row:mesh.uv) {
        row[0]=read_scalar<double>(in);
        row[1]=read_scalar<double>(in);
        if(!std::isfinite(row[0])||!std::isfinite(row[1]))
            throw std::runtime_error("MESH_UV_NONFINITE");
    }
    mesh.faces.resize(face_count);
    for(auto& face:mesh.faces) {
        for(auto& index:face) {
            index=read_scalar<std::uint32_t>(in);
            if(index>=vertex_count) throw std::runtime_error("MESH_FACE_INDEX_INVALID");
        }
    }
    return mesh;
}

std::vector<std::array<double,2>> read_positions(
    const std::string& path,
    std::uint32_t expected_vertices,
    int frame_index
) {
    std::ifstream in(path,std::ios::binary);
    if(!in) throw std::runtime_error("POSITIONS_OPEN_FAILED");
    char magic[8]{};
    in.read(magic,8);
    if(!in || std::string(magic,magic+5)!="RSVP1")
        throw std::runtime_error("POSITIONS_MAGIC_INVALID");
    const auto frame_count=read_scalar<std::uint32_t>(in);
    const auto vertex_count=read_scalar<std::uint32_t>(in);
    if(vertex_count!=expected_vertices||frame_index<0||
       static_cast<std::uint32_t>(frame_index)>=frame_count)
        throw std::runtime_error("POSITIONS_HEADER_INVALID");
    const std::uint64_t stride=static_cast<std::uint64_t>(vertex_count)*2u*sizeof(double);
    in.seekg(static_cast<std::streamoff>(8+sizeof(std::uint32_t)*2)+
             static_cast<std::streamoff>(stride*static_cast<std::uint64_t>(frame_index)),
             std::ios::beg);
    if(!in) throw std::runtime_error("POSITIONS_SEEK_FAILED");
    std::vector<std::array<double,2>> out(vertex_count);
    for(auto& p:out) {
        p[0]=read_scalar<double>(in);
        p[1]=read_scalar<double>(in);
        if(!std::isfinite(p[0])||!std::isfinite(p[1]))
            throw std::runtime_error("POSITION_NONFINITE");
    }
    return out;
}

Image load_png(const std::string& path) {
    FILE* fp=std::fopen(path.c_str(),"rb");
    if(!fp) throw std::runtime_error("PNG_OPEN_FAILED");
    png_structp png=png_create_read_struct(PNG_LIBPNG_VER_STRING,nullptr,nullptr,nullptr);
    png_infop info=png?png_create_info_struct(png):nullptr;
    if(!png||!info) {
        if(png) png_destroy_read_struct(&png,nullptr,nullptr);
        std::fclose(fp);
        throw std::runtime_error("PNG_INIT_FAILED");
    }
    if(setjmp(png_jmpbuf(png))) {
        png_destroy_read_struct(&png,&info,nullptr);
        std::fclose(fp);
        throw std::runtime_error("PNG_DECODE_FAILED");
    }
    png_init_io(png,fp);
    png_read_info(png,info);
    const auto width=png_get_image_width(png,info);
    const auto height=png_get_image_height(png,info);
    auto color=png_get_color_type(png,info);
    auto depth=png_get_bit_depth(png,info);
    if(depth==16) png_set_strip_16(png);
    if(color==PNG_COLOR_TYPE_PALETTE) png_set_palette_to_rgb(png);
    if(color==PNG_COLOR_TYPE_GRAY && depth<8) png_set_expand_gray_1_2_4_to_8(png);
    if(png_get_valid(png,info,PNG_INFO_tRNS)) png_set_tRNS_to_alpha(png);
    if(color==PNG_COLOR_TYPE_GRAY||color==PNG_COLOR_TYPE_GRAY_ALPHA) png_set_gray_to_rgb(png);
    if(!(color&PNG_COLOR_MASK_ALPHA) && !png_get_valid(png,info,PNG_INFO_tRNS))
        png_set_add_alpha(png,0xff,PNG_FILLER_AFTER);
    png_read_update_info(png,info);
    Image image;
    image.width=width; image.height=height;
    image.rgba.resize(static_cast<std::size_t>(width)*height*4u);
    std::vector<png_bytep> rows(height);
    for(std::uint32_t y=0;y<height;++y)
        rows[y]=image.rgba.data()+static_cast<std::size_t>(y)*width*4u;
    png_read_image(png,rows.data());
    png_read_end(png,nullptr);
    png_destroy_read_struct(&png,&info,nullptr);
    std::fclose(fp);
    return image;
}

inline double orient(
    const std::array<double,2>& a,
    const std::array<double,2>& b,
    double x,double y
) {
    return (b[0]-a[0])*(y-a[1])-(b[1]-a[1])*(x-a[0]);
}

inline bool top_left(
    const std::array<double,2>& a,
    const std::array<double,2>& b
) {
    const double dy=b[1]-a[1], dx=b[0]-a[0];
    return dy<0.0 || (std::abs(dy)<=1e-12 && dx>0.0);
}

std::array<double,4> sample_bilinear(const Image& image,double u,double v) {
    u=std::clamp(u,0.0,1.0);
    v=std::clamp(v,0.0,1.0);
    const double x=u*static_cast<double>(image.width-1);
    const double y=v*static_cast<double>(image.height-1);
    const auto x0=static_cast<std::uint32_t>(std::floor(x));
    const auto y0=static_cast<std::uint32_t>(std::floor(y));
    const auto x1=std::min(x0+1,image.width-1);
    const auto y1=std::min(y0+1,image.height-1);
    const double tx=x-x0, ty=y-y0;
    auto texel=[&](std::uint32_t xx,std::uint32_t yy,int c) {
        return static_cast<double>(
            image.rgba[(static_cast<std::size_t>(yy)*image.width+xx)*4u+c]
        )/255.0;
    };
    std::array<double,4> out{};
    for(int c=0;c<4;++c) {
        const double a=texel(x0,y0,c)*(1.0-tx)+texel(x1,y0,c)*tx;
        const double b=texel(x0,y1,c)*(1.0-tx)+texel(x1,y1,c)*tx;
        out[c]=a*(1.0-ty)+b*ty;
    }
    return out;
}

void write_bytes(const std::string& path,const std::vector<std::uint8_t>& bytes) {
    std::ofstream out(path,std::ios::binary);
    if(!out) throw std::runtime_error("OUTPUT_OPEN_FAILED");
    out.write(reinterpret_cast<const char*>(bytes.data()),static_cast<std::streamsize>(bytes.size()));
    if(!out) throw std::runtime_error("OUTPUT_WRITE_FAILED");
}

std::string arg_value(int argc,char** argv,const std::string& key,bool required=true) {
    for(int i=1;i+1<argc;++i) if(argv[i]==key) return argv[i+1];
    if(required) throw std::runtime_error("ARG_MISSING:"+key);
    return {};
}

int parse_int(const std::string& raw,const char* label) {
    std::size_t used=0;
    int v=0;
    try { v=std::stoi(raw,&used,10); }
    catch(...) { throw std::runtime_error(std::string(label)+"_INVALID"); }
    if(used!=raw.size()) throw std::runtime_error(std::string(label)+"_INVALID");
    return v;
}

} // namespace

int main(int argc,char** argv) {
    try {
        const auto mesh_path=arg_value(argc,argv,"--mesh");
        const auto positions_path=arg_value(argc,argv,"--positions");
        const auto texture_path=arg_value(argc,argv,"--texture");
        const auto output_path=arg_value(argc,argv,"--out-rgba");
        const int frame=parse_int(arg_value(argc,argv,"--frame"),"FRAME");
        const int out_w=parse_int(arg_value(argc,argv,"--width"),"WIDTH");
        const int out_h=parse_int(arg_value(argc,argv,"--height"),"HEIGHT");
        if(out_w<=0||out_h<=0) throw std::runtime_error("OUTPUT_DIMENSION_INVALID");

        const auto mesh=read_mesh(mesh_path);
        auto positions=read_positions(
            positions_path,
            static_cast<std::uint32_t>(mesh.uv.size()),
            frame
        );
        const auto texture=load_png(texture_path);
        if(texture.width!=mesh.source_width||texture.height!=mesh.source_height)
            throw std::runtime_error("TEXTURE_SOURCE_DIMENSION_DRIFT");

        const double sx=static_cast<double>(out_w)/static_cast<double>(mesh.source_width);
        const double sy=static_cast<double>(out_h)/static_cast<double>(mesh.source_height);
        for(auto& p:positions) { p[0]*=sx; p[1]*=sy; }

        std::vector<std::array<double,4>> accum(
            static_cast<std::size_t>(out_w)*out_h,
            {0.0,0.0,0.0,0.0}
        );
        std::uint64_t covered_samples=0;

        for(std::size_t fi=0;fi<mesh.faces.size();++fi) {
            const auto face=mesh.faces[fi];
            const auto& a=positions[face[0]];
            const auto& b=positions[face[1]];
            const auto& c=positions[face[2]];
            const double area=orient(a,b,c[0],c[1]);
            if(std::abs(area)<=1e-12) continue;
            const double sign=area>0.0?1.0:-1.0;
            const bool positive=area>0.0;
            const auto minx=std::max(0,static_cast<int>(std::floor(std::min({a[0],b[0],c[0]})-0.5)));
            const auto maxx=std::min(out_w-1,static_cast<int>(std::ceil(std::max({a[0],b[0],c[0]})-0.5)));
            const auto miny=std::max(0,static_cast<int>(std::floor(std::min({a[1],b[1],c[1]})-0.5)));
            const auto maxy=std::min(out_h-1,static_cast<int>(std::ceil(std::max({a[1],b[1],c[1]})-0.5)));
            if(minx>maxx||miny>maxy) continue;
            const bool tl0=positive?top_left(b,c):top_left(c,b);
            const bool tl1=positive?top_left(c,a):top_left(a,c);
            const bool tl2=positive?top_left(a,b):top_left(b,a);
            for(int y=miny;y<=maxy;++y) for(int x=minx;x<=maxx;++x) {
                const double px=x+0.5, py=y+0.5;
                const double q0=sign*orient(b,c,px,py);
                const double q1=sign*orient(c,a,px,py);
                const double q2=sign*orient(a,b,px,py);
                auto accept=[](double q,bool tl) {
                    return q>1e-12 || (std::abs(q)<=1e-12 && tl);
                };
                if(!accept(q0,tl0)||!accept(q1,tl1)||!accept(q2,tl2)) continue;
                const double w0=orient(b,c,px,py)/area;
                const double w1=orient(c,a,px,py)/area;
                const double w2=1.0-w0-w1;
                const auto& ua=mesh.uv[face[0]];
                const auto& ub=mesh.uv[face[1]];
                const auto& uc=mesh.uv[face[2]];
                const double u=w0*ua[0]+w1*ub[0]+w2*uc[0];
                const double v=w0*ua[1]+w1*ub[1]+w2*uc[1];
                const auto sample=sample_bilinear(texture,u,v);
                const double alpha=std::clamp(sample[3],0.0,1.0);
                auto& dst=accum[static_cast<std::size_t>(y)*out_w+x];
                const double one_minus=1.0-alpha;
                dst[0]=sample[0]*alpha+dst[0]*one_minus;
                dst[1]=sample[1]*alpha+dst[1]*one_minus;
                dst[2]=sample[2]*alpha+dst[2]*one_minus;
                dst[3]=alpha+dst[3]*one_minus;
                ++covered_samples;
            }
        }

        std::vector<std::uint8_t> out(static_cast<std::size_t>(out_w)*out_h*4u,0);
        for(std::size_t i=0;i<accum.size();++i) {
            const auto& p=accum[i];
            const double a=std::clamp(p[3],0.0,1.0);
            const double inv=a>1e-12?1.0/a:0.0;
            out[i*4u+0]=static_cast<std::uint8_t>(std::lround(std::clamp(p[0]*inv,0.0,1.0)*255.0));
            out[i*4u+1]=static_cast<std::uint8_t>(std::lround(std::clamp(p[1]*inv,0.0,1.0)*255.0));
            out[i*4u+2]=static_cast<std::uint8_t>(std::lround(std::clamp(p[2]*inv,0.0,1.0)*255.0));
            out[i*4u+3]=static_cast<std::uint8_t>(std::lround(a*255.0));
        }
        write_bytes(output_path,out);
        std::cout<<"REALSAS_VISUAL_MESH_RENDER_PASS"
                 <<" faces="<<mesh.faces.size()
                 <<" vertices="<<mesh.uv.size()
                 <<" covered_samples="<<covered_samples
                 <<" frame="<<frame<<"\n";
        return 0;
    } catch(const std::exception& e) {
        std::cerr<<"ERROR "<<e.what()<<"\n";
        return 2;
    }
}
