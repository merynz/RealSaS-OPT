#include "../src/runtime_v4_reference.h"
#include <png.h>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
void write_png(const std::string& path,uint32_t width,uint32_t height,const std::vector<uint8_t>&rgba){png_image image{};image.version=PNG_IMAGE_VERSION;image.width=width;image.height=height;image.format=PNG_FORMAT_RGBA;if(!png_image_write_to_file(&image,path.c_str(),0,rgba.data(),0,nullptr))throw std::runtime_error(std::string("PNG write failed: ")+image.message);}
}
int main(int argc,char**argv){if(argc<2){std::cerr<<"Usage: realsas_runtime_v4_demo <package.rss|dir> [--clip id] [--view V0] [--time seconds] [--out frame.png]\n";return 2;}std::string package=argv[1],clip_name="idle",view_name="V0",out="realsas_v4_frame.png";float time=0;for(int i=2;i<argc;++i){std::string key=argv[i];if(key=="--clip"&&i+1<argc)clip_name=argv[++i];else if(key=="--view"&&i+1<argc)view_name=argv[++i];else if(key=="--time"&&i+1<argc)time=std::stof(argv[++i]);else if(key=="--out"&&i+1<argc)out=argv[++i];else{std::cerr<<"Unknown argument: "<<key<<"\n";return 2;}}try{realsas::runtime_v4::ReferenceRuntime runtime(package);uint32_t ci=runtime.find_clip(clip_name),vi=runtime.find_view(view_name);if(ci==UINT32_MAX)throw std::runtime_error("Clip not found");if(vi==UINT32_MAX)throw std::runtime_error("View not found");auto rgba=runtime.render_rgba(ci,vi,time);std::filesystem::path target(out);if(!target.parent_path().empty())std::filesystem::create_directories(target.parent_path());write_png(out,runtime.view_width(vi),runtime.view_height(vi),rgba);std::cout<<"schema="<<runtime.binary_schema()<<"\n"<<"source_binding_sha256="<<runtime.source_binding_sha256()<<"\n"<<"rendered="<<out<<" view="<<view_name<<" clip="<<clip_name<<" time="<<time<<" renderer=RUNTIME_V4_SHARED_CANONICAL_DEPTH_REFERENCE\n";return 0;}catch(const std::exception&exc){std::cerr<<"RealSaS runtime-v4 error: "<<exc.what()<<"\n";return 1;}}
