#include "../src/graphics_v1.hpp"
#include <cassert>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

namespace {
void u32(std::vector<std::uint8_t>& out,std::uint32_t v){auto* p=reinterpret_cast<const std::uint8_t*>(&v);out.insert(out.end(),p,p+4);}
void text(std::vector<std::uint8_t>& out,const std::string& s){u32(out,static_cast<std::uint32_t>(s.size()));out.insert(out.end(),s.begin(),s.end());}
std::vector<std::uint8_t> fixture(){
    std::vector<std::uint8_t> b{'R','S','G','F','X','0','0','1'};u32(b,1);u32(b,3);u32(b,3);u32(b,1);u32(b,0);text(b,"mesh-hash");text(b,"compiled-graphics");text(b,"runtime-graphics");
    u32(b,0);u32(b,1);u32(b,2);
    text(b,"f0");u32(b,0);text(b,"f1");u32(b,2);text(b,"f2");u32(b,0);
    u32(b,0);u32(b,3);u32(b,0);u32(b,1);u32(b,2);u32(b,3);u32(b,0);u32(b,2);u32(b,1);u32(b,1);u32(b,0);u32(b,1);
    return b;
}
}
int main(){
    const auto state=realsas::graphics_v1::parse(fixture());
    assert(state.face_fragment_index.size()==3);assert(state.views.size()==1);assert(realsas::graphics_v1::face_visible(state,0,0));assert(state.support_band_pixels[1]==2);
    assert(realsas::graphics_v1::explicit_precedence(state,0,0,1));
    assert(realsas::graphics_v1::compare_layers(state,0,0,10.0,1,1.0,1e-12)<0);
    assert(realsas::graphics_v1::compare_layers(state,0,2,0.5,1,1.0,1e-12)<0);

    // Pixel court: f0 is explicitly in front of f1 despite contradictory physical depth.
    // f2 has no relation edge, so among currently admissible nodes physical depth decides.
    const std::vector<std::uint32_t> faces{0,1,2};
    const std::vector<double> depths{10.0,1.0,0.5};
    const auto order=realsas::graphics_v1::order_pixel_layers(state,0,faces,depths,1e-12);
    assert(order.size()==3);
    assert(faces[order[0]]==2); // unrelated, physically nearest
    assert(faces[order[1]]==0); // explicit f0 -> f1 now forces precedence
    assert(faces[order[2]]==1);

    std::cout<<"GRAPHICS_V1_NATIVE_CONSUMPTION_PASS\n";
}
