#include "../src/graphics_v1.hpp"
#include "../src/graphics_local_order_v2.hpp"
#include <cassert>
#include <cstring>
#include <iostream>
#include <stdexcept>
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

template<class F>
void require_error(F&& f,const std::string& expected){
    try{f();}
    catch(const std::runtime_error& e){assert(std::string(e.what())==expected);return;}
    assert(false&&"expected runtime_error");
}
}

int main(){
    const auto state=realsas::graphics_v1::parse(fixture());
    assert(state.face_fragment_index.size()==3);assert(state.views.size()==1);assert(realsas::graphics_v1::face_visible(state,0,0));assert(state.support_band_pixels[1]==2);
    assert(realsas::graphics_v1::explicit_precedence(state,0,0,1));
    assert(realsas::graphics_v1::compare_layers(state,0,0,10.0,1,1.0,1e-12)<0);
    assert(realsas::graphics_v1::compare_layers(state,0,2,0.5,1,1.0,1e-12)<0);

    // V1 pixel court: f0 is explicitly in front of f1 despite contradictory physical depth.
    // f2 has no relation edge, so among currently admissible nodes physical depth decides.
    const std::vector<std::uint32_t> faces{0,1,2};
    const std::vector<double> depths{10.0,1.0,0.5};
    const auto order=realsas::graphics_v1::order_pixel_layers(state,0,faces,depths,1e-12);
    assert(order.size()==3);
    assert(faces[order[0]]==2);
    assert(faces[order[1]]==0);
    assert(faces[order[2]]==1);

    // V2 local court: opposite precedence on two distinct rays is legal. The
    // relation is scoped to the current pixel and must never be unioned into a
    // view-global DAG.
    const std::vector<std::uint32_t> pair_faces{0,1};
    const std::vector<double> pair_depths{10.0,1.0};
    using Edge=realsas::graphics_v2::LocalOrderEdge;
    const auto ray_a=realsas::graphics_v2::order_pixel_layers_local(
        pair_faces,pair_depths,std::vector<Edge>{{0,1}},1e-12);
    const auto ray_b=realsas::graphics_v2::order_pixel_layers_local(
        pair_faces,pair_depths,std::vector<Edge>{{1,0}},1e-12);
    assert(pair_faces[ray_a[0]]==0&&pair_faces[ray_a[1]]==1);
    assert(pair_faces[ray_b[0]]==1&&pair_faces[ray_b[1]]==0);

    // A real depth tie without measured local precedence is not silently
    // resolved by face index. The consumer fails closed instead.
    require_error([&]{
        realsas::graphics_v2::order_pixel_layers_local(
            pair_faces,std::vector<double>{1.0,1.0},{},1e-9);
    },"GRAPHICS_V2_LOCAL_PIXEL_AMBIGUOUS_TIE");

    // Contradictory evidence inside the same ray is invalid even though opposite
    // evidence across two different rays is valid.
    require_error([&]{
        realsas::graphics_v2::order_pixel_layers_local(
            pair_faces,pair_depths,std::vector<Edge>{{0,1},{1,0}},1e-12);
    },"GRAPHICS_V2_LOCAL_EDGE_CONTRADICTION");

    std::cout<<"GRAPHICS_V1_NATIVE_CONSUMPTION_PASS\n";
    std::cout<<"GRAPHICS_V2_RAY_LOCAL_CONSUMPTION_PASS\n";
}
