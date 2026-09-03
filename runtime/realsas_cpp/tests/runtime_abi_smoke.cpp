#include "realsas/realsas_runtime.h"
#include <cmath>
#include <cstdint>

int main() {
    if (rs_runtime_abi_version() != 3u) return 1;
    const RsPostProcessSettings cfg = rs_runtime_default_post_process_settings();
    if (cfg.struct_version != 1u) return 2;
    if (!(cfg.exposure > 0.0f && cfg.saturation > 0.0f)) return 3;
    if (!(cfg.rim_strength >= 0.0f && cfg.bloom_strength >= 0.0f)) return 4;
    if (!std::isfinite(cfg.shadow_blur_radius) || !std::isfinite(cfg.vignette)) return 5;
    const uint32_t mixing_flag = RS_RUNTIME_FEATURE_CLIP_MIXING;
    if (mixing_flag != (1u << 9)) return 6;
    return 0;
}
