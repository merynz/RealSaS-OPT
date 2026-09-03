#ifndef REALSAS_RUNTIME_H
#define REALSAS_RUNTIME_H

#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32) && defined(REALSAS_RUNTIME_SHARED)
#  if defined(REALSAS_RUNTIME_BUILD)
#    define RS_API __declspec(dllexport)
#  else
#    define RS_API __declspec(dllimport)
#  endif
#else
#  define RS_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef struct RsRuntime RsRuntime;

/* Versioned engine-neutral settings for the optional conformance post-process. */
typedef struct RsPostProcessSettings {
    uint32_t struct_version;
    float shadow_strength;
    float shadow_offset_x;
    float shadow_offset_y;
    float shadow_blur_radius;
    float rim_strength;
    float rim_r;
    float rim_g;
    float rim_b;
    float bloom_strength;
    float bloom_radius;
    float exposure;
    float saturation;
    float vignette;
} RsPostProcessSettings;

typedef enum RsResult {
    RS_OK = 0,
    RS_ERROR_INVALID_ARGUMENT = 1,
    RS_ERROR_IO = 2,
    RS_ERROR_FORMAT = 3,
    RS_ERROR_NOT_FOUND = 4,
    RS_ERROR_BUFFER_TOO_SMALL = 5,
    RS_ERROR_INTERNAL = 6
} RsResult;

/* Bit values returned by rs_runtime_feature_flags. */
typedef enum RsRuntimeFeature {
    RS_RUNTIME_FEATURE_RGBA8_TEXTURES = 1u << 0,
    RS_RUNTIME_FEATURE_BAKED_MESH_FRAMES = 1u << 1,
    RS_RUNTIME_FEATURE_LINEAR_FRAME_INTERPOLATION = 1u << 2,
    RS_RUNTIME_FEATURE_DYNAMIC_DRAW_ORDER = 1u << 3,
    RS_RUNTIME_FEATURE_MULTI_VIEW = 1u << 4,
    RS_RUNTIME_FEATURE_SOFTWARE_REFERENCE_RENDERER = 1u << 5,
    RS_RUNTIME_FEATURE_CLIP_QUALIFICATION = 1u << 6,
    RS_RUNTIME_FEATURE_TEXTURE_CRC32 = 1u << 7,
    RS_RUNTIME_FEATURE_REFERENCE_POST_PROCESS = 1u << 8,
    RS_RUNTIME_FEATURE_CLIP_MIXING = 1u << 9
} RsRuntimeFeature;

/* Stable ABI contract version; independent from the .rsr binary schema. */
RS_API uint32_t rs_runtime_abi_version(void);

/* Opens a .rss/.realsas deployment archive or an unpacked package directory. */
RS_API RsResult rs_runtime_open(
    const char* package_path,
    RsRuntime** out_runtime,
    char* error_buffer,
    size_t error_buffer_size);

RS_API void rs_runtime_close(RsRuntime* runtime);

RS_API uint32_t rs_runtime_view_count(const RsRuntime* runtime);
RS_API const char* rs_runtime_view_id(const RsRuntime* runtime, uint32_t view_index);
RS_API uint32_t rs_runtime_view_width(const RsRuntime* runtime, uint32_t view_index);
RS_API uint32_t rs_runtime_view_height(const RsRuntime* runtime, uint32_t view_index);
RS_API const uint8_t* rs_runtime_view_texture_rgba(
    const RsRuntime* runtime,
    uint32_t view_index,
    size_t* out_byte_count);

RS_API uint32_t rs_runtime_mesh_count(const RsRuntime* runtime, uint32_t view_index);
RS_API const char* rs_runtime_mesh_id(
    const RsRuntime* runtime, uint32_t view_index, uint32_t mesh_index);
RS_API uint32_t rs_runtime_mesh_vertex_count(
    const RsRuntime* runtime, uint32_t view_index, uint32_t mesh_index);
RS_API uint32_t rs_runtime_mesh_triangle_count(
    const RsRuntime* runtime, uint32_t view_index, uint32_t mesh_index);

/* Writes x,y,u,v per vertex into out_xyuv. */
RS_API RsResult rs_runtime_mesh_rest_vertices(
    const RsRuntime* runtime,
    uint32_t view_index,
    uint32_t mesh_index,
    float* out_xyuv,
    size_t out_float_count);

/* Writes 3 uint32 indices per triangle into out_indices. */
RS_API RsResult rs_runtime_mesh_triangles(
    const RsRuntime* runtime,
    uint32_t view_index,
    uint32_t mesh_index,
    uint32_t* out_indices,
    size_t out_index_count);

RS_API uint32_t rs_runtime_clip_count(const RsRuntime* runtime);
RS_API const char* rs_runtime_clip_id(const RsRuntime* runtime, uint32_t clip_index);
RS_API const char* rs_runtime_clip_display_name(const RsRuntime* runtime, uint32_t clip_index);
RS_API const char* rs_runtime_clip_intent(const RsRuntime* runtime, uint32_t clip_index);
RS_API float rs_runtime_clip_duration(const RsRuntime* runtime, uint32_t clip_index);
RS_API float rs_runtime_clip_fps(const RsRuntime* runtime, uint32_t clip_index);
RS_API int rs_runtime_clip_loop(const RsRuntime* runtime, uint32_t clip_index);
RS_API int rs_runtime_clip_qualified(const RsRuntime* runtime, uint32_t clip_index);

/* Finds a clip by exact id, intent or display name. Returns UINT32_MAX on miss. */
RS_API uint32_t rs_runtime_find_clip(const RsRuntime* runtime, const char* id_or_name);
RS_API uint32_t rs_runtime_find_view(const RsRuntime* runtime, const char* view_id);

/* Samples x,y per vertex after baked-frame interpolation. */
RS_API RsResult rs_runtime_sample_mesh_vertices(
    const RsRuntime* runtime,
    uint32_t clip_index,
    uint32_t view_index,
    uint32_t mesh_index,
    float time_seconds,
    float* out_xy,
    size_t out_float_count);

/* Blends two already-baked clip samples in canonical vertex space. */
RS_API RsResult rs_runtime_sample_mesh_vertices_blend(
    const RsRuntime* runtime,
    uint32_t from_clip_index,
    float from_time_seconds,
    uint32_t to_clip_index,
    float to_time_seconds,
    float mix_alpha,
    uint32_t view_index,
    uint32_t mesh_index,
    float* out_xy,
    size_t out_float_count);

/* Returns the draw-order mesh index count for a sampled frame. */
RS_API uint32_t rs_runtime_sample_draw_order_count(
    const RsRuntime* runtime,
    uint32_t clip_index,
    uint32_t view_index,
    float time_seconds);

RS_API RsResult rs_runtime_sample_draw_order(
    const RsRuntime* runtime,
    uint32_t clip_index,
    uint32_t view_index,
    float time_seconds,
    uint32_t* out_mesh_indices,
    size_t out_index_count);

/* Draw order is discrete; the source order is used below alpha 0.5. */
RS_API uint32_t rs_runtime_sample_draw_order_blend_count(
    const RsRuntime* runtime,
    uint32_t from_clip_index,
    float from_time_seconds,
    uint32_t to_clip_index,
    float to_time_seconds,
    float mix_alpha,
    uint32_t view_index);

RS_API RsResult rs_runtime_sample_draw_order_blend(
    const RsRuntime* runtime,
    uint32_t from_clip_index,
    float from_time_seconds,
    uint32_t to_clip_index,
    float to_time_seconds,
    float mix_alpha,
    uint32_t view_index,
    uint32_t* out_mesh_indices,
    size_t out_index_count);

/* Reference software renderer. Host engines normally consume mesh data directly. */
RS_API RsResult rs_runtime_render_rgba(
    const RsRuntime* runtime,
    uint32_t clip_index,
    uint32_t view_index,
    float time_seconds,
    uint8_t* out_rgba,
    size_t out_byte_count);

/* Returns the canonical reference presentation profile. */
RS_API RsPostProcessSettings rs_runtime_default_post_process_settings(void);

/* Reference renderer plus optional deterministic presentation post-process. */
RS_API RsResult rs_runtime_render_rgba_ex(
    const RsRuntime* runtime,
    uint32_t clip_index,
    uint32_t view_index,
    float time_seconds,
    const RsPostProcessSettings* settings,
    uint8_t* out_rgba,
    size_t out_byte_count);

RS_API const char* rs_runtime_binary_schema(const RsRuntime* runtime);
RS_API const char* rs_runtime_source_manifest_sha256(const RsRuntime* runtime);
RS_API const char* rs_runtime_source_payload_merkle_sha256(const RsRuntime* runtime);
RS_API const char* rs_runtime_coordinate_system(const RsRuntime* runtime);
RS_API float rs_runtime_units_per_pixel(const RsRuntime* runtime);
RS_API uint32_t rs_runtime_feature_flags(const RsRuntime* runtime);
RS_API uint32_t rs_runtime_view_texture_crc32(const RsRuntime* runtime, uint32_t view_index);

#ifdef __cplusplus
}
#endif

#endif
