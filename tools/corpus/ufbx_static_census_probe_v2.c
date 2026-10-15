/* Exact-source FBX static census proposal. Build against pinned official ufbx.
 * ufbx is importer/proposal authority only: emitted triangulation MUST pass
 * RealSaS.PolygonTriangulationQualification.v1 before census admission. */
#include "ufbx.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void fail(const char *msg) { fprintf(stderr, "%s\n", msg); exit(1); }

static int material_id_for_slot(const ufbx_node *node, const ufbx_mesh *mesh, size_t slot) {
    const ufbx_material *mat = NULL;
    if (slot < node->materials.count) mat = node->materials.data[slot];
    else if (slot < mesh->materials.count) mat = mesh->materials.data[slot];
    return mat ? (int)mat->typed_id : -1;
}

static int dominant_owner_for_triangle(const ufbx_skin_deformer *skin, const uint32_t cp[3]) {
    if (!skin || skin->clusters.count == 0) return -1;
    size_t nc = skin->clusters.count;
    double *acc = (double*)calloc(nc, sizeof(double));
    if (!acc) fail("UFBX_STATIC_CENSUS_OOM");
    for (size_t vi = 0; vi < 3; vi++) {
        uint32_t vertex_id = cp[vi];
        if (vertex_id >= skin->vertices.count) continue;
        ufbx_skin_vertex sv = skin->vertices.data[vertex_id];
        double total = 0.0;
        for (size_t wi = 0; wi < sv.num_weights; wi++) {
            ufbx_skin_weight sw = skin->weights.data[sv.weight_begin + wi];
            if (sw.cluster_index < nc && sw.weight > 0.0) total += sw.weight;
        }
        if (total <= 0.0) continue;
        for (size_t wi = 0; wi < sv.num_weights; wi++) {
            ufbx_skin_weight sw = skin->weights.data[sv.weight_begin + wi];
            if (sw.cluster_index < nc && sw.weight > 0.0) {
                acc[sw.cluster_index] += (sw.weight / total) / 3.0;
            }
        }
    }
    size_t best = SIZE_MAX;
    double best_value = -1.0;
    for (size_t ci = 0; ci < nc; ci++) {
        if (acc[ci] > best_value) { best_value = acc[ci]; best = ci; }
    }
    int result = -1;
    if (best != SIZE_MAX && best_value > 0.0) {
        ufbx_skin_cluster *cluster = skin->clusters.data[best];
        if (cluster && cluster->bone_node) result = (int)cluster->bone_node->typed_id;
    }
    free(acc);
    return result;
}

int main(int argc, char **argv) {
    if (argc != 2) return 2;
    ufbx_load_opts opts = { 0 };
    ufbx_error error;
    opts.target_axes = ufbx_axes_right_handed_z_up;
    opts.target_unit_meters = 1.0;
    opts.evaluate_skinning = true;
    ufbx_scene *scene = ufbx_load_file(argv[1], &opts, &error);
    if (!scene) { fprintf(stderr, "ufbx load failed: %s\n", error.description.data); return 1; }

    /* Emit transformed control-point vertices, one block per mesh instance. */
    printf("{\"schema\":\"RealSaS.UFBXStaticCensusProbe.v2\",\"vertices\":[");
    size_t node_offsets_count = scene->nodes.count;
    size_t *node_offsets = (size_t*)calloc(node_offsets_count, sizeof(size_t));
    if (!node_offsets) fail("UFBX_STATIC_CENSUS_OOM");
    size_t vertex_total = 0;
    int first = 1;
    for (size_t ni = 0; ni < scene->nodes.count; ni++) {
        ufbx_node *node = scene->nodes.data[ni];
        ufbx_mesh *mesh = node->mesh;
        node_offsets[ni] = vertex_total;
        if (!mesh) continue;
        for (size_t vi = 0; vi < mesh->vertices.count; vi++) {
            ufbx_vec3 p = ufbx_transform_position(&node->geometry_to_world, mesh->vertices.data[vi]);
            printf("%s[%.17g,%.17g,%.17g]", first ? "" : ",", p.x, p.y, p.z);
            first = 0; vertex_total++;
        }
    }

    printf("],\"source_polygons\":[");
    size_t polygon_global = 0;
    first = 1;
    for (size_t ni = 0; ni < scene->nodes.count; ni++) {
        ufbx_node *node = scene->nodes.data[ni];
        ufbx_mesh *mesh = node->mesh;
        if (!mesh) continue;
        size_t offset = node_offsets[ni];
        for (size_t fi = 0; fi < mesh->faces.count; fi++, polygon_global++) {
            ufbx_face face = mesh->faces.data[fi];
            printf("%s[", first ? "" : ","); first = 0;
            for (size_t ci = 0; ci < face.num_indices; ci++) {
                uint32_t pv = mesh->vertex_indices.data[face.index_begin + ci];
                printf("%s%zu", ci ? "," : "", offset + (size_t)pv);
            }
            printf("]");
        }
    }

    printf("],\"faces\":[");
    first = 1;
    for (size_t ni = 0; ni < scene->nodes.count; ni++) {
        ufbx_node *node = scene->nodes.data[ni];
        ufbx_mesh *mesh = node->mesh;
        if (!mesh) continue;
        size_t offset = node_offsets[ni];
        size_t cap = mesh->max_face_triangles * 3;
        uint32_t *tri = (uint32_t*)malloc(cap * sizeof(uint32_t));
        if (!tri) fail("UFBX_STATIC_CENSUS_OOM");
        for (size_t fi = 0; fi < mesh->faces.count; fi++) {
            ufbx_face face = mesh->faces.data[fi];
            size_t num_tris = ufbx_triangulate_face(tri, cap, mesh, face);
            for (size_t ti = 0; ti < num_tris; ti++) {
                uint32_t a = mesh->vertex_indices.data[tri[ti*3+0]];
                uint32_t b = mesh->vertex_indices.data[tri[ti*3+1]];
                uint32_t c = mesh->vertex_indices.data[tri[ti*3+2]];
                printf("%s[%zu,%zu,%zu]", first ? "" : ",",
                    offset+(size_t)a, offset+(size_t)b, offset+(size_t)c);
                first = 0;
            }
        }
        free(tri);
    }

    printf("],\"triangle_polygon_id\":[");
    first = 1; polygon_global = 0;
    for (size_t ni = 0; ni < scene->nodes.count; ni++) {
        ufbx_mesh *mesh = scene->nodes.data[ni]->mesh;
        if (!mesh) continue;
        size_t cap = mesh->max_face_triangles * 3;
        uint32_t *tri = (uint32_t*)malloc(cap * sizeof(uint32_t));
        if (!tri) fail("UFBX_STATIC_CENSUS_OOM");
        for (size_t fi = 0; fi < mesh->faces.count; fi++, polygon_global++) {
            size_t num_tris = ufbx_triangulate_face(tri, cap, mesh, mesh->faces.data[fi]);
            for (size_t ti = 0; ti < num_tris; ti++) {
                printf("%s%zu", first ? "" : ",", polygon_global); first = 0;
            }
        }
        free(tri);
    }

    printf("],\"face_material_id\":[");
    first = 1;
    for (size_t ni = 0; ni < scene->nodes.count; ni++) {
        ufbx_node *node = scene->nodes.data[ni];
        ufbx_mesh *mesh = node->mesh;
        if (!mesh) continue;
        int *face_mat = (int*)malloc(mesh->faces.count * sizeof(int));
        if (!face_mat) fail("UFBX_STATIC_CENSUS_OOM");
        for (size_t fi = 0; fi < mesh->faces.count; fi++) face_mat[fi] = -1;
        for (size_t pi = 0; pi < mesh->material_parts.count; pi++) {
            ufbx_mesh_part *part = &mesh->material_parts.data[pi];
            int material_id = material_id_for_slot(node, mesh, part->index);
            for (size_t ii = 0; ii < part->face_indices.count; ii++) {
                uint32_t face_id = part->face_indices.data[ii];
                if (face_id < mesh->faces.count) face_mat[face_id] = material_id;
            }
        }
        size_t cap = mesh->max_face_triangles * 3;
        uint32_t *tri = (uint32_t*)malloc(cap * sizeof(uint32_t));
        if (!tri) fail("UFBX_STATIC_CENSUS_OOM");
        for (size_t fi = 0; fi < mesh->faces.count; fi++) {
            size_t num_tris = ufbx_triangulate_face(tri, cap, mesh, mesh->faces.data[fi]);
            for (size_t ti = 0; ti < num_tris; ti++) {
                printf("%s%d", first ? "" : ",", face_mat[fi]); first = 0;
            }
        }
        free(tri); free(face_mat);
    }

    printf("],\"face_owner_id\":[");
    first = 1;
    for (size_t ni = 0; ni < scene->nodes.count; ni++) {
        ufbx_mesh *mesh = scene->nodes.data[ni]->mesh;
        if (!mesh) continue;
        ufbx_skin_deformer *skin = mesh->skin_deformers.count ? mesh->skin_deformers.data[0] : NULL;
        size_t cap = mesh->max_face_triangles * 3;
        uint32_t *tri = (uint32_t*)malloc(cap * sizeof(uint32_t));
        if (!tri) fail("UFBX_STATIC_CENSUS_OOM");
        for (size_t fi = 0; fi < mesh->faces.count; fi++) {
            size_t num_tris = ufbx_triangulate_face(tri, cap, mesh, mesh->faces.data[fi]);
            for (size_t ti = 0; ti < num_tris; ti++) {
                uint32_t cp[3] = {
                    mesh->vertex_indices.data[tri[ti*3+0]],
                    mesh->vertex_indices.data[tri[ti*3+1]],
                    mesh->vertex_indices.data[tri[ti*3+2]],
                };
                int owner = dominant_owner_for_triangle(skin, cp);
                printf("%s%d", first ? "" : ",", owner); first = 0;
            }
        }
        free(tri);
    }

    printf("],\"vertex_count\":%zu,\"polygon_count\":%zu,", vertex_total, polygon_global);
    printf("\"material_count\":%zu,\"bone_count\":%zu,", scene->materials.count, scene->bones.count);
    printf("\"triangulation_proposal\":\"UFBX_TRIANGULATE_FACE_REQUIRES_REALSAS_QUALIFICATION\",");
    printf("\"owner_scope\":\"DIAGNOSTIC_DOMINANT_NORMALIZED_RAW_SKIN_JOINT\",\"native_motion_names\":[");
    for (size_t i = 0; i < scene->anim_stacks.count; i++) {
        if (i) putchar(',');
        ufbx_string s = scene->anim_stacks.data[i]->name;
        putchar('"'); for (size_t j = 0; j < s.length; j++) { char c=s.data[j]; if(c=='"'||c=='\\') putchar('\\'); putchar(c); } putchar('"');
    }
    printf("]}\n");
    free(node_offsets);
    ufbx_free_scene(scene);
    return 0;
}
