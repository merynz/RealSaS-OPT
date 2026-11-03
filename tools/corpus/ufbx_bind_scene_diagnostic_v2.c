/* Read-only source-authored FBX default world and cluster-bind world diagnostic.
   The historical independent ufbx_bind_scene_probe_v1.c remains unchanged.
   No source correction or family/admission PASS is issued by this program. */
#include "ufbx.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>

static void quoted(ufbx_string name) {
    putchar('"');
    for (size_t i = 0; i < name.length; ++i) {
        unsigned char c = (unsigned char)name.data[i];
        if (c == '"' || c == '\\') putchar('\\');
        if (c < 32) printf("\\u%04x", (unsigned)c);
        else putchar(c);
    }
    putchar('"');
}
static void vec(ufbx_vec3 v) {
    if (!isfinite(v.x) || !isfinite(v.y) || !isfinite(v.z)) {
        fprintf(stderr, "FBX_BIND_DIAG_NONFINITE_VERTEX\n"); exit(4);
    }
    printf("[%.17g,%.17g,%.17g]", v.x, v.y, v.z);
}
static int bind_sample(const ufbx_mesh *m, size_t i, ufbx_vec3 *out, double *sum) {
    if (m->skin_deformers.count != 1) return 0;
    const ufbx_skin_deformer *skin = m->skin_deformers.data[0];
    if (i >= skin->vertices.count) return 0;
    ufbx_skin_vertex sv = skin->vertices.data[i];
    if (!sv.num_weights || sv.weight_begin > skin->weights.count ||
        sv.num_weights > skin->weights.count - sv.weight_begin) return 0;
    ufbx_vec3 accum = {0};
    double total = 0.;
    for (size_t j = 0; j < sv.num_weights; ++j) {
        ufbx_skin_weight weight = skin->weights.data[sv.weight_begin + j];
        if (weight.cluster_index >= skin->clusters.count ||
            !isfinite(weight.weight) || weight.weight < 0.) return 0;
        const ufbx_skin_cluster *cluster = skin->clusters.data[weight.cluster_index];
        ufbx_vec3 bone = ufbx_transform_position(
            &cluster->geometry_to_bone, m->vertices.data[i]);
        ufbx_vec3 bind = ufbx_transform_position(
            &cluster->bind_to_world, bone);
        accum.x += weight.weight * bind.x;
        accum.y += weight.weight * bind.y;
        accum.z += weight.weight * bind.z;
        total += weight.weight;
    }
    if (!(total > 0.) || !isfinite(total)) return 0;
    /* This branch is solely diagnostic: count missing/un-normalized weights.
       Do not silently turn measured weights into approved skin truth. */
    out->x = accum.x / total;
    out->y = accum.y / total;
    out->z = accum.z / total;
    *sum = total;
    return 1;
}
/* For unskinned bone-parent meshes, compare current source joint frame
   with the authored skin cluster bind frame. Never choose ambiguous frames. */
static int parent_bind_candidate(const ufbx_scene *scene, const ufbx_node *mesh_node,
                                 ufbx_matrix *out, const ufbx_node **bone) {
    if (mesh_node->mesh->skin_deformers.count != 0) return 0;
    const ufbx_node *p = mesh_node->parent;
    while (p && p != scene->root_node) {
        if (p->bone) break;
        p = p->parent;
    }
    if (!p || p == scene->root_node) return 0;
    size_t match = 0;
    ufbx_matrix unique = {0};
    for (size_t mi=0; mi<scene->meshes.count; ++mi) {
        const ufbx_mesh *mesh=scene->meshes.data[mi];
        for (size_t si=0; si<mesh->skin_deformers.count; ++si) {
            const ufbx_skin_deformer *skin=mesh->skin_deformers.data[si];
            for (size_t ci=0; ci<skin->clusters.count; ++ci) {
                const ufbx_skin_cluster *cluster=skin->clusters.data[ci];
                if (cluster->bone_node != p) continue;
                if (!(fabs((double)ufbx_matrix_determinant(&p->node_to_world))>1e-12))
                    return 0;
                ufbx_matrix inverse=ufbx_matrix_invert(&p->node_to_world);
                ufbx_matrix relative=ufbx_matrix_mul(&inverse,&mesh_node->geometry_to_world);
                ufbx_matrix current=ufbx_matrix_mul(&cluster->bind_to_world,&relative);
                if (!match) unique=current;
                else {
                    for (int k=0; k<12; ++k)
                        if (fabs((double)(unique.v[k]-current.v[k]))>1e-9) return 0;
                }
                ++match;
            }
        }
    }
    if (!match) return 0;
    *out=unique; *bone=p;
    return 1;
}
int main(int argc, char **argv) {
    if (argc != 2) return 2;
    ufbx_load_opts opts = {0};
    ufbx_error error;
    opts.target_axes = ufbx_axes_right_handed_z_up;
    opts.target_unit_meters = 1.0;
    opts.evaluate_skinning = true;
    ufbx_scene *scene = ufbx_load_file(argv[1], &opts, &error);
    if (!scene) { fprintf(stderr,"ufbx load failed: %s\n",error.description.data); return 3; }
    printf("{\"schema\":\"RealSaS.FBXBindPositionDiagnostic.v1\",");
    printf("\"authority\":{\"training_authorized\":false,\"source_admitted\":false,");
    printf("\"candidate_bind_policy_selected\":false},\"bone_count\":%zu,\"objects\":[",scene->bones.count);
    size_t num = 0;
    for (size_t n=0; n<scene->nodes.count; ++n) {
        const ufbx_node *node=scene->nodes.data[n];
        const ufbx_mesh *m=node->mesh;
        if (!m) continue;
        if (num++) putchar(',');
        printf("{\"name\":"); quoted(node->name);
        printf(",\"vertex_count\":%zu,\"skin_deformer_count\":%zu,\"default_world\":[",
            m->vertices.count,m->skin_deformers.count);
        for (size_t i=0;i<m->vertices.count;++i) {
            if(i) putchar(',');
            vec(ufbx_transform_position(&node->geometry_to_world,m->vertices.data[i]));
        }
        printf("],\"bone_parent_bind_world\":[");
        ufbx_matrix parent_bind={0}; const ufbx_node *attached_bone=NULL;
        int eligible=parent_bind_candidate(scene,node,&parent_bind,&attached_bone);
        for (size_t vi=0;vi<m->vertices.count;++vi) {
            if(vi) putchar(',');
            if (eligible) vec(ufbx_transform_position(&parent_bind,m->vertices.data[vi]));
            else printf("null");
        }
        printf("],\"attached_bone\":");
        if (attached_bone) quoted(attached_bone->name); else printf("null");
        printf(",\"source_parent_node\":");
        if (node->parent) quoted(node->parent->name); else printf("null");
        printf("],\"cluster_bind_world\":[");
        size_t available=0, abnormal=0;
        for (size_t i=0;i<m->vertices.count;++i) {
            if(i) putchar(',');
            ufbx_vec3 bound={0}; double total=0.;
            if (bind_sample(m,i,&bound,&total)) {
                vec(bound);++available;
                if (fabs(total-1.)>1e-5) ++abnormal;
            } else printf("null");
        }
        printf("],\"cluster_bind_position_count\":%zu,\"nonunit_weight_vertex_count\":%zu}",
            available,abnormal);
    }
    printf("],\"source_files_mutated\":false}\n");
    ufbx_free_scene(scene);
    return 0;
}
