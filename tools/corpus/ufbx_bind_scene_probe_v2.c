/* Independent FBX bind-rest reader V2: correct unskinned bone-parent
   attachments using source-authored skin-cluster bind frames. Preserve skinned
   body coordinates, raw source bytes and historical V1 policy. */
#include "ufbx.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
static void str(ufbx_string s) {
    putchar('"');
    for (size_t i=0;i<s.length;i++) {
        unsigned char c=s.data[i];
        if(c=='"'||c=='\\') putchar('\\');
        if(c<32) printf("\\u%04x",c); else putchar(c);
    }
    putchar('"');
}
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
int main(int argc,char **argv) {
    if(argc!=2)return 2;
    ufbx_load_opts opts={0}; ufbx_error error;
    opts.target_axes=ufbx_axes_right_handed_z_up; opts.target_unit_meters=1.0;
    opts.evaluate_skinning=true;
    ufbx_scene *scene=ufbx_load_file(argv[1],&opts,&error);
    if(!scene){fprintf(stderr,"ufbx load failed: %s\n",error.description.data);return 1;}
    printf("{\"vertices\":[");
    size_t count=0, bone_parent_attachment_mesh_count=0;
    for(size_t n=0;n<scene->nodes.count;n++){
        ufbx_node *node=scene->nodes.data[n]; ufbx_mesh *m=node->mesh;
        if(!m)continue;
        ufbx_matrix authored_bind={0}; const ufbx_node *source_bone=NULL;
        int bound=parent_bind_candidate(scene,node,&authored_bind,&source_bone);
        const ufbx_node *ancestor=node->parent;
        int has_bone_parent=0;
        while (ancestor && ancestor!=scene->root_node) {
            if (ancestor->bone) {has_bone_parent=1;break;}
            ancestor=ancestor->parent;
        }
        if (m->skin_deformers.count==0 && has_bone_parent && !bound) {
            fprintf(stderr,"FBX_BIND_PARENT_BONE_UNRESOLVED:%.*s\\n",
                    (int)node->name.length,node->name.data);
            ufbx_free_scene(scene);return 8;
        }
        if (bound) ++bone_parent_attachment_mesh_count;
        const ufbx_matrix *world=bound?&authored_bind:&node->geometry_to_world;
        for(size_t v=0;v<m->vertices.count;v++){
            ufbx_vec3 p=ufbx_transform_position(world,m->vertices.data[v]);
            printf("%s[%.17g,%.17g,%.17g]",count++?",":"",p.x,p.y,p.z);
        }
    }
    printf("],\"faces\":["); size_t offset=0,tri=0;
    for(size_t n=0;n<scene->nodes.count;n++){
        ufbx_mesh *m=scene->nodes.data[n]->mesh;if(!m)continue;
        for(size_t f=0;f<m->faces.count;f++){
            ufbx_face face=m->faces.data[f];
            for(size_t j=1;j+1<face.num_indices;j++){
                uint32_t a=m->vertex_indices.data[face.index_begin];
                uint32_t b=m->vertex_indices.data[face.index_begin+j];
                uint32_t c=m->vertex_indices.data[face.index_begin+j+1];
                printf("%s[%zu,%zu,%zu]",tri++?",":"",offset+a,offset+b,offset+c);
            }
        }
        offset+=m->vertices.count;
    }
    printf("],\"objects\":[");
    size_t object_count=0;
    for (size_t n=0;n<scene->nodes.count;n++) {
        ufbx_node *node=scene->nodes.data[n];
        if (!node->mesh) continue;
        if(object_count++) putchar(',');
        printf("{\"name\":");str(node->name);
        printf(",\"vertex_count\":%zu,\"has_skin\":%s,\"source_parent_bone\":",
            node->mesh->vertices.count,node->mesh->skin_deformers.count?"true":"false");
        const ufbx_node *p=node->parent;
        while(p && p!=scene->root_node && !p->bone) p=p->parent;
        if(p && p!=scene->root_node) str(p->name);else printf("null");
        printf("}");
    }
    printf("],\"bone_count\":%zu,\"materials\":[",scene->bones.count);
    for(size_t i=0;i<scene->materials.count;i++){
        ufbx_material *m=scene->materials.data[i];
        if(i)putchar(',');printf("{\"name\":");str(m->name);
        ufbx_vec3 c=m->fbx.diffuse_color.value_vec3;
        printf(",\"diffuse\":[%.17g,%.17g,%.17g],\"transparency_factor\":%.17g}",
               c.x,c.y,c.z,m->fbx.transparency_factor.value_real);
    }
    printf("],\"native_motion_names\":[");
    for(size_t i=0;i<scene->anim_stacks.count;i++){if(i)putchar(',');str(scene->anim_stacks.data[i]->name);}
    printf("],\"weight_scope\":\"RAW_WEIGHTS_NOT_ASSUMED_NORMALIZED\",");
    printf("\"reader_schema\":\"RealSaS.SourceFBXBindRestReader.v2\",");
    printf("\"bone_parent_attachment_mesh_count\":%zu}\n",bone_parent_attachment_mesh_count);
    ufbx_free_scene(scene);return 0;
}
