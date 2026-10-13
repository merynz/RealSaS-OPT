/* Independent raw FBX reader. Build against pinned official ufbx, no Python FFI. */
#include "ufbx.h"
#include <stdio.h>
#include <stdlib.h>
static void str(ufbx_string s) {
    putchar('"');
    for (size_t i=0;i<s.length;i++) {
        unsigned char c=s.data[i];
        if(c=='"'||c=='\\') putchar('\\');
        if(c<32) printf("\\u%04x",c); else putchar(c);
    }
    putchar('"');
}
int main(int argc,char **argv) {
    if(argc!=2)return 2;
    ufbx_load_opts opts={0}; ufbx_error error;
    opts.target_axes=ufbx_axes_right_handed_z_up; opts.target_unit_meters=1.0;
    opts.evaluate_skinning=true;
    ufbx_scene *scene=ufbx_load_file(argv[1],&opts,&error);
    if(!scene){fprintf(stderr,"ufbx load failed: %s\n",error.description.data);return 1;}
    printf("{\"vertices\":[");
    size_t count=0;
    for(size_t n=0;n<scene->nodes.count;n++){
        ufbx_node *node=scene->nodes.data[n]; ufbx_mesh *m=node->mesh;
        if(!m)continue;
        for(size_t v=0;v<m->vertices.count;v++){
            ufbx_vec3 p=ufbx_transform_position(&node->geometry_to_world,m->vertices.data[v]);
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
    printf("],\"weight_scope\":\"RAW_WEIGHTS_NOT_ASSUMED_NORMALIZED\"}\n");
    ufbx_free_scene(scene);return 0;
}
