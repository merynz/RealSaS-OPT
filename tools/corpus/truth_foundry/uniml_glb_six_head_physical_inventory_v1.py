#!/usr/bin/env python3
"""Truth Foundry: actual canonical GLB physical census for six heads.

Consumes existing SHA-locked UniML3D canonical .glb files (never original source
truth). Enumerates meshes, mesh material/UV/skin attributes, triangle topology,
scene nodes, parent graph, rig bind/accessors and per-head evidence opportunities.
Never promotes to source-original, anatomical semantic, 2D art or train PASS.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
import struct

from tools.corpus.truth_foundry.glb_pixel_court_v1 import read_glb, audit

SCHEMA = 'RealSaS.UniML3DCanonicalPhysicalSixHeadCensus.v1'
HEADS = ('IRIS_GEOMETRY','IRIS_SEMANTIC','IRIS_GRAPHICS','TESSA','AXIS','MIRA')
MAX_COUNT = 3_000_000


def require(value, code):
    if not value:
        raise ValueError('CANONICAL_SIX_HEAD_'+code)


def hashed_file(p:Path) -> str:
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):
            h.update(b)
    return h.hexdigest()


def count_accessors(root, accessor_id, expected=None):
    accessors=root.get('accessors',[])
    if not isinstance(accessor_id,int) or not 0<=accessor_id<len(accessors):
        return None
    a=accessors[accessor_id]
    if expected is not None and a.get('type') != expected:
        return None
    v=a.get('count')
    if not isinstance(v,int) or v<0 or v>MAX_COUNT:
        return None
    return v


def examine(root, pixel):
    meshes=root.get('meshes',[])
    nodes=root.get('nodes',[])
    skins=root.get('skins',[])
    accessors=root.get('accessors',[])
    animations=root.get('animations',[])
    if not all(isinstance(v,list) for v in (meshes,nodes,skins,accessors,animations)):
        raise ValueError('INVALID_GLTF_ROOT_COLLECTION')
    pparents=Counter()
    for node in nodes:
        for child in node.get('children',[]):
            if not isinstance(child,int) or child<0 or child>=len(nodes):
                raise ValueError('BAD_CHILD_ID')
            pparents[child]+=1
    all_skin_joints=set()
    skin_joint_counts=[]
    skin_has_inv_bind=0
    skin_unknown_inverse_bind=0
    joint_names=[]
    for skin in skins:
        joints=skin.get('joints',[])
        if not isinstance(joints,list) or any(not isinstance(j,int) or not 0<=j<len(nodes) for j in joints):
            raise ValueError('BAD_SKIN_JOINT')
        skin_joint_counts.append(len(joints))
        all_skin_joints.update(joints)
        joint_names.extend(str(nodes[j].get('name',''))[:128] for j in joints)
        ibm=skin.get('inverseBindMatrices')
        if ibm is not None:
            if count_accessors(root,ibm, 'MAT4')==len(joints):
                skin_has_inv_bind+=1
            else:
                skin_unknown_inverse_bind+=1
        else:
            skin_unknown_inverse_bind+=1
    prims=0;tris=0;pos_vertices=0
    declared_skinned=0; joint_weight_pairs=0
    uv_count=0;normal_count=0;invalid_count=0; materials_bound=0
    topology_modes=Counter(); morph_targets=0; min_mesh_vertex_count=0
    all_weight_attribute_pairs=[]
    for mesh in meshes:
        for p in mesh.get('primitives',[]):
            prims+=1
            mode=p.get('mode',4)
            topology_modes[str(mode)]+=1
            attrs=p.get('attributes') or {}
            if not isinstance(attrs,dict):
                invalid_count+=1;continue
            n=count_accessors(root,attrs.get('POSITION'),'VEC3')
            if n is None:
                invalid_count+=1;continue
            pos_vertices+=n
            min_mesh_vertex_count+=n
            uv_count+=int(count_accessors(root,attrs.get('TEXCOORD_0'),'VEC2')==n)
            normal_count+=int(count_accessors(root,attrs.get('NORMAL'),'VEC3')==n)
            materials_bound+=int(isinstance(p.get('material'),int) and 0<=p['material']<len(root.get('materials',[])))
            idx=p.get('indices')
            faces=count_accessors(root,idx,'SCALAR') if idx is not None else n
            if mode==4 and faces is not None and faces%3==0:
                tris+=faces//3
            elif mode==4:
                invalid_count+=1
            joints_present=[]
            for j in (0,1):
                ja=count_accessors(root,attrs.get('JOINTS_'+str(j)), 'VEC4')
                wa=count_accessors(root,attrs.get('WEIGHTS_'+str(j)), 'VEC4')
                joints_present.append((ja,wa))
                if ja==n and wa==n:
                    joint_weight_pairs+=1
            declared_skinned+=int(joints_present[0]==(n,n))
            morph_targets+=len(p.get('targets',[]))
            all_weight_attribute_pairs.append({'primitive':prims-1,'weight_joint_counters':joints_present})
    anim_channels=0;rot_channels=0;trans_channels=0;scale_channels=0;weight_channels=0
    anim_target_nodes=set()
    invalid_anim_channels=0
    for anim in animations:
        samplers=anim.get('samplers',[])
        for ch in anim.get('channels',[]):
            anim_channels+=1
            s=ch.get('sampler')
            if not isinstance(s,int) or not 0<=s<len(samplers):
                invalid_anim_channels+=1;continue
            target=ch.get('target') or {}
            node=target.get('node'); prop=target.get('path')
            if not isinstance(node,int) or not 0<=node<len(nodes):
                invalid_anim_channels+=1;continue
            anim_target_nodes.add(node)
            rot_channels+=int(prop=='rotation')
            trans_channels+=int(prop=='translation')
            scale_channels+=int(prop=='scale')
            weight_channels+=int(prop=='weights')
    return {
        'meshes':len(meshes),'nodes':len(nodes),'node_graph_edges':sum(pparents.values()),
        'node_multi_parent_count':sum(x>1 for x in pparents.values()),
        'node_root_count':len(nodes)-len(pparents),
        'node_with_skin_count':sum('skin' in n for n in nodes),
        'node_with_mesh_count':sum('mesh' in n for n in nodes),
        'skins':len(skins),'skin_joints_total':sum(skin_joint_counts),
        'skin_joints_unique_node_count':len(all_skin_joints),
        'skin_inverse_bind_matrix_accessor_pass':skin_has_inv_bind,
        'skin_inverse_bind_matrix_unknown_or_bad_count':skin_unknown_inverse_bind,
        'source_named_joint_hint_count':sum(bool(n.strip()) for n in joint_names),
        'mesh_primitives':prims,'position_accessor_vertex_records':pos_vertices,
        'indices_triangle_count_estimate':tris,
        'normal_primitive_count':normal_count,'uv_primitive_count':uv_count,
        'primitive_with_material_count':materials_bound,
        'primitive_with_joints_weights_0_count':declared_skinned,
        'primitive_joint_weight_pair_count':joint_weight_pairs,
        'morph_target_count':morph_targets,
        'invalid_primitive_or_accessor_count':invalid_count,
        'primitive_modes':dict(topology_modes),
        'animation_clip_count':len(animations),'animation_channel_count':anim_channels,
        'animation_node_target_count':len(anim_target_nodes),
        'animation_rotation_channel_count':rot_channels,
        'animation_translation_channel_count':trans_channels,
        'animation_scale_channel_count':scale_channels,
        'animation_weight_channel_count':weight_channels,
        'animation_invalid_channel_count':invalid_anim_channels,
        'glb_decoded_image_count':pixel['decoded_image_count'],
        'glb_texture_uv_raster_candidate':pixel['candidate_pixel_rasterization_possible'],
        'glb_sha256':pixel['glb_sha256'],
        'independent_source_sha_match':False,
        'teacher_image_rendered':False,'author_appearance_preserved':False,
        'manifold_or_sdf_sign_qualified':False,
        'source_face_topology_parity':False,
        'source_skeleton_articulation_qualified':False,
        'source_skin_deform_parity':False,
        'part_semantic_concept_reviewed':False,
        'true_drawable_count_approved':False,
        'fit_training_authorized':False,
    }


def make_targets(source_key, stats):
    facts={
      'IRIS_GEOMETRY':[
        ('canonical_3d_position_vertices',stats['position_accessor_vertex_records']),
        ('canonical_glb_triangles',stats['indices_triangle_count_estimate']),
        ('original_source_face_truth','UNKNOWN'),
        ('canonical_camera_8_view_depth_normals','NOT_RENDERED'),
      ],
      'IRIS_SEMANTIC':[
        ('joint_name_source_hint_count',stats['source_named_joint_hint_count']),
        ('source_face_anatomical_mask','UNKNOWN'),
        ('source_face_part_instance_mask','UNKNOWN'),
        ('full_keep_hide_source_reveal','NOT_RENDERED'),
      ],
      'IRIS_GRAPHICS':[
        ('decoded_canonical_image_count',stats['glb_decoded_image_count']),
        ('textured_uv_primitive_count',stats['uv_primitive_count']),
        ('eight_source_native_rgba_views','NOT_RENDERED'),
        ('stylized_art_approval','UNKNOWN'),
      ],
      'TESSA':[
        ('canonical_indexed_triangle_count',stats['indices_triangle_count_estimate']),
        ('source_topology_face_correspondence','UNKNOWN'),
        ('deformation_bend_response_teacher','UNKNOWN'),
      ],
      'AXIS':[
        ('canonical_unique_joint_node_count',stats['skin_joints_unique_node_count']),
        ('native_rest_bind_accessor_count',stats['skin_inverse_bind_matrix_accessor_pass']),
        ('animation_channel_count',stats['animation_channel_count']),
        ('native_ik_fk_controller_truth','UNKNOWN'),
      ],
      'MIRA':[
        ('joint_skin_weight_primitive_count',stats['primitive_with_joints_weights_0_count']),
        ('skin_pose_deformation_authority','UNKNOWN'),
        ('weight_value_normalization_court','NOT_RUN'),
      ],
    }
    for head,targets in facts.items():
        for target,value in targets:
            yield {'source_key':source_key,'head':head,'target':target,'observed_value':value,
                   'evidence_provenance':'CANONICAL_GLB_STRUCTURAL_ONLY',
                   'head_target_training_admitted':'False','source_original_rights_court':'UNKNOWN',
                   'source_authority_court':'UNKNOWN','independent_truth_court':'NOT_RUN'}


def run(glb_dir:Path,manifest_csv:Path,out:Path,expected_count=84):
    glb_dir,manifest_csv,out=map(Path,(glb_dir,manifest_csv,out))
    with manifest_csv.open(newline='',encoding='utf-8') as f:
        rows=list(csv.DictReader(f))
    require(len(rows)==96,'MANIFEST_COUNT')
    out.mkdir(parents=True,exist_ok=True)
    census=[];targets=[];status=Counter()
    for m in rows:
        source_key='UNIML3D:'+m['object_type']
        if m['download_status']!='DOWNLOADED':
            status['missing_or_failed_downloads']+=1
            continue
        glb=glb_dir/(m['object_type']+'.glb')
        require(glb.is_file(),'MISSING_FILE_'+m['object_type'])
        digest=hashed_file(glb)
        require(digest==m['download_sha256'],'CANONICAL_GLB_SHA_MISMATCH')
        root,binary,got=read_glb(glb)
        require(got==digest,'CANONICAL_READER_SHA_MISMATCH')
        stat=examine(root,audit(glb))
        stat.update(source_key=source_key,provider_category_claim=m.get('category_provider_claim',''),
                    canonical_gltf_sha256=digest,original_source_licensed='UNKNOWN',
                    shader_style_court='NOT_RUN',source_frame_front_axis='UNKNOWN')
        census.append(stat)
        targets.extend(make_targets(source_key,stat))
        status['glbs_sha_verified']+=1
        status['canonical_triangle_meshes_observed']+=int(stat['indices_triangle_count_estimate']>0)
        status['canonical_skin_joint_structures_observed']+=int(stat['skin_joints_total']>0)
        status['canonical_weight_attribute_observed']+=int(stat['primitive_with_joints_weights_0_count']>0)
        status['canonical_embedded_pixel_raster_possible']+=int(stat['glb_texture_uv_raster_candidate'])
        status['canonical_animation_clip_observed']+=int(stat['animation_clip_count']>0)
    require(len(census)==expected_count,'EXPECTED_GLBS')
    require(all(x['head_target_training_admitted']=='False' for x in targets),'FALSE_TRAINING_PROMOTION')
    f1=out/'UNIML3D_CANONICAL_PHYSICAL_84_ASSETS_V1.csv'
    # Keep diagnostics as JSON for nested object-mode counts.
    with f1.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(census[0]));w.writeheader()
        for r in census:
            w.writerow({k:json.dumps(v,sort_keys=True) if isinstance(v,(dict,list)) else v for k,v in r.items()})
    f2=out/'UNIML3D_CANONICAL_SIX_HEAD_TARGETS_V1.csv'
    with f2.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(targets[0]));w.writeheader();w.writerows(targets)
    summary={
      'schema':SCHEMA,'source_manifest_sha256':hashed_file(manifest_csv),
      'source_glbs_canonically_sha_verified':len(census),
      'source_download_failures':len(rows)-len(census),
      'per_head_targets_written':len(targets),
      'observed_counters':dict(status),
      'original_source_rights_qualified':0,
      'geometry_sdf_sign_qualified':0,'semantic_anatomical_labels_approved':0,
      'graphics_stylized_art_approved':0,'tessa_source_topology_teacher_pass':0,
      'axis_ik_controls_pass':0,'mira_deformation_teacher_pass':0,
      'per_head_training_pass':{x:0 for x in HEADS},
      'data_exclusively_canonical_glb_not_original_asset':True,
      'raw_3d_files_duplicated_to_output':False,
      'outputs_sha256':{p.name:hashed_file(p) for p in (f1,f2)},
    }
    (out/'SUMMARY_V1.json').write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n')
    return summary


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--glb-dir',type=Path,required=True)
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    print(json.dumps(run(a.glb_dir,a.manifest,a.out),indent=2))

if __name__=='__main__': main()