"""Blender source-space teacher scene and isolated kinematic calibration.

Invoked by the released Platform adapter, not a notebook. Source names/weights/
control handles are privileged evaluator data. The producer does not choose
student target channels, an action covariance, physical constraints, semantic
attachments, observability, truth admission or canonical M/G/W authority.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import bpy
from mathutils import Matrix
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from tools.iris_vnext import blender_authored_teacher_export_v1 as source_transport
from tools.iris_vnext import blender_authored_teacher_export_v3 as geometry_transport


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def bound_ref(ref):
    path=Path(ref['path']).resolve()
    if not path.is_file() or sha256(path)!=ref['sha256']:raise ValueError('TEACHER_SCENE_BOUND_INPUT_DRIFT')
    return path


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()).hexdigest()


def _appearance_and_uv(meshes,rest):
    """Preserve sampler inputs; this is not a qualified shader/ray evaluator."""
    deps=bpy.context.evaluated_depsgraph_get();uv=[];uv_valid=[];materials=[];images={}
    for owner,obj in enumerate(meshes):
        evaluated=obj.evaluated_get(deps);mesh=evaluated.to_mesh(preserve_all_data_layers=True,depsgraph=deps)
        try:
            mesh.calc_loop_triangles();layer=mesh.uv_layers.active
            for tri in mesh.loop_triangles:
                if layer is None:
                    uv.append([[np.nan,np.nan]]*3);uv_valid.append(False)
                else:
                    value=[list(layer.data[i].uv) for i in tri.loops]
                    uv.append(value);uv_valid.append(bool(np.isfinite(value).all()))
        finally:evaluated.to_mesh_clear()
        for slot_index,slot in enumerate(obj.material_slots):
            material=slot.material
            row={'privileged_owner':owner,'slot':slot_index,'material_name':material.name if material else None,
                 'diffuse_rgba':list(material.diffuse_color) if material else None,'node_graph':None,
                 'surface_render_method':str(getattr(material,'surface_render_method','UNKNOWN')),
                 'sampler_qualification':'NOT_RUN'}
            if material and material.use_nodes and material.node_tree:
                nodes=[]
                for node in material.node_tree.nodes:
                    defaults={}
                    for socket in node.inputs:
                        value=getattr(socket,'default_value',None)
                        if isinstance(value,(float,int,bool,str)):defaults[socket.identifier]=value
                        elif value is not None:
                            try:defaults[socket.identifier]=list(value)
                            except TypeError:pass
                    properties={name:getattr(node,name) for name in ('interpolation','extension','projection','blend_type',
                                'operation','use_clamp','data_type','uv_map','attribute_name','vector_type')
                                if hasattr(node,name) and isinstance(getattr(node,name),(str,float,int,bool))}
                    nodes.append({'name':node.name,'type':node.bl_idname,'inputs':defaults,'properties':properties,
                                  'image':node.image.name if hasattr(node,'image') and node.image else None})
                    if hasattr(node,'image') and node.image:
                        image=node.image
                        if image.name not in images:
                            count=len(image.pixels);pixels=np.empty(count,dtype=np.float32);image.pixels.foreach_get(pixels)
                            width,height=map(int,image.size)
                            if width<=0 or height<=0 or count%(width*height):raise ValueError('TEACHER_SCENE_IMAGE_BUFFER_LAYOUT')
                            images[image.name]={'pixels':pixels,'size':[width,height],'buffer_channels':count//(width*height),
                                               'source_file_channels':int(image.channels),'buffer_is_float':bool(image.is_float),
                                               'colour_space':image.colorspace_settings.name,'alpha_mode':image.alpha_mode,
                                               'value_domain':'BLENDER_IMAGE_BUFFER__SOURCE_RENDERER_QUALIFICATION_REQUIRED'}
                row['node_graph']={'nodes':nodes,'links':[{'from_node':l.from_node.name,'from_socket':l.from_socket.identifier,
                         'to_node':l.to_node.name,'to_socket':l.to_socket.identifier} for l in material.node_tree.links]}
            materials.append(row)
    uv=np.asarray(uv,dtype=np.float64);uv_valid=np.asarray(uv_valid,dtype=bool)
    if uv.shape!=(len(rest['faces']),3,2):raise ValueError('TEACHER_SCENE_UV_TOPOLOGY_DRIFT')
    return uv,uv_valid,materials,images


def _control_inventory(armatures,source_sha):
    handles=[];lookup=[]
    for armature in armatures:
        for bone in sorted(armature.pose.bones,key=lambda x:x.name):
            for axis in range(3):
                key={'source_sha256':source_sha,'armature':armature.name,'bone':bone.name,'axis':axis}
                handles.append({'opaque_privileged_handle':digest(key),'kind':'NATIVE_POSE_BONE_LOCAL_ROTATION',
                     'units':'RADIAN','axis_in_native_rest_frame':axis,'source_lookup_privileged_only':key,
                     'authored_capability':'KINEMATIC_PARAMETER__NOT_PHYSICAL_DOF_OR_JOINT_LIMIT',
                     'source_constraints':[c.type for c in bone.constraints]})
                lookup.append((armature,bone,axis))
    return handles,lookup


def build(job,output):
    if job.get('schema')!='RealSaS.TeacherSceneCalibrationJob.v1' or job.get('training_allowed') is not False:
        raise ValueError('TEACHER_SCENE_JOB_SCOPE')
    source=bound_ref(job['source']);admission_path=bound_ref(job['observation_admission'])
    baseline_path=bound_ref(job['native_teacher']);snapshot_path=bound_ref(job['source_scene_snapshot'])
    policy_path=bound_ref(job['policy']);policy=json.loads(policy_path.read_text())
    equivalence=json.loads(bound_ref(job['source_equivalence_report']).read_text())
    if (equivalence.get('status')!='EXACT_SEALED_REPLAY' or not any(r.get('subject_id')==job['subject_key']
                and r.get('exact_arrays_equal') is True for r in equivalence.get('subjects',[]))):
        raise ValueError('TEACHER_SCENE_EXACT_NATIVE_REPLAY_REQUIRED')
    if policy.get('schema')!='RealSaS.TeacherKinematicCalibrationPolicy.v1':raise ValueError('TEACHER_SCENE_POLICY_SCHEMA')
    angles=np.asarray(policy['angle_amplitudes_radians'],dtype=np.float64)
    if angles.ndim!=1 or len(angles)==0 or not np.isfinite(angles).all() or np.any(angles<=0):
        raise ValueError('TEACHER_SCENE_EXPLICIT_AMPLITUDES_REQUIRED')
    admission=json.loads(admission_path.read_text());snapshot=json.loads(snapshot_path.read_text())
    if admission['source_sha256']!=job['source']['sha256'] or admission['subject_id']!=job['subject_key']:
        raise ValueError('TEACHER_SCENE_SOURCE_ADMISSION_DRIFT')
    source_transport._import_animated(source);source_transport._neutralize_animation()
    meshes=source_transport._mesh_objects();rest=geometry_transport._extract_snapshot(meshes,freeze_triangulation=True)
    with np.load(baseline_path,allow_pickle=False) as native:
        for role,key in [('vertices','rest_vertices_source_world'),('faces','faces'),('face_owner','face_owner'),('face_material','face_material')]:
            if not np.array_equal(rest[role],native[key]):raise ValueError('TEACHER_SCENE_SEALED_REST_DRIFT:'+role)
    uv,uv_valid,materials,images=_appearance_and_uv(meshes,rest)
    armatures=tuple(sorted((o for o in bpy.context.scene.objects if o.type=='ARMATURE'),key=lambda x:x.name))
    handles,controls=_control_inventory(armatures,job['source']['sha256'])
    if not controls:raise ValueError('TEACHER_SCENE_NO_SUPPORTED_KINEMATIC_CAPABILITY')
    object_basis={o.name:o.matrix_basis.copy() for o in bpy.context.scene.objects}
    roots=tuple(o for o in bpy.context.scene.objects if o.parent is None)
    shape_values=[(o.data.shape_keys,[key.value for key in o.data.shape_keys.key_blocks]) for o in meshes if o.data.shape_keys]
    def reset():
        source_transport._neutralize_animation()
        for o in bpy.context.scene.objects:o.matrix_basis=object_basis[o.name].copy()
        for obj in armatures:
            obj.data.pose_position='POSE'
            for bone in obj.pose.bones:bone.matrix_basis=Matrix.Identity(4)
        for keys,values in shape_values:
            for key,value in zip(keys.key_blocks,values):key.value=value
        bpy.context.view_layer.update()
    def evaluate():
        bpy.context.view_layer.update();current=geometry_transport._extract_snapshot(meshes,freeze_triangulation=False)
        if current['polygon_signature']!=rest['polygon_signature'] or not np.isfinite(current['vertices']).all():
            raise ValueError('TEACHER_SCENE_ACTION_TOPOLOGY_OR_NUMERIC_DRIFT')
        return current['vertices']
    reset();baseline=evaluate();reset();sham=evaluate()
    action_rows=[];responses=[];started=time.monotonic()
    total=len(controls)*len(angles)*2
    for ci,(armature,bone,axis) in enumerate(controls):
        for amplitude_index,amplitude in enumerate(angles):
            for sign in (-1,1):
                reset();bone.matrix_basis=Matrix.Rotation(float(sign*amplitude),4,'XYZ'[axis])
                responses.append(evaluate())
                action_rows.append({'control_index_privileged':ci,'amplitude_index':amplitude_index,'sign':sign,
                                    'kind':'ISOLATED_KINEMATIC_PARAMETER_PERTURBATION','units':'RADIAN',
                                    'amplitude':float(amplitude),'opaque_privileged_handle':handles[ci]['opaque_privileged_handle']})
                if len(action_rows)<=3 or len(action_rows)%30==0:
                    elapsed=time.monotonic()-started
                    print(f'[TEACHER_ACTION] {len(action_rows)}/{total} elapsed_s={elapsed:.2f} eta_s={elapsed/max(1,len(action_rows))*(total-len(action_rows)):.2f}',flush=True)
    reset();sham_after=evaluate()
    # A common root translation is a gauge control, not a learned attachment.
    translation=np.asarray(policy['root_translation_source_units'],dtype=np.float64)
    if translation.shape!=(3,) or not np.isfinite(translation).all():raise ValueError('TEACHER_SCENE_ROOT_CONTROL_POLICY')
    transform=Matrix.Translation(tuple(translation))
    for obj in roots:obj.matrix_world=transform@obj.matrix_world
    root_response=evaluate();reset()
    # Preserve source rest control frames and source skin weights as evaluator
    # inputs, never student labels or canonical Rig/Skin. Unsupported mechanisms
    # remain explicit; these weights alone do not prove skin/evaluator equivalence.
    bone_rows=[];bone_index={}
    for obj in armatures:
        for bone in sorted(obj.data.bones,key=lambda b:b.name):
            bone_index[(obj.name,bone.name)]=len(bone_rows)
            bone_rows.append({'armature':obj.name,'name':bone.name,'parent':bone.parent.name if bone.parent else None,
                              'rest_matrix_world':[list(r) for r in obj.matrix_world@bone.matrix_local]})
    w_offsets=[0];w_indices=[];w_values=[];weight_coverage=[]
    for obj in meshes:
        armature=obj.find_armature()
        for vertex in obj.data.vertices:
            found=[]
            if armature:
                for group in vertex.groups:
                    name=obj.vertex_groups[group.group].name;index=bone_index.get((armature.name,name))
                    if index is not None and group.weight>0:found.append((index,float(group.weight)))
            elif obj.parent and obj.parent.type=='ARMATURE' and obj.parent_type=='BONE':
                index=bone_index.get((obj.parent.name,obj.parent_bone))
                if index is not None:found=[(index,1.)]
            for index,weight in found:w_indices.append(index);w_values.append(weight)
            w_offsets.append(len(w_indices));weight_coverage.append(bool(found))
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    arrays={'rest_vertices_source_world':rest['vertices'],'faces':rest['faces'],'face_owner_privileged':rest['face_owner'],
            'face_material_privileged':rest['face_material'],'triangle_uv':uv,'triangle_uv_valid':uv_valid,
            'pose_identity_baseline':baseline,'sham_before':sham,'sham_after':sham_after,
            'isolated_position':np.asarray(responses,dtype=np.float64),'root_control_position':root_response,
            'source_weight_offsets':np.asarray(w_offsets,np.int64),'source_weight_bone_indices_privileged':np.asarray(w_indices,np.int64),
            'source_weight_values':np.asarray(w_values,np.float64),'source_weight_coverage':np.asarray(weight_coverage,bool)}
    if len(w_offsets)!=len(rest['vertices'])+1:raise ValueError('TEACHER_SCENE_SOURCE_WEIGHT_VERTEX_CORRESPONDENCE_UNSUPPORTED')
    np.savez_compressed(output/'geometry_and_response.npz',**arrays)
    image_arrays={};image_rows=[]
    for i,(name,image) in enumerate(sorted(images.items())):
        array_name=f'image_{i:04d}';image_arrays[array_name]=image.pop('pixels')
        image_rows.append({'privileged_image_name':name,'array':array_name,**image})
    np.savez_compressed(output/'image_pixels.npz',**image_arrays)
    appearance={'schema':'RealSaS.SourceAppearanceSamplerInputs.v1','materials':materials,'images':image_rows,
                'qualified_sampler':False,'linear_rgb_and_effective_alpha_measurement':'NOT_RUN'}
    (output/'appearance.json').write_text(json.dumps(appearance,sort_keys=True,indent=2,allow_nan=False)+'\n')
    # The packed source scene allows a renderer adapter to use the same evaluator
    # state instead of guessing a new material or skin implementation.
    source_transport._neutralize_animation();bpy.ops.file.pack_all()
    bpy.ops.wm.save_as_mainfile(filepath=str(output/'source_scene.blend'))
    files={p.name:{'path':str(p),'sha256':sha256(p),'size_bytes':p.stat().st_size} for p in output.iterdir() if p.is_file()}
    files['geometry_and_response.npz']['arrays']={k:{'dtype':v.dtype.str,'shape':list(v.shape)} for k,v in arrays.items()}
    scene={'schema':'RealSaS.TeacherSceneIR.v1','subject_key':job['subject_key'],'source_sha256':job['source']['sha256'],
           'job_sha256':digest(job),'evaluator_binary_sha256':job['evaluator_binary_sha256'],
           'topology_sha256':digest({'faces':rest['faces'].tolist(),'source_sha256':job['source']['sha256']}),
           'observation_bundle_sha256':job['observation_bundle_sha256'],'normalization_sha256':job['normalization_sha256'],
           'geometry_payloads':[files['geometry_and_response.npz']],
           'appearance_payloads':[files['appearance.json'],files['image_pixels.npz'],files['source_scene.blend']],
           'evaluator_source_commit':job['producer_source_commit'],'evaluator_config_sha256':job['policy']['sha256'],
           'native_clip_refs':[job['native_teacher']],'privileged_control_refs':handles,
           'constraint_capabilities':{'local_native_rotation_parameter':'MEASURED_CANDIDATE','physical_impulse':'UNAVAILABLE',
              'authored_physical_dof_or_joint_limit':'UNKNOWN','source_weight_lbs_equivalence':'NOT_QUALIFIED',
              'source_raster_sampler':'NOT_QUALIFIED','other_control_kinds':'UNSUPPORTED_BY_THIS_ADAPTER'},
           'coordinate_contract':{'frame':'SOURCE_WORLD','length_unit':'SOURCE_UNIT','time_unit':'NO_DYNAMIC_TIME_IN_STATIC_CALIBRATION',
                                  'source_world_to_normalized':snapshot['source_world_to_canonical']},
           'source_equivalence_report_sha256':job['source_equivalence_report']['sha256'],
           'calibration':{'actions':action_rows,'native_control_frames_privileged':bone_rows,
               'source_rest_to_pose_identity_max_abs':float(np.max(np.abs(rest['vertices']-baseline))),
               'sham_repeat_max_abs':float(np.max(np.abs(sham_after-sham))),
               'common_root_translation_max_abs':float(np.max(np.abs(root_response-(baseline+translation)))),
               'root_translation_source_units':translation.tolist(),'action_measure':'UNSEALED__NO_IMPLICIT_UNIFORM_LAW'},
           'authority':'UNQUALIFIED_DERIVED_MEASUREMENTS','student_input_allowed':False,'qualification_minted':False}
    (output/'TEACHER_SCENE.json').write_text(json.dumps(scene,sort_keys=True,indent=2,allow_nan=False)+'\n')
    print('[TEACHER_SCENE_WRITTEN]',job['subject_key'],len(action_rows),'actions; qualification pending',flush=True)


if __name__=='__main__':
    raw=sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else []
    p=argparse.ArgumentParser();p.add_argument('--job',required=True);p.add_argument('--output',required=True);a=p.parse_args(raw)
    build(json.loads(Path(a.job).read_text()),a.output)
