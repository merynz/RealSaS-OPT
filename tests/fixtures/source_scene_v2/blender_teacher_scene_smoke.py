"""Synthetic Blender witness for source reset, isolated rotation and root gauge."""
import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys

import bpy
import numpy as np

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from tools.truth_foundry.blender_teacher_scene_v1 import sha256
from compiler.realsas_compiler_services.orchestrator.adapters.truth_foundry_v1 import measure_teacher_scene_stage
from compiler.realsas_compiler_core.preproduct_authority_v1 import NormalizationDomainIR, normalization_domain_hash
from tools.iris_vnext import blender_authored_teacher_export_v1 as base
from tools.iris_vnext import blender_authored_teacher_export_v3 as transport


def ref(path):
    return {'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}


def write(path,value):
    path.write_text(json.dumps(value,sort_keys=True,indent=2)+'\n');return ref(path)


def main(folder):
    folder=Path(folder).resolve();folder.mkdir(parents=True,exist_ok=True)
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.object.armature_add();armature=bpy.context.object;bone=armature.data.bones[0].name
    bpy.ops.mesh.primitive_cube_add(size=1);mesh=bpy.context.object
    group=mesh.vertex_groups.new(name=bone);group.add(list(range(len(mesh.data.vertices))),1.,'REPLACE')
    modifier=mesh.modifiers.new('fixture_rig','ARMATURE');modifier.object=armature
    material=bpy.data.materials.new('fixture_material');material.use_nodes=True;mesh.data.materials.append(material)
    image=bpy.data.images.new('fixture_image',width=2,height=2,alpha=True)
    image.pixels.foreach_set(np.array([.2,.4,.6,1.]*4,dtype=np.float32))
    texture=material.node_tree.nodes.new('ShaderNodeTexImage');texture.image=image
    shader=next(n for n in material.node_tree.nodes if n.type=='BSDF_PRINCIPLED')
    material.node_tree.links.new(texture.outputs['Color'],shader.inputs['Base Color'])
    source=folder/'fixture.glb';bpy.ops.export_scene.gltf(filepath=str(source),export_format='GLB',export_animations=False)
    snapshots=[]
    for _ in range(2):
        base._import_animated(source);base._neutralize_animation()
        snapshots.append(transport._extract_snapshot(base._mesh_objects(),freeze_triangulation=True))
    for key in ('vertices','faces','face_owner','face_material'):
        assert np.array_equal(snapshots[0][key],snapshots[1][key]),key
    rest=snapshots[0];native=folder/'native.npz'
    np.savez_compressed(native,rest_vertices_source_world=rest['vertices'],faces=rest['faces'],
                        face_owner=rest['face_owner'],face_material=rest['face_material'])
    policy=write(folder/'policy.json',{'schema':'RealSaS.TeacherKinematicCalibrationPolicy.v1',
               'angle_amplitudes_radians':[.01],'root_translation_source_units':[.1,-.2,.3]})
    report=write(folder/'replay.json',{'schema':'RealSaS.NativeTeacherReplayReport.v1','status':'EXACT_SEALED_REPLAY',
                 'subjects':[{'subject_id':'synthetic_fixture','exact_arrays_equal':True}],
                 'scope':'SYNTHETIC_SOURCE_REIMPORT_ONLY__NOT_REAL_CORPUS'})
    admission=write(folder/'admission.json',{'source_sha256':ref(source)['sha256'],'subject_id':'synthetic_fixture'})
    snapshot=write(folder/'snapshot.json',{'source_world_to_canonical':np.eye(4).tolist()})
    # Exercise the actual Compiler wire record -> released adapter -> Blender
    # boundary. Calling build(job) directly had missed a wrong normalization key.
    norm=NormalizationDomainIR('1'*64,'2'*64,(0.,0.,0.),1.,(-1.,-1.,-1.),(1.,1.,1.),'REALSAS_OBJECT_FRAME','')
    norm=replace(norm,normalization_hash=normalization_domain_hash(norm))
    parents=[('01_SOURCE_BYTES_SEALED','RealSaS.SourceByteSeal.v1',{'schema':'RealSaS.SourceByteSeal.v1','files':[ref(source)]}),
             ('03_SOURCE_MECHANICAL_AUDIT','RealSaS.SourceMechanicalAuditIR.v1',{'schema':'RealSaS.SourceMechanicalAuditIR.v1'}),
             ('08_NORMALIZATION_DOMAIN_QUALIFIED',norm.schema_version,norm.to_dict())]
    ledger=[]
    for sid,schema,value in parents:
        bound=write(folder/(sid+'.json'),value)
        ledger.append({'id':sid,'status':'PASS','outputs':[{**bound,'schema':schema}]})
    index=write(folder/'native-index.json',{'schema':'RealSaS.NativeTeacherBaselineBindings.v1','subjects':[
        {'subject_key':'synthetic_fixture','native_teacher':ref(native)}]})
    binary=Path(bpy.app.binary_path).resolve()
    cfg={'mode':'SOURCE_KINEMATIC_CALIBRATION_ONLY','native_baselines':index,'policy':policy,
         'source_equivalence_report':report,'evaluator_binary':{'path':str(binary),'sha256':sha256(binary)},
         'evaluator_version':'Blender '+bpy.app.version_string,'maximum_seconds_per_subject':60,'producer_source_commit':'3'*40}
    result=measure_teacher_scene_stage({'repo_root':ROOT,'run_root':folder/'synthetic-run',
        'stage':{'id':'47_SOURCE_RESPONSE_CALIBRATION','policy':{'product_pass_authority':False}},
        'ledger':{'execution_class':'WITNESS','stages':ledger},'run_manifest':{'subject_id':'synthetic_fixture',
            'truth_foundry':cfg,'source_files':[{'path':str(source),'expected_sha256':ref(source)['sha256'],'source_kind':'PRIMARY'}],
            'source_audit':{'iris_observation_admission':admission,'source_scene_snapshot':snapshot}}})
    assert result['status']=='PASS'
    out=Path(next(r['path'] for r in result['outputs'] if r['schema']=='RealSaS.TeacherSceneIR.v1')).parent
    scene=json.loads((out/'TEACHER_SCENE.json').read_text())
    assert scene['qualification_minted'] is False and scene['student_input_allowed'] is False
    assert scene['observation_bundle_sha256']==norm.observation_set_binding_hash
    assert scene['normalization_sha256']==norm.normalization_hash
    assert scene['calibration']['sham_repeat_max_abs']==0
    assert scene['calibration']['source_rest_to_pose_identity_max_abs']<1e-6
    assert scene['calibration']['common_root_translation_max_abs']<1e-6
    appearance=json.loads((out/'appearance.json').read_text())
    assert appearance['images'] and appearance['qualified_sampler'] is False
    with np.load(out/'image_pixels.npz',allow_pickle=False) as pixels:
        for image in appearance['images']:
            assert pixels[image['array']].size==np.prod(image['size'])*image['buffer_channels']
    with np.load(out/'geometry_and_response.npz',allow_pickle=False) as arrays:
        assert arrays['isolated_position'].shape[0]==6
        assert np.array_equal(arrays['sham_before'],arrays['sham_after'])
        assert np.any(np.abs(arrays['isolated_position']-arrays['pose_identity_baseline'])>1e-4)
        frame=np.asarray(scene['calibration']['native_control_frames_privileged'][0]['rest_matrix_world'])
        original=arrays['pose_identity_baseline']
        assert arrays['source_weight_bone_indices_privileged'].max()<len(scene['calibration']['native_control_frames_privileged'])
        for index,action in enumerate(scene['calibration']['actions']):
            axis=scene['privileged_control_refs'][action['control_index_privileged']]['axis_in_native_rest_frame']
            from mathutils import Matrix
            transform=frame@np.asarray(Matrix.Rotation(action['sign']*action['amplitude'],4,'XYZ'[axis]))@np.linalg.inv(frame)
            expected=original@transform[:3,:3].T+transform[:3,3]
            np.testing.assert_allclose(arrays['isolated_position'][index],expected,atol=1e-6,rtol=0)
    (folder/'SMOKE_REPORT.json').write_text(json.dumps({'schema':'RealSaS.TeacherSceneSyntheticWitness.v1',
        'source_reset_isolated_rotation_and_root_gauge':True,'actual_compiler_normalization_wire_and_adapter':True,
        'real_corpus_qualified':False,'training_executed':False},indent=2)+'\n')
    print('TEACHER_SCENE_SYNTHETIC_WITNESS_PASS',flush=True)


if __name__=='__main__':
    raw=sys.argv[sys.argv.index('--')+1:];p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args(raw);main(a.output)
