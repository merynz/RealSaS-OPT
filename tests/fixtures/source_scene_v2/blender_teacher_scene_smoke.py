"""Synthetic Blender witness for source reset, isolated rotation and root gauge."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import bpy
import numpy as np

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from tools.truth_foundry.blender_teacher_scene_v1 import build
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
    job={'schema':'RealSaS.TeacherSceneCalibrationJob.v1','subject_key':'synthetic_fixture','source':ref(source),
         'observation_admission':admission,'native_teacher':ref(native),'source_scene_snapshot':snapshot,
         'source_equivalence_report':report,'policy':policy,'observation_bundle_sha256':'1'*64,
         'normalization_sha256':'2'*64,'producer_source_commit':'3'*40,'evaluator_binary_sha256':'4'*64,'training_allowed':False}
    out=folder/'teacher';build(job,out)
    scene=json.loads((out/'TEACHER_SCENE.json').read_text())
    assert scene['qualification_minted'] is False and scene['student_input_allowed'] is False
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
        'source_reset_isolated_rotation_and_root_gauge':True,'real_corpus_qualified':False,'training_executed':False},indent=2)+'\n')
    print('TEACHER_SCENE_SYNTHETIC_WITNESS_PASS',flush=True)


if __name__=='__main__':
    raw=sys.argv[sys.argv.index('--')+1:];p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args(raw);main(a.output)
