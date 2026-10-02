"""Same frozen G3 and motion-frame court for a freshly inferred rig and skin."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
from compiler.realsas_compiler_core.artifact_codec_v2 import (
    rigging_surface_from_dict,qualified_skeleton_from_dict,qualified_skin_from_dict,
    canonical_mesh_candidate_from_dict,qualified_camera_set_from_dict,mesh_policy_from_dict)
from compiler.realsas_compiler_core.deformation_envelope_derivation_v1 import derive_deformation_envelope_v1
from compiler.realsas_compiler_core.mesh.deformation_stress_v2 import run_g3_local_frame_micro_stress_v2
from compiler.realsas_compiler_core.mesh.deformation_stress_v1 import _candidate_skin_matrix
from compiler.realsas_compiler_core.mesh.dynamic_frame_court_v1 import measure_dynamic_frame_geometry_v1,sampled_pose_hash_v1
from compiler.realsas_compiler_core.motion_dynamic_proof_v2 import _joint_pose_v2
from compiler.realsas_compiler_services.orchestrator.adapters.adapter_io import stage_output_payload,sha256_file
from tools.demo.render_knight_motion_preview_v1 import _ctx,_skin,_tracks_for_clip
from tools.audit_knight_canonical_caa_mechanics_geometry_lock_v1 import CLIPS,FULL_MOTION_SAMPLES


def read(path):return json.loads(Path(path).read_text())
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--authority-root',type=Path,required=True);ap.add_argument('--run-id',required=True)
    ap.add_argument('--surface-json',type=Path,required=True);ap.add_argument('--candidate-json',type=Path,required=True);ap.add_argument('--inference-dir',type=Path,required=True)
    args=ap.parse_args();out=args.inference_dir
    ctx=_ctx(args.authority_root,args.run_id)
    surface=rigging_surface_from_dict(read(args.surface_json));candidate=canonical_mesh_candidate_from_dict(read(args.candidate_json))
    skeleton=qualified_skeleton_from_dict(read(out/'fresh_qualified_skeleton.json'));skin=qualified_skin_from_dict(read(out/'fresh_qualified_skin.json'))
    cameraset=qualified_camera_set_from_dict(stage_output_payload(ctx,'05_CAMERA_CONTRACT_SOLVED','RealSaS.QualifiedCameraSetIR.v1'))
    policy=mesh_policy_from_dict(stage_output_payload(ctx,'18_CANONICAL_MESH_ADDRESSING_BUILD','RealSaS.MeshQualificationPolicyIR.v1'))
    cameras=tuple(sorted(cameraset.cameras,key=lambda c:c.view_index))
    axis,envelope=derive_deformation_envelope_v1(skeleton=skeleton,camera_set=cameraset)
    g3=run_g3_local_frame_micro_stress_v2(candidate,surface=surface,skeleton=skeleton,skin=skin,envelope=envelope,cameras=cameras,policy=policy)
    (out/'fresh_g3_report.json').write_text(json.dumps(g3.to_dict(),sort_keys=True)+'\n')
    print('FRESH_G3='+json.dumps({'passed':g3.passed,'max_condition':g3.maximum_condition_number}),flush=True)
    rest,weights,faces=_candidate_skin_matrix(candidate,surface=surface,skeleton=skeleton,skin=skin);faces=np.asarray(faces)
    joint_ids=tuple(j.canonical_joint_id for j in skeleton.joints);source_report=read('canonical/KNIGHT_MOTION_SOURCE_ACTION_DIAGNOSTIC_20260927.json')
    rows=[];poses=[];mapping_rows={};source_hashes={}
    for clip in CLIPS:
        path=ctx['run_root']/'inputs/motion/quaternius_knight_v1'/(clip+'.motion.json');payload=read(path);source_hashes[clip]=sha256_file(path)
        tracks,mapping=_tracks_for_clip(payload,skeleton,cameras,source_report);mapping_rows[clip]=mapping
        times=np.linspace(0.,float(payload['duration_seconds']),FULL_MOTION_SAMPLES,endpoint=not bool(payload.get('loop')))
        for i,t in enumerate(times):
            mats,_,resthash=_joint_pose_v2(skeleton=skeleton,tracks=tracks,time_seconds=float(t),cameras=cameras)
            posed=_skin(rest,weights,joint_ids,mats);poses.append(posed)
            rows.append({'clip_id':clip,'frame_index':i,'time_seconds':float(t),
                'sampled_pose_hash':sampled_pose_hash_v1(clip_id=clip,time_seconds=float(t),rest_frame_set_hash=resthash,skin_matrices=mats,posed=posed),
                'geometry':measure_dynamic_frame_geometry_v1(rest=rest,posed=posed,faces=faces,policy=policy)})
        print('FRESH_MOTION_CLIP='+json.dumps({'clip':clip,'failed_frames':sum(not r['geometry']['passed'] for r in rows if r['clip_id']==clip)}),flush=True)
    np.savez_compressed(out/'fresh_sampled_geometry.npz',rest=rest,weights=weights,faces=faces,poses=np.asarray(poses))
    passed=g3.passed and all(r['geometry']['passed'] for r in rows)
    report={'status':'PASS_SAMPLED_CONDITIONING_ONLY' if passed else 'FAIL_FRESH_INFERENCE_DYNAMICS',
        'candidate_lineage_hash':candidate.candidate_lineage_hash,'surface_lineage_hash':surface.geometry_lineage_hash,
        'skeleton_lineage_hash':skeleton.skeleton_lineage_hash,'skin_lineage_hash':skin.skin_lineage_hash,
        'rig_inference_report_sha256':sha256_file(out/'rig_REPORT.json'),'skin_inference_report_sha256':sha256_file(out/'skin_REPORT.json'),
        'policy_hash':policy.qualification_policy_lineage_hash,'g3_passed':g3.passed,
        'frame_count':len(rows),'failed_motion_frame_count':sum(not r['geometry']['passed'] for r in rows),
        'maximum_motion_edge_ratio':max(r['geometry']['maximum_edge_ratio'] for r in rows),
        'motion':rows,'motion_source_sha256':source_hashes,'new_rig_retarget_mapping':mapping_rows,
        'retarget_rule_changed':False,'intersection_proof_executed':False,'product_authority_minted':False}
    (out/'DYNAMIC_REPORT.json').write_text(json.dumps(report,sort_keys=True)+'\n')
    print('FRESH_DYNAMIC_RESULT='+json.dumps({k:report[k] for k in ['status','g3_passed','frame_count','failed_motion_frame_count','maximum_motion_edge_ratio']}),flush=True)
    if not passed:raise SystemExit(2)


if __name__=='__main__':main()
