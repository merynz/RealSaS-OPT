"""CPU preflight for the frozen-Arachne mechanical residual adapter on exact Knight V9.

This court does not train. It proves that current product-space tensors bind:
fresh V9 Arachne field + rich product conditioning + current compiler candidate
sparse supports + exact skeleton-derived mechanical probe transforms + teacher-valid
semantic tether.

The output bundle is the exact input contract later consumed by the A100 fit.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from compiler.realsas_compiler_core.artifact_codec_v2 import (
    canonical_mesh_candidate_from_dict,
    mesh_policy_from_dict,
    qualified_camera_set_from_dict,
    qualified_skeleton_from_dict,
    qualified_skin_from_dict,
    rigging_surface_from_dict,
)
from compiler.realsas_compiler_core.deformation_envelope_derivation_v1 import (
    derive_deformation_envelope_v1,
)
from compiler.realsas_compiler_core.joint_frames_v1 import derive_joint_frames_from_skeleton
from compiler.realsas_compiler_core.mesh.deformation_stress_v2 import _pose_skin_matrices
from compiler.realsas_compiler_core.mesh.skin_topology_compatibility_v1 import _stress_angle
from compiler.realsas_compiler_core.product_mesh_skin_v1 import _skin_support_coefficients
from compiler.realsas_compiler_services.orchestrator.adapters.adapter_io import stage_output_payload
from models.arachne.v3.conditioning_v3 import ArachneRichConditioningAdapterV3
from models.arachne.v4.mechanical_consequence_loss_v1 import mechanical_consequence_loss_v1
from models.arachne.v4.mechanical_residual_adapter_v1 import MechanicalResidualAdapterV1
from tools.audit_v9_teacher_projection_oracle_v1 import teacher_to_skin
from tools.demo.render_knight_motion_preview_v1 import _ctx


def read(path):
    return json.loads(Path(path).read_text())


def skin_matrix(surface, skeleton, skin):
    sids=tuple(str(x.surface_id) for x in surface.surface_nodes)
    jids=tuple(str(x.canonical_joint_id) for x in sorted(skeleton.joints,key=lambda j:str(j.canonical_joint_id)))
    rows={str(x.surface_id):x for x in skin.rows}
    ji={jid:i for i,jid in enumerate(jids)}
    out=np.zeros((len(sids),len(jids)),np.float32)
    for r,sid in enumerate(sids):
        row=rows[sid]
        for jid,w in row.influences:
            out[r,ji[str(jid)]]=float(w)
    if not np.allclose(out.sum(1),1.0,atol=1e-6,rtol=0.0):
        raise RuntimeError("ARACHNE_MECH_PREFLIGHT_SKIN_SIMPLEX")
    return sids,jids,out


def sparse_candidate_support(candidate, surface_ids):
    si={sid:i for i,sid in enumerate(surface_ids)}
    supports=[tuple(_skin_support_coefficients(v)) for v in candidate.vertices]
    K=max(len(x) for x in supports)
    idx=np.full((len(supports),K),-1,np.int64)
    coeff=np.zeros((len(supports),K),np.float32)
    for vi,rows in enumerate(supports):
        for k,(sid,c) in enumerate(rows):
            if str(sid) not in si:
                raise RuntimeError("ARACHNE_MECH_PREFLIGHT_SUPPORT_OUTSIDE_SURFACE")
            idx[vi,k]=si[str(sid)]
            coeff[vi,k]=float(c)
    if not np.allclose(coeff.sum(1),1.0,atol=1e-6,rtol=0.0):
        raise RuntimeError("ARACHNE_MECH_PREFLIGHT_SUPPORT_SIMPLEX")
    return idx,coeff


def candidate_geometry(candidate):
    rest=np.asarray([v.P for v in candidate.vertices],np.float32)
    vi={str(v.candidate_vertex_id):i for i,v in enumerate(candidate.vertices)}
    faces=np.asarray([[vi[str(x)] for x in face] for face in candidate.faces],np.int64)
    return rest,faces


def probe_transforms(skeleton,cameras,envelope,*,max_probes):
    frames=derive_joint_frames_from_skeleton(skeleton,cameras=cameras)
    joint_ids=tuple(str(x.canonical_joint_id) for x in sorted(skeleton.joints,key=lambda j:str(j.canonical_joint_id)))
    deg=float(_stress_angle(envelope))
    rows=[("REST",None,None,0.0)]
    for jid in sorted(joint_ids):
        for axis in range(3):
            for sign in (-1.0,1.0):
                rows.append((f"{jid}:LOCAL_{axis}:{sign*deg:+g}",jid,axis,sign*deg))
    rows=rows[:max(1,int(max_probes))]
    mats=[]
    ids=[]
    for pid,jid,axis,d in rows:
        by=_pose_skin_matrices(skeleton,frames,joint_id=jid,local_axis_index=axis,degrees=d)
        mats.append(np.stack([by[x] for x in joint_ids],axis=0).astype(np.float32))
        ids.append(pid)
    return joint_ids,tuple(ids),np.stack(mats,axis=0)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--authority-root",type=Path,required=True)
    ap.add_argument("--run-id",required=True)
    ap.add_argument("--surface-json",type=Path,required=True)
    ap.add_argument("--candidate-json",type=Path,required=True)
    ap.add_argument("--fresh-dir",type=Path,required=True)
    ap.add_argument("--teacher-bank",type=Path,required=True)
    ap.add_argument("--out-dir",type=Path,required=True)
    ap.add_argument("--probe-count",type=int,default=8)
    a=ap.parse_args()

    ctx=_ctx(a.authority_root,a.run_id)
    surface=rigging_surface_from_dict(read(a.surface_json))
    candidate=canonical_mesh_candidate_from_dict(read(a.candidate_json))
    skeleton=qualified_skeleton_from_dict(read(a.fresh_dir/"fresh_qualified_skeleton.json"))
    predicted=qualified_skin_from_dict(read(a.fresh_dir/"fresh_qualified_skin.json"))

    camera_set=qualified_camera_set_from_dict(
        stage_output_payload(ctx,"05_CAMERA_CONTRACT_SOLVED","RealSaS.QualifiedCameraSetIR.v1")
    )
    cameras=tuple(sorted(camera_set.cameras,key=lambda x:int(x.view_index)))
    _,envelope=derive_deformation_envelope_v1(skeleton=skeleton,camera_set=camera_set)
    policy=mesh_policy_from_dict(
        stage_output_payload(ctx,"18_CANONICAL_MESH_ADDRESSING_BUILD","RealSaS.MeshQualificationPolicyIR.v1")
    )

    cond=ArachneRichConditioningAdapterV3()( (surface,), (skeleton,) )
    sids,jids,base=skin_matrix(surface,skeleton,predicted)
    if tuple(cond.surface_ids[0])!=sids or tuple(cond.joint_ids[0])!=jids:
        raise RuntimeError("ARACHNE_MECH_PREFLIGHT_CONDITIONING_AXIS_DRIFT")

    teacher_skin,_,valid,_,_,_=teacher_to_skin(surface,skeleton,a.teacher_bank)
    _,teacher_jids,teacher=skin_matrix(surface,skeleton,teacher_skin)
    if teacher_jids!=jids:
        raise RuntimeError("ARACHNE_MECH_PREFLIGHT_TEACHER_JOINT_DRIFT")

    support_idx,support_coeff=sparse_candidate_support(candidate,sids)
    rest,faces=candidate_geometry(candidate)
    probe_jids,probe_ids,transforms=probe_transforms(
        skeleton,cameras,envelope,max_probes=a.probe_count
    )
    if probe_jids!=jids:
        raise RuntimeError("ARACHNE_MECH_PREFLIGHT_PROBE_JOINT_DRIFT")

    model=MechanicalResidualAdapterV1().eval()
    with torch.no_grad():
        out,delta=model(
            base_weights=torch.from_numpy(base[None]),
            surface_geometry7=torch.from_numpy(cond.geometry7.astype(np.float32)),
            pair_geometry=torch.from_numpy(cond.pair_geometry.astype(np.float32)),
            surface_mask=torch.from_numpy(cond.surface_mask),
            joint_mask=torch.from_numpy(cond.joint_mask),
            surface_chunk_size=256,
        )
        identity_l1=(out-torch.from_numpy(base[None])).abs().sum(-1)
        loss=mechanical_consequence_loss_v1(
            out,
            candidate_support_indices=torch.from_numpy(support_idx[None]),
            candidate_support_coefficients=torch.from_numpy(support_coeff[None]),
            candidate_rest_vertices=torch.from_numpy(rest[None]),
            candidate_faces=torch.from_numpy(faces),
            probe_transforms=torch.from_numpy(transforms[None]),
            base_surface_weights=torch.from_numpy(base[None]),
            teacher_surface_weights=torch.from_numpy(teacher[None]),
            teacher_valid_mask=torch.from_numpy(valid[None]),
            max_edge_ratio=4.0,
            min_area_ratio=float(policy.g3_min_dynamic_area_ratio),
            max_area_ratio=float(policy.g3_max_dynamic_area_ratio),
            max_condition_number=float(policy.g3_max_dynamic_condition_number),
            trust_weight=0.25,
            teacher_weight=1.0,
            mechanical_weight=1.0,
        )

    a.out_dir.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(
        a.out_dir/"KNIGHT_V9_ARACHNE_MECHANICAL_ADAPTER_BUNDLE_V1.npz",
        base_weights=base.astype(np.float32),
        teacher_weights=teacher.astype(np.float32),
        teacher_valid_mask=np.asarray(valid,np.uint8),
        geometry7=cond.geometry7[0].astype(np.float32),
        pair_geometry=cond.pair_geometry[0].astype(np.float32),
        surface_mask=cond.surface_mask[0].astype(np.uint8),
        joint_mask=cond.joint_mask[0].astype(np.uint8),
        candidate_support_indices=support_idx,
        candidate_support_coefficients=support_coeff,
        candidate_rest_vertices=rest,
        candidate_faces=faces,
        probe_transforms=transforms,
        surface_ids=np.asarray(sids,dtype="U128"),
        joint_ids=np.asarray(jids,dtype="U128"),
        probe_ids=np.asarray(probe_ids,dtype="U192"),
    )
    report={
        "schema":"RealSaS.KnightV9ArachneMechanicalAdapterCPUPreflight.v1",
        "status":"PASS",
        "training_performed":False,
        "product_authority_minted":False,
        "surface_rows":len(sids),
        "joints":len(jids),
        "candidate_vertices":len(rest),
        "candidate_faces":len(faces),
        "max_candidate_support_width":int(support_idx.shape[1]),
        "teacher_valid_rows":int(np.count_nonzero(valid)),
        "teacher_total_rows":int(len(valid)),
        "teacher_coverage":float(np.mean(valid)),
        "adapter_trainable_parameters":int(model.parameter_count),
        "adapter_identity_row_l1_max":float(identity_l1.max()),
        "adapter_delta_abs_max":float(delta.abs().max()),
        "probe_count":len(probe_ids),
        "probe_ids":probe_ids,
        "initial_loss":{
            k:float(v) for k,v in loss.items()
            if torch.is_tensor(v) and v.ndim==0
        },
        "claim_boundary":[
            "This court performs no optimization.",
            "The residual adapter is zero-init and must reproduce the frozen fresh V9 Arachne field.",
            "Teacher supervision is bound only through the V9 teacher-valid mask.",
            "Compiler candidate transfer uses sparse typed support coefficients, including future harmonic multi-support.",
            "Exact full G3B/G3/motion remain compiler qualification authority after fitting.",
        ],
    }
    (a.out_dir/"REPORT.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    print("ARACHNE_MECH_PREFLIGHT="+json.dumps(report,sort_keys=True),flush=True)


if __name__=="__main__":
    main()
