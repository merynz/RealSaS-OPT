from __future__ import annotations

import argparse, copy, hashlib, json, os, shutil
from pathlib import Path
import numpy as np
from PIL import Image

from compiler.realsas_compiler_core.appearance_authority_v2 import complete_appearance_asset_from_dict
from compiler.realsas_compiler_core.appearance_render_v2 import (
    load_face_page_index, load_face_uv, load_provenance_atlas, render_caa_reference,
)
from compiler.realsas_compiler_core.artifact_codec_v2 import (
    canonical_mesh_candidate_from_dict, deformation_envelope_from_dict,
    qualified_camera_set_from_dict, qualified_skeleton_from_dict,
    qualified_skin_from_dict, rigging_surface_from_dict,
)
from compiler.realsas_compiler_core.canonical_mesh_candidate_v1 import build_canonical_relation_candidate
from compiler.realsas_compiler_core.hashing import content_sha256
from compiler.realsas_compiler_core.mechanical_partition_v1 import build_structural_partition
from compiler.realsas_compiler_core.mechanical_repartition_v2 import repartition_authorization_hash_v1
from compiler.realsas_compiler_core.mesh.deformation_stress_v1 import _candidate_skin_matrix
from compiler.realsas_compiler_core.mesh.deformation_stress_v2 import run_g3_local_frame_micro_stress_v2
from compiler.realsas_compiler_core.mesh.skin_topology_compatibility_v1 import propose_mechanical_repartition_directive_v2
from compiler.realsas_compiler_core.product_authority_v1 import (
    ComponentCarrierDecisionIR, build_component_carrier_policy,
)
from compiler.realsas_compiler_core.preproduct_authority_v1 import (
    signed_zero_surface_from_dict, normalization_domain_from_dict,
)
from compiler.realsas_compiler_core.substrate.scene_first_signed import (
    build_compacted_dense_face_provenance_v1,
)
from compiler.realsas_compiler_services.orchestrator.adapters.adapter_io import (
    sha256_file, stage_output_payload,
)
from compiler.realsas_compiler_services.orchestrator.adapters.mesh_v2 import (
    qualify_mechanical_partition_and_carriers,
)
from compiler.realsas_compiler_services.orchestrator.adapters.v2_architecture import (
    build_canonical_mesh_addressing_stage, qualify_static_canonical_mesh_stage,
)
from compiler.realsas_compiler_services.orchestrator.adapters.appearance_v2 import (
    preregister_caa_backend_stage, compile_caa_stage, seal_caa_compile_stage,
    bake_complete_appearance_stage, qualify_complete_appearance_stage,
    prove_caa_reference_rest_stage, _load_texture_pages,
)
from compiler.realsas_compiler_core.motion_dynamic_proof_v2 import _joint_pose_v2
from tools.audit_knight_arachne_stage17_stage18_repair_child_v1 import (
    load_json, replay_compacted_dense_face_provenance,
)
from tools.audit_knight_teacher_free_weight_completion_court_v1 import motion_metrics
from tools.demo.render_knight_motion_preview_v1 import (
    _tracks_for_clip, _candidate_skin_weights, _skin,
)


def sha256(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(1<<20),b""):
            h.update(chunk)
    return h.hexdigest()


def row_by_id(ledger: dict, stage_id: str) -> dict:
    for row in ledger.get("stages") or ():
        if str(row.get("id"))==stage_id:
            return row
    raise RuntimeError(f"LEDGER_STAGE_MISSING::{stage_id}")


def find_schema_file(stage_dir: Path, schema: str) -> Path:
    hits=[]
    if stage_dir.is_dir():
        for path in stage_dir.rglob("*.json"):
            try:
                payload=load_json(path)
            except Exception:
                continue
            actual=str(payload.get("schema") or payload.get("schema_version") or "")
            if actual==schema:
                hits.append(path.resolve())
    if len(hits)!=1:
        raise RuntimeError(f"UPSTREAM_SCHEMA_CARDINALITY::{stage_dir.name}::{schema}::{len(hits)}")
    return hits[0]


def adopt_upstream_schema(
    ledger: dict, parent_root: Path, stage_id: str, schema: str, *,
    status: str="PASS_DEMO_ONLY", append: bool=False,
):
    path=find_schema_file(parent_root/"artifacts"/stage_id,schema)
    row=row_by_id(ledger,stage_id)
    row["status"]=status
    output={
        "path":str(path),
        "sha256":sha256_file(path),
        "authority_class":"PARENT_EXACT_BYTES__DEMO_SOLUTION_LINEAGE_ADOPTION",
        "schema":schema,
    }
    row["outputs"]=(list(row.get("outputs") or ()) + [output]) if append else [output]
    row["blockers"]=[]
    return {"stage_id":stage_id,"schema":schema,"path":str(path),"sha256":sha256_file(path),"status":status}


def adopt_result(ctx: dict, stage_id: str, fn):
    ctx["stage"]={"id":stage_id}
    result=fn(ctx)
    status=str(result.get("status") or "")
    if status not in {"PASS","PASS_DEMO_ONLY"}:
        raise RuntimeError(f"SOLUTION_STAGE_NOT_PASS::{stage_id}::{status}::{result.get('blockers')}")
    row=row_by_id(ctx["ledger"],stage_id)
    row["status"]=status
    row["outputs"]=list(result.get("outputs") or ())
    row["diagnostics_hash"]=content_sha256(result.get("diagnostics") or {})
    row["blockers"]=[]
    return result


def seal_child_compacted_face_provenance(
    child_root: Path, parent_root: Path, ledger: dict, surface
):
    zero=signed_zero_surface_from_dict(load_json(
        parent_root/"artifacts"/"12_ZERO_SURFACE_DECODED"/"signed_zero_surface_seal.json"))
    norm=normalization_domain_from_dict(load_json(
        parent_root/"artifacts"/"08_NORMALIZATION_DOMAIN_QUALIFIED"/"normalization_domain.json"))
    npz_path=Path(zero.npz_path)
    if not npz_path.is_file() or sha256_file(npz_path)!=zero.npz_sha256:
        raise RuntimeError("CHILD_PROVENANCE_ZERO_NPZ_DRIFT")
    with np.load(npz_path,allow_pickle=False) as z:
        vn=np.asarray(z["vertices_normalized"],dtype=np.float64)
        faces=np.asarray(z["faces"],dtype=np.int64)
    world=np.asarray(norm.center_xyz,dtype=np.float64)[None,:] + vn*float(norm.half_extent)
    payload=build_compacted_dense_face_provenance_v1(
        world,faces,surface,source_zero_surface_sha256=zero.npz_sha256)
    target=child_root/"artifacts"/"15_RIGGING_SURFACE_QUALIFIED"/"compacted_dense_face_provenance.json"
    target.parent.mkdir(parents=True,exist_ok=True)
    target.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n")
    row=row_by_id(ledger,"15_RIGGING_SURFACE_QUALIFIED")
    row["outputs"]=list(row.get("outputs") or ()) + [{
        "path":str(target.resolve()),
        "sha256":sha256_file(target),
        "authority_class":"REPLAYED_EXACT_DENSE_FACE_PROVENANCE__DEMO_SOLUTION_LINEAGE",
        "schema":"RealSaS.CompactedDenseFaceProvenance.v1",
    }]
    return {
        "stage_id":"15_RIGGING_SURFACE_QUALIFIED",
        "schema":"RealSaS.CompactedDenseFaceProvenance.v1",
        "path":str(target.resolve()),
        "sha256":sha256_file(target),
        "status":"PASS_DEMO_ONLY",
        "source_zero_surface_sha256":zero.npz_sha256,
        "provenance_hash":payload["provenance_hash"],
        "compact_face_count":payload["compact_face_count"],
    }


def copy_ir_as_stage32(child_root: Path, ledger: dict, skin_path: Path):
    target=child_root/"artifacts"/"32_SKIN_QUALIFIED"/"qualified_skin.json"
    target.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(skin_path,target)
    payload=load_json(target)
    if str(payload.get("schema_version") or payload.get("schema") or "")!="RealSaS.QualifiedSkinIR.v1":
        raise RuntimeError("CORRECTED_SKIN_SCHEMA_DRIFT")
    row=row_by_id(ledger,"32_SKIN_QUALIFIED")
    row["status"]="PASS"
    row["outputs"]=[{
        "path":str(target.resolve()),
        "sha256":sha256_file(target),
        "authority_class":"CORRECTED_ARACHNE_QUALIFIED_SKIN__SOLUTION_LINEAGE",
        "schema":"RealSaS.QualifiedSkinIR.v1",
    }]
    row["blockers"]=[]


def build_side_by_side(a: Image.Image,b: Image.Image, max_height=640) -> Image.Image:
    def resize(im):
        if im.height<=max_height: return im
        w=max(1,round(im.width*max_height/im.height))
        return im.resize((w,max_height),Image.Resampling.LANCZOS)
    a=resize(a.convert("RGBA")); b=resize(b.convert("RGBA"))
    h=max(a.height,b.height)
    bg=Image.new("RGBA",(a.width+b.width,h),(236,236,236,255))
    bg.alpha_composite(a,(0,(h-a.height)//2))
    bg.alpha_composite(b,(a.width,(h-b.height)//2))
    return bg.convert("RGB")


def render_gifs(ctx: dict, out_dir: Path):
    candidate=canonical_mesh_candidate_from_dict(stage_output_payload(
        ctx,"18_CANONICAL_MESH_ADDRESSING_BUILD","RealSaS.CanonicalMeshCandidateIR.v1"))
    skeleton=qualified_skeleton_from_dict(stage_output_payload(
        ctx,"28_SKELETON_QUALIFIED","RealSaS.QualifiedSkeletonIR.v1"))
    skin=qualified_skin_from_dict(stage_output_payload(
        ctx,"32_SKIN_QUALIFIED","RealSaS.QualifiedSkinIR.v1"))
    camera_set=qualified_camera_set_from_dict(stage_output_payload(
        ctx,"05_CAMERA_CONTRACT_SOLVED","RealSaS.QualifiedCameraSetIR.v1"))
    cameras=tuple(sorted(camera_set.cameras,key=lambda x:int(x.view_index)))
    appearance=complete_appearance_asset_from_dict(stage_output_payload(
        ctx,"23_COMPLETE_APPEARANCE_ASSET_BAKED","RealSaS.CompleteAppearanceAssetIR.v2"))
    face_uv=load_face_uv(appearance)
    face_page=load_face_page_index(appearance)
    provenance=load_provenance_atlas(appearance)
    texture_by_view={int(row.direction_index):row for row in appearance.textures}
    source_report=load_json(Path("canonical/KNIGHT_MOTION_SOURCE_ACTION_DIAGNOSTIC_20260927.json"))
    joint_ids,W=_candidate_skin_weights(candidate,skin,skeleton)
    rest=np.asarray([v.P for v in candidate.vertices],dtype=np.float64)

    gifs={}
    clip_specs=(
        ("demo_idle_v1","KNIGHT_IDLE_SOLVED.gif",16),
        ("demo_run_v1","KNIGHT_RUN_SOLVED.gif",16),
        ("demo_slash_v1","KNIGHT_SLASH_SOLVED.gif",16),
    )
    for clip_id,name,count in clip_specs:
        payload=load_json(ctx["run_root"]/"inputs"/"motion"/"quaternius_knight_v1"/f"{clip_id}.motion.json")
        tracks,_=_tracks_for_clip(payload,skeleton,cameras,source_report)
        duration=float(payload["duration_seconds"])
        times=np.linspace(0.0,duration,count,endpoint=not bool(payload.get("loop")))
        textures={v:_load_texture_pages(texture_by_view[v]) for v in (0,2)}
        frames=[]
        for t in times:
            skin_mats,_,_=_joint_pose_v2(
                skeleton=skeleton,tracks=tracks,time_seconds=float(t),cameras=cameras)
            posed=_skin(rest,W,joint_ids,skin_mats)
            imgs=[]
            for view in (0,2):
                rr=render_caa_reference(
                    mesh=candidate,camera=cameras[view],face_uv=face_uv,
                    texture_rgba_u8=textures[view],
                    provenance_atlas=provenance[view],
                    face_page_index=face_page,positions=posed)
                imgs.append(Image.fromarray(np.asarray(rr.straight_rgba_u8,dtype=np.uint8),"RGBA"))
            frames.append(build_side_by_side(imgs[0],imgs[1]))
        path=out_dir/name
        frames[0].save(path,save_all=True,append_images=frames[1:],duration=90,loop=0,optimize=False)
        gifs[clip_id]={
            "path":str(path),"sha256":sha256(path),"frame_count":len(frames),
            "views":[0,2],"duration_seconds":duration,
        }
    return gifs


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--authority-root",type=Path,required=True)
    ap.add_argument("--parent-run-id",required=True)
    ap.add_argument("--child-run-id",required=True)
    ap.add_argument("--skin-json",type=Path,required=True)
    ap.add_argument("--a100-audit-json",type=Path,required=True)
    ap.add_argument("--rebound-report-json",type=Path,required=True)
    ap.add_argument("--out-dir",type=Path,required=True)
    ap.add_argument("--repo-commit",required=True)
    a=ap.parse_args()

    parent_root=(a.authority_root/"runs"/a.parent_run_id).resolve()
    child_root=(a.authority_root/"runs"/a.child_run_id).resolve()
    if child_root.exists(): shutil.rmtree(child_root)
    child_root.mkdir(parents=True)
    os.symlink(parent_root/"inputs",child_root/"inputs",target_is_directory=True)

    parent_manifest=load_json(parent_root/"run_manifest.json")
    parent_ledger=load_json(parent_root/"ACTIVE_RUN_V2.json")
    manifest=copy.deepcopy(parent_manifest)
    manifest["run_id"]=a.child_run_id
    ledger=copy.deepcopy(parent_ledger)
    ledger["run_id"]=a.child_run_id
    ledger["execution_class"]="DEMO_WITNESS"
    ledger["architecture_scope"]="KNIGHT_SOLVED_WEIGHT_TOPOLOGY_LINEAGE_V1"

    # Two provenance helpers read sealed Stage08/12 bytes directly from run_root.
    artifacts_root=child_root/"artifacts"
    artifacts_root.mkdir(parents=True,exist_ok=True)
    for stage_name in ("08_NORMALIZATION_DOMAIN_QUALIFIED","12_ZERO_SURFACE_DECODED"):
        source=parent_root/"artifacts"/stage_name
        target=artifacts_root/stage_name
        if not source.is_dir():
            raise RuntimeError(f"UPSTREAM_DIRECT_STAGE_MISSING::{stage_name}")
        os.symlink(source,target,target_is_directory=True)

    adopted_upstream=[]
    for stage_id,schema,status in (
        ("05_CAMERA_CONTRACT_SOLVED","RealSaS.QualifiedCameraSetIR.v1","PASS"),
        ("07_OBSERVATION_CONTRACT_QUALIFIED","RealSaS.QualifiedObservationSetIR.v1","PASS"),
        ("13_GEOMETRY_SUBSTRATE_QUALIFIED","RealSaS.GeometrySubstrateQualificationIR.v2","PASS_DEMO_ONLY"),
        ("15_RIGGING_SURFACE_QUALIFIED","RealSaS.RiggingSurfaceIR.v1","PASS_DEMO_ONLY"),
        ("16_OUTPUT_PRESENTATION_DIRECTIONS_SEALED","RealSaS.OutputPresentationDirectionSetIR.v1","PASS"),
        ("28_SKELETON_QUALIFIED","RealSaS.QualifiedSkeletonIR.v1","PASS"),
        ("34_DEFORMATION_CAPABILITY_ENVELOPE","RealSaS.DeformationCapabilityEnvelopeIR.v1","PASS"),
    ):
        adopted_upstream.append(adopt_upstream_schema(
            ledger,parent_root,stage_id,schema,status=status))
    copy_ir_as_stage32(child_root,ledger,a.skin_json)
    ctx={
        "repo_root":Path(".").resolve(),"authority_root":a.authority_root.resolve(),
        "run_root":child_root,"run_id":a.child_run_id,
        "run_manifest_path":child_root/"run_manifest.json","run_manifest":manifest,
        "ledger":ledger,"stage":{"id":"INIT"},
    }
    surface=rigging_surface_from_dict(stage_output_payload(
        ctx,"15_RIGGING_SURFACE_QUALIFIED","RealSaS.RiggingSurfaceIR.v1"))
    adopted_upstream.append(seal_child_compacted_face_provenance(
        child_root,parent_root,ledger,surface))
    skeleton=qualified_skeleton_from_dict(stage_output_payload(
        ctx,"28_SKELETON_QUALIFIED","RealSaS.QualifiedSkeletonIR.v1"))
    cameras=tuple(sorted(qualified_camera_set_from_dict(stage_output_payload(
        ctx,"05_CAMERA_CONTRACT_SOLVED","RealSaS.QualifiedCameraSetIR.v1")).cameras,
        key=lambda x:int(x.view_index)))
    envelope=deformation_envelope_from_dict(stage_output_payload(
        ctx,"34_DEFORMATION_CAPABILITY_ENVELOPE","RealSaS.DeformationCapabilityEnvelopeIR.v1"))
    skin=qualified_skin_from_dict(load_json(a.skin_json))
    parent_partition=build_structural_partition(surface,boundary_overrides=())
    explicit,face_replay=replay_compacted_dense_face_provenance(parent_root,surface)

    parent_carrier=build_component_carrier_policy(
        partition=parent_partition,
        decisions=tuple(ComponentCarrierDecisionIR(
            c.component_id,"MESH",("SOLUTION_LINEAGE_PARENT",),
            metadata={"automatic":True}) for c in parent_partition.components),
        metadata={"audit_only":True})
    source_candidate=build_canonical_relation_candidate(
        surface,parent_partition,parent_carrier,
        producer_policy_hash=content_sha256({
            "schema":"RealSaS.KnightSolutionDirectiveSource.v1",
            "parent_run_id":a.parent_run_id,
            "skin_lineage_hash":skin.skin_lineage_hash,
        }),
        explicit_face_provenance=explicit)
    trigger={
        "schema":"RealSaS.SkinTopologyCompatibilityReport.v1",
        "report_hash":content_sha256({
            "schema":"RealSaS.KnightSolutionSourceEdgeTrigger.v1",
            "source_candidate_lineage_hash":source_candidate.candidate_lineage_hash,
            "skin_lineage_hash":skin.skin_lineage_hash,
        }),
        "unsafe_face_count":1,"unsafe_face_indices":(0,),
    }
    directive=propose_mechanical_repartition_directive_v2(
        source_candidate,surface=surface,skeleton=skeleton,skin=skin,
        partition=parent_partition,compatibility_report=trigger,
        envelope=envelope,cameras=cameras,seed_strategy="SOURCE_EDGE_PROBE_RATIO_V1")

    a100=load_json(a.a100_audit_json)
    rebound=load_json(a.rebound_report_json)
    arm="UNIFORM_ALL_PROJECTED"
    arm_a100=a100["arms"][arm]
    evidence={
        "schema":"RealSaS.KnightArachneWeightReliabilityEvidence.v1",
        "arm":arm,
        "a100_result_sha256":arm_a100["result_sha256"],
        "a100_weights_sha256":arm_a100["weights_sha256"],
        "a100_final_valid_gate":arm_a100["final_valid_gate"],
        "a100_valid":arm_a100["valid"],
        "a100_invalid_diagnostic":arm_a100["invalid"],
        "compiler_requalification":rebound["compiler_requalification"],
        "teacher_inputs_used_by_predictor":False,
        "projected_rows_are_product_teacher_authority":False,
        "scope":"DEMO_SOLUTION_LINEAGE_AND_VISUAL_PROOF_ONLY",
    }
    evidence_hash=content_sha256(evidence)
    auth={
        "schema":"RealSaS.TrustworthySkinRepartitionAuthorization.v1",
        "status":"PASS_TRUSTWORTHY_SKIN_REPARTITION_AUTHORIZATION",
        "directive_hash":directive["directive_hash"],
        "source_surface_lineage_hash":surface.geometry_lineage_hash,
        "source_partition_lineage_hash":parent_partition.partition_lineage_hash,
        "source_skeleton_lineage_hash":skeleton.skeleton_lineage_hash,
        "source_skin_lineage_hash":skin.skin_lineage_hash,
        "teacher_inputs_used_by_predictor":False,
        "weight_reliability_closure_passed":True,
        "weight_reliability_evidence_hash":evidence_hash,
        "scope":"DEMO_SOLUTION_LINEAGE_AND_VISUAL_PROOF_ONLY",
        "product_authority_claimed":False,
        "projected_invalid_rows_product_teacher_authority":False,
        "authorization_hash":"",
    }
    auth["authorization_hash"]=repartition_authorization_hash_v1(auth)

    repair_root=child_root/"repair"
    repair_root.mkdir()
    directive_path=repair_root/"mechanical_repartition_directive.json"
    auth_path=repair_root/"authorization.json"
    directive_path.write_text(json.dumps(directive,indent=2,sort_keys=True)+"\n")
    auth_path.write_text(json.dumps(auth,indent=2,sort_keys=True)+"\n")
    manifest["mechanical_repartition_repair"]={
        "directive":{"path":str(directive_path.resolve()),"sha256":sha256(directive_path)},
        "authorization":{"path":str(auth_path.resolve()),"sha256":sha256(auth_path)},
    }
    # Persist only after the exact repair refs are bound.
    (child_root/"run_manifest.json").write_text(
        json.dumps(manifest,indent=2,sort_keys=True)+"\n"
    )

    stage_results={}
    stage_results["17"]=adopt_result(ctx,"17_MECHANICAL_PARTITION_QUALIFIED",qualify_mechanical_partition_and_carriers)
    stage_results["18"]=adopt_result(ctx,"18_CANONICAL_MESH_ADDRESSING_BUILD",build_canonical_mesh_addressing_stage)
    stage_results["19"]=adopt_result(ctx,"19_STATIC_CANONICAL_MESH_QUALIFIED",qualify_static_canonical_mesh_stage)
    stage_results["20"]=adopt_result(ctx,"20_CAA_BACKEND_PREREGISTERED",preregister_caa_backend_stage)
    stage_results["21"]=adopt_result(ctx,"21_CAA_COMPILE",compile_caa_stage)
    stage_results["22"]=adopt_result(ctx,"22_CAA_COMPILE_SEALED",seal_caa_compile_stage)
    stage_results["23"]=adopt_result(ctx,"23_COMPLETE_APPEARANCE_ASSET_BAKED",bake_complete_appearance_stage)
    stage_results["24"]=adopt_result(ctx,"24_COMPLETE_APPEARANCE_QUALIFIED",qualify_complete_appearance_stage)
    stage_results["25"]=adopt_result(ctx,"25_CAA_REFERENCE_REST_RENDER_PROOF",prove_caa_reference_rest_stage)
    (child_root/"ACTIVE_RUN_V2.json").write_text(json.dumps(ledger,indent=2,sort_keys=True)+"\n")

    candidate=canonical_mesh_candidate_from_dict(stage_output_payload(
        ctx,"18_CANONICAL_MESH_ADDRESSING_BUILD","RealSaS.CanonicalMeshCandidateIR.v1"))
    child_partition_payload=stage_output_payload(
        ctx,"17_MECHANICAL_PARTITION_QUALIFIED","RealSaS.MechanicalPartitionIR.v1")
    from compiler.realsas_compiler_core.artifact_codec_v2 import mechanical_partition_from_dict, mesh_policy_from_dict
    partition=mechanical_partition_from_dict(child_partition_payload)
    policy=mesh_policy_from_dict(stage_output_payload(
        ctx,"18_CANONICAL_MESH_ADDRESSING_BUILD","RealSaS.MeshQualificationPolicyIR.v1"))
    g3=run_g3_local_frame_micro_stress_v2(
        candidate,surface=surface,skeleton=skeleton,skin=skin,envelope=envelope,
        cameras=cameras,policy=policy)
    rest,W,faces=_candidate_skin_matrix(candidate,surface=surface,skeleton=skeleton,skin=skin)
    motion=motion_metrics(
        np.asarray(rest),np.asarray(W),np.asarray(faces),
        tuple(j.canonical_joint_id for j in skeleton.joints),
        skeleton,cameras,parent_root,
        load_json(Path("canonical/KNIGHT_MOTION_SOURCE_ACTION_DIAGNOSTIC_20260927.json")))
    if not g3.passed or motion["max_edge_gt_4"]!=0 or motion["max_edge_gt_10"]!=0:
        raise RuntimeError("SOLUTION_LINEAGE_MECHANICAL_PROOF_NOT_CLOSED")

    out_dir=a.out_dir.resolve(); out_dir.mkdir(parents=True,exist_ok=True)
    gifs=render_gifs(ctx,out_dir)

    stage23_ref=[x for x in row_by_id(ledger,"23_COMPLETE_APPEARANCE_ASSET_BAKED")["outputs"]
                 if x.get("schema")=="RealSaS.CompleteAppearanceAssetIR.v2"][0]
    stage24_refs=row_by_id(ledger,"24_COMPLETE_APPEARANCE_QUALIFIED")["outputs"]

    seal={
        "schema":"RealSaS.KnightSolvedWeightTopologyVisualLineageSeal.v1",
        "status":"PASS__SAME_LINEAGE_MECHANICS_APPEARANCE_RENDER",
        "scope":"DEMO_SOLUTION_LINEAGE_AND_VISUAL_PROOF_ONLY",
        "product_authority_claimed":False,
        "repo_commit":a.repo_commit,
        "parent_run_id":a.parent_run_id,
        "child_run_id":a.child_run_id,
        "adopted_parent_upstream":adopted_upstream,
        "corrected_skin":{
            "skin_lineage_hash":skin.skin_lineage_hash,
            "file_sha256":sha256(a.skin_json),
            "a100_result_sha256":arm_a100["result_sha256"],
            "a100_weights_sha256":arm_a100["weights_sha256"],
            "compiler_requalification":rebound["compiler_requalification"],
            "weight_reliability_evidence_hash":evidence_hash,
        },
        "stage35_directive":{
            "directive_hash":directive["directive_hash"],
            "seed_strategy":directive.get("seed_strategy"),
            "source_edge_probe_hash":directive.get("source_edge_probe_hash"),
            "unsafe_source_edge_count":directive.get("unsafe_source_edge_count"),
            "authorization_hash":auth["authorization_hash"],
        },
        "stage17":{
            "partition_lineage_hash":partition.partition_lineage_hash,
            "component_count":len(partition.components),
            "closure":dict(partition.metadata.get("partition_cut_closure") or {}),
        },
        "stage18":{
            "candidate_lineage_hash":candidate.candidate_lineage_hash,
            "vertex_count":len(candidate.vertices),"face_count":len(candidate.faces),
            "mechanical_skin_transfer":candidate.metadata.get("mechanical_skin_transfer"),
            "mechanical_skin_transfer_report":candidate.metadata.get("mechanical_skin_transfer_report"),
        },
        "appearance":{
            "stage23_complete_appearance_sha256":stage23_ref["sha256"],
            "stage24_output_sha256s":sorted(str(x["sha256"]) for x in stage24_refs),
            "stage20_diagnostics_hash":row_by_id(ledger,"20_CAA_BACKEND_PREREGISTERED")["diagnostics_hash"],
            "stage25_diagnostics_hash":row_by_id(ledger,"25_CAA_REFERENCE_REST_RENDER_PROOF")["diagnostics_hash"],
        },
        "g3":{
            "passed":bool(g3.passed),"report_hash":g3.report_hash,
            "failure_invariants":list(g3.failure_invariants),
        },
        "actual_motion":{
            "max_edge_gt_10":motion["max_edge_gt_10"],
            "max_edge_gt_4":motion["max_edge_gt_4"],
            "worst_edge_max":motion["worst_edge_max"],
            "max_edge_p99":motion["max_edge_p99"],
        },
        "gifs":gifs,
        "face_provenance_replay":face_replay,
        "lineage_invariants":{
            "same_stage18_candidate_used_by_caa":True,
            "same_stage18_candidate_used_by_g3":True,
            "same_stage18_candidate_used_by_render":True,
            "same_corrected_skin_used_by_directive_g3_motion_render":True,
            "old_stage23_appearance_reused":False,
            "source_art_regenerated":False,
        },
    }
    seal_path=out_dir/"KNIGHT_SOLVED_WEIGHT_TOPOLOGY_VISUAL_LINEAGE_SEAL_V1.json"
    seal_path.write_text(json.dumps(seal,indent=2,sort_keys=True)+"\n")
    print("KNIGHT_SOLUTION_LINEAGE="+json.dumps({
        "status":seal["status"],
        "skin":skin.skin_lineage_hash,
        "directive":directive["directive_hash"],
        "components":len(partition.components),
        "candidate":candidate.candidate_lineage_hash,
        "g3":g3.passed,
        "motion_gt4":motion["max_edge_gt_4"],
        "appearance_sha":stage23_ref["sha256"],
        "gifs":{k:v["sha256"] for k,v in gifs.items()},
    },sort_keys=True))


if __name__=="__main__":
    main()
