"""Verified observation -> frozen DINO features and image-derived F* shards.

Historical FIT8 remains the default call surface.  Truth139 reuses the exact same observation/F*
construction through a generic roster plus a preparation-only policy; no training policy or split
authority is created by this module.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
import numpy as np
from PIL import Image
import torch
from models.iris.training.cohort_v1 import file_hash, value_hash, atomic_json,validate_knight_recipe,source_identity
from models.iris.training.apparatus_v1 import query_pools, certified_ray_pools
from tools.corpus.prepare_iris_observations_v1 import DEFAULT_ROSTER, load_roster
from tools.training.iris_exterior_audit_v1 import exterior_scan
from models.iris.v2.dinov2_foundation_v2 import load_exact_dinov2_s_v2, extract_dinov2_s_patch_map_flat_v2
from models.iris.v5.source_signed_field_v5 import source_signed_distance_fields_v5, source_signed_field_torch_v5

ROOT = Path(__file__).resolve().parents[2]
FIT8_POLICY_PATH = ROOT/"canonical/FIT8_IRIS_SHARED_TRAINING_POLICY_V1.json"
TRUTH139_PREPARATION_POLICY_PATH = ROOT/"canonical/TRUTH139_IRIS_SHARD_PREPARATION_POLICY_V1.json"
PREPARATION_POLICY_SCHEMA = "RealSaS.IrisShardPreparationPolicy.v1"


def verify_observations(input_root, row):
    folder = Path(input_root)/row["subject_key"]
    proof_path = folder/"IRIS_OBSERVATION_ADMISSION.json"
    proof = json.loads(proof_path.read_text())
    if (proof["status"] != "PASS_IRIS_BIND_OBSERVATIONS" or proof["subject_id"] != row["subject_key"]
            or proof["source_sha256"] != row["source"]["sha256"]):
        raise ValueError("PREPARE_UNADMITTED_OBSERVATION")
    policy = json.loads((ROOT/"canonical/FIT8_IRIS_OBSERVATION_POLICY_V1.json").read_text())
    if proof["policy"] != policy or len(proof["views"]) != 8:
        raise ValueError("PREPARE_OBSERVATION_POLICY_DRIFT")
    candidate_path = folder/"SOURCE_REST_RENDER_CANDIDATE.json"
    camera_path = folder/"camera_bundle.json"
    if file_hash(candidate_path) != proof["candidate_sha256"] or file_hash(camera_path) != proof["camera_sha256"]:
        raise ValueError("PREPARE_OBSERVATION_RECEIPT_DRIFT")
    candidate = json.loads(candidate_path.read_text())
    snapshot_path = folder/"SOURCE_SCENE_SNAPSHOT.json"
    if file_hash(snapshot_path) != candidate["source_scene_snapshot"]["sha256"]:
        raise ValueError("PREPARE_SCENE_SNAPSHOT_DRIFT")
    from tools.observation.source_scene_contract_v2 import validate_imported_scene_snapshot
    snapshot = json.loads(snapshot_path.read_text())
    validate_imported_scene_snapshot(snapshot)
    bundle = json.loads(camera_path.read_text())
    norm = np.asarray(snapshot["source_world_to_canonical"])
    expected = np.eye(4)
    expected[:3,:3] /= bundle["common_half_extent"]
    expected[:3,3] = -np.asarray(bundle["common_center_xyz"])/bundle["common_half_extent"]
    if not np.allclose(norm,expected,atol=1e-6,rtol=1e-6):
        raise ValueError("PREPARE_CAMERA_NORMALIZATION_DRIFT")
    for view in proof["views"]:
        i = view["view_index"]
        if (file_hash(folder/f"V{i}.png") != view["raster_sha256"]
                or file_hash(folder/f"V{i}.mask.bin") != view["mask_sha256"]
                or view["maximum_mask_gap_px"] > policy["maximum_mask_gap_px"]
                or view["border_margin_px"] < policy["minimum_border_margin_px"]):
            raise ValueError("PREPARE_OBSERVATION_RASTER_OR_COURT_DRIFT")
    return folder, proof, value_hash({"proof_sha256":file_hash(proof_path),
                                     "source_sha256":proof["source_sha256"],
                                     "camera_sha256":proof["camera_sha256"],
                                     "views":proof["views"]})


def _validate_preparation_policy(policy, roster_schema):
    schema=str(policy.get("schema") or "")
    if roster_schema=="RealSaS.FIT8DiversitySelection.v1":
        if schema!="RealSaS.SharedIrisTrainingPolicy.v1":
            raise ValueError("PREPARE_FIT8_POLICY_SCHEMA")
        validate_knight_recipe(policy,source_identity(ROOT))
        return "LEGACY_ROSTER_ORDINAL_OFFSET_1000000"
    if roster_schema!="RealSaS.Truth139ApparatusManifest.v1":
        raise ValueError("PREPARE_ROSTER_SCHEMA_UNSUPPORTED")
    if schema!=PREPARATION_POLICY_SCHEMA or policy.get("training_authorized") is not False:
        raise ValueError("PREPARE_TRUTH139_POLICY_SCHEMA_OR_AUTHORITY")
    if policy.get("per_subject_seed_semantics")!="SHA256_SUBJECT_ID_LOW32_PLUS_BASE_SEED__ROSTER_ORDER_INDEPENDENT":
        raise ValueError("PREPARE_TRUTH139_SEED_SEMANTICS")
    source_policy=FIT8_POLICY_PATH
    if (policy.get("source_fit8_policy_repo_path")!="canonical/FIT8_IRIS_SHARED_TRAINING_POLICY_V1.json"
            or policy.get("source_fit8_policy_sha256")!=file_hash(source_policy)):
        raise ValueError("PREPARE_TRUTH139_SOURCE_POLICY_DRIFT")
    required=(
        "seed","near_pool_count","heldout_count","near_zero_band","straddle_pool_count",
        "straddle_delta","straddle_base_band","replay_global_count","replay_near_count",
        "heldout_near_count","heldout_straddle_count","silhouette_pool_per_stratum",
        "source_boundary_band_px","source_background_support_px","silhouette_depth_samples",
        "silhouette_softmin_tau","pool_rejection_max_draw","pool_progress_draw_interval",
        "silhouette_pool_seed",
    )
    if any(name not in policy for name in required):
        raise ValueError("PREPARE_TRUTH139_POLICY_FIELD_MISSING")
    return policy["per_subject_seed_semantics"]


def _subject_seed(policy,row,index,seed_semantics):
    if seed_semantics=="LEGACY_ROSTER_ORDINAL_OFFSET_1000000":
        return int(policy["seed"])+int(index)*1000000, int(policy["silhouette_pool_seed"])+int(index)*1000000
    token=int(hashlib.sha256(str(row["subject_key"]).encode()).hexdigest()[:8],16)
    return (int(policy["seed"])+token)%(2**63-1), (int(policy["silhouette_pool_seed"])+token)%(2**63-1)


def prepare(input_root, output_root, source_dir, weight_path, policy, device="cuda", persist_root=None,
            roster_path=DEFAULT_ROSTER):
    roster_path,roster,roster_sha=load_roster(roster_path)
    seed_semantics=_validate_preparation_policy(policy,roster["schema"])
    output_root = Path(output_root); output_root.mkdir(parents=True, exist_ok=True)
    # All proofs are checked before loading the backbone or creating any shards.
    verified = [verify_observations(input_root, row) for row in roster["selected"]]
    model, authority = load_exact_dinov2_s_v2(source_dir, weight_path, device=device)
    subjects = []
    for index, (row, (folder, proof, binding)) in enumerate(zip(roster["selected"], verified)):
        start = time.monotonic()
        print(f"[PREPARE_BEGIN] {index+1}/{len(verified)} {row['subject_key']} verified_observation={binding[:12]}",flush=True)
        name = row["subject_key"]+".pt"
        path = output_root/name
        query_seed,ray_seed=_subject_seed(policy,row,index,seed_semantics)
        identity = value_hash({"observation_binding":binding, "preparation_policy":policy,
                               "roster_identity_sha256":roster_sha,"subject_id":row["subject_key"],
                               "query_seed":query_seed,"ray_seed":ray_seed,"seed_semantics":seed_semantics,
                               "foundation":authority.authority.authority_hash,
                               "preparation_source_sha256":file_hash(__file__),
                               "apparatus_source_sha256":file_hash(ROOT/"models/iris/training/apparatus_v1.py"),
                               "exterior_source_sha256":file_hash(ROOT/"tools/training/iris_exterior_audit_v1.py")})
        meta_path = path.with_suffix(".json")
        persisted = Path(persist_root)/name if persist_root else None
        persisted_meta = persisted.with_suffix(".json") if persisted else None
        if not path.exists() and persisted and persisted.is_file() and persisted_meta.is_file():
            cached = json.loads(persisted_meta.read_text())
            if cached["preparation_identity"] == identity and file_hash(persisted) == cached["shard_sha256"]:
                shutil.copyfile(persisted,path);shutil.copyfile(persisted_meta,meta_path)
        if path.exists() and meta_path.exists():
            cached = json.loads(meta_path.read_text())
            if cached["preparation_identity"] == identity and file_hash(path) == cached["shard_sha256"]:
                subjects.append(cached)
                print(f"[CACHE] {row['subject_key']} verified_shard={cached['shard_sha256'][:12]}",flush=True)
                continue
        cameras = json.loads((folder/"camera_bundle.json").read_text())
        rows = cameras["cameras"]
        center = np.asarray(cameras["common_center_xyz"],np.float32)
        half = float(cameras["common_half_extent"])
        tokens, masks = [], []
        for i in range(8):
            rgba = np.asarray(Image.open(folder/f"V{i}.png").convert("RGBA"), dtype=np.float32)/255.
            rgb = np.rint((rgba[:,:,:3]*rgba[:,:,3:4] + 1.-rgba[:,:,3:4])*255.).astype(np.uint8)
            native = torch.from_numpy(rgb.copy()).permute(2,0,1)[None].to(device)
            with torch.no_grad():
                patches = extract_dinov2_s_patch_map_flat_v2(model, native)
            tokens.append(patches.flatten(2).transpose(1,2).cpu())
            masks.append(np.frombuffer((folder/f"V{i}.mask.bin").read_bytes(), np.uint8).reshape(1024,1024)>0)
        image_tokens = torch.stack(tokens,dim=1).float()
        print(f"[PREPARE_DINO] {row['subject_key']} views=8 tokens=1369x384 seconds={time.monotonic()-start:.1f}",flush=True)
        axis = (torch.arange(37,dtype=torch.float32)+.5)*(2./37)-1.
        yy,xx = torch.meshgrid(axis,axis,indexing="ij")
        token_xy = torch.stack([xx,yy],dim=-1).reshape(1,1,1369,2).expand(1,8,-1,-1).clone()
        origins = np.asarray([r["origin"] for r in rows],np.float32)
        right = np.asarray([r["right"] for r in rows],np.float32)
        up = np.asarray([r["screen_up"] for r in rows],np.float32)
        cam_half = np.asarray([r["half_extent"] for r in rows],np.float32)
        features = np.concatenate([(origins-center)/half, right, up,
                                   np.asarray([r["forward"] for r in rows],np.float32),
                                   (cam_half/half)[:,None]],axis=1)
        signed = source_signed_distance_fields_v5(np.stack(masks))
        boundary_pixels = [torch.from_numpy(np.argwhere(np.abs(signed[v])<=policy["source_boundary_band_px"])[:,[1,0]].copy())
                           for v in range(8)]
        if not all(len(pixels) for pixels in boundary_pixels):
            raise ValueError("PREPARE_EMPTY_BOUNDARY_RAY_POOL")
        kwargs = dict(signed_2d=signed, center_xyz=center, normalization_half_extent=half,
                      camera_origins=origins, camera_right=right, camera_screen_up=up, camera_half_extent=cam_half)
        shard = {"subject_id":row["subject_key"],"observation_binding":binding,
                 "image_tokens":image_tokens,"token_xy":token_xy,
                 "camera_features":torch.from_numpy(features)[None],
                 "signed_2d":torch.from_numpy(signed),"center_xyz":torch.from_numpy(center),
                 "normalization_half_extent":half,"camera_origins":torch.from_numpy(origins),
                 "camera_right":torch.from_numpy(right),"camera_screen_up":torch.from_numpy(up),
                 "camera_forward":torch.from_numpy(np.asarray([r["forward"] for r in rows],np.float32)),
                 "boundary_pixels":boundary_pixels,
                 "camera_half_extent":torch.from_numpy(cam_half),"resolution":1024}
        shard.update(query_pools(shard,policy,query_seed,device))
        print(f"[PREPARE_QUERIES] {row['subject_key']} near={len(shard['near_points'])} straddle={len(shard['straddle_points'])} heldout={len(shard['heldout_points'])} seconds={time.monotonic()-start:.1f}",flush=True)
        shard.update(certified_ray_pools(shard,policy,ray_seed,device))
        print(f"[PREPARE_RAYS] {row['subject_key']} certified_rays={len(shard['ray_xy'])} strata=32 seconds={time.monotonic()-start:.1f}",flush=True)
        print(f"[PREPARE_R512_BEGIN] {row['subject_key']} source-only census",flush=True)
        shard['source_exterior_census']=exterior_scan(shard,resolution=512,device=device)
        print(f"[PREPARE_R512_DONE] {row['subject_key']} certified={shard['source_exterior_census']['certified_exterior_count']} seconds={time.monotonic()-start:.1f}",flush=True)
        heldout=shard["heldout_points"].numpy();heldout_targets=shard["heldout_targets"].numpy()
        parity=source_signed_field_torch_v5(torch.from_numpy(heldout[:4096]),**kwargs).cpu().numpy()
        parity_error=float(np.max(np.abs(parity-heldout_targets[:4096])))
        if parity_error>2e-6:
            raise ValueError("PREPARE_NUMPY_TORCH_FSTAR_PARITY")
        temp = path.with_suffix(".staged")
        torch.save(shard,temp);os.replace(temp,path)
        meta = {"subject_id":row["subject_key"],"shard_file":name,"shard_sha256":file_hash(path),
                "observation_binding":binding,"preparation_identity":identity,
                "fstar_cpu_parity_maximum_abs":parity_error,
                "source_sha256":row["source"]["sha256"],
                "query_seed":int(query_seed),"ray_seed":int(ray_seed),"seed_semantics":seed_semantics,
                "foundation_authority_sha256":authority.authority.authority_hash}
        meta["ray_court"]=shard["ray_court"]
        meta['source_exterior_census']=shard['source_exterior_census']
        atomic_json(meta_path,meta)
        if persisted:
            persisted.parent.mkdir(parents=True,exist_ok=True)
            staged = persisted.with_suffix(".staged")
            shutil.copyfile(path,staged)
            if file_hash(staged) != meta["shard_sha256"]:
                raise ValueError("PREPARE_DRIVE_SHARD_COPY_HASH")
            os.replace(staged,persisted);atomic_json(persisted_meta,meta)
        subjects.append(meta)
        print(f"[PREPARE] {row['subject_key']} views=8 tokens=1369x384 near={len(shard['near_points'])} "
              f"heldout={len(heldout)} Fstar_parity={parity_error:.2g} seconds={time.monotonic()-start:.1f}",flush=True)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    manifest = {"schema":"RealSaS.SharedIrisCohortShards.v1","subjects":subjects,
                "subject_count":len(subjects),"teacher_inputs_in_forward":False,"policy_sha256":value_hash(policy),
                "preparation_policy_schema":policy["schema"],"roster_schema":roster["schema"],
                "roster_identity_sha256":roster_sha,"seed_semantics":seed_semantics,
                "preparation_source_sha256":file_hash(__file__),
                "apparatus_source_sha256":file_hash(ROOT/"models/iris/training/apparatus_v1.py"),
                "observation_scope":"IRIS_BIND_IMAGE_DERIVED_FSTAR",
                "full_mechanical_truth_admitted":False,"training_authorized":False,
                "go_qualification_required":True}
    atomic_json(output_root/"COHORT_MANIFEST.json",manifest)
    if persist_root:
        atomic_json(Path(persist_root)/"COHORT_MANIFEST.json",manifest)
    return manifest


def main():
    p=argparse.ArgumentParser()
    for flag in ("input-root","output-root","source-dir","weight-path"):
        p.add_argument("--"+flag,type=Path,required=True)
    p.add_argument("--device",default="cuda")
    p.add_argument("--near-pool-count",type=int)
    p.add_argument("--heldout-count",type=int)
    p.add_argument("--roster",type=Path,default=DEFAULT_ROSTER)
    p.add_argument("--policy",type=Path)
    args=p.parse_args()
    _,roster,_=load_roster(args.roster)
    default_policy=TRUTH139_PREPARATION_POLICY_PATH if roster["schema"]=="RealSaS.Truth139ApparatusManifest.v1" else FIT8_POLICY_PATH
    policy_path=(args.policy or default_policy).resolve()
    policy=json.loads(policy_path.read_text())
    if args.near_pool_count:
        policy["near_pool_count"]=args.near_pool_count
    if args.heldout_count:
        policy["heldout_count"]=args.heldout_count
    prepare(args.input_root,args.output_root,args.source_dir,args.weight_path,policy,args.device,roster_path=args.roster)
if __name__=="__main__":
    main()
