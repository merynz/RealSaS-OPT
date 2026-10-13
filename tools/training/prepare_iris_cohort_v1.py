"""Verified observation -> frozen DINO features and image-derived F* shards."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import shutil
import time
import numpy as np
from PIL import Image
import torch
from models.iris.training.cohort_v1 import file_hash, value_hash, atomic_json
from models.iris.v2.dinov2_foundation_v2 import load_exact_dinov2_s_v2, extract_dinov2_s_patch_map_flat_v2
from models.iris.v5.source_signed_field_v5 import source_signed_distance_fields_v5, source_signed_field_numpy_v5, source_signed_field_torch_v5

ROOT = Path(__file__).resolve().parents[2]

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

def prepare(input_root, output_root, source_dir, weight_path, policy, device="cuda", persist_root=None):
    output_root = Path(output_root); output_root.mkdir(parents=True, exist_ok=True)
    roster = json.loads((ROOT/"canonical/FIT8_DIVERSITY_SELECTION_V1.json").read_text())
    # All proofs are checked before loading the backbone or creating any shards.
    verified = [verify_observations(input_root, row) for row in roster["selected"]]
    model, authority = load_exact_dinov2_s_v2(source_dir, weight_path, device=device)
    subjects = []
    for index, (row, (folder, proof, binding)) in enumerate(zip(roster["selected"], verified)):
        start = time.monotonic()
        name = row["subject_key"]+".pt"
        path = output_root/name
        identity = value_hash({"observation_binding":binding, "policy":policy,
                               "foundation":authority.authority.authority_hash,
                               "preparation_source_sha256":file_hash(__file__)})
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
        rng = np.random.default_rng(policy["seed"]+index*2+1000)
        near, near_count, rounds = [], 0, 0
        while near_count < policy["near_pool_count"]:
            query = rng.uniform(-1.,1.,(131072,3)).astype(np.float32)
            target = source_signed_field_numpy_v5(query,**kwargs)
            chosen = query[np.abs(target)<=policy["near_zero_band"]]
            near.append(chosen);near_count += len(chosen);rounds+=1
            if rounds > 512:
                raise RuntimeError("PREPARE_NEAR_ZERO_POOL_EXHAUSTED")
        near = np.concatenate(near)[:policy["near_pool_count"]]
        heldout_rng = np.random.default_rng(policy["seed"]+index*2+1001)
        heldout = heldout_rng.uniform(-1.,1.,(policy["heldout_count"],3)).astype(np.float32)
        heldout_targets = source_signed_field_numpy_v5(heldout,**kwargs)
        parity = source_signed_field_torch_v5(torch.from_numpy(heldout[:4096]), **kwargs).cpu().numpy()
        parity_error = float(np.max(np.abs(parity-heldout_targets[:4096])))
        if parity_error > 2e-6:
            raise ValueError("PREPARE_NUMPY_TORCH_FSTAR_PARITY")
        # Exact duplicate checks use byte addresses, independent of rounded display.
        if set(map(bytes,near)).intersection(map(bytes,heldout)):
            raise ValueError("PREPARE_HELDOUT_QUERY_LEAKAGE")
        shard = {"subject_id":row["subject_key"],"observation_binding":binding,
                 "image_tokens":image_tokens,"token_xy":token_xy,
                 "camera_features":torch.from_numpy(features)[None],
                 "signed_2d":torch.from_numpy(signed),"center_xyz":torch.from_numpy(center),
                 "normalization_half_extent":half,"camera_origins":torch.from_numpy(origins),
                 "camera_right":torch.from_numpy(right),"camera_screen_up":torch.from_numpy(up),
                 "camera_forward":torch.from_numpy(np.asarray([r["forward"] for r in rows],np.float32)),
                 "boundary_pixels":boundary_pixels,
                 "camera_half_extent":torch.from_numpy(cam_half),"resolution":1024,
                 "near_points":torch.from_numpy(near),"heldout_points":torch.from_numpy(heldout),
                 "heldout_targets":torch.from_numpy(heldout_targets)}
        temp = path.with_suffix(".staged")
        torch.save(shard,temp);os.replace(temp,path)
        meta = {"subject_id":row["subject_key"],"shard_file":name,"shard_sha256":file_hash(path),
                "observation_binding":binding,"preparation_identity":identity,
                "fstar_cpu_parity_maximum_abs":parity_error,
                "source_sha256":row["source"]["sha256"],
                "foundation_authority_sha256":authority.authority.authority_hash}
        atomic_json(meta_path,meta)
        if persisted:
            persisted.parent.mkdir(parents=True,exist_ok=True)
            staged = persisted.with_suffix(".staged")
            shutil.copyfile(path,staged)
            if file_hash(staged) != meta["shard_sha256"]:
                raise ValueError("PREPARE_DRIVE_SHARD_COPY_HASH")
            os.replace(staged,persisted);atomic_json(persisted_meta,meta)
        subjects.append(meta)
        print(f"[PREPARE] {row['subject_key']} views=8 tokens=1369x384 near={len(near)} "
              f"heldout={len(heldout)} Fstar_parity={parity_error:.2g} seconds={time.monotonic()-start:.1f}",flush=True)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    manifest = {"schema":"RealSaS.SharedIrisCohortShards.v1","subjects":subjects,
                "teacher_inputs_in_forward":False,"policy_sha256":value_hash(policy),
                "preparation_source_sha256":file_hash(__file__),
                "observation_scope":"IRIS_BIND_IMAGE_DERIVED_FSTAR",
                "full_mechanical_truth_admitted":False,"go_qualification_required":True}
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
    args=p.parse_args()
    policy=json.loads((ROOT/"canonical/FIT8_IRIS_SHARED_TRAINING_POLICY_V1.json").read_text())
    if args.near_pool_count:
        policy["near_pool_count"]=args.near_pool_count
    if args.heldout_count:
        policy["heldout_count"]=args.heldout_count
    prepare(args.input_root,args.output_root,args.source_dir,args.weight_path,policy,args.device)
if __name__=="__main__":
    main()
