from __future__ import annotations

from compiler.realsas_compiler_core.appearance import build_observed_appearance_binding
from compiler.realsas_compiler_core.hashing import content_sha256


def observation_hashes_from_master_manifest(manifest) -> dict[int,str]:
    out={}
    for view in manifest.views:
        out[int(view.view_index)]=content_sha256({"asset_id":manifest.asset_id,"image_relpath":view.image_relpath,"image_sha256":view.image_sha256,"raster_relpath":view.raster_relpath,"raster_sha256":view.raster_sha256,"camera_sha256":view.camera_sha256})
    if set(out)!=set(range(8)): raise ValueError("appearance manifest requires exact 8 views")
    return out


def build_master_observed_appearance_v1(*,manifest,surface,mesh,target_view_index:int):
    views={int(v.view_index):v for v in manifest.views}
    if int(target_view_index) not in views: raise ValueError("target view absent from manifest")
    obs=observation_hashes_from_master_manifest(manifest); target=views[int(target_view_index)]
    atlas_hash=content_sha256({"image_sha256":target.image_sha256,"image_relpath":target.image_relpath})
    return build_observed_appearance_binding(surface=surface,mesh=mesh,target_view_index=int(target_view_index),camera_binding_hash=target.camera_sha256,observation_hash_by_view=obs,atlas_payload_hash=atlas_hash)
