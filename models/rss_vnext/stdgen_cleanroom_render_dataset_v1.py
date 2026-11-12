from __future__ import annotations

"""Strict 3D-rendered StdGEN canonical/multiview truth materialization.

This is a *consumer* for asset renderer outputs (Blender/GLTF pipeline).
Every image must have been rendered from an independently admitted 3D asset
and have a same-asset, same-pose, same-camera RGB/normal/alpha/depth
counterpart. No diffusion-generated completions are accepted as targets.
See CleanroomDiffusionRenderRecordV1 for lineage, rigged/static pose
eligibility and family-level train/val/unseen separation.

Only the six-elevation=0-degree orthographic reference views are eligible
for the StdGEN/Era3D baseline: azimuth [-90,-45,0,45,90,180]. Other camera
patterns require a different independently qualified model contract.
"""

from dataclasses import fields
from hashlib import sha256
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

from .stdgen_cleanroom_diffusion_training_v1 import (
    CleanroomDiffusionRenderRecordV1,
    validate_cleanroom_render_corpus_v1,
)


_SCHEMA="RealSaS.RSSvNext.StdGENCleanroomRenderedSixViewDataset.v1"
_CAMERA_SCHEMA="STDGEN_ZERO_ELEVATION_SIX_ORTHOGRAPHIC_AZIMUTH_V1"
_AZIMUTHS=(-90,-45,0,45,90,180)
_FILE_KEYS=(
    "input_scene_rgb","canonical_target_rgb",
    "multiview_target_rgb","multiview_target_normal",
    "multiview_target_alpha","multiview_target_depth",
)
_TUPLE_FIELDS=frozenset({
    "input_pose_ids","input_scene_rgb_sha256",
    "canonical_target_rgb_sha256",
    "multiview_target_rgb_sha256","multiview_target_normal_sha256",
    "multiview_target_alpha_sha256","multiview_target_depth_sha256",
})


def _record_from_json(value: Any) -> CleanroomDiffusionRenderRecordV1:
    if type(value) is not dict:
        raise ValueError("RSS_VNEXT_STDGEN_RENDER_RECORD_NOT_OBJECT")
    attributes=dict(value)
    for key in _TUPLE_FIELDS:
        if key in attributes and type(attributes[key]) is list:
            attributes[key]=tuple(attributes[key])
    try:
        return CleanroomDiffusionRenderRecordV1(**attributes)
    except (TypeError,ValueError) as exc:
        raise ValueError("RSS_VNEXT_STDGEN_RENDER_RECORD_INVALID") from exc


def _inside(root: Path, relative: str) -> Path:
    if (
        type(relative) is not str or not relative.endswith(".png")
        or not relative or relative.startswith("/")
        or "\\" in relative or "\x00" in relative
    ):
        raise ValueError("RSS_VNEXT_STDGEN_RENDER_PATH_INVALID")
    path=Path(relative)
    if any(part in (".","..") for part in path.parts):
        raise ValueError("RSS_VNEXT_STDGEN_RENDER_PATH_TRAVERSAL")
    candidate=root
    for part in path.parts:
        candidate=candidate/part
        if candidate.is_symlink():
            raise ValueError("RSS_VNEXT_STDGEN_RENDER_SYMLINK_FORBIDDEN")
    final=candidate.resolve(strict=True)
    if not final.is_relative_to(root.resolve()) or not final.is_file():
        raise ValueError("RSS_VNEXT_STDGEN_RENDER_PATH_OUTSIDE_ROOT")
    return final


def _digest(file: Path) -> str:
    d=sha256()
    with file.open("rb") as stream:
        while chunk:=stream.read(4*1024*1024):
            d.update(chunk)
    return d.hexdigest()


def _image(file: Path, *, mode: str) -> np.ndarray:
    with Image.open(file) as im:
        im.load()
        if im.format!="PNG" or im.width>2048 or im.height>2048 or min(im.size)<16:
            raise ValueError("RSS_VNEXT_STDGEN_RENDER_IMAGE_DOMAIN")
        arr=np.array(im.convert(mode),dtype=np.uint8)
    return arr


class StdGENCleanroomRenderedSixViewDatasetV1(Dataset):
    """Train/validation samples with verified 3D-source canonical supervision.

    The training reference is the *rendered canonical source* (not the user
    image). This is StdGEN's 1-reference Era3D baseline. The existing
    paired-user-source conditioner remains an ablation in another module.
    """
    def __init__(self, manifest: str | Path, *, split: str="train") -> None:
        super().__init__()
        manifest=Path(manifest)
        if not manifest.is_file() or manifest.is_symlink():
            raise ValueError("RSS_VNEXT_STDGEN_RENDER_MANIFEST_MISSING")
        try:
            raw=json.loads(manifest.read_text("utf-8"))
        except (ValueError,UnicodeError) as exc:
            raise ValueError("RSS_VNEXT_STDGEN_RENDER_MANIFEST_JSON_INVALID") from exc
        if (
            type(raw) is not dict or raw.get("schema")!=_SCHEMA
            or raw.get("camera_schema")!=_CAMERA_SCHEMA
            or raw.get("azimuth_degrees")!=list(_AZIMUTHS)
            or raw.get("all_images_from_admitted_3d_assets") is not True
            or raw.get("contains_diffusion_generated_training_targets") is not False
            or type(raw.get("items")) is not list or not raw["items"]
        ):
            raise ValueError("RSS_VNEXT_STDGEN_RENDER_MANIFEST_CONTRACT")
        self.root=manifest.parent.resolve()
        self.entries=[]
        records=[]
        for item in raw["items"]:
            if type(item) is not dict:
                raise ValueError("RSS_VNEXT_STDGEN_RENDER_ITEM_INVALID")
            record=_record_from_json(item.get("truth_record"))
            files=item.get("files")
            if type(files) is not dict or set(files)!=set(_FILE_KEYS):
                raise ValueError("RSS_VNEXT_STDGEN_RENDER_MODALITIES_INVALID")
            if record.generated_diffusion_images_are_training_targets:
                raise ValueError("RSS_VNEXT_STDGEN_GENERATED_PSEUDOTRUTH_REJECTED")
            expected={
                "input_scene_rgb":record.input_scene_rgb_sha256,
                "canonical_target_rgb":record.canonical_target_rgb_sha256,
                "multiview_target_rgb":record.multiview_target_rgb_sha256,
                "multiview_target_normal":record.multiview_target_normal_sha256,
                "multiview_target_alpha":record.multiview_target_alpha_sha256,
                "multiview_target_depth":record.multiview_target_depth_sha256,
            }
            resolved={}
            for key,hashes in expected.items():
                paths=files[key]
                if type(paths) is not list or len(paths)!=len(hashes):
                    raise ValueError("RSS_VNEXT_STDGEN_RENDER_MODALITY_LENGTH_DRIFT")
                resolved[key]=[]
                for path,expected_digest in zip(paths,hashes):
                    full=_inside(self.root,path)
                    if _digest(full)!=expected_digest:
                        raise ValueError("RSS_VNEXT_STDGEN_RENDER_TRUTH_HASH_DRIFT")
                    resolved[key].append(full)
            records.append(record)
            if record.split==split:
                if not record.source_pose_truth_available:
                    # Only canonical-render identity is valid for stage 2,
                    # but rigless static objects cannot supervise stage 1.
                    pass
                self.entries.append((record,resolved))
        self.corpus_report=validate_cleanroom_render_corpus_v1(records)
        self.manifest_sha256=_digest(manifest)
        if not self.entries:
            raise ValueError("RSS_VNEXT_STDGEN_RENDER_SPLIT_EMPTY")
        self.split=split

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int) -> dict[str,Any]:
        record,paths=self.entries[index]
        # Hash again when read: an on-disk render changed during a running
        # training job must never silently become another asset's teacher.
        arrays={}
        expected={
            "canonical_target_rgb":record.canonical_target_rgb_sha256,
            "multiview_target_rgb":record.multiview_target_rgb_sha256,
            "multiview_target_normal":record.multiview_target_normal_sha256,
            "multiview_target_alpha":record.multiview_target_alpha_sha256,
        }
        for key,digests in expected.items():
            mode="L" if key=="multiview_target_alpha" else "RGBA"
            arrays[key]=[]
            for file,hexdigest in zip(paths[key],digests):
                if _digest(file)!=hexdigest:
                    raise ValueError("RSS_VNEXT_STDGEN_RENDER_TRUTH_HASH_CHANGED_DURING_TRAINING")
                arrays[key].append(_image(file,mode=mode))
        ref=arrays["canonical_target_rgb"][0]
        height,width=ref.shape[:2]
        for key,values in arrays.items():
            if any(x.shape[:2]!=(height,width) for x in values):
                raise ValueError("RSS_VNEXT_STDGEN_RENDER_CROSS_CAMERA_RESOLUTION_DRIFT")

        def rgb(value: np.ndarray) -> torch.Tensor:
            alpha=value[:,:,3:4].astype(np.float32)/255.
            color=value[:,:,:3].astype(np.float32)/255.
            # White composite as in StdGEN/Era3D training.
            color=color*alpha+(1.-alpha)
            return torch.from_numpy(color.transpose(2,0,1).copy())
        def mask(value: np.ndarray) -> torch.Tensor:
            return torch.from_numpy(
                (value.astype(np.float32)/255.)[None].copy()
            )
        rgb_target=torch.stack([rgb(x) for x in arrays["multiview_target_rgb"]])
        normal_target=torch.stack([rgb(x) for x in arrays["multiview_target_normal"]])
        valid=torch.stack([mask(x) for x in arrays["multiview_target_alpha"]])
        return {
            "reference_rgb":rgb(ref),
            "target_rgb":rgb_target,
            "target_normal_encoded_rgb":normal_target,
            "target_rgb_valid":valid,
            "target_normal_valid":valid.clone(),
            "source_asset_lineage_sha256":record.source_asset_lineage_sha256,
            "morphology_id":record.morphology_id,
            "split":record.split,
        }


__all__=["StdGENCleanroomRenderedSixViewDatasetV1"]
