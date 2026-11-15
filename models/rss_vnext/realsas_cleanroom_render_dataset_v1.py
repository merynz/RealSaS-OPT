from __future__ import annotations

"""Strict 3D-rendered RealSaS canonical/multiview truth materialization.

This is a *consumer* for asset renderer outputs (Blender/GLTF pipeline).
Every image must have been rendered from an independently admitted 3D asset
and have a same-asset, same-pose, same-camera RGB/normal/alpha/depth
counterpart. No diffusion-generated completions are accepted as targets.
See CleanroomDiffusionRenderRecordV1 for lineage, rigged/static pose
eligibility and family-level train/val/unseen separation.

Only the six-elevation=0-degree orthographic reference views are eligible
for the published six-view research baseline: azimuth [-90,-45,0,45,90,180]. Other camera
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

from .realsas_cleanroom_diffusion_training_v1 import (
    CleanroomDiffusionRenderRecordV1,
    validate_cleanroom_render_corpus_v1,
    STAGE_CANONICAL, STAGE_MULTIVIEW,
)
from .realsas_virtual_camera_policy_v1 import target_canonical_and_six_cameras_v1


_SCHEMA="RealSaS.RSSvNext.RealSaSCleanroomRenderedSixViewDataset.v1"
_CAMERA_SCHEMA="REALSAS_ZERO_ELEVATION_SIX_ORTHOGRAPHIC_AZIMUTH_V1"
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
        raise ValueError("RSS_VNEXT_REALSAS_RENDER_RECORD_NOT_OBJECT")
    attributes=dict(value)
    for key in _TUPLE_FIELDS:
        if key in attributes and type(attributes[key]) is list:
            attributes[key]=tuple(attributes[key])
    try:
        return CleanroomDiffusionRenderRecordV1(**attributes)
    except (TypeError,ValueError) as exc:
        raise ValueError("RSS_VNEXT_REALSAS_RENDER_RECORD_INVALID") from exc


def _inside(root: Path, relative: str) -> Path:
    if (
        type(relative) is not str or not relative.endswith(".png")
        or not relative or relative.startswith("/")
        or "\\" in relative or "\x00" in relative
    ):
        raise ValueError("RSS_VNEXT_REALSAS_RENDER_PATH_INVALID")
    path=Path(relative)
    if any(part in (".","..") for part in path.parts):
        raise ValueError("RSS_VNEXT_REALSAS_RENDER_PATH_TRAVERSAL")
    candidate=root
    for part in path.parts:
        candidate=candidate/part
        if candidate.is_symlink():
            raise ValueError("RSS_VNEXT_REALSAS_RENDER_SYMLINK_FORBIDDEN")
    final=candidate.resolve(strict=True)
    if not final.is_relative_to(root.resolve()) or not final.is_file():
        raise ValueError("RSS_VNEXT_REALSAS_RENDER_PATH_OUTSIDE_ROOT")
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
            raise ValueError("RSS_VNEXT_REALSAS_RENDER_IMAGE_DOMAIN")
        if mode=="MASK":
            # Renderer may store a grayscale mask or use the PNG alpha
            # channel. Luminance of an RGBA PNG is NOT its coverage.
            if im.mode in ("RGBA","LA") or "transparency" in im.info:
                arr=np.array(im.convert("RGBA"),dtype=np.uint8)[:,:,3]
            else:
                arr=np.array(im.convert("L"),dtype=np.uint8)
        else:
            arr=np.array(im.convert(mode),dtype=np.uint8)
    return arr


class RealSaSCleanroomRenderedSixViewDatasetV1(Dataset):
    """Source-rendered, pose-qualified Stage A / Stage B training examples.

    Stage A: TWO arbitrary source-pose RGBA -> TWO canonical pose RGB targets.
    Stage B: TWO canonical-pose RGBA -> SIX RGB plus SIX aligned normal targets.

    The original customer source camera is NEVER inferred from a target
    render. A fixed virtual target camera policy is shared with inference.
    The old six-view API is preserved by default for research compatibility.
    """
    def __init__(
        self, manifest: str | Path, *, split: str="train",
        stage: str=STAGE_MULTIVIEW,
    ) -> None:
        super().__init__()
        if stage not in (STAGE_CANONICAL, STAGE_MULTIVIEW):
            raise ValueError("RSS_VNEXT_REALSAS_RENDER_STAGE_INVALID")
        self.stage=stage
        manifest=Path(manifest)
        if not manifest.is_file() or manifest.is_symlink():
            raise ValueError("RSS_VNEXT_REALSAS_RENDER_MANIFEST_MISSING")
        try:
            raw=json.loads(manifest.read_text("utf-8"))
        except (ValueError,UnicodeError) as exc:
            raise ValueError("RSS_VNEXT_REALSAS_RENDER_MANIFEST_JSON_INVALID") from exc
        if (
            type(raw) is not dict or raw.get("schema")!=_SCHEMA
            or raw.get("camera_schema")!=_CAMERA_SCHEMA
            or raw.get("azimuth_degrees")!=list(_AZIMUTHS)
            or raw.get("all_images_from_admitted_3d_assets") is not True
            or raw.get("contains_diffusion_generated_training_targets") is not False
            or type(raw.get("items")) is not list or not raw["items"]
        ):
            raise ValueError("RSS_VNEXT_REALSAS_RENDER_MANIFEST_CONTRACT")
        self.root=manifest.parent.resolve()
        self.entries=[]
        records=[]
        for item in raw["items"]:
            if type(item) is not dict:
                raise ValueError("RSS_VNEXT_REALSAS_RENDER_ITEM_INVALID")
            record=_record_from_json(item.get("truth_record"))
            files=item.get("files")
            if type(files) is not dict or set(files)!=set(_FILE_KEYS):
                raise ValueError("RSS_VNEXT_REALSAS_RENDER_MODALITIES_INVALID")
            if record.generated_diffusion_images_are_training_targets:
                raise ValueError("RSS_VNEXT_REALSAS_GENERATED_PSEUDOTRUTH_REJECTED")
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
                    raise ValueError("RSS_VNEXT_REALSAS_RENDER_MODALITY_LENGTH_DRIFT")
                resolved[key]=[]
                for path,expected_digest in zip(paths,hashes):
                    full=_inside(self.root,path)
                    if _digest(full)!=expected_digest:
                        raise ValueError("RSS_VNEXT_REALSAS_RENDER_TRUTH_HASH_DRIFT")
                    resolved[key].append(full)
            records.append(record)
            if record.split==split:
                # Static assets legitimately provide six-view construction,
                # but they CANNOT teach arbitrary-pose -> canonical-pose.
                if stage==STAGE_CANONICAL and not record.source_pose_truth_available:
                    continue
                self.entries.append((record,resolved))
        self.corpus_report=validate_cleanroom_render_corpus_v1(records)
        self.manifest_sha256=_digest(manifest)
        if not self.entries:
            raise ValueError("RSS_VNEXT_REALSAS_RENDER_SPLIT_EMPTY")
        self.split=split

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int) -> dict[str,Any]:
        record,paths=self.entries[index]
        # Rehash the exact source/target files used by THIS stage on EVERY
        # read: source images are model INPUTS, never unseen teacher labels.
        keys=(
            ("input_scene_rgb","canonical_target_rgb")
            if self.stage==STAGE_CANONICAL
            else (
                "canonical_target_rgb","multiview_target_rgb",
                "multiview_target_normal","multiview_target_alpha",
            )
        )
        hashes={
            "input_scene_rgb":record.input_scene_rgb_sha256,
            "canonical_target_rgb":record.canonical_target_rgb_sha256,
            "multiview_target_rgb":record.multiview_target_rgb_sha256,
            "multiview_target_normal":record.multiview_target_normal_sha256,
            "multiview_target_alpha":record.multiview_target_alpha_sha256,
        }
        arrays={}
        for key in keys:
            mode="MASK" if key=="multiview_target_alpha" else "RGBA"
            arrays[key]=[]
            for file,hexdigest in zip(paths[key],hashes[key]):
                if _digest(file)!=hexdigest:
                    raise ValueError(
                        "RSS_VNEXT_REALSAS_RENDER_TRUTH_HASH_CHANGED_DURING_TRAINING"
                    )
                arrays[key].append(_image(file,mode=mode))

        # Two distinct original *files* are mandatory; a training item must
        # not silently fold both source views into one canonical reference.
        source=(
            arrays["input_scene_rgb"] if self.stage==STAGE_CANONICAL
            else arrays["canonical_target_rgb"]
        )
        targets=(
            arrays["canonical_target_rgb"] if self.stage==STAGE_CANONICAL
            else arrays["multiview_target_rgb"]
        )
        if len(source)!=2 or len(targets)!=(2 if self.stage==STAGE_CANONICAL else 6):
            raise ValueError("RSS_VNEXT_REALSAS_STAGE_VIEW_CARDINALITY_DRIFT")
        height,width=source[0].shape[:2]
        if any(
            image.shape[:2]!=(height,width)
            for values in arrays.values() for image in values
        ):
            raise ValueError("RSS_VNEXT_REALSAS_RENDER_CROSS_CAMERA_RESOLUTION_DRIFT")

        def rgb(value: np.ndarray) -> torch.Tensor:
            alpha=value[:,:,3:4].astype(np.float32)/255.
            color=value[:,:,:3].astype(np.float32)/255.
            # Match existing RealSaS/Era3D white-composite RGB convention.
            color=color*alpha+(1.-alpha)
            return torch.from_numpy(color.transpose(2,0,1).copy())

        def rgba(value: np.ndarray) -> torch.Tensor:
            return torch.from_numpy(
                (value.astype(np.float32)/255.).transpose(2,0,1).copy()
            )

        def mask(value: np.ndarray) -> torch.Tensor:
            return torch.from_numpy(
                (value.astype(np.float32)/255.)[None].copy()
            )

        canonical_cameras,six_cameras=target_canonical_and_six_cameras_v1(1)
        target_cameras=(
            canonical_cameras[0]
            if self.stage==STAGE_CANONICAL else
            torch.cat((six_cameras[0],six_cameras[0]),dim=0)
        )
        modalities=torch.zeros(
            (2 if self.stage==STAGE_CANONICAL else 12,),dtype=torch.long,
        )
        if self.stage==STAGE_MULTIVIEW:
            modalities[6:]=1
            target_rgb=torch.stack([rgb(x) for x in targets])
            target_normal=torch.stack([
                rgb(x) for x in arrays["multiview_target_normal"]
            ])
            target_valid=torch.stack([
                mask(x) for x in arrays["multiview_target_alpha"]
            ])
        else:
            # The canonical target PNGs are independently rendered in the
            # source asset's canonical pose, NOT diffusion predictions.
            target_rgb=torch.stack([rgb(x) for x in targets])
            target_valid=torch.stack([
                mask(x[:,:,3]) for x in targets
            ])
            target_normal=None

        if not bool((target_valid>0).any()):
            raise ValueError("RSS_VNEXT_REALSAS_STAGE_NO_QUALIFIED_TARGET_PIXELS")
        sample={
            "stage":self.stage,
            "source_rgba":torch.stack([rgba(x) for x in source]),
            "target_rgb":target_rgb,
            "target_rgb_valid":target_valid,
            "target_camera_features":target_cameras,
            "target_modalities":modalities,
            "target_virtual_camera_authority":
                "FIXED_RENDER_TARGET_CAMERA_POLICY_NOT_SOURCE_OBSERVATION",
            "source_pair_authority":(
                "SOURCE_3D_RENDERED_TWO_ARBITRARY_POSES"
                if self.stage==STAGE_CANONICAL else
                "SOURCE_3D_RENDERED_TWO_CANONICAL_IMAGES"
            ),
            "original_source_camera_observed":False,
            "source_asset_lineage_sha256":record.source_asset_lineage_sha256,
            "dedup_family_sha256":record.dedup_family_sha256,
            "rights_admission_sha256":record.rights_admission_sha256,
            "canonical_pose_definition_sha256":
                record.neutral_pose_definition_sha256,
            "input_pose_ids":record.input_pose_ids,
            "morphology_id":record.morphology_id,
            "split":record.split,
        }
        if self.stage==STAGE_MULTIVIEW:
            # Legacy single-reference baseline remains for old consumers,
            # but the actual TWO-view conditioner must use source_rgba.
            sample.update({
                "reference_rgb":rgb(source[0]),
                "target_normal_encoded_rgb":target_normal,
                "target_normal_valid":target_valid.clone(),
            })
        return sample


class RealSaSCleanroomRenderedCanonicalPoseDatasetV1(
    RealSaSCleanroomRenderedSixViewDatasetV1
):
    """Stage A: strictly two arbitrary-pose renders -> canonical pair."""
    def __init__(self, manifest: str | Path, *, split: str="train") -> None:
        super().__init__(manifest,split=split,stage=STAGE_CANONICAL)


__all__=[
    "RealSaSCleanroomRenderedSixViewDatasetV1",
    "RealSaSCleanroomRenderedCanonicalPoseDatasetV1",
]
