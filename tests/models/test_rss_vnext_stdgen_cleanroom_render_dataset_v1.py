from __future__ import annotations

from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest
import torch

from models.rss_vnext.stdgen_cleanroom_render_dataset_v1 import (
    StdGENCleanroomRenderedSixViewDatasetV1,
)


def _sha(value: bytes) -> str:
    return sha256(value).hexdigest()


def _materialize(tmp_path,*,extra_unseen=False):
    root=tmp_path
    items=[]
    for asset,split in [("a","train")]+([("b","unseen")] if extra_unseen else []):
        hashes={}
        paths={}
        names=(
            ("input_scene_rgb",2),("canonical_target_rgb",2),
            ("multiview_target_rgb",6),("multiview_target_normal",6),
            ("multiview_target_alpha",6),("multiview_target_depth",6),
        )
        for key,count in names:
            hashes[key]=[]
            paths[key]=[]
            for i in range(count):
                name=f"{asset}_{key}_{i}.png"
                pixel=np.zeros((16,16,4),dtype=np.uint8)
                pixel[...,0]=42+i
                pixel[...,3]=255
                if key=="multiview_target_alpha":
                    pixel[0,:,3]=0
                if key=="multiview_target_depth":
                    pixel[...,0]=255
                im=Image.fromarray(pixel,"RGBA")
                im.save(root/name)
                hashes[key].append(_sha((root/name).read_bytes()))
                paths[key].append(name)
        record=dict(
            source_asset_lineage_sha256=_sha(asset.encode()),
            dedup_family_sha256=_sha(("family-"+asset).encode()),
            rights_admission_sha256=_sha(b"admitted"),
            morphology_id="generic_winged",
            neutral_pose_definition_sha256=_sha(b"pose-policy"),
            split=split,
            input_pose_ids=["posed-A","posed-B"],
            input_scene_rgb_sha256=hashes["input_scene_rgb"],
            canonical_target_rgb_sha256=hashes["canonical_target_rgb"],
            multiview_target_rgb_sha256=hashes["multiview_target_rgb"],
            multiview_target_normal_sha256=hashes["multiview_target_normal"],
            multiview_target_alpha_sha256=hashes["multiview_target_alpha"],
            multiview_target_depth_sha256=hashes["multiview_target_depth"],
            renderer_version_sha256=_sha(b"blender-renderer-v1"),
            source_pose_truth_available=True,
            source_asset_training_rights_independently_admitted=True,
        )
        items.append({"truth_record":record,"files":paths})
    manifest=dict(
        schema="RealSaS.RSSvNext.StdGENCleanroomRenderedSixViewDataset.v1",
        camera_schema="STDGEN_ZERO_ELEVATION_SIX_ORTHOGRAPHIC_AZIMUTH_V1",
        azimuth_degrees=[-90,-45,0,45,90,180],
        all_images_from_admitted_3d_assets=True,
        contains_diffusion_generated_training_targets=False,
        items=items,
    )
    file=root/"manifest.json"
    file.write_text(json.dumps(manifest))
    return file


def test_admitted_3d_render_truth_feeds_era3d_six_view_batch(tmp_path):
    manifest=_materialize(tmp_path,extra_unseen=True)
    train=StdGENCleanroomRenderedSixViewDatasetV1(manifest,split="train")
    item=train[0]
    assert len(train)==1
    assert item["reference_rgb"].shape==(3,16,16)
    assert item["target_rgb"].shape==(6,3,16,16)
    assert item["target_normal_encoded_rgb"].shape==(6,3,16,16)
    assert item["target_rgb_valid"].shape==(6,1,16,16)
    assert torch.all(item["target_rgb_valid"][:,0,0,:]==0.)
    assert len(train.manifest_sha256)==64
    assert train.corpus_report["counts_by_split"]["train"]==1
    assert train.corpus_report["counts_by_split"]["unseen"]==1
    unseen=StdGENCleanroomRenderedSixViewDatasetV1(manifest,split="unseen")
    assert unseen[0]["source_asset_lineage_sha256"]!=item["source_asset_lineage_sha256"]


def test_changed_render_file_rejected_before_training_and_on_epoch_read(tmp_path):
    manifest=_materialize(tmp_path)
    ds=StdGENCleanroomRenderedSixViewDatasetV1(manifest)
    png=tmp_path/"a_multiview_target_rgb_0.png"
    old=png.read_bytes()
    png.write_bytes(old+b"altered")
    with pytest.raises(ValueError,match="HASH_CHANGED_DURING_TRAINING"):
        ds[0]
    with pytest.raises(ValueError,match="TRUTH_HASH_DRIFT"):
        StdGENCleanroomRenderedSixViewDatasetV1(manifest)


def test_pseudotruth_mislabelled_camera_or_cross_family_holdout_rejected(tmp_path):
    file=_materialize(tmp_path,extra_unseen=True)
    data=json.loads(file.read_text())
    data["contains_diffusion_generated_training_targets"]=True
    file.write_text(json.dumps(data))
    with pytest.raises(ValueError,match="MANIFEST_CONTRACT"):
        StdGENCleanroomRenderedSixViewDatasetV1(file)
    data["contains_diffusion_generated_training_targets"]=False
    data["azimuth_degrees"]=[0,45,90,135,180,225]
    file.write_text(json.dumps(data))
    with pytest.raises(ValueError,match="MANIFEST_CONTRACT"):
        StdGENCleanroomRenderedSixViewDatasetV1(file)
    data["azimuth_degrees"]=[-90,-45,0,45,90,180]
    data["items"][1]["truth_record"]["dedup_family_sha256"] = data["items"][0]["truth_record"]["dedup_family_sha256"]
    file.write_text(json.dumps(data))
    with pytest.raises(ValueError,match="TRAIN_UNSEEN_LINEAGE_LEAK"):
        StdGENCleanroomRenderedSixViewDatasetV1(file)


def test_manifest_rejects_escape_path_without_reading_outside_repository(tmp_path):
    file=_materialize(tmp_path)
    data=json.loads(file.read_text())
    data["items"][0]["files"]["input_scene_rgb"][0]="../secret.png"
    file.write_text(json.dumps(data))
    with pytest.raises(ValueError,match="TRAVERSAL"):
        StdGENCleanroomRenderedSixViewDatasetV1(file)
