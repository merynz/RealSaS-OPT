from __future__ import annotations

from dataclasses import replace
import numpy as np
import pytest

from compiler.realsas_compiler_core.rss_vnext_cleanroom_diffusion_source_v1 import (
    RSSvNextRawTwoSceneObservationsV1,
    seal_cleanroom_diffusion_prediction_v1,
    validate_cleanroom_diffusion_prediction_v1,
    STAGE_CANONICAL,
    STAGE_MULTIVIEW,
)
from compiler.realsas_compiler_core.types import QualificationError
from compiler.realsas_compiler_core.rss_vnext_source_observation_v1 import (
    canonicalize_two_view_source_v1,
)


def raw_two_scenes():
    # Unknown source camera, unequal source resolutions, RGB photo with
    # contextual scenery in view 0; RGBA game art in view 1.
    a=np.full((3,21,29),201,dtype=np.uint8)
    b=np.zeros((4,31,19),dtype=np.uint8)
    b[0]=110
    b[3,5:26,3:16]=255
    return RSSvNextRawTwoSceneObservationsV1(
        frames_chw=(a,b),view_hints=("OTHER","UNKNOWN"),
        source_rights_claim_sha256="1"*64,
    )


def candidate_views(count=2):
    rgba=np.zeros((count,4,20,26),dtype=np.uint8)
    rgba[:,0,5:15,8:18]=197
    rgba[:,3,5:15,8:18]=255
    cam=np.zeros((count,13),dtype=np.float32)
    cam[:,3]=1.
    cam[:,7]=1.
    cam[:,11]=1.
    cam[:,12]=1.
    confidence=np.full((count,1,20,26),.7,dtype=np.float32)
    return rgba,cam,confidence


def canonical(source=None):
    source=source or raw_two_scenes()
    rgba,cam,confidence=candidate_views()
    return seal_cleanroom_diffusion_prediction_v1(
        source=source, generator_checkpoint_sha256="2"*64,
        stage=STAGE_CANONICAL, generated_rgba=rgba,
        target_camera_features=cam, generated_confidence=confidence,
        generation_seed=17,
    )


def test_raw_arbitrary_scenes_do_not_need_precalibrated_front_back_cameras():
    raw=raw_two_scenes()
    assert raw.frames_chw[0].shape==(3,21,29)
    assert raw.frames_chw[1].shape==(4,31,19)
    assert raw.view_hints==("OTHER","UNKNOWN")
    assert len(raw.source_binding_sha256)==64
    before=raw.source_binding_sha256
    with pytest.raises(ValueError):
        raw.frames_chw[0][0,0,0]=0
    assert raw.source_binding_sha256==before
    # Do not mistake generated normal pose views for original measured cameras.
    with pytest.raises((TypeError,AttributeError,QualificationError)):
        canonicalize_two_view_source_v1(raw.frames_chw, None,
            source_rights_claim_sha256="1"*64)


def test_generated_canonical_and_multiview_are_independent_and_unauthorized():
    source=raw_two_scenes()
    canon=canonical(source)
    assert canon.stage==STAGE_CANONICAL
    assert canon.generated_view_count==2
    assert canon.source_pixel_authority is False
    assert canon.camera_calibration_observed is False
    assert canon.canonical_mesh_authority is False
    assert canon.source_surface_or_depth_authority is False
    assert canon.qualified_3d_character is False
    rgba,cams,confidence=candidate_views(6)
    normals=np.zeros((6,3,20,26),dtype=np.float32)
    normals[:,2]=1.
    multi=seal_cleanroom_diffusion_prediction_v1(
        source=source,generator_checkpoint_sha256="3"*64,
        stage=STAGE_MULTIVIEW,generated_rgba=rgba,
        target_camera_features=cams,generated_confidence=confidence,
        generated_normals=normals,generation_seed=99,parent_canonical=canon,
    )
    validate_cleanroom_diffusion_prediction_v1(multi,source=source,parent_canonical=canon)
    assert multi.parent_canonical_prediction_sha256==canon.prediction_hash
    assert multi.source_binding_sha256==source.source_binding_sha256
    assert source.frames_chw[0][0,0,0]==201
    with pytest.raises(QualificationError,match="MULTIVIEW_REQUIRES_CANONICAL_PARENT"):
        validate_cleanroom_diffusion_prediction_v1(multi,source=source)
    with pytest.raises(QualificationError,match="MULTIVIEW_CANONICAL_PARENT_DRIFT"):
        validate_cleanroom_diffusion_prediction_v1(
            multi,source=source,parent_canonical=canonical(
                RSSvNextRawTwoSceneObservationsV1(
                    frames_chw=tuple(x.copy() for x in source.frames_chw),
                    view_hints=("UNKNOWN","UNKNOWN"),
                    source_rights_claim_sha256="1"*64,
                ),
            ),
        )


def test_rehashing_or_tampering_cannot_mint_observed_pixels_or_geometry():
    source=raw_two_scenes()
    canon=canonical(source)
    mutated=replace(canon,source_pixel_authority=True)
    with pytest.raises(QualificationError,match="AUTHORITY"):
        validate_cleanroom_diffusion_prediction_v1(mutated,source=source)
    mutated=replace(canon,canonical_mesh_authority=True)
    with pytest.raises(QualificationError,match="AUTHORITY"):
        validate_cleanroom_diffusion_prediction_v1(mutated,source=source)
    rgba=canon.generated_rgba.copy()
    rgba[0,0,0,0]=15
    mutated=replace(canon,generated_rgba=rgba)
    with pytest.raises(QualificationError,match="PREDICTION_CONTENT_DRIFT"):
        validate_cleanroom_diffusion_prediction_v1(mutated,source=source)
    with pytest.raises(QualificationError,match="PREDICTION_ORIGINAL_SOURCE_LINEAGE_DRIFT"):
        validate_cleanroom_diffusion_prediction_v1(
            canon,source=RSSvNextRawTwoSceneObservationsV1(
                frames_chw=tuple(x.copy() for x in source.frames_chw),
                view_hints=source.view_hints,
                source_rights_claim_sha256="f"*64,
            ),
        )


def test_generated_corrupted_normals_or_confidence_fail_closed():
    src=raw_two_scenes()
    canon=canonical(src)
    rgba,cams,conf=candidate_views(6)
    bad=conf.copy()
    bad[0,0,1,1]=float("nan")
    normals=np.zeros((6,3,20,26),dtype=np.float32)
    with pytest.raises(QualificationError,match="UNCERTAINTY"):
        seal_cleanroom_diffusion_prediction_v1(
            source=src,generator_checkpoint_sha256="3"*64,
            stage=STAGE_MULTIVIEW,generated_rgba=rgba,
            target_camera_features=cams,generated_confidence=bad,
            generated_normals=normals,generation_seed=5,parent_canonical=canon,
        )
    bad_normals=normals.copy()
    bad_normals[1,0,1,1]=float("inf")
    with pytest.raises(QualificationError,match="NORMAL_DOMAIN"):
        seal_cleanroom_diffusion_prediction_v1(
            source=src,generator_checkpoint_sha256="3"*64,
            stage=STAGE_MULTIVIEW,generated_rgba=rgba,
            target_camera_features=cams,generated_confidence=conf,
            generated_normals=bad_normals,generation_seed=5,parent_canonical=canon,
        )
