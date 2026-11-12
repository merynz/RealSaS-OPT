from __future__ import annotations

from dataclasses import replace
import numpy as np
import pytest
import torch

from compiler.realsas_compiler_core.rss_vnext_inference_evidence_v1 import (
    RSSvNextInferenceEvidenceIRV1, seal_inference_evidence_v1,
)
from compiler.realsas_compiler_core.rss_vnext_source_observation_v1 import (
    canonicalize_two_view_source_v1,
)
from compiler.realsas_compiler_core.rss_vnext_source_evidence_binding_v1 import (
    validate_rss_vnext_source_evidence_binding_v1,
)
from compiler.realsas_compiler_core.rss_vnext_source_ray_replay_v1 import (
    replay_original_source_rays_v1,
)
from compiler.realsas_compiler_core.types import QualificationError
from models.rss_vnext.inference_evidence_export_v1 import _source_tensor_digest
from models.rss_vnext.source_ray_evidence import sample_front_back_source_rays
from tests.models.test_rss_vnext_inference_checkpoint_bridge_v1 import _mesh


def _case():
    h,w=16,22
    yy,xx=np.mgrid[:h,:w]
    img=np.empty((2,4,h,w),dtype=np.uint8)
    for v in (0,1):
        img[v,0]=(11*xx+7*yy+v*31)%255
        img[v,1]=(3*xx+29*yy+v*17)%255
        img[v,2]=(19*xx+5*yy+v*13)%255
        img[v,3]=255
    cam=np.zeros((2,13),dtype=np.float32)
    cam[0,[3,7,11,12]]=1.
    cam[1,[3,7,12]]=[-1.,1.,1.]
    cam[1,11]=-1.
    source=canonicalize_two_view_source_v1(
        img,cam,source_rights_claim_sha256="e"*64,
    )
    rgba,cameras=source.to_model_tensors()
    xyz=torch.tensor([[[-.75,.60,0.],[.4,-.35,.5],[.2,.2,-.5],[2.,0.,0.]]])
    sample=sample_front_back_source_rays(rgba,cameras,xyz)
    mesh=_mesh()
    mesh["rest_xyz"]=xyz[0].tolist()
    mesh["base_xyz"]=xyz[0].tolist()
    mesh["local_features"]=mesh["local_features"][:1]*4
    mesh["faces"]=[[0,1,2]]
    mesh["vertex_components"]=[0]*4
    mesh["face_components"]=[0]
    for key in (
        "normals_xyz","semantic_logits","presentation_role_logits",
        "affordance_logits","graphics_logits","static_quality_logits",
    ):
        mesh[key]=mesh[key][:1]*4
    mesh["unsigned_distance"]=[.1]*4
    mesh["sign_logits"]=[.7]*4
    mesh["source_rays"]={
        "rgba":sample.rgba[0].tolist(),
        "valid":sample.projection_valid[0].tolist(),
        "alpha":sample.silhouette_alpha[0].tolist(),
        "appearance_authority":"ORIGINAL_RGBA_CAMERA_RAY_NOT_POINT_TEXTURE",
        "depth_authority":"UNKNOWN_NO_SOURCE_DEPTH",
    }
    ev=seal_inference_evidence_v1(RSSvNextInferenceEvidenceIRV1(
        checkpoint_sha256="b"*64,
        observed_front_sha256=_source_tensor_digest(rgba[0,0]),
        observed_back_sha256=_source_tensor_digest(rgba[0,1]),
        camera_binding_sha256=_source_tensor_digest(cameras),
        mesh_status="UNQUALIFIED_MESH_CANDIDATE",mesh=mesh,
    ))
    bound=validate_rss_vnext_source_evidence_binding_v1(
        source=source,evidence=ev,
        expected_source_binding_sha256=source.source_binding_sha256,
    )
    return source,ev,bound


def test_two_original_views_replay_torch_samples_and_preserve_uncertainty():
    source,evidence,bound=_case()
    result=replay_original_source_rays_v1(
        source=source,evidence=evidence,source_identity_court=bound,
    )
    assert result.checked_ray_count==8
    assert result.max_channel_error<.001
    assert not result.source_surface_ownership
    assert not result.source_depth_or_visibility
    assert not result.material_or_drawable_authority


def test_source_sample_rehash_cannot_hide_changed_rgba():
    src,ev,_=_case()
    mesh=dict(ev.mesh)
    rays=dict(mesh["source_rays"])
    data=[[list(view) for view in row] for row in rays["rgba"]]
    data[0][0][0]+=.05
    rays["rgba"]=data
    mesh["source_rays"]=rays
    tampered=seal_inference_evidence_v1(replace(ev,mesh=mesh,evidence_hash=""))
    court=validate_rss_vnext_source_evidence_binding_v1(
        source=src,evidence=tampered,
        expected_source_binding_sha256=src.source_binding_sha256,
    )
    with pytest.raises(QualificationError,match="ORIGINAL_PIXEL_SAMPLE_DRIFT"):
        replay_original_source_rays_v1(
            source=src,evidence=tampered,source_identity_court=court,
        )


def test_topology_abstain_does_not_synthesize_sampled_color():
    src,ev,_=_case()
    empty=seal_inference_evidence_v1(replace(
        ev,mesh=None,mesh_status="ABSTAIN",reason="NO_TOPOLOGY",evidence_hash="",
    ))
    court=validate_rss_vnext_source_evidence_binding_v1(
        source=src,evidence=empty,
        expected_source_binding_sha256=src.source_binding_sha256,
    )
    result=replay_original_source_rays_v1(
        source=src,evidence=empty,source_identity_court=court,
    )
    assert result.state=="ABSTAIN_NO_TOPOLOGY"
    assert result.checked_ray_count==0
    assert not result.canonical_mesh_authority
